"""Picks a model for each turn and falls back when one fails.

Model ids are discovered from each provider at startup (and every 30 minutes)
and matched against preference patterns, newest version first, so a renamed or
retired model doesn't take the assistant down. Gemini lists models it won't
serve to new keys, so discovery isn't trusted blindly either: a model that
answers "unavailable" is benched for hours, a rate-limited one for a minute.
"""
import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from .errors import AuthFailed, ModelUnavailable, Overloaded, ProviderError, RateLimited
from .observability import model_call_span, record_usage, stats
from .providers import ChatProvider, Event, TextDelta, TurnEnd

log = logging.getLogger("axiom.router")

# Order matters: fast and capable first, then the other provider, then the small fallbacks.
PREFERENCES: list[tuple[str, str]] = [
    ("groq", r"openai/gpt-oss-120b"),
    ("cerebras", r"gpt-oss-120b"),  # same model, separate free quota
    # Groq's per-minute token limit is per model, so its second model doubles the budget
    ("groq", r"qwen/qwen\d+(\.\d+)*-\d+b"),
    ("gemini", r"gemini-\d+(\.\d+)?-flash"),
    ("gemini", r"gemini-flash-latest"),
    ("cerebras", r"qwen-3-\d+b-a\d+b-instruct.*"),
    ("gemini", r"gemini-\d+(\.\d+)?-flash-lite"),
    # last: in evals it answered from thin air instead of from the facts it was given
    ("groq", r"openai/gpt-oss-20b"),
]
PER_PATTERN = 3  # newest three versions of each family, e.g. gemini-3.8-flash, 3.7, 3.6

SHORT_WAIT_S = 3.0  # wait this long for the same model rather than fall back
COOLDOWN_S = {RateLimited: 60, Overloaded: 30, ModelUnavailable: 6 * 3600, AuthFailed: 3600}


@dataclass(frozen=True)
class Candidate:
    provider: ChatProvider
    model: str

    @property
    def key(self) -> str:
        return f"{self.provider.name}/{self.model}"


@dataclass(frozen=True)
class Answering:
    """Sent before a model's events: who is answering, and on which attempt."""

    candidate: Candidate
    attempt: int


def _version_key(model: str) -> tuple:
    return tuple(int(n) for n in re.findall(r"\d+", model))


def rank(available: dict[str, list[str]], providers: dict[str, ChatProvider]) -> list[Candidate]:
    ranked, seen = [], set()
    for provider_name, pattern in PREFERENCES:
        if provider_name not in providers:
            continue
        matches = [m for m in available.get(provider_name, []) if re.fullmatch(pattern, m)]
        for model in sorted(matches, key=_version_key, reverse=True)[:PER_PATTERN]:
            if (provider_name, model) not in seen:
                seen.add((provider_name, model))
                ranked.append(Candidate(providers[provider_name], model))
    return ranked


class Router:
    def __init__(self, providers: list[ChatProvider]):
        self.providers = {p.name: p for p in providers}
        self.candidates: list[Candidate] = []
        self.benched_until: dict[str, float] = {}

    async def refresh(self) -> None:
        available = {}
        for name, provider in self.providers.items():
            try:
                available[name] = await provider.list_models()
            except Exception as e:  # one provider being down shouldn't empty the list
                log.warning("model discovery failed for %s: %s", name, e)
        if available:
            self.candidates = rank(available, self.providers)
        log.info("candidates: %s", [c.key for c in self.candidates])

    async def refresh_forever(self, every_s: int = 1800) -> None:
        while True:
            await asyncio.sleep(every_s)
            await self.refresh()

    def _usable(self, now: float) -> list[Candidate]:
        return [c for c in self.candidates if self.benched_until.get(c.key, 0) <= now]

    def _bench(self, candidate: Candidate, error: ProviderError) -> None:
        stats.failures[f"{candidate.key} ({type(error).__name__})"] += 1
        seconds = COOLDOWN_S.get(type(error), 0)
        if isinstance(error, RateLimited) and error.retry_after is not None:
            # bench for as long as the provider asked, not a flat minute
            seconds = min(max(error.retry_after, 1.0), 6 * 3600)
        if not seconds:
            return
        if isinstance(error, AuthFailed):  # the key is bad for every model of that provider
            for c in self.candidates:
                if c.provider is candidate.provider:
                    self.benched_until[c.key] = time.time() + seconds
        self.benched_until[candidate.key] = time.time() + seconds

    async def stream(self, messages: list[dict], tools: list[dict], max_tokens: int = 1200) -> AsyncIterator[Event | Answering]:
        """Yields who's answering (Answering), then that model's events.

        Falls back only before any text has gone out: once a visitor has seen
        half an answer, restarting with another model would repeat or contradict it.
        """
        attempts = 0
        last_error: ProviderError | None = None
        queue = list(self._usable(time.time()))
        waited: set[str] = set()
        while queue:
            candidate = queue.pop(0)
            attempts += 1
            sent_text = False
            with model_call_span(candidate.provider.name, candidate.model) as span:
                try:
                    announced = False
                    async for event in candidate.provider.stream(candidate.model, messages, tools, max_tokens):
                        if not announced:
                            yield Answering(candidate, attempts)
                            announced = True
                        if isinstance(event, TextDelta):
                            sent_text = True
                        if isinstance(event, TurnEnd):
                            record_usage(span, event.usage, event.finish_reason)
                        yield event
                    span.set_attribute("axiom.fallback_attempts", attempts - 1)
                    return
                except ProviderError as error:
                    span.set_attribute("error.type", type(error).__name__)
                    log.warning("%s failed: %s", candidate.key, error)
                    last_error = error
                    if not error.retryable or sent_text:
                        raise
                    if (isinstance(error, RateLimited) and error.retry_after is not None
                            and error.retry_after <= SHORT_WAIT_S and candidate.key not in waited):
                        # a per-minute limit that frees up in a moment: the best model is worth a short wait
                        waited.add(candidate.key)
                        stats.failures[f"{candidate.key} (waited)"] += 1
                        await asyncio.sleep(error.retry_after + 0.1)
                        queue.insert(0, candidate)
                        continue
                    self._bench(candidate, error)
        raise last_error or Overloaded("router", "*", "no model available right now")
