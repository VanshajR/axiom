"""Provider adapters.

Same idea as the adapter in LLM Insight Engine (one interface per provider,
no tracing inside), extended from single prompts to chat with tool calls and
streaming. Groq and Gemini both speak the OpenAI chat-completions dialect, so
one adapter covers both; a provider that doesn't would get its own class.
"""
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx

from .errors import Overloaded, classify


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON, as the model wrote it
    # provider-specific data that must be echoed back with the call, e.g. Gemini's
    # thought_signature; without it Gemini rejects the next turn with a 400
    extra: dict | None = None


@dataclass
class TurnEnd:
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)


Event = TextDelta | TurnEnd


class ChatProvider(ABC):
    name: str

    @abstractmethod
    async def list_models(self) -> list[str]: ...

    @abstractmethod
    def stream(self, model: str, messages: list[dict], tools: list[dict], max_tokens: int = 1200) -> AsyncIterator[Event]: ...


# A healthy model starts streaming within a few seconds (Gemini's thinking included).
# An overloaded one can sit silent for minutes, so give up after 20s of silence and
# let the router try the next model; the silent one gets benched for a while.
STREAM_TIMEOUT = httpx.Timeout(20.0, connect=10.0)

# https://ai.google.dev/gemini-api/docs/thought-signatures (calls not made by Gemini)
FOREIGN_CALL = {"google": {"thought_signature": "skip_thought_signature_validator"}}


class OpenAICompatible(ChatProvider):
    def __init__(self, name: str, base_url: str, api_key: str, http: httpx.AsyncClient, echoes_extra: bool = False):
        self.name = name
        self.echoes_extra = echoes_extra  # whether this provider understands tool-call extra_content
        self.base_url = base_url.rstrip("/")
        self.http = http
        self._headers = {"Authorization": f"Bearer {api_key}"}

    async def list_models(self) -> list[str]:
        r = await self.http.get(f"{self.base_url}/models", headers=self._headers, timeout=15)
        if r.status_code != 200:
            raise classify(self.name, "*", r.status_code, r.text)
        # Gemini's compat endpoint answers "models/gemini-...", Groq answers bare ids
        return [m["id"].removeprefix("models/") for m in r.json().get("data", [])]

    async def stream(self, model: str, messages: list[dict], tools: list[dict], max_tokens: int = 1200) -> AsyncIterator[Event]:
        body = {"model": model, "messages": self._prepare(messages), "stream": True, "temperature": 0.3, "max_tokens": max_tokens}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        calls: dict[int, dict] = {}  # tool calls arrive in pieces, keyed by index
        finish, usage = None, {}
        try:
            async with self.http.stream(
                "POST", f"{self.base_url}/chat/completions", json=body, headers=self._headers, timeout=STREAM_TIMEOUT
            ) as r:
                if r.status_code != 200:
                    raise classify(self.name, model, r.status_code, (await r.aread()).decode(errors="replace"))
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    usage = chunk.get("usage") or chunk.get("x_groq", {}).get("usage") or usage
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            yield TextDelta(delta["content"])
                        for piece in delta.get("tool_calls") or []:
                            slot = calls.setdefault(piece.get("index", len(calls)), {"id": "", "name": "", "arguments": "", "extra": None})
                            slot["id"] = piece.get("id") or slot["id"]
                            slot["extra"] = piece.get("extra_content") or slot["extra"]
                            fn = piece.get("function") or {}
                            slot["name"] += fn.get("name") or ""
                            slot["arguments"] += fn.get("arguments") or ""
                        finish = choice.get("finish_reason") or finish
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as e:
            raise Overloaded(self.name, model, type(e).__name__) from e

        tool_calls = [
            ToolCall(id=c["id"] or f"call_{i}", name=c["name"], arguments=c["arguments"] or "{}", extra=c["extra"])
            for i, c in sorted(calls.items())
        ]
        yield TurnEnd(tool_calls=tool_calls, finish_reason=finish, usage=usage)


    def _prepare(self, messages: list[dict]) -> list[dict]:
        """Earlier turns may have come from another provider; drop extras this one would reject."""
        if self.echoes_extra:
            # Gemini wants a thought signature on every earlier tool call. Calls another
            # provider made have none, so they get the placeholder Google documents for
            # exactly that case (a request that falls over to Gemini mid-answer).
            return [
                {**m, "tool_calls": [c if c.get("extra_content") else {**c, "extra_content": FOREIGN_CALL}
                                     for c in m["tool_calls"]]}
                if m.get("tool_calls") else m
                for m in messages
            ]
        cleaned = []
        for m in messages:
            if m.get("tool_calls"):
                m = {**m, "tool_calls": [{k: v for k, v in c.items() if k != "extra_content"} for c in m["tool_calls"]]}
            cleaned.append(m)
        return cleaned


def build_providers(settings, http: httpx.AsyncClient) -> list[ChatProvider]:
    providers = []
    if settings.groq_api_key:
        providers.append(OpenAICompatible("groq", "https://api.groq.com/openai/v1", settings.groq_api_key, http))
    if settings.cerebras_api_key:
        providers.append(OpenAICompatible("cerebras", "https://api.cerebras.ai/v1", settings.cerebras_api_key, http))
    if settings.gemini_api_key:
        providers.append(OpenAICompatible(
            "gemini", "https://generativelanguage.googleapis.com/v1beta/openai", settings.gemini_api_key, http,
            echoes_extra=True,
        ))
    return providers
