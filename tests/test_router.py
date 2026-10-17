import pytest

from axiom.errors import BadRequest, ModelUnavailable, Overloaded, RateLimited, classify
from axiom.providers import OpenAICompatible, TextDelta
from axiom.router import Answering, Router, rank

from .conftest import FakeProvider, say


async def collect(router, messages=None):
    events = []
    async for e in router.stream(messages or [{"role": "user", "content": "hi"}], []):
        events.append(e)
    return events


def test_rank_prefers_newest_and_follows_preference_order():
    groq = FakeProvider("groq", [], [])
    gemini = FakeProvider("gemini", [], [])
    ranked = rank(
        {
            "groq": ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "whisper-large"],
            "gemini": ["gemini-2.5-flash", "gemini-3.8-flash", "gemini-3.10-flash", "gemini-3.8-flash-tts"],
        },
        {"groq": groq, "gemini": gemini},
    )
    keys = [c.key for c in ranked]
    assert keys[0] == "groq/openai/gpt-oss-120b"
    # numeric, not string, ordering: 3.10 is newer than 3.8
    assert keys.index("gemini/gemini-3.10-flash") < keys.index("gemini/gemini-3.8-flash")
    assert not any("whisper" in k or "tts" in k for k in keys)


def test_rank_skips_providers_without_keys():
    gemini = FakeProvider("gemini", [], [])
    ranked = rank({"gemini": ["gemini-3.5-flash"]}, {"gemini": gemini})
    assert [c.key for c in ranked] == ["gemini/gemini-3.5-flash"]


async def test_falls_back_and_benches_the_failed_model():
    groq = FakeProvider("groq", ["openai/gpt-oss-120b"], [RateLimited("groq", "m", "slow down", 429)])
    gemini = FakeProvider("gemini", ["gemini-3.5-flash"], [say("hello")])
    router = Router([groq, gemini])
    await router.refresh()

    events = await collect(router)
    answering = [e for e in events if isinstance(e, Answering)]
    assert answering[0].candidate.key == "gemini/gemini-3.5-flash" and answering[0].attempt == 2
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["hello"]
    assert router.benched_until["groq/openai/gpt-oss-120b"] > 0

    # benched: the next request goes straight to Gemini without touching Groq
    gemini.script.append(say("again"))
    await collect(router)
    assert len(groq.seen) == 1


async def test_bad_request_moves_on_without_benching():
    gemini = FakeProvider("gemini", ["gemini-3.6-flash"], [BadRequest("gemini", "m", "thought_signature", 400)])
    groq = FakeProvider("groq", ["openai/gpt-oss-20b"], [say("ok")])
    router = Router([groq, gemini])
    await router.refresh()
    router.candidates.reverse()  # gemini first for this test
    events = await collect(router)
    assert any(isinstance(e, TextDelta) for e in events)
    assert "gemini/gemini-3.6-flash" not in router.benched_until


async def test_no_fallback_after_text_has_been_sent():
    class HalfAnswer(FakeProvider):
        async def stream(self, model, messages, tools, max_tokens=1200):
            yield TextDelta("partial")
            raise Overloaded(self.name, model, "dropped")

    router = Router([HalfAnswer("groq", ["openai/gpt-oss-120b"], []), FakeProvider("gemini", ["gemini-3.5-flash"], [say("x")])])
    await router.refresh()
    with pytest.raises(Overloaded):
        await collect(router)


async def test_everything_down_raises():
    router = Router([FakeProvider("groq", ["openai/gpt-oss-120b"], [Overloaded("groq", "m", "503", 503)])])
    await router.refresh()
    with pytest.raises(Overloaded):
        await collect(router)


@pytest.mark.parametrize(
    "status, body, kind",
    [
        (429, "", RateLimited),
        (404, "", ModelUnavailable),
        (400, "This model is no longer available to new users", ModelUnavailable),
        (400, "model_decommissioned", ModelUnavailable),
        (503, "high demand", Overloaded),
        (400, "Function call is missing a thought_signature", BadRequest),
    ],
)
def test_classify(status, body, kind):
    assert isinstance(classify("p", "m", status, body), kind)


def test_extra_content_only_goes_to_providers_that_understand_it():
    history = [{"role": "assistant", "tool_calls": [{"id": "1", "function": {}, "extra_content": {"google": {"thought_signature": "x"}}}]}]
    groq = OpenAICompatible("groq", "http://x", "k", http=None)
    gemini = OpenAICompatible("gemini", "http://x", "k", http=None, echoes_extra=True)
    assert "extra_content" not in groq._prepare(history)[0]["tool_calls"][0]
    assert "extra_content" in gemini._prepare(history)[0]["tool_calls"][0]


def test_gemini_accepts_tool_calls_another_provider_made():
    # Groq called a tool, hit its limit, and the answer moved to Gemini
    history = [{"role": "assistant", "tool_calls": [{"id": "1", "function": {}}]}]
    gemini = OpenAICompatible("gemini", "http://x", "k", http=None, echoes_extra=True)
    sent = gemini._prepare(history)[0]["tool_calls"][0]
    assert sent["extra_content"]["google"]["thought_signature"] == "skip_thought_signature_validator"
    assert "extra_content" not in history[0]["tool_calls"][0]  # the shared history is left alone


def test_cerebras_joins_the_ranking_when_it_has_a_key():
    cerebras = FakeProvider("cerebras", [], [])
    groq = FakeProvider("groq", [], [])
    ranked = rank(
        {"groq": ["openai/gpt-oss-120b"], "cerebras": ["gpt-oss-120b", "qwen-3-235b-a22b-instruct-2507", "llama3.1-8b"]},
        {"groq": groq, "cerebras": cerebras},
    )
    keys = [c.key for c in ranked]
    assert keys[:2] == ["groq/openai/gpt-oss-120b", "cerebras/gpt-oss-120b"]
    assert "cerebras/qwen-3-235b-a22b-instruct-2507" in keys and "cerebras/llama3.1-8b" not in keys


@pytest.mark.parametrize(
    "body, headers, seconds",
    [
        ("Rate limit reached ... Please try again in 472.5ms. Need more tokens?", None, 0.4725),
        ("Please try again in 1m26.4s.", None, 86.4),
        ('{"error": {"details": [{"retryDelay": "37s"}]}}', None, 37.0),
        ("slow down", {"retry-after": "2"}, 2.0),
        ("slow down", None, None),
    ],
)
def test_rate_limits_say_how_long_to_wait(body, headers, seconds):
    error = classify("groq", "m", 429, body, headers)
    assert isinstance(error, RateLimited)
    assert error.retry_after == pytest.approx(seconds) if seconds is not None else error.retry_after is None


async def test_a_short_rate_limit_waits_for_the_same_model_instead_of_falling_back():
    groq = FakeProvider("groq", ["openai/gpt-oss-120b"], [
        RateLimited("groq", "m", "try again in 50ms", 429, retry_after=0.05),
        say("from groq"),
    ])
    gemini = FakeProvider("gemini", ["gemini-3.5-flash"], [say("from gemini")])
    router = Router([groq, gemini])
    await router.refresh()
    events = await collect(router)
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["from groq"]
    assert gemini.seen == [] and "groq/openai/gpt-oss-120b" not in router.benched_until


async def test_a_long_rate_limit_benches_only_as_long_as_asked():
    groq = FakeProvider("groq", ["openai/gpt-oss-120b"], [RateLimited("groq", "m", "x", 429, retry_after=20)])
    gemini = FakeProvider("gemini", ["gemini-3.5-flash"], [say("from gemini")])
    router = Router([groq, gemini])
    await router.refresh()
    import time
    before = time.time()
    await collect(router)
    assert 18 < router.benched_until["groq/openai/gpt-oss-120b"] - before < 22
