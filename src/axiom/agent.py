"""The agent loop: model -> MCP tools -> model, until there's an answer.

Same shape as the loop in OpenResponses-MCP, but tool results go back as real
`tool` messages instead of being pasted into a growing prompt (that workaround
was only needed because the Responses runtime couldn't accept them).
"""
import json
import time
from collections.abc import AsyncIterator

from . import tools
from .config import settings
from .observability import stats, tracer
from .providers import TextDelta, TurnEnd
from .router import Answering, Router

SYSTEM_PROMPT = """You are Axiom, the assistant on Vanshaj Raghuvanshi's portfolio site. Visitors are mostly recruiters and engineers.

Facts:
- Answer only from your tools. If they don't cover something, say you don't know and point to the contact section (#contact). Never guess numbers, dates, employers or skills.
- Use the tools for anything about Vanshaj; don't rely on earlier turns for facts you can look up again.

Voice:
- Plain, warm and direct, like a colleague who knows his work. Refer to him as Vanshaj or he.
- No emoji headers, no sign-offs, no "great question".
- Keep it short: a sentence or two for simple questions. Expand only when asked for detail.

Format (markdown):
- For a project or a role, give one line on what it is, then **Impact** and **Technical** bullets from the tool's impact and technical lists, keeping the **bold** metrics. Leave out a heading if its list is empty.
- End answers about something on the page with a link to its page_anchor, written as [Show on page](#anchor).
- Link to code, live apps and profiles with the URLs the tools return.

If someone asks about you, Axiom: you're a project of his too ("axiom" in the projects), answer from that entry and say "that's me".
If a question isn't about Vanshaj, give a one-line friendly redirect."""


async def run(message: str, history: list[dict], router: Router) -> AsyncIterator[dict]:
    """Yields events for the client: text, tool, done, error."""
    started = time.perf_counter()
    first_token_at = None
    stats.requests += 1

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # only plain turns from the visitor's side; tool traffic stays server-side
    for turn in history[-10:]:
        if turn.get("role") in ("user", "assistant") and isinstance(turn.get("content"), str):
            messages.append({"role": turn["role"], "content": turn["content"][: settings.max_message_chars * 2]})
    messages.append({"role": "user", "content": message})
    schemas = await tools.openai_tool_schemas()

    answered_by, attempts, tool_count = None, 1, 0
    with tracer.start_as_current_span("axiom.chat") as span:
        try:
            for _turn in range(settings.max_turns):
                text, end = "", None
                async for event in router.stream(messages, schemas):
                    if isinstance(event, Answering):
                        answered_by, attempts = event.candidate.key, max(attempts, event.attempt)
                    elif isinstance(event, TextDelta):
                        first_token_at = first_token_at or time.perf_counter()
                        text += event.text
                        yield {"type": "text", "text": event.text}
                    elif isinstance(event, TurnEnd):
                        end = event

                if not end or not end.tool_calls:
                    break

                messages.append({
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments},
                         **({"extra_content": c.extra} if c.extra else {})}
                        for c in end.tool_calls
                    ],
                })
                for call in end.tool_calls:
                    tool_count += 1
                    stats.tool_calls[call.name] += 1
                    yield {"type": "tool", "name": call.name}
                    try:
                        args = json.loads(call.arguments or "{}")
                        result = await tools.call(call.name, args if isinstance(args, dict) else {})
                    except Exception as e:  # a bad call shouldn't end the conversation; the model can recover
                        result = json.dumps({"error": f"{type(e).__name__}: {e}"})
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            else:
                yield {"type": "text", "text": "\n\nI went in circles on that one. Could you ask it another way?"}

            latency_ms = (time.perf_counter() - started) * 1000
            stats.latencies_ms.append(latency_ms)
            if first_token_at:
                stats.first_token_ms.append((first_token_at - started) * 1000)
            if answered_by:
                stats.answered_by[answered_by] += 1
            if attempts > 1:
                stats.rescued_by_fallback += 1
            span.set_attribute("axiom.answered_by", answered_by or "")
            span.set_attribute("axiom.tool_calls", tool_count)
            yield {"type": "done", "model": answered_by, "attempts": attempts,
                   "tools": tool_count, "latency_ms": round(latency_ms)}
        except Exception as e:
            stats.failed += 1
            span.record_exception(e)
            yield {"type": "error", "message": "Axiom can't reach a model right now. Please try again in a minute."}
