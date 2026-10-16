# Decisions

Why Axiom is built the way it is, roughly in the order the questions came up.

## Built on my own projects, not a framework
Axiom could have been LangChain plus a hosted vector store. Instead it reuses two things I'd already built: the provider adapter and OpenTelemetry tracing from [LLM Insight Engine](https://github.com/VanshajR/LLM-Observability), and the MCP tool pattern from [OpenResponses-MCP](https://github.com/VanshajR/OpenResponses-MCP). The whole agent fits in a few hundred lines I can explain line by line, and both earlier projects get a real workload instead of a demo script.

## No retrieval
Everything Axiom knows is one ~55 KB JSON file the site already publishes. The tools return exactly the part a question needs (one project, one role), so there's no chunking, no embeddings and no stale index. If it ever outgrows that, the tools are the place to add search, not the prompt.

## Tool results as real messages
OpenResponses-MCP pasted earlier tool results back into a growing prompt, because the runtime it targeted couldn't accept tool messages. Groq and Gemini both can, so Axiom sends proper `tool` messages. The model sees structured results, and the conversation stays the shape the APIs expect.

## Model ids are discovered, not configured
In a few weeks Groq retired two Llama models and replaced `qwen3.6` with `qwen3.8`, and Gemini started refusing `gemini-2.5-flash` to new keys while still listing it. A hardcoded model list would have broken three times. The router matches model families by pattern, newest version first, refreshes every 30 minutes, and treats a "no longer available" answer as a reason to bench that model for hours.

## Fallback only before the first word
If a model fails mid-answer, switching to another would show the visitor half an answer followed by a different one. So the router falls back freely until text starts streaming, and after that an error is just an error.

## A 400 isn't always our fault
Gemini's thinking models attach a `thought_signature` to tool calls and reject the next turn without it, as a 400. Groq rejects the same field. So 400s move on to the next model without benching, and the adapter keeps the signature for Gemini and strips it for everyone else, because one conversation can hop providers.

## Hugging Face Space, reused
New Docker Spaces need a paid plan since 2026, but existing ones still run on free CPU. Axiom lives in a Space I already had. Its proxy answers CORS preflights for every origin, so the origin allowlist is enforced again inside `/chat`; the per-visitor rate limit is what actually protects the free model quotas from scripts.

## The MCP server is public
The tools only expose what the site already shows, so there's nothing to protect by keeping them private. Anyone with an MCP client can ask about me through them, which is a better demo of MCP than any screenshot.
