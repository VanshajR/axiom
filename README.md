# Axiom

[![ci](https://github.com/VanshajR/axiom/actions/workflows/ci.yml/badge.svg)](https://github.com/VanshajR/axiom/actions/workflows/ci.yml)
[![MCP](https://img.shields.io/badge/MCP-public%20server-orange)](#use-it-from-an-mcp-client)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

The assistant on [vanshajraghuvanshi.dev](https://vanshajraghuvanshi.dev). Ask it about my projects, work or skills and it answers from the same data the site is built from, linking back to the part of the page it's talking about.

It's also where two of my other projects run for real:

- **[OpenResponses-MCP](https://github.com/VanshajR/OpenResponses-MCP)**: Axiom's tools are an MCP server. The agent discovers and calls them through MCP, and the same server is public at `/mcp`, so any MCP client (Claude, Cursor, ...) can ask about me too.
- **[LLM Insight Engine](https://github.com/VanshajR/LLM-Observability)**: every model call goes through a provider adapter and is traced as an OpenTelemetry span with GenAI semantic conventions, exported to Langfuse.

## How a question gets answered

```
visitor ── POST /chat ──▶ agent loop ──▶ router ──▶ Groq / Gemini (OpenAI-compatible APIs)
                            │   ▲            │
                            ▼   │            └─ discovers models at startup, falls back on
                       MCP tools (in-process)   rate limits, 503s and retired models
                            │
                       profile.json (the site's data)
```

1. The agent sends the conversation and the MCP tool list to the router.
2. The router tries the best available model. A 429, 503, timeout or "model no longer available" benches that model for a while and the next one gets the turn. Once text has started streaming it won't switch, so a visitor never sees two half-answers.
3. Tool calls run against the MCP server; results go back as real `tool` messages until the model answers.
4. The answer streams back as server-sent events, with which model answered and how many tries it took.

## Things that went wrong and shaped it

- **Model ids don't last.** In the weeks this was built, Groq retired two Llama models and replaced `qwen3.6` with `qwen3.8`, and Gemini started refusing `gemini-2.5-flash` to new keys while still listing it. So nothing here names an exact model: the router matches families by pattern, newest version first, and treats "listed" as a hint rather than a promise.
- **Gemini's thinking models sign their tool calls.** Each call carries a `thought_signature` that has to come back on the next turn or the request is rejected. Groq rejects the same field. The adapter keeps it for Gemini and strips it for everyone else, since one conversation can hop providers mid-way.
- **Free tiers are shared by every visitor**, so there's a per-visitor rate limit and an origin allowlist.

## Numbers

`GET /stats` reports rolling p50/p95 latency, time to first token, which models answered, provider failures, and how many requests fallback rescued. (Figures go here once it has real traffic.)

## Use it from an MCP client

The tools are a public, read-only MCP server (streamable HTTP):

```
https://targaryen001-wyvern.hf.space/mcp/
```

In Cursor or any client that takes a URL:

```json
{ "mcpServers": { "axiom": { "url": "https://targaryen001-wyvern.hf.space/mcp/" } } }
```

In Claude: Settings → Connectors → Add custom connector, with the same URL.

## Why it's built this way

[docs/decisions.md](docs/decisions.md) covers the choices: no framework, no retrieval, discovered model ids, when fallback is allowed, and the Gemini quirks.

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate      # .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"
cp .env.example .env                                # add the two free keys
uvicorn axiom.app:app --port 8100
pytest                                              # no network, no keys needed
```

Deployed to a Hugging Face Docker Space with `scripts/deploy_space.py`; a GitHub Actions cron pings `/health` so it doesn't sleep.
