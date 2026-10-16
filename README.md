# Axiom

[![ci](https://github.com/VanshajR/axiom/actions/workflows/ci.yml/badge.svg)](https://github.com/VanshajR/axiom/actions/workflows/ci.yml)
[![MCP](https://img.shields.io/badge/MCP-public%20server-orange)](#use-it-from-an-mcp-client)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

The assistant on [vanshajraghuvanshi.dev](https://vanshajraghuvanshi.dev). Ask it about my projects, work or skills and it answers from the same data the site is built from, linking back to the part of the page it's talking about.

It's also where two of my other projects run for real:

- **[OpenResponses-MCP](https://github.com/VanshajR/OpenResponses-MCP)**: Axiom's tools are an MCP server, and the agent calls them through MCP.
- **[LLM Insight Engine](https://github.com/VanshajR/LLM-Observability)**: every model call is traced with OpenTelemetry and exported to Langfuse.

Requests go to Groq or Gemini, whichever is up; if one is rate-limited or down, the next one answers.

## Use it from an MCP client

The tools are a public, read-only MCP server:

```json
{ "mcpServers": { "axiom": { "url": "https://targaryen001-wyvern.hf.space/mcp/" } } }
```

In Claude: Settings → Connectors → Add custom connector, with the same URL.

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate      # .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"
cp .env.example .env                                # free keys from Groq and Google AI Studio
uvicorn axiom.app:app --port 8100
pytest
```

Why it's built the way it is: [docs/decisions.md](docs/decisions.md).
