"""HTTP API.

  POST /chat    streams an answer as server-sent events (text, tool, done, error)
  GET  /health  status plus the models currently in rotation
  GET  /stats   rolling numbers: latency, fallback rescues, tool usage
       /mcp     the same tools as a public MCP server (streamable HTTP)
"""
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field

from . import agent, tools
from .config import settings
from .observability import init_tracing, stats
from .profile_data import ProfileStore
from .providers import build_providers
from .ratelimit import RateLimiter
from .router import Router

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

# Public and read-only: the tools only return what the site already publishes, so
# the DNS-rebinding guard (meant for local MCP servers) would only block the
# server's own hostname.
mcp_http = tools.mcp.streamable_http_app(
    streamable_http_path="/",
    stateless_http=True,
    json_response=True,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


_router: Router | None = None
_starting = asyncio.Lock()


async def ready() -> Router:
    """Set everything up on first use. Long-running servers do it at startup (lifespan);
    on serverless the first request does it, and a warm instance keeps it."""
    global _router
    if _router is not None:
        return _router
    async with _starting:
        if _router is None:
            init_tracing()
            http = httpx.AsyncClient()
            tools.store = ProfileStore(settings.profile_url, http)
            router = Router(build_providers(settings, http))
            await router.refresh()
            asyncio.create_task(router.refresh_forever())
            mcp_up = asyncio.Event()
            asyncio.create_task(_serve_mcp(mcp_up))
            await mcp_up.wait()
            _router = router
    return _router


async def _serve_mcp(up: asyncio.Event) -> None:
    # the MCP transport needs its session manager running for as long as the app lives
    async with tools.mcp.session_manager.run():
        up.set()
        await asyncio.Event().wait()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ready()
    yield


app = FastAPI(title="Axiom", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.mount("/mcp", mcp_http)


class ExactMcpPath:
    """Serve POST /mcp itself. A mount only matches /mcp/..., so Starlette would answer
    /mcp with a redirect, and MCP clients generally don't follow redirects on a POST."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] == "/mcp":
            scope = {**scope, "path": "/mcp/", "raw_path": b"/mcp/"}
        if scope["type"] == "http" and scope["path"].startswith("/mcp"):
            await ready()
        await self.app(scope, receive, send)


app.add_middleware(ExactMcpPath)
limiter = RateLimiter(settings.requests_per_minute)


class Turn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[Turn] = Field(default_factory=list, max_length=40)
    mode: Literal["chat", "contact"] = "chat"


def _visitor(request: Request) -> str:
    # behind a proxy (Vercel, Docker hosts) the real address is the first X-Forwarded-For entry
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


@app.post("/chat")
async def chat(body: ChatRequest, request: Request):
    # Some proxies answer CORS preflights themselves and allow any origin, so
    # the allowlist is enforced here too. (A script can fake Origin; the rate
    # limit below is what protects the free quotas from those.)
    origin = request.headers.get("origin")
    if origin and origin not in settings.allowed_origins:
        raise HTTPException(403, "Axiom only answers on vanshajraghuvanshi.dev.")
    if len(body.message) > settings.max_message_chars:
        raise HTTPException(413, f"Keep messages under {settings.max_message_chars} characters.")
    if not limiter.allow(_visitor(request)):
        raise HTTPException(429, "That's a lot of questions at once. Give it a minute.")

    router = await ready()

    async def events():
        async for event in agent.run(body.message, [t.model_dump() for t in body.history], router, body.mode):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/health")
async def health(request: Request):
    router = await ready()
    return {"ok": bool(router.candidates), "models": [c.key for c in router.candidates]}


@app.get("/stats")
async def get_stats():
    return stats.summary()
