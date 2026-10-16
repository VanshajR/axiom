"""Axiom's tools, defined once as an MCP server.

The agent discovers and calls them through MCP (in-process, no subprocess),
the same pattern as OpenResponses-MCP. Because they're a real MCP server, the
app also exposes them at /mcp, so any MCP client can ask about Vanshaj too.

Tools return compact JSON built only from profile.json. Page anchors ("#project-...")
let answers link to the matching card on the site.
"""
import difflib
import json
import re

from mcp.server.mcpserver import MCPServer

from .profile_data import ProfileStore

store: ProfileStore | None = None  # set by the app at startup

mcp = MCPServer(
    "axiom",
    instructions="Facts about Vanshaj Raghuvanshi: projects, experience, education, skills and contact, "
    "from the data his portfolio site is built from.",
)

BOLD = re.compile(r"\*\*(.+?)\*\*")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _on_site(project: dict) -> bool:
    return project.get("site", project.get("status") != "in-progress")


def _split(highlights: list[dict]) -> dict:
    """Highlights with a **metric** are impact; the rest describe how it was built."""
    impact = [h["text"] for h in highlights if BOLD.search(h["text"])]
    technical = [h["text"] for h in highlights if not BOLD.search(h["text"])]
    return {"impact": impact, "technical": technical}


def _find(name: str, items: list[dict], *keys: str) -> dict | None:
    wanted = name.lower().strip()
    names = {}
    for item in items:
        for key in keys:
            if item.get(key):
                names[str(item[key]).lower()] = item
    if wanted in names:
        return names[wanted]
    for label, item in names.items():
        if wanted in label or (len(label) > 3 and label in wanted):
            return item
    close = difflib.get_close_matches(wanted, list(names), n=1, cutoff=0.6)
    return names[close[0]] if close else None


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


@mcp.tool()
async def get_overview() -> str:
    """Who Vanshaj is: headline, short bio, whether he's open to work, and links. Start here for broad questions."""
    data = await store.get()
    me = data["profile"]
    latest = next((e for e in data["experience"] if e["kind"] == "work"), None)
    return _json({
        "name": me["name"],
        "headline": me["headline"],
        "about": me.get("about") or me["bio"],
        "open_to_work": me.get("open_to_work", False),
        "most_recent_role": latest and f"{latest.get('site_role') or latest['role']}, {latest['org']}",
        "links": {**me["links"], "email": me["email"], "website": me["website"]},
    })


@mcp.tool()
async def list_projects(area: str = "", include_earlier: bool = False) -> str:
    """List Vanshaj's projects, main ones first.

    area: optional filter, one of ai, backend, web, systems, tools, games.
    include_earlier: also list older coursework-style projects.
    """
    data = await store.get()
    projects = [p for p in data["projects"] if _on_site(p) and (include_earlier or p["tier"] != "archive")]
    if area:
        projects = [p for p in projects if area.lower() in p.get("areas", [])]
    return _json([
        {
            "name": p["name"],
            "tier": "main" if p["tier"] == "flagship" else "other" if p["tier"] == "side" else "earlier",
            "status": p["status"],
            "areas": p.get("areas", []),
            "summary": p["summary"],
            "page_anchor": f"#project-{_slug(p['name'])}",
        }
        for p in projects
    ])


@mcp.tool()
async def get_project(name: str) -> str:
    """Full details of one project: what it is, impact, how it's built, stack, links and usage numbers."""
    data = await store.get()
    p = _find(name, [p for p in data["projects"] if _on_site(p)], "name", "slug")
    if not p:
        return _json({"error": f"no project called {name!r}", "known": [p["name"] for p in data["projects"] if _on_site(p)]})
    return _json({
        "name": p["name"],
        "status": p["status"],
        "note": p.get("note"),
        "summary": p["summary"],
        **_split(p["highlights"]),
        "stack": p["tech"],
        "links": p.get("links", {}),
        "usage": {k: v for k, v in (p.get("stats") or {}).items() if k not in ("checked", "window")} or None,
        "page_anchor": f"#project-{_slug(p['name'])}",
    })


@mcp.tool()
async def get_experience(organization: str = "") -> str:
    """Vanshaj's work and leadership roles. With an organization (e.g. "DXFactor", "DRDO"), details of that role."""
    data = await store.get()
    roles = data["experience"]
    if organization:
        e = _find(organization, roles, "org", "short", "id")
        if not e:
            return _json({"error": f"no role at {organization!r}", "known": [r["org"] for r in roles]})
        return _json({
            "role": e.get("site_role") or e["role"],
            "organization": e["org"],
            "kind": e["kind"],
            "start": e["start"],
            "end": e.get("end") or "present",
            "location": e.get("location"),
            "summary": e.get("summary"),
            **_split(e["highlights"]),
            "page_anchor": f"#experience-{e['id']}",
        })
    return _json([
        {
            "role": r.get("site_role") or r["role"],
            "organization": r["org"],
            "kind": r["kind"],
            "start": r["start"],
            "end": r.get("end") or "present",
            "page_anchor": f"#experience-{r['id']}",
        }
        for r in roles
    ])


@mcp.tool()
async def get_education() -> str:
    """Degrees and schooling, with scores."""
    data = await store.get()
    return _json([
        {k: e.get(k) for k in ("degree", "institution", "start", "end", "score", "notes")} | {"page_anchor": f"#education-{e['id']}"}
        for e in data["education"]
    ])


@mcp.tool()
async def get_skills(skill: str = "") -> str:
    """Skills grouped by area. With a skill (e.g. "Rust", "MCP"), which of his projects use it."""
    data = await store.get()
    if not skill:
        return _json(data["skills"]["groups"])
    wanted = re.sub(r"[^a-z0-9+#]", "", skill.lower())
    used = [
        p["name"] for p in data["projects"]
        if _on_site(p) and any(re.sub(r"[^a-z0-9+#]", "", t.lower()) == wanted for t in p["tech"])
    ]
    listed = any(re.sub(r"[^a-z0-9+#]", "", s.lower()) == wanted for group in data["skills"]["groups"].values() for s in group)
    return _json({"skill": skill, "in_skill_list": listed, "used_in_projects": used})


@mcp.tool()
async def get_easter_eggs() -> str:
    """Fun and hidden things to try on Vanshaj's portfolio site."""
    data = await store.get()
    return _json((data.get("site") or {}).get("easter_eggs", []))


@mcp.tool()
async def get_contact() -> str:
    """How to reach Vanshaj. The site's contact form (or Axiom's email flow) can also send him a message."""
    data = await store.get()
    me = data["profile"]
    return _json({"email": me["email"], "linkedin": me["links"].get("linkedin"), "github": me["links"].get("github"),
                  "contact_section": "#contact", "phone": "not shared publicly; ask by email"})


async def openai_tool_schemas() -> list[dict]:
    """The MCP tool list, translated to the function-calling schema both providers accept."""
    return [
        {"type": "function", "function": {"name": t.name, "description": t.description or "", "parameters": t.input_schema}}
        for t in await mcp.list_tools()
    ]


async def call(name: str, arguments: dict) -> str:
    result = await mcp.call_tool(name, arguments)
    text = "".join(getattr(c, "text", "") for c in result.content)
    return text if not result.is_error else _json({"error": text})
