import json

from axiom import agent, tools
from axiom.router import Router

from .conftest import FakeProvider, call, say


async def run(router, message="hi"):
    events = []
    async for e in agent.run(message, [], router):
        events.append(e)
    return events


async def test_tools_are_discovered_through_mcp():
    names = {s["function"]["name"] for s in await tools.openai_tool_schemas()}
    assert {"get_overview", "list_projects", "get_project", "get_experience", "get_skills", "get_contact"} <= names


async def test_project_details_split_impact_and_technical():
    result = json.loads(await tools.call("get_project", {"name": "ev digital twin"}))
    assert result["name"] == "EV Digital Twin Platform"
    assert any("**601 vehicles**" in h for h in result["impact"])
    assert all("**" not in h for h in result["technical"])
    assert result["page_anchor"] == "#project-ev-digital-twin-platform"


async def test_in_progress_projects_stay_hidden_unless_marked_for_the_site():
    names = [p["name"] for p in json.loads(await tools.call("list_projects", {}))]
    assert "Axiom" in names  # site: true
    assert "AniKit" not in names  # in progress, not on the site
    assert "error" in json.loads(await tools.call("get_project", {"name": "AniKit"}))


async def test_experience_uses_the_site_wording_and_finds_short_names():
    drdo = json.loads(await tools.call("get_experience", {"organization": "drdo"}))
    assert drdo["role"] == "AI Research and Development Intern"
    assert drdo["page_anchor"] == "#experience-drdo"


async def test_skill_lookup_lists_projects_that_use_it():
    rust = json.loads(await tools.call("get_skills", {"skill": "rust"}))
    assert rust["used_in_projects"] == ["depthpad"]


async def test_contact_never_shares_a_phone_number(profile):
    assert "phone" not in json.dumps(profile["profile"])
    contact = json.loads(await tools.call("get_contact", {}))
    assert contact["phone"].startswith("not shared")


async def test_agent_calls_a_tool_then_answers():
    provider = FakeProvider("groq", ["openai/gpt-oss-120b"], [
        call("get_project", {"name": "depthpad"}),
        say("depthpad turns key depth into stick travel."),
    ])
    router = Router([provider])
    await router.refresh()
    events = await run(router, "what is depthpad?")

    assert [e["name"] for e in events if e["type"] == "tool"] == ["get_project"]
    assert "".join(e["text"] for e in events if e["type"] == "text").startswith("depthpad")
    done = events[-1]
    assert done["type"] == "done" and done["tools"] == 1 and done["model"] == "groq/openai/gpt-oss-120b"

    # the second turn saw the tool result as a proper tool message, not pasted text
    second_turn = provider.seen[1][1]
    tool_msg = next(m for m in second_turn if m["role"] == "tool")
    assert json.loads(tool_msg["content"])["name"] == "depthpad"


async def test_agent_echoes_gemini_thought_signatures():
    signature = {"google": {"thought_signature": "sig"}}
    provider = FakeProvider("gemini", ["gemini-3.6-flash"], [call("get_overview", {}, extra=signature), say("done")])
    router = Router([provider])
    await router.refresh()
    await run(router)
    assistant = next(m for m in provider.seen[1][1] if m.get("tool_calls"))
    assert assistant["tool_calls"][0]["extra_content"] == signature


async def test_agent_survives_a_broken_tool_call():
    provider = FakeProvider("groq", ["openai/gpt-oss-120b"], [call("get_project", "not an object"), say("sorry")])
    router = Router([provider])
    await router.refresh()
    events = await run(router)
    assert events[-1]["type"] == "done"


async def test_agent_reports_an_error_when_no_model_answers():
    router = Router([])
    events = await run(router)
    assert events[-1]["type"] == "error"


async def test_profile_falls_back_to_the_bundled_snapshot(tmp_path, monkeypatch):
    from axiom.profile_data import ProfileStore

    snapshot = tmp_path / "snap.json"
    snapshot.write_text('{"profile": {"name": "from snapshot"}}', encoding="utf-8")
    monkeypatch.setenv("PROFILE_SNAPSHOT", str(snapshot))
    monkeypatch.delenv("PROFILE_PATH", raising=False)

    class Unreachable:
        async def get(self, *a, **k):
            raise ConnectionError("site down")

    store = ProfileStore("https://example.invalid/profile.json", http=Unreachable())
    assert (await store.get())["profile"]["name"] == "from snapshot"


def test_chat_rejects_other_sites_even_if_a_proxy_allowed_the_preflight(monkeypatch):
    from fastapi.testclient import TestClient
    from axiom import app as app_module

    import dataclasses

    site_only = dataclasses.replace(app_module.settings, allowed_origins=["https://vanshajraghuvanshi.dev"])
    monkeypatch.setattr(app_module, "settings", site_only)
    client = TestClient(app_module.app)  # no lifespan: the request is rejected before any model is needed
    r = client.post("/chat", json={"message": "hi"}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


async def test_contact_mode_uses_its_own_prompt_and_no_tools():
    class Recorder(FakeProvider):
        async def stream(self, model, messages, tools):
            self.tools_offered = tools
            async for e in super().stream(model, messages, tools):
                yield e

    provider = Recorder("groq", ["openai/gpt-oss-120b"], [say("Happy to help. What's your name?")])
    router = Router([provider])
    await router.refresh()
    events = [e async for e in agent.run("I want to email him", [], router, mode="contact")]
    assert provider.tools_offered == []
    assert "CONTACT_READY" in provider.seen[0][1][0]["content"]
    assert events[-1]["type"] == "done"
