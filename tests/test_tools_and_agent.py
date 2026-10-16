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


async def test_contact_mode_only_offers_send_email():
    class Recorder(FakeProvider):
        async def stream(self, model, messages, tools, max_tokens=1200):
            self.tools_offered = tools
            async for e in super().stream(model, messages, tools, max_tokens):
                yield e

    provider = Recorder("groq", ["openai/gpt-oss-120b"], [say("Happy to help. What's your name?")])
    router = Router([provider])
    await router.refresh()
    events = [e async for e in agent.run("I want to email him", [], router, mode="contact")]
    assert [t["function"]["name"] for t in provider.tools_offered] == ["send_email"]
    assert "send_email" in provider.seen[0][1][0]["content"]
    assert events[-1]["type"] == "done"


def test_persona_comes_from_the_site_data(profile):
    prompt = agent.system_prompt(profile)
    assert "Vanshaj Raghuvanshi designed and built you" in prompt
    assert profile["site"]["assistant"]["personality"] in prompt
    assert "start_email_to_vanshaj" in prompt and "show_on_page" in prompt


async def test_site_actions_reach_the_client_as_events():
    provider = FakeProvider("groq", ["openai/gpt-oss-120b"], [
        call("show_on_page", {"target": "Projects"}, id_="a"),
        say("Here are his projects."),
    ])
    router = Router([provider])
    await router.refresh()
    events = [e async for e in agent.run("take me to his projects", [], router)]

    actions = [e for e in events if e["type"] == "action"]
    assert actions == [{"type": "action", "name": "show_on_page", "args": {"target": "projects"}}]
    # the model was told it happened, so it doesn't claim something it didn't do
    tool_msg = next(m for m in provider.seen[1][1] if m["role"] == "tool")
    assert json.loads(tool_msg["content"])["ok"] is True


async def test_page_anchors_keep_their_hash_and_bad_cards_are_refused():
    from axiom import site_actions

    assert site_actions.normalise("show_on_page", {"target": "project-depthpad"}) == {"target": "#project-depthpad"}
    assert site_actions.normalise("show_profile_card", {"platform": "myspace"}) is None


async def test_site_actions_are_not_on_the_public_mcp_server():
    names = {s["function"]["name"] for s in await tools.openai_tool_schemas()}
    assert "show_on_page" not in names and "start_email_to_vanshaj" not in names
    assert "get_easter_eggs" in names


async def test_easter_eggs_come_from_the_data(profile):
    eggs = json.loads(await tools.call("get_easter_eggs", {}))
    assert eggs == profile["site"]["easter_eggs"]


async def test_email_is_sent_through_a_checked_tool_call():
    good = {"name": "Priya", "email": "priya@acme.io", "company": "Acme", "role": "Recruiter",
            "purpose": "hiring", "message": "Hi Vanshaj, we would love to talk about an AI engineer role."}
    provider = FakeProvider("groq", ["openai/gpt-oss-120b"], [
        call("send_email", {**good, "email": "priya@acme"}, id_="bad"),  # not a valid address
        call("send_email", good, id_="ok"),
        say("Thanks Priya, it is on its way."),
    ])
    router = Router([provider])
    await router.refresh()
    said = "Hi Vanshaj, we would love to talk about an AI engineer role."
    events = [e async for e in agent.run(said, [], router, mode="contact")]

    assert [e for e in events if e["type"] == "action"] == [{"type": "action", "name": "send_email", "args": good}]
    # the bad address went back to the model as something to fix, not to the site
    first_result = next(m for m in provider.seen[1][1] if m["role"] == "tool")
    assert "isn't a valid email" in first_result["content"]


def test_check_email_requires_name_address_and_message():
    from axiom.site_actions import check_email

    assert check_email({"name": "", "email": "a@b.co", "message": "hello there"})[1] == "missing their name"
    assert check_email({"name": "A", "email": "a@b.co", "message": ""})[1] == "missing the message itself"
    fields, problem = check_email({"name": "A", "email": "a@b.co", "message": "hello there"})
    assert problem is None and fields["purpose"] == "general"


def test_drafts_must_come_from_what_the_visitor_wrote():
    from axiom.site_actions import check_email

    visitor = "Priya. priya@acme.io. Acme AI, recruiter. We're hiring an AI engineer for agent tooling, open to a call next week?"
    ok = {"name": "Priya", "email": "priya@acme.io",
          "message": "Hi Vanshaj, we're hiring an AI engineer for agent tooling at Acme AI. Are you open to a call next week?"}
    invented = {"name": "Priya", "email": "priya@acme.io",
                "message": "I was impressed by your full-stack development and product design work and have an exciting senior role."}
    assert check_email(ok, visitor)[1] is None
    assert "made up" in check_email(invented, visitor)[1]


async def test_email_flow_only_shows_the_final_attempt():
    provider = FakeProvider("groq", ["openai/gpt-oss-120b"], [
        ("Drafting now...", [__import__("axiom.providers", fromlist=["ToolCall"]).ToolCall(
            id="x", name="send_email", arguments='{"name": "A", "email": "a@b.co", "message": "an invented pitch about synergy"}')]),
        say("What would you like to say to Vanshaj? (Waiting for your message...)......"),
    ])
    router = Router([provider])
    await router.refresh()
    events = [e async for e in agent.run("Acme, recruiter", [], router, mode="contact")]
    text = "".join(e["text"] for e in events if e["type"] == "text")
    assert text == "What would you like to say to Vanshaj?"


def test_tidy_drops_repeats_and_keeps_replies_short():
    rambling = ("What would you like to say to Vanshaj? Whenever you're ready, just type it. "
                "Whenever you're ready, just type it. Just let me know the text. Whenever you're ready, just type it.")
    assert agent.tidy(rambling) == "What would you like to say to Vanshaj?"
    asked_twice = "Thanks, Priya. What's your email address? And what company are you with?"
    assert agent.tidy(asked_twice) == "Thanks, Priya. What's your email address?"


async def test_starting_the_email_flow_hands_over_in_the_same_request():
    said = ("I'm Priya (priya@acme.io), a recruiter at Acme. "
            "Tell him we're hiring an AI engineer and would love a call next week.")
    fields = {"name": "Priya", "email": "priya@acme.io", "company": "Acme", "role": "recruiter",
              "purpose": "hiring", "message": "We're hiring an AI engineer and would love a call next week."}
    provider = FakeProvider("groq", ["openai/gpt-oss-120b"], [
        call("start_email_to_vanshaj", {}, id_="start"),
        call("send_email", fields, id_="send"),
        say("Here's your draft."),
    ])
    router = Router([provider])
    await router.refresh()
    events = [e async for e in agent.run(said, [], router)]

    actions = [e["name"] for e in events if e["type"] == "action"]
    assert actions == ["start_email_to_vanshaj", "send_email"]
    # the second turn ran under the email prompt with only send_email on offer
    second_messages = provider.seen[1][1]
    assert second_messages[0]["content"] == agent.CONTACT_PROMPT
    assert not any(m.get("tool_call_id") == "start" for m in second_messages)


async def test_public_mcp_answers_at_mcp_without_a_redirect():
    import httpx
    from axiom import app as app_module

    async with tools.mcp.session_manager.run():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                                  headers={"Accept": "application/json, text/event-stream"})
    assert r.status_code == 200
    names = {t["name"] for t in r.json()["result"]["tools"]}
    assert "get_project" in names and "send_email" not in names


async def test_project_list_keeps_summaries_to_one_sentence():
    listed = json.loads(await tools.call("list_projects", {}))
    assert listed and all(p["summary"].count(". ") == 0 for p in listed)
