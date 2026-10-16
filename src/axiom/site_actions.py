"""Tools that make the site do something, rather than look something up.

The model calls these like any other tool, which is how Axiom now handles
intent: "take me to his projects", "I'd like to email him", "show me his
LinkedIn" each map to a call here instead of a keyword list on the client.
The agent turns a call into an `action` event for the site to carry out, and
tells the model it happened. They aren't on the public MCP server: outside the
site there's no page to scroll and no chat to open.
"""
import json
import re

SECTIONS = ["hero", "about", "experience", "education", "skills", "projects", "contact"]

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "show_on_page",
            "description": "Scroll the visitor's page to a section or to a specific card and highlight it. "
            "Use when they ask to go to, see or be shown something on the site.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": f"A section ({', '.join(SECTIONS)}) or a page_anchor returned by another tool, "
                        "e.g. #project-depthpad or #experience-drdo.",
                    }
                },
                "required": ["target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_email_to_vanshaj",
            "description": "Open the in-chat flow that takes the visitor's details and message and emails it to Vanshaj. "
            "Use when they want to contact, email, hire or reach out to him, or leave him a message.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_resume",
            "description": "Open Vanshaj's resume in the site's built-in viewer.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_profile_card",
            "description": "Show an interactive card for one of Vanshaj's profiles in the chat.",
            "parameters": {
                "type": "object",
                "properties": {"platform": {"type": "string", "enum": ["linkedin", "github", "instagram", "email"]}},
                "required": ["platform"],
            },
        },
    },
]

# Only offered in the email flow: sending is a tool call the site carries out
# (through EmailJS), checked here first, instead of a tag the site has to parse.
SEND_EMAIL = {
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "Show the visitor the finished message as a draft they can send or edit. Call once you have their name, a valid email and the message.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string", "description": "The visitor's email address, for Vanshaj's reply"},
                "company": {"type": "string", "description": "Optional"},
                "role": {"type": "string", "description": "Optional"},
                "purpose": {"type": "string", "enum": ["hiring", "collaboration", "project inquiry", "general"]},
                "message": {"type": "string", "description": "The confirmed message, in the visitor's voice"},
            },
            "required": ["name", "email", "message"],
        },
    },
}

NAMES = {s["function"]["name"] for s in SCHEMAS} | {"send_email"}
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.I)

# what the model is told after the site has carried an action out
RESULTS = {
    "show_on_page": "Done: the page has scrolled there and highlighted it. Mention it in a few words.",
    "start_email_to_vanshaj": "Done: the email flow is open. Say you'll help them write to him and ask for their name.",
    "open_resume": "Done: the resume viewer is open. Say so in a few words.",
    "show_profile_card": "Done: the card is showing in the chat. Add at most one short line.",
    "send_email": "The site is showing them this draft with Send and Edit buttons. Reply with one short line telling them to check it and press Send, or say what to change. Don't ask them to type yes.",
}


STOPWORDS = set("""a an and are as at be but by for from have he hi hello his i i'm im in is it its i'd
i'll me my of on or our so that the their them there this to us was we were with would you your
vanshaj dear regards best thanks thank""".split())


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z'\-]{2,}", text.lower()) if w not in STOPWORDS}


def grounded(message: str, visitor_text: str) -> bool:
    """Is the draft built from what the visitor wrote, rather than invented?

    Models are good at polishing and bad at admitting they don't know, so a
    draft that's mostly words the visitor never used gets sent back. Light
    rewording passes; a made-up pitch doesn't.
    """
    drafted = _words(message)
    if not drafted:
        return False
    return len(drafted & _words(visitor_text)) / len(drafted) >= 0.5


def check_email(args: dict, visitor_text: str | None = None) -> tuple[dict | None, str | None]:
    """The message as it will be sent, or the problem to put to the visitor."""
    fields = {k: str(args.get(k) or "").strip() for k in ("name", "email", "company", "role", "purpose", "message")}
    if not fields["name"]:
        return None, "missing their name"
    if not EMAIL.match(fields["email"]):
        return None, f"{fields['email'] or 'no email'} isn't a valid email address; ask for it again"
    if len(fields["message"]) < 5:
        return None, "missing the message itself"
    if visitor_text is not None and not grounded(fields["message"], visitor_text):
        return None, ("the message has to come from what the visitor wrote, not be made up. "
                      "Ask them what they'd like to say, then draft it closely from their words")
    fields["purpose"] = fields["purpose"] or "general"
    return fields, None


def normalise(name: str, args: dict) -> dict | None:
    """Clean up the model's arguments; None means the call makes no sense."""
    if name == "show_on_page":
        target = str(args.get("target", "")).strip().lower()
        if not target:
            return None
        if target.lstrip("#") in SECTIONS:
            return {"target": target.lstrip("#")}
        return {"target": "#" + target.lstrip("#")}
    if name == "show_profile_card":
        platform = str(args.get("platform", "")).lower()
        return {"platform": platform} if platform in ("linkedin", "github", "instagram", "email") else None
    return {}


def result(name: str) -> str:
    return json.dumps({"ok": True, "note": RESULTS[name]})
