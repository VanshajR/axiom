import json
import pathlib

import pytest

from axiom import tools
from axiom.profile_data import ProfileStore
from axiom.providers import ChatProvider, TextDelta, ToolCall, TurnEnd

FIXTURE = pathlib.Path(__file__).parent / "profile.json"


@pytest.fixture(autouse=True)
def profile():
    """A trimmed copy of the real profile.json shape."""
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    store = ProfileStore(url="unused")
    store.set(data)
    tools.store = store
    return data


class FakeProvider(ChatProvider):
    """Plays back scripted turns; a script entry that is an exception gets raised."""

    def __init__(self, name, models, script):
        self.name, self.models, self.script, self.seen = name, models, list(script), []

    async def list_models(self):
        return self.models

    async def stream(self, model, messages, tools, max_tokens=1200):
        self.seen.append((model, messages))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        text, calls = step
        if text:
            yield TextDelta(text)
        yield TurnEnd(tool_calls=calls)


def say(text):
    return (text, [])


def call(name, args, extra=None, id_="c1"):
    return ("", [ToolCall(id=id_, name=name, arguments=json.dumps(args), extra=extra)])
