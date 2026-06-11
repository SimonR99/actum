"""Turn-loop behavior: replies are kept, provider errors are surfaced."""

import asyncio
import json

import pytest

from actum.agent import RobotAgent
from actum.tools import RobotTools


class StubProvider:
    name = "stub"

    def __init__(self, reply="", error=None):
        self.reply = reply
        self.error = error
        self.sent = []

    def start(self, system_prompt, tools):
        pass

    def send(self, content):
        self.sent.append(content)
        if self.error is not None:
            raise self.error
        return self.reply

    def close(self):
        pass


@pytest.fixture
def agent(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "robot": {"backend": "fake"},
                "memory": {"path": str(tmp_path / "memory.json")},
            }
        ),
        encoding="utf-8",
    )
    agent = RobotAgent(config_path)
    agent.tools = RobotTools(agent)
    return agent


def test_turn_keeps_final_plain_text_reply(agent):
    agent.provider = StubProvider(reply="The dock is two meters ahead.")

    actions = asyncio.run(agent.process_event({"source": "chat", "text": "where is the dock?"}))

    assert actions[-1]["type"] == "reply"
    assert actions[-1]["text"] == "The dock is two meters ahead."
    # The reply also becomes the episode summary instead of a generic line.
    assert agent.runtime.memory.episodes[-1].summary == "The dock is two meters ahead."


def test_turn_surfaces_provider_error(agent):
    agent.provider = StubProvider(error=RuntimeError("connection reset"))
    received = []
    agent._broadcast = lambda payload: received.append(payload)

    actions = asyncio.run(agent.process_event({"source": "chat", "text": "hello"}))

    assert actions == []
    assert received[-1]["error"] == "connection reset"
    assert any(e["type"] == "turn.error" for e in agent.runtime.events.tail())


def test_turn_ignored_when_models_not_loaded(agent):
    agent.provider = None
    received = []
    agent._broadcast = lambda payload: received.append(payload)

    actions = asyncio.run(agent.process_event({"source": "chat", "text": "hello"}))

    assert actions == []
    assert received[-1]["ignored"] is True
