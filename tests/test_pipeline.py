"""End-to-end validation of the agentic pipeline.

Drives the real event flow — event bus → companion gate → turn loop → tools →
backend → runtime state → broadcast — with a scripted provider standing in for
the LLM, plus the background tick (cron, task continuation, idle waiting) and
conversation reset.
"""

import asyncio
import json
import time

import pytest

from actum.agent import RobotAgent, _MAX_TASK_CONTINUATIONS
from actum.tools import RobotTools


class ScriptedProvider:
    """Stands in for the LLM: each send() runs the next script entry."""

    name = "scripted"

    def __init__(self, script=None):
        self.script = list(script or [])
        self.sent = []
        self.resets = 0

    def start(self, system_prompt, tools):
        self.tools = {tool.__name__: tool for tool in tools}

    def send(self, content):
        self.sent.append(content)
        if not self.script:
            return ""
        step = self.script.pop(0)
        return step(self.tools) if callable(step) else step

    def reset(self):
        self.resets += 1

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
                "behavior_loop": {"idle_review": False},
            }
        ),
        encoding="utf-8",
    )
    agent = RobotAgent(config_path)
    agent.tools = RobotTools(agent)
    agent.runtime.connect()
    agent._last_deliberate_at = time.time()  # background_loop sets this at start

    def install(provider):
        provider.start("", agent.tools.get_tools())
        agent.provider = provider
        return provider

    agent.install_provider = install
    return agent


def test_full_turn_chains_tools_through_backend(agent):
    def run_task(tools):
        tools["set_plan"]("greet the operator", "1. wave\n2. say hello")
        tools["mark_step"]("step-1")
        tools["navigate"]("forward", 0.3)
        tools["speak"]("Hello there!")
        tools["done"]("Greeted the operator.")
        return ""

    agent.install_provider(ScriptedProvider([run_task]))
    received = []
    agent._broadcast = lambda payload: received.append(payload)
    # Speech goes through TTS hardware; stub it out but keep the queue logic.
    spoken = []

    async def fake_speak(text):
        spoken.append(text)

    agent._speak = fake_speak

    actions = asyncio.run(agent.process_event({"source": "chat", "text": "say hi and step closer"}))

    types = [a["type"] for a in actions]
    assert types == ["plan", "step", "navigate", "speak", "done"]
    assert agent.runtime.intent.status == "done"
    assert agent.runtime.intent.summary == "Greeted the operator."
    assert spoken == ["Hello there!"]
    # Backend really moved (fake backend records pose).
    assert agent.runtime.backend.pose["x"] == pytest.approx(0.3)
    # Tool graph captured the backend result.
    nav_nodes = [n for n in agent.runtime.tool_graph if n["type"] == "navigate"]
    assert nav_nodes and nav_nodes[-1]["result"]["ok"] is True
    # Episode memory recorded the done summary.
    assert agent.runtime.memory.episodes[-1].summary == "Greeted the operator."
    assert received[-1]["error"] == ""


def test_companion_gate_blocks_low_importance_vision(agent):
    agent.install_provider(ScriptedProvider())
    received = []
    agent._broadcast = lambda payload: received.append(payload)

    actions = asyncio.run(agent.process_event({"source": "vision", "importance": 0.1}))

    assert actions == []
    assert received[-1]["ignored"] is True
    assert not agent.provider.sent  # the LLM was never invoked


def test_vision_turn_updates_scene_summary(agent):
    def review(tools):
        tools["done"]("A person is sitting at the desk near the window.")
        return ""

    agent.install_provider(ScriptedProvider([review]))
    asyncio.run(agent.process_event({"source": "vision", "importance": 0.9, "force": True}))

    assert agent.runtime.scene["summary"] == "A person is sitting at the desk near the window."
    assert agent.runtime.scene["source"] == "vision"
    assert agent.runtime.snapshot()["scene"]["summary"]


def test_background_tick_runs_due_cron_jobs(agent):
    agent.install_provider(ScriptedProvider())
    job = agent.runtime.cron.add("hourly check", 3600, "Check the room.")
    job.next_run_at = time.time() - 1  # force due

    asyncio.run(agent._background_tick())

    event = agent.event_bus.get_nowait()
    assert event["source"] == "cron"
    assert event["text"] == "Check the room."
    assert event["force"] is True


def test_background_tick_continues_stalled_task(agent):
    agent.install_provider(ScriptedProvider())
    agent.runtime.set_plan("tidy the desk", "1. look\n2. report")
    agent.runtime.intent.status = "active"
    agent.runtime.intent.updated_at = time.time() - 3600  # long stalled

    asyncio.run(agent._background_tick())

    event = agent.event_bus.get_nowait()
    assert event["source"] == "continue"
    assert event["text"] == "tidy the desk"
    assert agent._continuation_attempts == 1


def test_stalled_task_fails_after_max_continuations(agent):
    agent.install_provider(ScriptedProvider())
    agent.runtime.set_plan("impossible task", "1. wait forever")
    agent.runtime.intent.status = "active"
    agent.runtime.intent.updated_at = time.time() - 3600
    agent._continued_goal = "impossible task"
    agent._continuation_attempts = _MAX_TASK_CONTINUATIONS

    asyncio.run(agent._background_tick())

    assert agent.event_bus.empty() or agent.event_bus.get_nowait()["source"] != "continue"
    assert agent.runtime.intent.status == "blocked"


def test_background_tick_sets_waiting_when_idle(agent):
    agent.install_provider(ScriptedProvider())

    asyncio.run(agent._background_tick())

    assert agent.runtime.behavior.active_node_id == "wait"
    assert agent.event_bus.empty()


def test_conversation_reset_clears_session_but_keeps_memory(agent):
    agent.install_provider(ScriptedProvider())
    agent.runtime.memory.remember_fact("dock", "by the window")
    agent.runtime.set_plan("old goal", "1. step")
    agent._continuation_attempts = 2

    ok, message = agent.reset_conversation()

    assert ok is True
    assert agent.provider.resets == 1
    assert agent.runtime.intent.status == "idle"
    assert agent.runtime.intent.goal == ""
    assert agent._continuation_attempts == 0
    assert agent.runtime.memory.recall_fact("dock") == "by the window"
    assert any(e["type"] == "conversation.reset" for e in agent.runtime.events.tail())


def test_speak_mutes_microphone_during_playback(agent):
    calls = []

    class StubMic:
        def pause(self):
            calls.append("pause")

        def resume(self):
            calls.append("resume")

    async def fake_unmuted(text):
        calls.append(f"speak:{text}")

    agent.mic = StubMic()
    agent._speak_unmuted = fake_unmuted

    asyncio.run(agent._speak("hello"))

    assert calls == ["pause", "speak:hello", "resume"]


def test_openai_provider_reset_keeps_system_prompt():
    from actum.inference.openai import OpenAIProvider

    provider = OpenAIProvider(model="test", pop_pending_frame=lambda: None)
    provider._messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    provider.reset()
    assert provider._messages == [{"role": "system", "content": "sys"}]
