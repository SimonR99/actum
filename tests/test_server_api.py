"""HTTP surface validation: readiness gating, command queueing, conversation reset."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from actum.agent import RobotAgent
from actum.server import attach_server, _set_server_status
from actum.tools import RobotTools


class StubProvider:
    name = "stub"

    def __init__(self):
        self.resets = 0

    def start(self, system_prompt, tools):
        pass

    def send(self, content):
        return ""

    def reset(self):
        self.resets += 1

    def close(self):
        pass


@pytest.fixture
def client(tmp_path):
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
    agent.provider = StubProvider()
    app = FastAPI()
    app.state.agent = agent
    attach_server(agent, app)
    return TestClient(app), agent, app


def test_health_reports_not_ready_then_ready(client):
    http, agent, app = client
    assert http.get("/health").status_code == 503

    _set_server_status(app, "ready", ready=True)
    payload = http.get("/health").json()
    assert payload["ready"] is True


def test_command_rejected_until_ready_then_queued(client):
    http, agent, app = client
    assert http.post("/command", json={"text": "hello"}).status_code == 503

    _set_server_status(app, "ready", ready=True)
    response = http.post("/command", json={"text": "hello"})
    assert response.status_code == 200
    event = agent.event_bus.get_nowait()
    assert event["source"] == "chat"
    assert event["text"] == "hello"


def test_empty_command_is_rejected(client):
    http, agent, app = client
    _set_server_status(app, "ready", ready=True)
    assert http.post("/command", json={"text": "  "}).status_code == 400


def test_trigger_validates_source(client):
    http, agent, app = client
    _set_server_status(app, "ready", ready=True)
    assert http.post("/trigger/nonsense", json={}).status_code == 400

    response = http.post("/trigger/chat", json={"text": "look around"})
    assert response.status_code == 200
    assert agent.event_bus.get_nowait()["source"] == "chat"


def test_conversation_reset_endpoint(client):
    http, agent, app = client
    assert http.post("/conversation/reset").status_code == 503

    _set_server_status(app, "ready", ready=True)
    agent.runtime.set_plan("goal", "1. step")
    response = http.post("/conversation/reset")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert agent.provider.resets == 1
    assert payload["state"]["intent"]["status"] == "idle"


def test_state_snapshot_includes_scene_and_profile(client):
    http, agent, app = client
    agent.runtime.set_scene("Desk with a laptop.")
    state = http.get("/state").json()

    assert state["scene"]["summary"] == "Desk with a laptop."
    assert state["profile"]["active"]
    assert state["backend"] == "fake"
