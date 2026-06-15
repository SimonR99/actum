from actum.tools import RobotTools
from actum.runtime import RobotRuntime


class FakeAgent:
    def __init__(self, runtime=None, config=None):
        self.runtime = runtime
        self.config = config or (runtime.config if runtime is not None else {})
        self._pending_speech = []

    def queue_speech(self, text):
        self._pending_speech.append(text)


def test_navigate_uses_backend_when_available():
    runtime = RobotRuntime({"robot": {"backend": "fake"}}, "testbot")
    runtime.connect()
    tools = RobotTools(FakeAgent(runtime))

    result = tools.navigate("forward", 0.25)

    assert "Simulated move forward" in result
    assert runtime.backend.actions[-1]["action"] == "drive"
    assert tools.actions_taken[-1]["backend"] == "fake"
    assert tools.actions_taken[-1]["ok"] is True


def test_rotate_records_backend_result_in_graph():
    runtime = RobotRuntime({"robot": {"backend": "fake"}}, "testbot")
    runtime.connect()
    tools = RobotTools(FakeAgent(runtime))

    result = tools.rotate(45)

    assert "Simulated rotate" in result
    assert runtime.tool_graph[-1]["result"]["ok"] is True
    assert runtime.tool_graph[-1]["type"] == "rotate"


def test_navigate_without_runtime_reports_no_backend():
    tools = RobotTools(FakeAgent())

    result = tools.navigate("stop")

    assert "No robot backend configured" in result
    assert tools.actions_taken[-1]["backend"] is None


def test_set_plan_updates_runtime_intent():
    runtime = RobotRuntime({"robot": {"backend": "fake"}}, "testbot")
    tools = RobotTools(FakeAgent(runtime))

    result = tools.set_plan("pick up the cube", "1. look for cube\n2. approach cube")

    assert "2 steps" in result
    assert runtime.intent.goal == "pick up the cube"
    assert runtime.intent.steps[0].status == "active"


def test_remember_writes_to_runtime_memory(tmp_path):
    runtime = RobotRuntime(
        {
            "robot": {"backend": "fake"},
            "memory": {"path": str(tmp_path / "memory.json")},
        },
        "testbot",
    )
    agent = FakeAgent(runtime)
    tools = RobotTools(agent)

    result = tools.remember("favorite_tool", "torque wrench")

    assert "Stored" in result
    assert runtime.memory.recall_fact("favorite_tool") == "torque wrench"
    assert tools.recall("favorite_tool") == "favorite_tool = torque wrench"


def test_memory_observation_tools_record_structured_entries(tmp_path):
    runtime = RobotRuntime(
        {
            "robot": {"backend": "fake"},
            "memory": {"path": str(tmp_path / "memory.json")},
        },
        "testbot",
    )
    tools = RobotTools(FakeAgent(runtime))

    result = tools.record_observation("Saw the dock beside the desk.", "dock, desk")
    spatial = tools.remember_spatial_note("Dock is left of the desk.", "desk")

    assert "Recorded observation" in result
    assert "Recorded spatial note" in spatial
    assert runtime.memory.recent(1)[0]["summary"] == "Saw the dock beside the desk."
    assert runtime.memory.spatial_notes[-1].data["place"] == "desk"


def test_behavior_map_body_and_cron_tools_update_runtime(tmp_path):
    runtime = RobotRuntime(
        {
            "robot": {"backend": "fake"},
            "memory": {"path": str(tmp_path / "memory.json")},
        },
        "testbot",
    )
    tools = RobotTools(FakeAgent(runtime))

    behavior = tools.set_behavior_tree(
        "watch the room",
        '[{"id":"wait","label":"Wait for trigger","kind":"wait"},{"id":"look","label":"Review camera","kind":"vision"}]',
    )
    node = tools.mark_behavior_node("look", "active", "camera review")
    mapped = tools.record_map_observation(
        "Desk is in front of the dock.", "desk", confidence=0.9
    )
    body = tools.update_body_perception(
        "Arm is clear.", posture="ready", contacts="left foot,right foot"
    )
    cron = tools.schedule_job("room check", 120, "Review the room.")

    assert "2 node" in behavior
    assert "marked active" in node
    assert "map-1" in mapped
    assert body == "Body perception updated."
    assert "room check" in cron
    assert runtime.behavior.active_node_id == "look"
    assert runtime.spatial_map.observations[-1].place == "desk"
    assert runtime.body.contacts == ["left foot", "right foot"]
    assert runtime.cron.jobs[-1].instruction == "Review the room."


def test_web_fetch_rejects_non_http_urls():
    runtime = RobotRuntime({"robot": {"backend": "fake"}}, "testbot")
    tools = RobotTools(FakeAgent(runtime))

    result = tools.web_fetch("file:///etc/passwd")

    assert "Unsupported URL scheme" in result
    assert runtime.tool_graph[-1]["type"] == "web_fetch"
    assert runtime.tool_graph[-1]["result"]["ok"] is False


def test_list_mcp_servers_reports_disabled_config():
    runtime = RobotRuntime(
        {"robot": {"backend": "fake"}, "mcp": {"enabled": False}}, "testbot"
    )
    tools = RobotTools(FakeAgent(runtime, runtime.config))

    result = tools.list_mcp_servers()

    assert "No MCP servers are enabled" in result
    assert runtime.tool_graph[-1]["type"] == "list_mcp_servers"
    assert runtime.tool_graph[-1]["result"]["ok"] is True


def test_get_tools_respects_runtime_tool_settings():
    runtime = RobotRuntime(
        {"robot": {"backend": "fake"}, "tools": {"enabled": ["done", "look"]}},
        "testbot",
    )
    tools = RobotTools(FakeAgent(runtime))

    names = {tool.__name__ for tool in tools.get_tools()}

    assert names == {"done", "look"}
