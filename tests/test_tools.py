from robo.tools import RobotTools
from robo.runtime import RobotRuntime


class FakeAgent:
    def __init__(self, runtime=None):
        self.runtime = runtime
        self._pending_speech = []
        self.memory = {}


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
