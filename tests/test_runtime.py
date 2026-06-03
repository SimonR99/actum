from robo.runtime import RobotRuntime


def test_runtime_snapshot_exposes_capabilities_and_intent():
    runtime = RobotRuntime({"robot": {"backend": "fake"}}, "testbot")
    runtime.set_plan("inspect the table", "look\nreport status")

    snapshot = runtime.snapshot()

    names = {item["name"] for item in snapshot["capabilities"]}
    assert "navigate" in names
    assert snapshot["intent"]["goal"] == "inspect the table"
    assert snapshot["backend"] == "fake"
