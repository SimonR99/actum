from actum.runtime import RobotRuntime


def test_runtime_snapshot_exposes_capabilities_and_intent():
    runtime = RobotRuntime({"robot": {"backend": "fake"}}, "testbot")
    runtime.set_plan("inspect the table", "look\nreport status")

    snapshot = runtime.snapshot()

    names = {item["name"] for item in snapshot["capabilities"]}
    assert "navigate" in names
    assert "web_fetch" in names
    assert "list_mcp_servers" in names
    assert snapshot["intent"]["goal"] == "inspect the table"
    assert snapshot["backend"] == "fake"
    assert snapshot["robot_config"]["backend"] == "fake"
    assert snapshot["companion"]["always_on"] is True
    assert "memory" in snapshot
    assert "behavior" in snapshot
    assert "cron" in snapshot
    assert "map" in snapshot
    assert "body" in snapshot
    assert "settings" in snapshot
    assert snapshot["behavior"]["nodes"][0]["label"] == "look"
    assert snapshot["personality"]["name"] == "testbot"


def test_runtime_records_cron_map_body_and_settings():
    runtime = RobotRuntime({"robot": {"backend": "fake"}}, "testbot")

    job = runtime.add_cron_job("look around", 30, "Check the room.")
    obs = runtime.record_map_observation("Dock is beside the desk.", place="desk", confidence=0.8)
    runtime.update_body_perception("Standing with empty hands.", posture="standing", holding="nothing")
    runtime.settings.set_model_provider("openai", model="example-model", api_key="test-key", enabled=True)
    runtime.settings.set_tool_enabled("navigate", False)

    snapshot = runtime.snapshot()

    assert job["id"] == "look-around"
    assert obs["place"] == "desk"
    assert snapshot["cron"]["jobs"][0]["instruction"] == "Check the room."
    assert snapshot["map"]["observations"][0]["summary"] == "Dock is beside the desk."
    assert snapshot["body"]["summary"] == "Standing with empty hands."
    assert snapshot["settings"]["models"]["active_provider"] == "openai"
    assert snapshot["settings"]["models"]["providers"]["openai"]["api_key_configured"] is True
    assert "api_key" not in snapshot["settings"]["models"]["providers"]["openai"]
    assert "navigate" not in snapshot["settings"]["tools"]["enabled"]
