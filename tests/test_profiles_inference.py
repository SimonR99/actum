from actum.core.profile import ProfileManager
from actum.inference import build_provider
from actum.inference.base import build_tool_schema
from actum.runtime import RobotRuntime


def test_profile_manager_resolves_and_switches():
    pm = ProfileManager({"active_profile": "power_saver"})
    assert pm.resolved["compute"] == "cpu"
    assert pm.resolved["provider"] == "local"

    pm.set_active("fast")
    assert pm.resolved["provider"] == "openai"
    assert pm.resolved["compute"] == "gpu"


def test_unknown_profile_falls_back_to_balanced():
    pm = ProfileManager({"active_profile": "does-not-exist"})
    assert pm.active_name == "balanced"


def test_runtime_applies_profile_to_provider_and_loop_rates():
    runtime = RobotRuntime({"robot": {"backend": "fake"}, "active_profile": "fast"}, "bot")

    assert runtime.settings.models["active_provider"] == "openai"
    assert runtime.compute_backend == "gpu"
    assert runtime.behavior.tick_seconds == 6.0
    assert runtime.deliberate_seconds == 120.0
    assert runtime.snapshot()["profile"]["active"] == "fast"


def test_set_profile_updates_loop_rates():
    runtime = RobotRuntime({"robot": {"backend": "fake"}, "active_profile": "balanced"}, "bot")
    ok, _ = runtime.set_profile("power_saver")

    assert ok is True
    assert runtime.behavior.tick_seconds == 30.0
    assert runtime.behavior.idle_review_seconds == 120.0


def test_build_provider_selects_litert_and_openai():
    runtime = RobotRuntime({"robot": {"backend": "fake"}, "active_profile": "balanced"}, "bot")

    local = build_provider(
        runtime.settings,
        compute="cpu",
        pop_pending_frame=lambda: None,
        resolve_model_path=lambda: "model.litertlm",
    )
    assert local.name == "litert"

    runtime.settings.set_model_provider("openai", model="gpt-4o-mini")
    cloud = build_provider(
        runtime.settings,
        compute="gpu",
        pop_pending_frame=lambda: None,
        resolve_model_path=lambda: "model.litertlm",
    )
    assert cloud.name == "openai"


def test_tool_schema_skips_self_and_marks_required():
    def gripper(self, action: str, force: float = 1.0) -> str:
        """Control the gripper."""
        return ""

    schema = build_tool_schema(gripper)["function"]
    assert schema["name"] == "gripper"
    assert "self" not in schema["parameters"]["properties"]
    assert schema["parameters"]["properties"]["force"]["type"] == "number"
    assert schema["parameters"]["required"] == ["action"]
