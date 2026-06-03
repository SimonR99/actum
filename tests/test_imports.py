import importlib


def test_lightweight_submodules_import_without_model_runtime():
    for module_name in (
        "actum",
        "actum.backends",
        "actum.backends.laptop",
        "actum.core",
        "actum.core.autonomy",
        "actum.core.companion",
        "actum.core.memory",
        "actum.core.settings",
        "actum.integrations.mcp",
        "actum.integrations.web",
        "actum.perception",
        "actum.tools",
        "actum.tts",
        "actum.agent",
    ):
        importlib.import_module(module_name)
