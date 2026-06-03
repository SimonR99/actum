import importlib


def test_lightweight_submodules_import_without_model_runtime():
    for module_name in (
        "robo",
        "robo.backends",
        "robo.core",
        "robo.perception",
        "robo.tools",
        "robo.tts",
        "robo.agent",
    ):
        importlib.import_module(module_name)
