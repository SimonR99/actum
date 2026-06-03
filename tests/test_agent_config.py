import json

from actum.agent import RobotAgent, _load_config, _resolve_model_path
from actum.backends.factory import create_backend


def test_resolve_model_path_expands_wildcard(monkeypatch, tmp_path):
    model_dir = tmp_path / "hub" / "snapshots" / "abc123"
    model_dir.mkdir(parents=True)
    model_file = model_dir / "gemma.litertlm"
    model_file.write_text("fake", encoding="utf-8")

    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "hub" / "snapshots" / "*" / "gemma.litertlm"))

    assert _resolve_model_path() == str(model_file)


def test_legacy_hardware_config_maps_to_unitree_backend(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"hardware": {"enabled": True, "type": "unitree_g1", "network_interface": "eth0"}}),
        encoding="utf-8",
    )

    cfg = _load_config(config_path)
    backend = create_backend(cfg)

    assert backend.name == "unitree_g1"


def test_default_config_uses_laptop_backend(tmp_path):
    cfg = _load_config(tmp_path / "missing.json")
    backend = create_backend(cfg)

    assert backend.name == "laptop"
    assert cfg["personality"]["name"] == "dino"
    assert cfg["companion"]["always_on"] is True
    assert cfg["memory"]["enabled"] is True


def test_personality_name_overrides_legacy_name(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"name": "legacy", "personality": {"name": "Nova"}, "memory": {"path": str(tmp_path / "memory.json")}}),
        encoding="utf-8",
    )

    agent = RobotAgent(config_path)

    assert agent.get_name() == "Nova"


def test_robot_settings_update_name_backend_and_persist(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "old",
                "personality": {"name": "old"},
                "memory": {"path": str(tmp_path / "memory.json")},
                "robot": {"backend": "laptop", "laptop": {"webcam": True}},
            }
        ),
        encoding="utf-8",
    )
    agent = RobotAgent(config_path)

    ok, message = agent.set_robot_settings(
        name="dino",
        robot_config={"backend": "fake"},
        persist=True,
    )

    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert ok is True
    assert "fake" in message
    assert agent.get_name() == "dino"
    assert agent.runtime.backend.name == "fake"
    assert saved["name"] == "dino"
    assert saved["personality"]["name"] == "dino"
    assert saved["robot"]["backend"] == "fake"
