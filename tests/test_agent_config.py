import json

from robo.agent import _load_config, _resolve_model_path
from robo.backends.factory import create_backend


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
