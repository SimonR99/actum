"""Backend construction from config.json."""

from __future__ import annotations

from typing import Any

from robo.backends.base import RobotBackend
from robo.backends.fake import FakeBackend
from robo.backends.lekiwi import LeKiwiBackend
from robo.backends.unitree_g1 import UnitreeG1Backend


def _legacy_backend_config(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    hardware = config.get("hardware", {})
    if isinstance(hardware, dict) and hardware.get("enabled"):
        backend = str(hardware.get("type") or "unitree_g1").lower()
        if backend == "unitree_g1":
            return "unitree_g1", dict(hardware)
    return "fake", {}


def create_backend(config: dict[str, Any]) -> RobotBackend:
    robot_cfg = config.get("robot", {}) if isinstance(config.get("robot"), dict) else {}
    backend_name = str(robot_cfg.get("backend") or "").lower().strip()
    if backend_name:
        backend_cfg = robot_cfg.get(backend_name, {}) if isinstance(robot_cfg.get(backend_name), dict) else {}
    else:
        backend_name, backend_cfg = _legacy_backend_config(config)

    if backend_name in {"fake", "sim", "simulation"}:
        return FakeBackend(backend_cfg)
    if backend_name == "unitree_g1":
        return UnitreeG1Backend(backend_cfg)
    if backend_name == "lekiwi":
        return LeKiwiBackend(backend_cfg)
    raise ValueError(f"Unsupported robot backend: {backend_name!r}")
