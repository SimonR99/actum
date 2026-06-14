"""Backend construction from config.json."""

from __future__ import annotations

from typing import Any

from actum.backends.base import RobotBackend
from actum.backends.fake import FakeBackend
from actum.backends.laptop import LaptopBackend
from actum.backends.lekiwi import LeKiwiBackend
from actum.backends.unitree_g1 import UnitreeG1Backend
from actum.backends.ros2 import ROS2Backend


def create_backend(config: dict[str, Any]) -> RobotBackend:
    robot_cfg = config.get("robot", {}) if isinstance(config.get("robot"), dict) else {}
    backend_name = str(robot_cfg.get("backend") or "laptop").lower().strip()
    backend_cfg = (
        robot_cfg.get(backend_name, {})
        if isinstance(robot_cfg.get(backend_name), dict)
        else {}
    )

    if backend_name in {"laptop", "local", "companion"}:
        return LaptopBackend(backend_cfg)
    if backend_name in {"fake", "sim", "simulation"}:
        return FakeBackend(backend_cfg)
    if backend_name == "unitree_g1":
        return UnitreeG1Backend(backend_cfg)
    if backend_name == "lekiwi":
        return LeKiwiBackend(backend_cfg)
    if backend_name == "ros2":
        return ROS2Backend(backend_cfg)
    raise ValueError(f"Unsupported robot backend: {backend_name!r}")
