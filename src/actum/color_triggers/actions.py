"""Map detected color groups to robot or text actions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from actum.backends.base import RobotBackend
from actum.core.schema import ActionResult


ActionCallback = Callable[[str, dict[str, Any], dict[str, Any]], None]


@dataclass
class TriggerAction:
    """One command bound to a color group name."""

    type: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TriggerAction:
        if not isinstance(data, dict):
            raise ValueError("Action must be an object")
        action_type = str(data.get("type", "")).strip()
        if not action_type:
            raise ValueError("Action requires a type")
        params = {k: v for k, v in data.items() if k != "type"}
        return cls(type=action_type, params=params)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, **self.params}


@dataclass
class ColorTriggerConfig:
    enabled: bool = False
    calibration_path: str = "config/color_triggers_calibration.json"
    detect_every_frames: int = 10
    cooldown_seconds: float = 3.0
    show_debug: bool = False
    actions: dict[str, TriggerAction] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls, data: dict[str, Any] | None, base_dir: Path | None = None
    ) -> ColorTriggerConfig:
        if not data:
            return cls()
        actions: dict[str, TriggerAction] = {}
        for group_name, payload in (data.get("actions") or {}).items():
            actions[str(group_name)] = TriggerAction.from_dict(payload)
        calibration = str(data.get("calibration", cls.calibration_path))
        if base_dir and not Path(calibration).is_absolute():
            calibration = str((base_dir / calibration).resolve())
        return cls(
            enabled=bool(data.get("enabled", False)),
            calibration_path=calibration,
            detect_every_frames=max(1, int(data.get("detect_every_frames", 10))),
            cooldown_seconds=max(0.0, float(data.get("cooldown_seconds", 3.0))),
            show_debug=bool(data.get("show_debug", False)),
            actions=actions,
        )

    def to_dict(self, calibration_path: str | None = None) -> dict[str, Any]:
        calibration = calibration_path or self.calibration_path
        return {
            "enabled": self.enabled,
            "calibration": calibration,
            "detect_every_frames": self.detect_every_frames,
            "cooldown_seconds": self.cooldown_seconds,
            "show_debug": self.show_debug,
            "actions": {
                group: action.to_dict() for group, action in self.actions.items()
            },
        }


def load_trigger_config(path: Path) -> ColorTriggerConfig:
    if not path.exists():
        return ColorTriggerConfig()
    payload = json.loads(path.read_text())
    return ColorTriggerConfig.from_dict(payload, base_dir=path.parent.parent)


class ActionExecutor:
    """Run a configured trigger action through the robot backend."""

    def __init__(
        self,
        backend: RobotBackend | None = None,
        on_action: ActionCallback | None = None,
    ):
        self.backend = backend
        self.on_action = on_action

    def execute(
        self,
        group_name: str,
        action: TriggerAction,
        context: dict[str, Any] | None = None,
    ) -> ActionResult | None:
        context = context or {}
        action_type = action.type.lower()
        params = dict(action.params)

        if self.on_action is not None:
            self.on_action(group_name, {"type": action_type, **params}, context)

        if action_type in ("log", "notify"):
            message = str(params.get("text", params.get("message", group_name)))
            print(f"[color_trigger] {group_name}: {message}")
            return None

        if action_type == "speak":
            text = str(params.get("text", "")).strip()
            if not text:
                raise ValueError(f"speak action for {group_name} requires text")
            if self.backend is None:
                print(f"[color_trigger] speak (no backend): {text}")
                return None
            return self.backend.speak(text)

        if action_type == "drive":
            direction = str(params.get("direction", "forward"))
            distance_m = float(params.get("distance_m", 0.3))
            if self.backend is None:
                print(f"[color_trigger] drive (no backend): {direction} {distance_m}m")
                return None
            return self.backend.drive(direction, distance_m=distance_m)

        if action_type == "rotate":
            degrees = float(params.get("degrees", 90))
            if self.backend is None:
                print(f"[color_trigger] rotate (no backend): {degrees}°")
                return None
            return self.backend.rotate(degrees)

        if action_type == "gesture":
            name = str(params.get("name", "")).strip()
            if not name:
                raise ValueError(f"gesture action for {group_name} requires name")
            if self.backend is None:
                print(f"[color_trigger] gesture (no backend): {name}")
                return None
            return self.backend.gesture(name)

        if action_type == "stop":
            if self.backend is None:
                print(f"[color_trigger] stop (no backend)")
                return None
            return self.backend.stop()

        if action_type == "prompt":
            text = str(params.get("text", "")).strip()
            if not text:
                raise ValueError(f"prompt action for {group_name} requires text")
            print(f"[color_trigger] prompt: {text}")
            return None

        raise ValueError(f"Unknown color trigger action type: {action_type!r}")
