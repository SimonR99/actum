"""Deterministic backend for local development, tests, and CI."""

from __future__ import annotations

from typing import Any

from actum.backends.base import RobotBackend
from actum.core.schema import RobotState, now


class FakeBackend(RobotBackend):
    name = "fake"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.actions: list[dict[str, Any]] = []
        self.pose = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}

    def _remember(self, action: str, **data: Any):
        self.actions.append({"action": action, "time": now(), **data})

    def get_state(self) -> RobotState:
        return RobotState(
            backend=self.name,
            connected=self.connected,
            mode="simulated",
            battery_percent=100.0,
            pose=dict(self.pose),
            metadata={"actions": len(self.actions)},
        )

    def speak(self, text: str):
        started = now()
        self._remember("speak", text=text)
        return self._result("speak", True, f"Simulated speech: {text}", started, text=text)

    def drive(self, direction: str, distance_m: float = 0.5):
        started = now()
        distance = float(distance_m)
        if direction == "forward":
            self.pose["x"] += distance
        elif direction == "backward":
            self.pose["x"] -= distance
        elif direction == "left":
            self.pose["y"] += distance
        elif direction == "right":
            self.pose["y"] -= distance
        elif direction != "stop":
            return self._result("drive", False, f"Invalid direction: {direction}", started, direction=direction, distance_m=distance)
        self._remember("drive", direction=direction, distance_m=distance)
        return self._result("drive", True, f"Simulated move {direction} {distance:.2f} m.", started, direction=direction, distance_m=distance)

    def rotate(self, degrees: float):
        started = now()
        self.pose["yaw_deg"] += float(degrees)
        self._remember("rotate", degrees=float(degrees))
        return self._result("rotate", True, f"Simulated rotate {degrees:+.0f} deg.", started, degrees=float(degrees))

    def gripper(self, action: str):
        started = now()
        self._remember("gripper", gripper_action=action)
        return self._result("gripper", True, f"Simulated gripper {action}.", started, gripper_action=action)

    def gesture(self, name: str):
        started = now()
        self._remember("gesture", gesture=name)
        return self._result("gesture", True, f"Simulated gesture {name}.", started, gesture=name)

    def stop(self):
        started = now()
        self._remember("stop")
        return self._result("stop", True, "Simulated stop.", started)
