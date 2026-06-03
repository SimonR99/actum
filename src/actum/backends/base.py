"""Backend protocol for physical and simulated robots."""

from __future__ import annotations

from abc import ABC
from typing import Any

from actum.core.schema import ActionResult, RobotState, now


class RobotBackend(ABC):
    name = "base"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.connected = False

    def connect(self) -> bool:
        self.connected = True
        return True

    def close(self):
        self.connected = False

    def get_state(self) -> RobotState:
        return RobotState(backend=self.name, connected=self.connected)

    def _result(
        self,
        action: str,
        ok: bool,
        message: str,
        started_at: float,
        **data: Any,
    ) -> ActionResult:
        return ActionResult(
            action=action,
            ok=ok,
            message=message,
            data=data,
            started_at=started_at,
            ended_at=now(),
        )

    def speak(self, text: str) -> ActionResult:
        started = now()
        return self._result("speak", False, "Backend does not provide speech.", started, text=text)

    def drive(self, direction: str, distance_m: float = 0.5) -> ActionResult:
        started = now()
        return self._result("drive", False, "Backend does not provide locomotion.", started, direction=direction, distance_m=distance_m)

    def rotate(self, degrees: float) -> ActionResult:
        started = now()
        return self._result("rotate", False, "Backend does not provide rotation.", started, degrees=degrees)

    def gripper(self, action: str) -> ActionResult:
        started = now()
        return self._result("gripper", False, "Backend does not provide a gripper.", started, gripper_action=action)

    def gesture(self, name: str) -> ActionResult:
        started = now()
        return self._result("gesture", False, "Backend does not provide gestures.", started, gesture=name)

    def set_led(self, r: int, g: int, b: int) -> ActionResult:
        started = now()
        return self._result("set_led", False, "Backend does not provide LEDs.", started, r=r, g=g, b=b)

    def stop(self) -> ActionResult:
        started = now()
        return self._result("stop", True, "No active backend motion.", started)
