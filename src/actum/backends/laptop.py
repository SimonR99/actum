"""Laptop backend for an always-on local companion runtime."""

from __future__ import annotations

import os
import platform
import socket
from typing import Any

from actum.backends.base import RobotBackend
from actum.core.schema import RobotState, now


class LaptopBackend(RobotBackend):
    """Local computer backend with webcam, microphone, and speaker access.

    The laptop backend is intentionally non-mobile: it lets the agent perceive,
    speak, browse, and use configured software tools without pretending that the
    laptop can drive, gesture, or manipulate objects.
    """

    name = "laptop"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.webcam_enabled = bool(self.config.get("webcam", True))
        self.microphone_enabled = bool(self.config.get("microphone", True))
        self.speaker_enabled = bool(self.config.get("speaker", True))

    def connect(self) -> bool:
        self.connected = True
        return True

    def get_state(self) -> RobotState:
        return RobotState(
            backend=self.name,
            connected=self.connected,
            mode="companion",
            battery_percent=None,
            metadata={
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "pid": os.getpid(),
                "webcam": self.webcam_enabled,
                "microphone": self.microphone_enabled,
                "speaker": self.speaker_enabled,
                "mobility": "stationary",
            },
        )

    def speak(self, text: str):
        started = now()
        if not self.speaker_enabled:
            return self._result(
                "speak", False, "Laptop speaker output is disabled.", started, text=text
            )
        return self._result(
            "speak",
            True,
            "Speech will play through the local TTS pipeline.",
            started,
            text=text,
        )

    def drive(self, direction: str, distance_m: float = 0.5):
        started = now()
        return self._result(
            "drive",
            False,
            "Laptop backend is stationary and cannot navigate.",
            started,
            direction=direction,
            distance_m=float(distance_m),
        )

    def rotate(self, degrees: float):
        started = now()
        return self._result(
            "rotate",
            False,
            "Laptop backend is stationary and cannot rotate.",
            started,
            degrees=float(degrees),
        )

    def gripper(self, action: str):
        started = now()
        return self._result(
            "gripper",
            False,
            "Laptop backend does not provide an end-effector.",
            started,
            gripper_action=action,
        )

    def gesture(self, name: str):
        started = now()
        return self._result(
            "gesture",
            False,
            "Laptop backend does not provide robot gestures.",
            started,
            gesture=name,
        )
