"""Unitree G1 backend.

This adapter intentionally does not vendor Unitree's SDK. Install it separately:

    pip install git+https://github.com/unitreerobotics/unitree_sdk2_python.git
"""

from __future__ import annotations

import time
from math import radians
from typing import Any

from robo.backends.base import RobotBackend
from robo.core.schema import RobotState, now


GESTURES: list[str] = []

_MAX_DRIVE_DISTANCE_M = 1.5
_MAX_MOTION_DURATION_S = 4.0
_DRIVE_SPEED_MPS = 0.25
_ROTATE_SPEED_RADPS = 0.35


class UnitreeG1Backend(RobotBackend):
    name = "unitree_g1"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._interface = self.config.get("network_interface", "eth0")
        self._speaker_id = int(self.config.get("speaker_id", 0))
        self._volume = int(self.config.get("volume", 80))
        self._audio = None
        self._loco = None
        self._arm = None

    def connect(self) -> bool:
        if self.connected:
            return True
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient, action_map
            from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
            from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        except ImportError as exc:
            print(f"[unitree] SDK not installed: {exc}")
            print("[unitree] Install with: pip install git+https://github.com/unitreerobotics/unitree_sdk2_python.git")
            return False

        print(f"[unitree] Connecting via interface {self._interface!r}")
        ChannelFactoryInitialize(0, self._interface)

        self._audio = AudioClient()
        self._audio.SetTimeout(10.0)
        self._audio.Init()

        self._loco = LocoClient()
        self._loco.SetTimeout(10.0)
        self._loco.Init()

        self._arm = G1ArmActionClient()
        self._arm.SetTimeout(10.0)
        self._arm.Init()

        global GESTURES
        GESTURES = list(action_map.keys())
        self._audio.SetVolume(self._volume)
        self.connected = True
        return True

    def get_state(self) -> RobotState:
        return RobotState(
            backend=self.name,
            connected=self.connected,
            mode="hardware" if self.connected else "offline",
            metadata={"network_interface": self._interface, "gestures": GESTURES},
        )

    def speak(self, text: str):
        started = now()
        if not self._audio:
            return self._result("speak", False, "Unitree audio client is not connected.", started, text=text)
        code = self._audio.TtsMaker(text, self._speaker_id)
        return self._result("speak", code == 0, "Unitree TTS queued." if code == 0 else f"Unitree TTS failed: {code}", started, text=text, code=code)

    def _set_velocity(self, vx: float, vy: float, omega: float, duration: float = 0.5) -> bool:
        if not self._loco:
            return False
        code = self._loco.SetVelocity(vx, vy, omega, duration)
        if code != 0:
            print(f"[unitree] SetVelocity failed (code={code})")
        return code == 0

    def drive(self, direction: str, distance_m: float = 0.5):
        started = now()
        if not self._loco:
            return self._result("drive", False, "Unitree locomotion client is not connected.", started, direction=direction, distance_m=distance_m)
        if direction == "stop":
            ok = self._set_velocity(0.0, 0.0, 0.0, 0.1)
            return self._result("drive", ok, "Stopped." if ok else "Stop failed.", started, direction=direction, distance_m=0.0)

        distance = min(max(abs(float(distance_m)), 0.0), _MAX_DRIVE_DISTANCE_M)
        duration = min(distance / _DRIVE_SPEED_MPS, _MAX_MOTION_DURATION_S) if distance else 0.0
        vx, vy = 0.0, 0.0
        if direction == "forward":
            vx = _DRIVE_SPEED_MPS
        elif direction == "backward":
            vx = -_DRIVE_SPEED_MPS
        elif direction == "left":
            vy = _DRIVE_SPEED_MPS
        elif direction == "right":
            vy = -_DRIVE_SPEED_MPS
        else:
            return self._result("drive", False, f"Invalid direction: {direction}", started, direction=direction, distance_m=distance)

        ok = self._set_velocity(vx, vy, 0.0, duration)
        if ok and duration:
            time.sleep(duration)
            self._set_velocity(0.0, 0.0, 0.0, 0.1)
        return self._result("drive", ok, f"Moved {direction} {distance:.2f} m." if ok else "Move failed.", started, direction=direction, distance_m=distance, duration_s=duration)

    def rotate(self, degrees: float):
        started = now()
        if not self._loco:
            return self._result("rotate", False, "Unitree locomotion client is not connected.", started, degrees=degrees)
        angle = max(min(float(degrees), 180.0), -180.0)
        duration = min(abs(radians(angle)) / _ROTATE_SPEED_RADPS, _MAX_MOTION_DURATION_S) if angle else 0.0
        omega = -_ROTATE_SPEED_RADPS if angle > 0 else _ROTATE_SPEED_RADPS
        ok = True if angle == 0 else self._set_velocity(0.0, 0.0, omega, duration)
        if ok and duration:
            time.sleep(duration)
            self._set_velocity(0.0, 0.0, 0.0, 0.1)
        return self._result("rotate", ok, f"Rotated {angle:+.0f} deg." if ok else "Rotate failed.", started, degrees=angle, duration_s=duration)

    def gesture(self, name: str):
        started = now()
        if not self._arm:
            return self._result("gesture", False, "Unitree arm client is not connected.", started, gesture=name)
        try:
            from unitree_sdk2py.g1.arm.g1_arm_action_client import action_map
        except ImportError:
            return self._result("gesture", False, "Unitree SDK action map is unavailable.", started, gesture=name)

        action_id = action_map.get(name)
        if action_id is None:
            return self._result("gesture", False, f"Unknown gesture: {name}", started, gesture=name, available=sorted(action_map.keys()))
        code = self._arm.ExecuteAction(action_id)
        return self._result("gesture", code == 0, f"Gesture {name} executed." if code == 0 else f"Gesture failed: {code}", started, gesture=name, action_id=action_id, code=code)

    def set_led(self, r: int, g: int, b: int):
        started = now()
        if not self._audio:
            return self._result("set_led", False, "Unitree audio/LED client is not connected.", started, r=r, g=g, b=b)
        code = self._audio.LedControl(r, g, b)
        return self._result("set_led", code == 0, "LED updated." if code == 0 else f"LED update failed: {code}", started, r=r, g=g, b=b, code=code)

    def stop(self):
        started = now()
        ok = self._set_velocity(0.0, 0.0, 0.0, 0.1)
        return self._result("stop", ok, "Stopped." if ok else "Stop failed.", started)
