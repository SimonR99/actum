"""LeKiwi backend through Hugging Face LeRobot.

The LeRobot API has moved over time, so this adapter keeps imports lazy and
isolated. It should fail with a clear install/config message instead of making
the whole agent impossible to import.
"""

from __future__ import annotations

from typing import Any

from actum.backends.base import RobotBackend
from actum.core.schema import RobotState, now


class LeKiwiBackend(RobotBackend):
    name = "lekiwi"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._robot = None
        self._remote_ip = self.config.get("remote_ip", "127.0.0.1")
        self._port = int(self.config.get("port", 5555))
        self._id = self.config.get("id", "lekiwi")

    def connect(self) -> bool:
        if self.connected:
            return True
        try:
            try:
                from lerobot.robots.lekiwi import LeKiwiClientConfig, LeKiwiClient
            except ImportError:
                from lerobot.common.robots.lekiwi import (
                    LeKiwiClient,
                    LeKiwiClientConfig,
                )
        except ImportError as exc:
            print(f"[lekiwi] LeRobot LeKiwi client is not installed: {exc}")
            print(
                '[lekiwi] Install LeRobot from source or with: pip install "lerobot[lekiwi]"'
            )
            return False

        cfg = LeKiwiClientConfig(
            remote_ip=self._remote_ip, port=self._port, id=self._id
        )
        self._robot = LeKiwiClient(cfg)
        if hasattr(self._robot, "connect"):
            self._robot.connect()
        self.connected = True
        return True

    def close(self):
        if self._robot and hasattr(self._robot, "disconnect"):
            self._robot.disconnect()
        self.connected = False

    def get_state(self) -> RobotState:
        metadata: dict[str, Any] = {
            "remote_ip": self._remote_ip,
            "port": self._port,
            "id": self._id,
        }
        if self._robot and hasattr(self._robot, "get_observation"):
            try:
                obs = self._robot.get_observation()
                metadata["observation_keys"] = (
                    sorted(obs.keys()) if isinstance(obs, dict) else str(type(obs))
                )
            except Exception as exc:
                metadata["observation_error"] = str(exc)
        return RobotState(
            backend=self.name,
            connected=self.connected,
            mode="hardware" if self.connected else "offline",
            metadata=metadata,
        )

    def _send_action(self, action: dict[str, Any]):
        if not self._robot or not hasattr(self._robot, "send_action"):
            return False, "LeKiwi robot is not connected or send_action is unavailable."
        self._robot.send_action(action)
        return True, "LeKiwi action sent."

    def drive(self, direction: str, distance_m: float = 0.5):
        started = now()
        ok, message = self._send_action(
            {"base": {"direction": direction, "distance_m": distance_m}}
        )
        return self._result(
            "drive", ok, message, started, direction=direction, distance_m=distance_m
        )

    def rotate(self, degrees: float):
        started = now()
        ok, message = self._send_action({"base": {"rotate_deg": degrees}})
        return self._result("rotate", ok, message, started, degrees=degrees)

    def gripper(self, action: str):
        started = now()
        ok, message = self._send_action({"arm": {"gripper": action}})
        return self._result("gripper", ok, message, started, gripper_action=action)

    def stop(self):
        started = now()
        ok, message = self._send_action({"base": {"direction": "stop"}})
        return self._result("stop", ok, message, started)
