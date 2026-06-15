"""LeKiwi backend — direct Feetech motor bus (lerobot 0.1.x API).

Talks to the physical hardware over USB serial using the lerobot
FeetechMotorsBus + LeKiwi helpers. No remote server required.

Install lerobot by mounting /home/simon/lerobot as /lerobot in the container;
the entrypoint script will pip-install it automatically.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from actum.backends.base import RobotBackend
from actum.core.schema import ActionResult, RobotState, now

# ── Kinematics (inlined from MobileManipulator to avoid ZeroMQ init) ───────

_XY_SPEED = 0.2  # m/s  (medium speed level)
_THETA_SPEED = 60.0  # deg/s
_WHEEL_RADIUS = 0.05  # m
_BASE_RADIUS = 0.125  # m


def _degps_to_raw(degps: float) -> int:
    steps_per_deg = 4096.0 / 360.0
    speed_int = min(int(round(abs(degps) * steps_per_deg)), 0x7FFF)
    return (speed_int | 0x8000) if degps < 0 else (speed_int & 0x7FFF)


def _body_to_wheel_raw(x: float, y: float, theta_deg: float) -> list[int]:
    """Body-frame (x, y m/s, theta deg/s) → [left, back, right] raw speeds."""
    v = np.array([x, y, theta_deg * math.pi / 180.0])
    angles = np.radians([300, 180, 60])
    m = np.array([[math.cos(a), math.sin(a), _BASE_RADIUS] for a in angles])
    wheel_degps = (m.dot(v) / _WHEEL_RADIUS) * (180.0 / math.pi)
    steps_per_deg = 4096.0 / 360.0
    raw_floats = [abs(d) * steps_per_deg for d in wheel_degps]
    peak = max(raw_floats)
    if peak > 3000:
        wheel_degps = wheel_degps * (3000 / peak)
    return [_degps_to_raw(d) for d in wheel_degps]


# ── Motor layout ────────────────────────────────────────────────────────────

_WHEELS = ["left_wheel", "back_wheel", "right_wheel"]
_ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
_GRIPPER = "gripper"

# STS3215 position range: 0–4095, center = 2048.
# Gesture sequences: list of ([pan, lift, elbow, wrist_flex, wrist_roll], delay_s)
_GESTURES: dict[str, list[tuple[list[int], float]]] = {
    "home": [
        ([2048, 2048, 2048, 2048, 2048], 0.8),
    ],
    "yes": [
        ([2048, 1100, 2740, 2048, 2048], 0.2),
        ([2048, 1100, 2940, 2048, 2048], 0.2),
        ([2048, 1100, 2740, 2048, 2048], 0.2),
        ([2048, 1100, 2940, 2048, 2048], 0.2),
        ([2048, 1100, 2740, 2048, 2048], 0.1),
    ],
    "no": [
        ([2048, 1100, 2940, 2048, 2048], 0.5),
        ([1848, 1100, 2940, 2048, 2048], 0.2),
        ([2248, 1100, 2940, 2048, 2048], 0.2),
        ([1848, 1100, 2940, 2048, 2048], 0.2),
        ([2248, 1100, 2940, 2048, 2048], 0.2),
        ([2048, 1100, 2940, 2048, 2048], 0.2),
    ],
    "wave": [
        ([2048, 1400, 2400, 2048, 2048], 0.4),
        ([1800, 1400, 2400, 2200, 2048], 0.2),
        ([2300, 1400, 2400, 1800, 2048], 0.2),
        ([1800, 1400, 2400, 2200, 2048], 0.2),
        ([2300, 1400, 2400, 1800, 2048], 0.2),
        ([2048, 2048, 2048, 2048, 2048], 0.5),
    ],
    "point": [
        ([2048, 1200, 3200, 1500, 2048], 0.6),
    ],
}


_ALL_MOTORS = _ARM_JOINTS + [_GRIPPER] + _WHEELS


class LeKiwiBackend(RobotBackend):
    name = "lekiwi"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._bus = None
        self._lekiwi = None
        self._serial_port: str = self.config.get("serial_port", "/dev/ttyACM0")
        self._present_motors: set[str] = set()

    def _probe_present_motors(self) -> set[str]:
        """Ping each configured motor ID; SyncWrite does not fail on missing hardware."""
        import scservo_sdk as scs

        present: set[str] = set()
        for name, (motor_id, _model) in self._bus.motors.items():
            _model_num, comm, _err = self._bus.packet_handler.ping(
                self._bus.port_handler, motor_id
            )
            if comm == scs.COMM_SUCCESS:
                present.add(name)
        return present

    def _require_motors(
        self, names: list[str], action: str, started_at: float
    ) -> ActionResult | None:
        missing = [name for name in names if name not in self._present_motors]
        if not missing:
            return None
        return self._result(
            action,
            False,
            f"Motor(s) not on bus: {missing}. Responding: {sorted(self._present_motors)}",
            started_at,
            missing_motors=missing,
            present_motors=sorted(self._present_motors),
        )

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        if self.connected:
            return True
        try:
            from lerobot.common.robot_devices.motors.feetech import (
                FeetechMotorsBus,
                TorqueMode,
            )
            from lerobot.common.robot_devices.robots.configs import LeKiwiRobotConfig
            from lerobot.common.robot_devices.robots.mobile_manipulator import LeKiwi
        except ImportError as exc:
            print(f"[lekiwi] lerobot not installed: {exc}")
            print("[lekiwi] Mount /home/simon/lerobot as /lerobot in the container.")
            return False

        try:
            cfg = LeKiwiRobotConfig()
            cfg.leader_arms = {}
            bus_cfg = cfg.follower_arms["main"]
            bus_cfg.port = self._serial_port

            self._bus = FeetechMotorsBus(bus_cfg)
            self._bus.connect()

            self._present_motors = self._probe_present_motors()
            missing = sorted(set(_ALL_MOTORS) - self._present_motors)
            if missing:
                print(
                    f"[lekiwi] missing motors on bus (no response): {missing}"
                )

            arm_present = [
                name for name in _ARM_JOINTS + [_GRIPPER] if name in self._present_motors
            ]
            if arm_present:
                # Enable torque on reachable arm joints (position mode by default)
                self._bus.write(
                    "Torque_Enable", TorqueMode.ENABLED.value, arm_present
                )

            # LeKiwi.__init__ sets wheels to velocity mode (Mode=1)
            self._lekiwi = LeKiwi(self._bus)

            self.connected = True
            print(
                f"[lekiwi] connected on {self._serial_port} "
                f"({len(self._present_motors)}/{len(_ALL_MOTORS)} motors)"
            )
            return True
        except Exception as exc:
            print(f"[lekiwi] connect failed: {exc}")
            self._bus = None
            self._lekiwi = None
            return False

    def close(self):
        if self._lekiwi:
            try:
                self._lekiwi.stop()
            except Exception:
                pass
        if self._bus:
            try:
                from lerobot.common.robot_devices.motors.feetech import TorqueMode

                arm_present = [
                    name
                    for name in _ARM_JOINTS + [_GRIPPER]
                    if name in self._present_motors
                ]
                if arm_present:
                    self._bus.write(
                        "Torque_Enable",
                        TorqueMode.DISABLED.value,
                        arm_present,
                    )
                self._bus.disconnect()
            except Exception:
                pass
        self._bus = None
        self._lekiwi = None
        self._present_motors = set()
        self.connected = False

    def get_state(self) -> RobotState:
        missing = sorted(set(_ALL_MOTORS) - self._present_motors)
        return RobotState(
            backend=self.name,
            connected=self.connected,
            mode="hardware" if self.connected else "offline",
            metadata={
                "serial_port": self._serial_port,
                "present_motors": sorted(self._present_motors),
                "missing_motors": missing,
            },
        )

    # ── Motion ───────────────────────────────────────────────────────────────

    def drive(self, direction: str, distance_m: float = 0.5) -> ActionResult:
        started = now()
        if not self.connected or not self._lekiwi:
            return self._result(
                "drive", False, "Not connected.", started, direction=direction
            )

        vx = vy = 0.0
        if direction == "forward":
            vy = _XY_SPEED
        elif direction == "backward":
            vy = -_XY_SPEED
        elif direction == "left":
            vx = -_XY_SPEED
        elif direction == "right":
            vx = _XY_SPEED
        else:
            return self._result(
                "drive", False, f"Unknown direction: {direction!r}", started
            )

        speeds = _body_to_wheel_raw(vx, vy, 0.0)
        duration = distance_m / _XY_SPEED
        self._lekiwi.motor_bus.write("Goal_Speed", speeds, _WHEELS)
        time.sleep(duration)
        self._lekiwi.stop()
        return self._result(
            "drive",
            True,
            f"Drove {direction} {distance_m} m",
            started,
            direction=direction,
            distance_m=distance_m,
        )

    def rotate(self, degrees: float) -> ActionResult:
        started = now()
        if not self.connected or not self._lekiwi:
            return self._result(
                "rotate", False, "Not connected.", started, degrees=degrees
            )

        theta = _THETA_SPEED if degrees > 0 else -_THETA_SPEED
        speeds = _body_to_wheel_raw(0.0, 0.0, theta)
        duration = abs(degrees) / _THETA_SPEED
        self._lekiwi.motor_bus.write("Goal_Speed", speeds, _WHEELS)
        time.sleep(duration)
        self._lekiwi.stop()
        return self._result(
            "rotate", True, f"Rotated {degrees}°", started, degrees=degrees
        )

    def stop(self) -> ActionResult:
        started = now()
        if self._lekiwi:
            try:
                self._lekiwi.stop()
            except Exception as exc:
                return self._result("stop", False, str(exc), started)
        return self._result("stop", True, "Stopped.", started)

    # ── Arm ──────────────────────────────────────────────────────────────────

    def gesture(self, name: str) -> ActionResult:
        started = now()
        if not self.connected or not self._bus:
            return self._result(
                "gesture", False, "Not connected.", started, gesture=name
            )

        seq = _GESTURES.get(name.lower())
        if seq is None:
            available = list(_GESTURES.keys())
            return self._result(
                "gesture",
                False,
                f"Unknown gesture {name!r}. Available: {available}",
                started,
            )

        for positions, delay in seq:
            names = [
                joint
                for joint in _ARM_JOINTS
                if joint in self._present_motors
            ]
            if not names:
                return self._result(
                    "gesture",
                    False,
                    "No arm motors on bus.",
                    started,
                    gesture=name,
                )
            values = [positions[_ARM_JOINTS.index(joint)] for joint in names]
            self._bus.write("Goal_Position", values, names)
            time.sleep(delay)
        return self._result("gesture", True, f"Gesture: {name}", started, gesture=name)

    def gripper(self, action: str) -> ActionResult:
        started = now()
        if not self.connected or not self._bus:
            return self._result(
                "gripper", False, "Not connected.", started, gripper_action=action
            )

        blocked = self._require_motors([_GRIPPER], "gripper", started)
        if blocked is not None:
            blocked.data["gripper_action"] = action
            return blocked

        pos = 1200 if action.lower() in ("open", "release") else 2800
        try:
            before = int(self._bus.read("Present_Position", _GRIPPER)[0])
        except Exception as exc:
            return self._result(
                "gripper",
                False,
                f"Gripper not reachable: {exc}",
                started,
                gripper_action=action,
            )

        self._bus.write("Goal_Position", pos, _GRIPPER)
        time.sleep(0.8)
        try:
            after = int(self._bus.read("Present_Position", _GRIPPER)[0])
        except Exception as exc:
            return self._result(
                "gripper",
                False,
                f"Gripper command sent but read failed: {exc}",
                started,
                gripper_action=action,
            )

        if abs(after - before) < 20 and abs(after - pos) > 50:
            return self._result(
                "gripper",
                False,
                f"Gripper did not move (position stayed near {after}).",
                started,
                gripper_action=action,
                present_position=after,
                goal_position=pos,
            )

        return self._result(
            "gripper", True, f"Gripper {action}", started, gripper_action=action
        )

    def arm_pose(self, positions: dict[str, int]) -> ActionResult:
        """Move individual arm joints to target positions (0–4095, center=2048)."""
        started = now()
        if not self.connected or not self._bus:
            return self._result("arm_pose", False, "Not connected.", started)

        valid = set(_ARM_JOINTS + [_GRIPPER])
        unknown = [k for k in positions if k not in valid]
        if unknown:
            return self._result(
                "arm_pose",
                False,
                f"Unknown joints: {unknown}. Valid: {sorted(valid)}",
                started,
            )

        blocked = self._require_motors(list(positions.keys()), "arm_pose", started)
        if blocked is not None:
            blocked.data["positions"] = positions
            return blocked

        for joint, pos in positions.items():
            pos = max(0, min(4095, int(pos)))
            self._bus.write("Goal_Position", pos, joint)
        time.sleep(0.4)
        return self._result(
            "arm_pose", True, f"Arm posed: {positions}", started, positions=positions
        )
