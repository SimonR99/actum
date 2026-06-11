"""ROS 2 robot backend for Actum."""

from __future__ import annotations

import math
import threading
import time
from typing import Any

from actum.backends.base import RobotBackend
from actum.core.schema import ActionResult, RobotState, now


def quaternion_to_euler_yaw(x: float, y: float, z: float, w: float) -> float:
    """Compute yaw (rotation around z-axis in radians) from quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class ROS2Backend(RobotBackend):
    name = "ros2"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._node_name = str(self.config.get("node_name", "actum_node")).strip() or "actum_node"
        self._cmd_vel_topic = str(self.config.get("cmd_vel_topic", "/cmd_vel")).strip() or "/cmd_vel"
        self._odom_topic = str(self.config.get("odom_topic", "/odom")).strip() or "/odom"
        self._joint_states_topic = str(self.config.get("joint_states_topic", "/joint_states")).strip() or "/joint_states"
        self._gripper_topic = str(self.config.get("gripper_topic", "/gripper_cmd")).strip() or "/gripper_cmd"

        self._linear_speed = float(self.config.get("linear_speed", 0.25))
        self._angular_speed = float(self.config.get("angular_speed", 0.5))

        self._node = None
        self._executor = None
        self._thread = None
        self._lock = threading.Lock()

        # State cache
        self._pose = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}
        self._joints: dict[str, float] = {}

        # Publishers
        self._cmd_vel_pub = None
        self._gripper_pub = None

        # Subscribers
        self._odom_sub = None
        self._joint_sub = None

    def connect(self) -> bool:
        if self.connected:
            return True

        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from geometry_msgs.msg import Twist
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import JointState
            from std_msgs.msg import String
        except ImportError as exc:
            print(f"[ros2] ROS 2 python dependencies not installed or sourced: {exc}")
            print("[ros2] Ensure ROS 2 environment is sourced (e.g. source /opt/ros/jazzy/setup.bash)")
            return False

        try:
            if not rclpy.ok():
                rclpy.init()

            self._node = rclpy.create_node(self._node_name)
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)

            # Create publishers
            self._cmd_vel_pub = self._node.create_publisher(Twist, self._cmd_vel_topic, 10)
            self._gripper_pub = self._node.create_publisher(String, self._gripper_topic, 10)

            # Create subscribers
            self._odom_sub = self._node.create_subscription(
                Odometry, self._odom_topic, self._odom_callback, 10
            )
            self._joint_sub = self._node.create_subscription(
                JointState, self._joint_states_topic, self._joint_callback, 10
            )

            # Spin in background thread
            self._thread = threading.Thread(target=self._executor.spin, daemon=True)
            self._thread.start()

            self.connected = True
            print(f"[ros2] Node {self._node_name!r} connected successfully.")
            return True
        except Exception as exc:
            print(f"[ros2] Connection error: {exc}")
            self.close()
            return False

    def close(self):
        import rclpy
        with self._lock:
            if self._executor:
                self._executor.shutdown()
            if self._node:
                self._node.destroy_node()
            self._node = None
            self._executor = None
            self._thread = None
            self.connected = False
            self._cmd_vel_pub = None
            self._gripper_pub = None
            self._odom_sub = None
            self._joint_sub = None

    def _odom_callback(self, msg: Any):
        """Update pose cache from Odometry message."""
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        yaw_rad = quaternion_to_euler_yaw(orientation.x, orientation.y, orientation.z, orientation.w)
        with self._lock:
            self._pose = {
                "x": float(position.x),
                "y": float(position.y),
                "yaw_deg": float(math.degrees(yaw_rad)),
            }

    def _joint_callback(self, msg: Any):
        """Update joints cache from JointState message."""
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                self._joints[str(name)] = float(pos)

    def get_state(self) -> RobotState:
        with self._lock:
            pose = dict(self._pose)
            joints = dict(self._joints)
        return RobotState(
            backend=self.name,
            connected=self.connected,
            mode="hardware" if self.connected else "offline",
            pose=pose,
            joints=joints,
            metadata={
                "node_name": self._node_name,
                "cmd_vel_topic": self._cmd_vel_topic,
                "odom_topic": self._odom_topic,
                "joint_states_topic": self._joint_states_topic,
                "gripper_topic": self._gripper_topic,
            },
        )

    def speak(self, text: str):
        started = now()
        # ROS 2 doesn't have a default audio message standard; fallback to local TTS
        return self._result(
            "speak",
            True,
            f"Speech falls back to local TTS engine: {text}",
            started,
            text=text,
        )

    def drive(self, direction: str, distance_m: float = 0.5):
        started = now()
        if not self.connected or not self._cmd_vel_pub:
            return self._result(
                "drive",
                False,
                "ROS 2 backend is not connected.",
                started,
                direction=direction,
                distance_m=distance_m,
            )

        if direction == "stop":
            self.stop()
            return self._result("drive", True, "Stopped.", started, direction=direction, distance_m=0.0)

        distance = abs(float(distance_m))
        duration = distance / self._linear_speed if self._linear_speed > 0 else 0.0

        vx, vy = 0.0, 0.0
        if direction == "forward":
            vx = self._linear_speed
        elif direction == "backward":
            vx = -self._linear_speed
        elif direction == "left":
            vy = self._linear_speed
        elif direction == "right":
            vy = -self._linear_speed
        else:
            return self._result(
                "drive",
                False,
                f"Invalid direction: {direction}",
                started,
                direction=direction,
                distance_m=distance,
            )

        from geometry_msgs.msg import Twist
        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy

        # Publish Twist message periodically
        rate = 10.0
        steps = int(duration * rate)
        for _ in range(max(1, steps)):
            self._cmd_vel_pub.publish(msg)
            time.sleep(1.0 / rate)

        self.stop()
        return self._result(
            "drive",
            True,
            f"Published drive {direction} {distance:.2f} m.",
            started,
            direction=direction,
            distance_m=distance,
            duration_s=duration,
        )

    def rotate(self, degrees: float):
        started = now()
        if not self.connected or not self._cmd_vel_pub:
            return self._result(
                "rotate",
                False,
                "ROS 2 backend is not connected.",
                started,
                degrees=degrees,
            )

        angle = float(degrees)
        duration = abs(math.radians(angle)) / self._angular_speed if self._angular_speed > 0 else 0.0
        omega = -self._angular_speed if angle > 0 else self._angular_speed

        from geometry_msgs.msg import Twist
        msg = Twist()
        msg.angular.z = omega

        rate = 10.0
        steps = int(duration * rate)
        for _ in range(max(1, steps)):
            self._cmd_vel_pub.publish(msg)
            time.sleep(1.0 / rate)

        self.stop()
        return self._result(
            "rotate",
            True,
            f"Published rotate {angle:+.0f} deg.",
            started,
            degrees=angle,
            duration_s=duration,
        )

    def gripper(self, action: str):
        started = now()
        if not self.connected or not self._gripper_pub:
            return self._result(
                "gripper",
                False,
                "ROS 2 backend is not connected.",
                started,
                gripper_action=action,
            )

        from std_msgs.msg import String
        msg = String()
        msg.data = action
        self._gripper_pub.publish(msg)

        return self._result(
            "gripper",
            True,
            f"Published gripper action {action!r}.",
            started,
            gripper_action=action,
        )

    def stop(self):
        started = now()
        if self.connected and self._cmd_vel_pub:
            from geometry_msgs.msg import Twist
            msg = Twist()
            self._cmd_vel_pub.publish(msg)
            return self._result("stop", True, "Published zero velocity stop.", started)
        return self._result("stop", False, "ROS 2 backend not connected.", started)
