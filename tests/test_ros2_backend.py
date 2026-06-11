import sys
import types
from unittest.mock import MagicMock

import pytest
from actum.backends.ros2 import ROS2Backend, quaternion_to_euler_yaw


class FakeTwist:
    def __init__(self):
        class Vector3:
            def __init__(self):
                self.x = 0.0
                self.y = 0.0
                self.z = 0.0
        self.linear = Vector3()
        self.angular = Vector3()


class FakeString:
    def __init__(self):
        self.data = ""


class FakeOdometry:
    def __init__(self):
        class Pose:
            def __init__(self):
                class PoseWithCovariance:
                    def __init__(self):
                        class Point:
                            def __init__(self):
                                self.x = 1.0
                                self.y = 2.0
                                self.z = 0.0
                        class Quaternion:
                            def __init__(self):
                                self.x = 0.0
                                self.y = 0.0
                                self.z = 0.0
                                self.w = 1.0
                        self.position = Point()
                        self.orientation = Quaternion()
                self.pose = PoseWithCovariance()
        self.pose = Pose()


class FakeJointState:
    def __init__(self):
        self.name = ["joint1", "joint2"]
        self.position = [0.5, -0.2]


@pytest.fixture
def mock_ros2(monkeypatch):
    """Sets up a complete set of mocked ROS 2 modules."""
    mock_rclpy = MagicMock()
    mock_rclpy.ok.return_value = False

    mock_node = MagicMock()
    mock_rclpy.create_node.return_value = mock_node

    mock_exec = MagicMock()
    mock_rclpy.executors.SingleThreadedExecutor.return_value = mock_exec

    # Mock python modules
    sys.modules["rclpy"] = mock_rclpy
    sys.modules["rclpy.executors"] = types.SimpleNamespace(SingleThreadedExecutor=mock_rclpy.executors.SingleThreadedExecutor)
    sys.modules["geometry_msgs"] = types.SimpleNamespace()
    sys.modules["geometry_msgs.msg"] = types.SimpleNamespace(Twist=FakeTwist)
    sys.modules["nav_msgs"] = types.SimpleNamespace()
    sys.modules["nav_msgs.msg"] = types.SimpleNamespace(Odometry=FakeOdometry)
    sys.modules["sensor_msgs"] = types.SimpleNamespace()
    sys.modules["sensor_msgs.msg"] = types.SimpleNamespace(JointState=FakeJointState)
    sys.modules["std_msgs"] = types.SimpleNamespace()
    sys.modules["std_msgs.msg"] = types.SimpleNamespace(String=FakeString)

    yield {
        "rclpy": mock_rclpy,
        "node": mock_node,
        "executor": mock_exec,
    }

    # Clean up sys.modules
    for mod in ["rclpy", "rclpy.executors", "geometry_msgs", "geometry_msgs.msg", "nav_msgs", "nav_msgs.msg", "sensor_msgs", "sensor_msgs.msg", "std_msgs", "std_msgs.msg"]:
        sys.modules.pop(mod, None)


def test_quaternion_to_euler_yaw():
    # Facing forward (yaw = 0)
    yaw = quaternion_to_euler_yaw(0.0, 0.0, 0.0, 1.0)
    assert abs(yaw) < 1e-5

    # Facing left (+90 deg / +1.5708 rad)
    # w = cos(45 deg) = 0.7071, z = sin(45 deg) = 0.7071
    yaw_left = quaternion_to_euler_yaw(0.0, 0.0, 0.7071068, 0.7071068)
    assert abs(yaw_left - 1.570796) < 1e-4


def test_ros2_backend_config_defaults():
    backend = ROS2Backend()
    assert backend._node_name == "actum_node"
    assert backend._cmd_vel_topic == "/cmd_vel"
    assert backend._odom_topic == "/odom"
    assert backend._joint_states_topic == "/joint_states"
    assert backend._gripper_topic == "/gripper_cmd"


def test_ros2_backend_connect_and_disconnect(mock_ros2):
    backend = ROS2Backend({
        "node_name": "test_actum_node",
        "cmd_vel_topic": "/cmd_vel_test",
    })
    
    assert backend.connected is False
    assert backend.connect() is True
    assert backend.connected is True

    mock_ros2["rclpy"].init.assert_called_once()
    mock_ros2["rclpy"].create_node.assert_called_once_with("test_actum_node")

    # Check publishers & subscribers
    assert backend._cmd_vel_pub is not None
    assert backend._gripper_pub is not None
    assert backend._odom_sub is not None
    assert backend._joint_sub is not None

    state = backend.get_state()
    assert state.connected is True
    assert state.metadata["node_name"] == "test_actum_node"
    assert state.metadata["cmd_vel_topic"] == "/cmd_vel_test"

    backend.close()
    assert backend.connected is False
    assert backend._node is None


def test_ros2_backend_callbacks(mock_ros2):
    backend = ROS2Backend()
    assert backend.connect() is True

    # Test odom callback
    msg_odom = FakeOdometry()
    # Let's set a yaw (e.g. facing left / 90 degrees)
    msg_odom.pose.pose.orientation.z = 0.7071068
    msg_odom.pose.pose.orientation.w = 0.7071068
    msg_odom.pose.pose.position.x = 5.0
    msg_odom.pose.pose.position.y = -3.0

    backend._odom_callback(msg_odom)
    state = backend.get_state()
    assert state.pose["x"] == 5.0
    assert state.pose["y"] == -3.0
    assert abs(state.pose["yaw_deg"] - 90.0) < 1e-2

    # Test joint states callback
    msg_joints = FakeJointState()
    backend._joint_callback(msg_joints)
    state = backend.get_state()
    assert state.joints["joint1"] == 0.5
    assert state.joints["joint2"] == -0.2


def test_ros2_backend_drive(mock_ros2, monkeypatch):
    monkeypatch.setattr("actum.backends.ros2.time.sleep", lambda _x: None)
    
    backend = ROS2Backend()
    assert backend.connect() is True

    # Mock publish on cmd_vel
    published_msgs = []
    backend._cmd_vel_pub.publish = published_msgs.append

    res = backend.drive("forward", 0.5)
    assert res.ok is True
    assert len(published_msgs) > 0
    # First message should move forward
    assert published_msgs[0].linear.x == 0.25
    assert published_msgs[0].linear.y == 0.0
    # Last message should be stop (zero velocity)
    assert published_msgs[-1].linear.x == 0.0
    assert published_msgs[-1].linear.y == 0.0


def test_ros2_backend_rotate(mock_ros2, monkeypatch):
    monkeypatch.setattr("actum.backends.ros2.time.sleep", lambda _x: None)
    
    backend = ROS2Backend()
    assert backend.connect() is True

    # Mock publish on cmd_vel
    published_msgs = []
    backend._cmd_vel_pub.publish = published_msgs.append

    res = backend.rotate(45.0)  # Rotate clockwise (negative yaw velocity)
    assert res.ok is True
    assert len(published_msgs) > 0
    assert published_msgs[0].angular.z < 0.0  # clockwise is negative
    assert published_msgs[-1].angular.z == 0.0


def test_ros2_backend_gripper(mock_ros2):
    backend = ROS2Backend()
    assert backend.connect() is True

    published_msgs = []
    backend._gripper_pub.publish = published_msgs.append

    res = backend.gripper("open")
    assert res.ok is True
    assert len(published_msgs) == 1
    assert published_msgs[0].data == "open"
