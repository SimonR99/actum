from robo.backends.unitree_g1 import UnitreeG1Backend


class FakeLoco:
    def __init__(self):
        self.calls = []

    def SetVelocity(self, vx, vy, omega, duration):
        self.calls.append(("velocity", vx, vy, omega, duration))
        return 0

    def SetTaskId(self, task_id):
        self.calls.append(("task", task_id))
        return 0


def test_unitree_drive_sends_bounded_velocity_and_stop(monkeypatch):
    monkeypatch.setattr("robo.backends.unitree_g1.time.sleep", lambda _duration: None)
    loco = FakeLoco()
    robot = UnitreeG1Backend({"network_interface": "eth0"})
    robot._loco = loco

    result = robot.drive("forward", 0.5)

    assert result.ok is True
    assert loco.calls[0] == ("velocity", 0.25, 0.0, 0.0, 2.0)
    assert loco.calls[-1] == ("velocity", 0.0, 0.0, 0.0, 0.1)


def test_unitree_gesture_uses_action_map(monkeypatch):
    class FakeArm:
        def ExecuteAction(self, action_id):
            self.action_id = action_id
            return 0

    import types
    import sys

    module_name = "unitree_sdk2py.g1.arm.g1_arm_action_client"
    fake_module = types.SimpleNamespace(action_map={"high wave": 26})
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    loco = FakeLoco()
    robot = UnitreeG1Backend({"network_interface": "eth0"})
    robot._loco = loco
    robot._arm = FakeArm()

    result = robot.gesture("high wave")

    assert result.ok is True
    assert result.data["action_id"] == 26
