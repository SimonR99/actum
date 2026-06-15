#!/usr/bin/env python3
"""Run lerobot ik_arm.py move_to_pose test (keyboard import stubbed)."""

from __future__ import annotations

import sys
from types import ModuleType

sys.modules["keyboard"] = ModuleType("keyboard")

import numpy as np

from ik_arm import IKArmController

TARGET = np.array([0.2, 0.0, 0.15])


def main() -> int:
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
    print(f"IK test on {port}, target xyz (m): {TARGET.tolist()}")
    controller = IKArmController(port=port)
    try:
        success = controller.move_to_pose(TARGET)
        print("RESULT:", "success" if success else "failed")
        return 0 if success else 1
    finally:
        controller.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
