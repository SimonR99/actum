#!/usr/bin/env python3
"""Validate RobotTools actions against LeKiwi hardware."""

from __future__ import annotations

import sys
import time

from actum.runtime import RobotRuntime
from actum.tools import RobotTools


class FakeAgent:
    def __init__(self, runtime):
        self.runtime = runtime
        self.config = runtime.config

    def queue_speech(self, text):
        pass


def main() -> int:
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
    runtime = RobotRuntime(
        {"robot": {"backend": "lekiwi", "lekiwi": {"serial_port": port}}},
        "validator",
    )
    if not runtime.connect():
        print("Runtime connect failed")
        return 1

    tools = RobotTools(FakeAgent(runtime))
    checks = [
        ("navigate forward", lambda: tools.navigate("forward", 0.05)),
        ("navigate backward", lambda: tools.navigate("backward", 0.05)),
        ("rotate +10", lambda: tools.rotate(10)),
        ("rotate -10", lambda: tools.rotate(-10)),
        ("wave home", lambda: tools.wave("home")),
        ("wave yes", lambda: tools.wave("yes")),
        ("gripper open", lambda: tools.gripper("open")),
        ("gripper close", lambda: tools.gripper("close")),
        ("arm_pose", lambda: tools.arm_pose(2048, 2048, 2048, 2048, 2048)),
        ("navigate stop (expected fail)", lambda: tools.navigate("stop")),
    ]

    print("--- RobotTools via lekiwi backend ---")
    failed = 0
    for name, fn in checks:
        msg = fn()
        if "expected fail" in name:
            ok = "Unknown direction" in msg or "Invalid" in msg
        else:
            ok = (
                "No robot backend" not in msg
                and "Not connected" not in msg
                and "failed" not in msg.lower()
            )
        status = "OK" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"[{status}] {name}: {msg}")
        time.sleep(0.3)

    runtime.close()
    print(f"\nTools layer: {len(checks) - failed}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
