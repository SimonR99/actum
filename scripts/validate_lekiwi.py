#!/usr/bin/env python3
"""Validate LeKiwi physical actions against connected hardware."""

from __future__ import annotations

import sys
import time

from actum.backends.lekiwi import LeKiwiBackend, _GESTURES


def main() -> int:
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
    results: list[tuple[str, str, str]] = []

    def test(name: str, fn):
        try:
            ok, msg = fn()
            status = "OK" if ok else "FAIL"
            results.append((name, status, msg))
            print(f"[{status}] {name}: {msg}")
        except Exception as exc:
            results.append((name, "ERROR", str(exc)))
            print(f"[ERROR] {name}: {exc}")

    backend = LeKiwiBackend({"serial_port": port})
    if not backend.connect():
        print(f"Cannot connect to LeKiwi on {port}")
        return 1

    print(f"Connected on {port}\n--- Validation ---")

    for direction in ("forward", "backward", "left", "right"):
        test(
            f"drive({direction}, 0.05m)",
            lambda d=direction: ((r := backend.drive(d, 0.05)).ok, r.message),
        )
        time.sleep(0.3)

    test("rotate(+15°)", lambda: ((r := backend.rotate(15)).ok, r.message))
    time.sleep(0.3)
    test("rotate(-15°)", lambda: ((r := backend.rotate(-15)).ok, r.message))
    time.sleep(0.3)
    test("stop()", lambda: ((r := backend.stop()).ok, r.message))

    for gesture in _GESTURES:
        test(
            f"gesture({gesture})",
            lambda g=gesture: ((r := backend.gesture(g)).ok, r.message),
        )
        time.sleep(0.5)

    test("gripper(open)", lambda: ((r := backend.gripper("open")).ok, r.message))
    time.sleep(0.3)
    test("gripper(close)", lambda: ((r := backend.gripper("close")).ok, r.message))
    time.sleep(0.3)
    test("gripper(open)", lambda: ((r := backend.gripper("open")).ok, r.message))

    test(
        "arm_pose(center)",
        lambda: (
            (
                r := backend.arm_pose(
                    {
                        "shoulder_pan": 2048,
                        "shoulder_lift": 2048,
                        "elbow_flex": 2048,
                        "wrist_flex": 2048,
                        "wrist_roll": 2048,
                    }
                )
            ).ok,
            r.message,
        ),
    )
    time.sleep(0.3)
    test(
        "arm_pose(pan+100)",
        lambda: ((r := backend.arm_pose({"shoulder_pan": 2148})).ok, r.message),
    )
    time.sleep(0.3)
    test(
        "arm_pose(home via center)",
        lambda: ((r := backend.arm_pose({"shoulder_pan": 2048})).ok, r.message),
    )

    test(
        "gesture(invalid) expected fail",
        lambda: (not (r := backend.gesture("high wave")).ok, r.message),
    )
    test(
        "drive(stop) expected fail",
        lambda: (not (r := backend.drive("stop")).ok, r.message),
    )

    backend.close()

    print("\n--- Summary ---")
    passed = sum(1 for _, status, _ in results if status == "OK")
    failed = len(results) - passed
    print(f"Passed: {passed}/{len(results)}  Failed/Error: {failed}/{len(results)}")
    if failed:
        print("\nFailures:")
        for name, status, msg in results:
            if status != "OK":
                print(f"  - {name}: [{status}] {msg}")
        return 1

    print("All actions validated successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
