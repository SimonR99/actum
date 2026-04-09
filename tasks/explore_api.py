"""Explore the Unitree G1 SDK API — connection, locomotion, and arm gestures.

Tests: DDS connection, LocoClient WaveHand, G1ArmActionClient gestures.

Usage:
    python tasks/explore_api.py <network_interface>
    python tasks/explore_api.py eth0
"""

import sys
import time

# ── Setup ──────────────────────────────────────────────────────────────────────

if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <network_interface>")
    print("  Example: python tasks/explore_api.py eth0")
    sys.exit(1)

sys.path.insert(0, "unitree_sdk2_python")

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient, action_map


def check(label: str, code: int):
    status = "OK" if code == 0 else f"FAILED (code={code})"
    print(f"  [{status}] {label}")
    return code == 0


print("=" * 55)
print("Unitree G1 SDK API explorer")
print("=" * 55)

# ── Connect ────────────────────────────────────────────────────────────────────

print("\n[1] Initialising DDS channel …")
ChannelFactoryInitialize(0, sys.argv[1])
print(f"    Network interface: {sys.argv[1]}")

# ── LocoClient ─────────────────────────────────────────────────────────────────

print("\n[2] LocoClient (locomotion + arm tasks)")
loco = LocoClient()
loco.SetTimeout(10.0)
loco.Init()
print("    Connected to 'sport' service.")

print("\n  WaveHand (turn_flag=False) …")
print("  WARNING: robot will physically wave — make sure it has space.")
input("  Press Enter to continue (Ctrl-C to skip) …")
code = loco.WaveHand(turn_flag=False)
check("WaveHand", code)
time.sleep(3)

print("\n  ShakeHand …")
input("  Press Enter to continue (Ctrl-C to skip) …")
code = loco.ShakeHand(stage=0)
check("ShakeHand(stage=0)", code)
time.sleep(2)
code = loco.ShakeHand(stage=-1)  # toggle off
check("ShakeHand(stage=-1)", code)
time.sleep(1)

# ── G1ArmActionClient ──────────────────────────────────────────────────────────

print("\n[3] G1ArmActionClient (pre-defined gestures)")
arm = G1ArmActionClient()
arm.SetTimeout(10.0)
arm.Init()
print("    Connected to 'arm' service.")

print("\n  Available gestures in action_map:")
for name, action_id in sorted(action_map.items(), key=lambda x: x[1]):
    print(f"    {action_id:3d}  {name}")

print("\n  Testing 'high wave' gesture …")
input("  Press Enter to continue (Ctrl-C to skip) …")
action_id = action_map.get("high wave")
if action_id is not None:
    code = arm.ExecuteAction(action_id)
    check(f"ExecuteAction({action_id} = 'high wave')", code)
    time.sleep(4)
else:
    print("  [SKIP] 'high wave' not in action_map — check SDK version")

print("\n  Releasing arm (returning to neutral) …")
release_id = action_map.get("release arm", 99)
code = arm.ExecuteAction(release_id)
check(f"ExecuteAction({release_id} = 'release arm')", code)
time.sleep(2)

# ── Summary ────────────────────────────────────────────────────────────────────

print("\n[Done] API exploration complete.")
print("\nKey findings:")
print(f"  - LocoClient.WaveHand()       → quick wave via locomotion service")
print(f"  - G1ArmActionClient.ExecuteAction(id) → gesture library ({len(action_map)} gestures)")
print(f"  - Gesture IDs: {dict(list(sorted(action_map.items(), key=lambda x:x[1]))[:5])} …")
