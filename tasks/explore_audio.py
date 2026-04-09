"""Explore the Unitree G1 audio API.

Tests: volume get/set, TTS (both speakers), LED control, PCM stream playback.

Usage:
    python tasks/explore_audio.py <network_interface>
    python tasks/explore_audio.py eth0

The network interface is the one connected to the robot (check with `ip link`).
"""

import sys
import time

# ── Setup ──────────────────────────────────────────────────────────────────────

if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <network_interface>")
    print("  Example: python tasks/explore_audio.py eth0")
    sys.exit(1)

sys.path.insert(0, "unitree_sdk2_python")

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient


def check(label: str, code: int, extra: str = ""):
    status = "OK" if code == 0 else f"FAILED (code={code})"
    print(f"  [{status}] {label}" + (f" — {extra}" if extra else ""))
    return code == 0


print("=" * 55)
print("Unitree G1 Audio API explorer")
print("=" * 55)

# ── Connect ────────────────────────────────────────────────────────────────────

print("\n[1] Connecting …")
ChannelFactoryInitialize(0, sys.argv[1])

client = AudioClient()
client.SetTimeout(10.0)
client.Init()
print("    Connected to 'voice' service.")

# ── Volume ─────────────────────────────────────────────────────────────────────

print("\n[2] Volume")
code, vol = client.GetVolume()
if code == 0:
    print(f"  [OK ] GetVolume → {vol}")
else:
    print(f"  [ERR] GetVolume failed (code={code})")

code = client.SetVolume(80)
check("SetVolume(80)", code)

# ── Text-to-speech ─────────────────────────────────────────────────────────────

print("\n[3] Text-to-speech (TtsMaker)")
print("  Testing speaker 0 …")
code = client.TtsMaker("Hello! I am the Unitree G1 robot.", 0)
check("TtsMaker(speaker_id=0)", code)
time.sleep(3)

print("  Testing speaker 1 …")
code = client.TtsMaker("Testing voice number one.", 1)
check("TtsMaker(speaker_id=1)", code)
time.sleep(3)

# ── PCM stream playback ────────────────────────────────────────────────────────

print("\n[4] PCM stream playback (PlayStream / PlayStop)")
print("  Generating a 440 Hz test tone (1 s, 16-bit 16 kHz mono) …")
try:
    import numpy as np
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 32767 * 0.5).astype("<i2")
    pcm_bytes = tone.tobytes()

    app_name = "robo_explore"
    stream_id = "tone_test"
    chunk = 16000 * 2  # 0.5 s per chunk

    for i in range(0, len(pcm_bytes), chunk):
        code, _ = client.PlayStream(app_name, stream_id, pcm_bytes[i : i + chunk])
        if code != 0:
            print(f"  [ERR] PlayStream chunk failed (code={code})")
            break
    else:
        print("  [OK ] PlayStream — tone sent")
    time.sleep(1.2)

    code = client.PlayStop(app_name)
    check("PlayStop", code)

except ImportError:
    print("  [SKIP] numpy not available — skipping PCM stream test")

# ── LED ────────────────────────────────────────────────────────────────────────

print("\n[5] LED control (LedControl)")
for label, r, g, b in [("red", 255, 0, 0), ("green", 0, 255, 0), ("blue", 0, 0, 255), ("off", 0, 0, 0)]:
    code = client.LedControl(r, g, b)
    check(f"LedControl({label})", code)
    time.sleep(0.5)

# ── Summary ────────────────────────────────────────────────────────────────────

print("\n[Done] Audio API exploration complete.")
