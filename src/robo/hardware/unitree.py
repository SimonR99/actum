"""Unitree G1 hardware backend.

Wraps the Unitree SDK2 Python clients into a clean interface used by the agent.
The unitree_sdk2_python package must be on sys.path — either installed or present
as a sibling directory (the agent/server handle this automatically if
unitree_sdk2_python/ exists in the workspace root).

Provides:
  - TTS via the robot's built-in speaker (AudioClient.TtsMaker)
  - PCM audio stream (AudioClient.PlayStream)
  - Gestures: wave, shake hand, clap, … (G1ArmActionClient + LocoClient)
  - LED control
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


# Ensure the SDK is importable from the workspace root sibling directory.
_SDK_PATH = Path(__file__).parents[4] / "unitree_sdk2_python"
if _SDK_PATH.exists() and str(_SDK_PATH) not in sys.path:
    sys.path.insert(0, str(_SDK_PATH))


# All gesture names available via G1ArmActionClient.
# Populated from action_map after first successful import.
GESTURES: list[str] = []


class UnitreeG1:
    """Hardware backend for the Unitree G1 humanoid robot.

    Usage::

        hw = UnitreeG1(network_interface="eth0", speaker_id=0, volume=80)
        hw.connect()          # initialises DDS; call once at startup
        hw.speak("Hello!")    # robot's built-in TTS
        hw.wave()             # pre-defined wave gesture
    """

    def __init__(
        self,
        network_interface: str,
        speaker_id: int = 0,
        volume: int = 80,
    ):
        self._interface = network_interface
        self._speaker_id = speaker_id
        self._volume = volume

        self._audio = None
        self._loco = None
        self._arm = None
        self._connected = False

    # ── Connection ─────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Initialise DDS and connect to the robot services.

        Safe to call multiple times — subsequent calls are no-ops.
        Returns True on success, False if the SDK is unavailable.
        """
        if self._connected:
            return True

        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
            from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
            from unitree_sdk2py.g1.arm.g1_arm_action_client import (
                G1ArmActionClient,
                action_map,
            )
        except ImportError as e:
            print(f"[unitree] SDK not available: {e}")
            print(f"[unitree] Expected SDK at: {_SDK_PATH}")
            return False

        print(f"[unitree] Connecting via interface '{self._interface}' …")
        ChannelFactoryInitialize(0, self._interface)

        self._audio = AudioClient()
        self._audio.SetTimeout(10.0)
        self._audio.Init()

        self._loco = LocoClient()
        self._loco.SetTimeout(10.0)
        self._loco.Init()

        self._arm = G1ArmActionClient()
        self._arm.SetTimeout(10.0)
        self._arm.Init()

        # Populate global gesture list from the SDK's action_map
        global GESTURES
        GESTURES = list(action_map.keys())

        # Apply initial volume
        self._audio.SetVolume(self._volume)

        self._connected = True
        print(f"[unitree] Connected. {len(GESTURES)} gestures available.")
        return True

    # ── Audio ──────────────────────────────────────────────────────────────

    def speak(self, text: str) -> bool:
        """Speak text using the robot's built-in TTS engine.

        Supports English and Chinese. Blocks until synthesis starts (not
        until playback finishes — allow ~1 s/sentence before the next call).
        """
        if not self._audio:
            print(f"[unitree] speak: not connected — would say: {text!r}")
            return False
        code = self._audio.TtsMaker(text, self._speaker_id)
        if code != 0:
            print(f"[unitree] TtsMaker failed (code={code})")
        return code == 0

    def play_pcm(self, pcm_bytes: bytes, app_name: str = "robo", chunk_size: int = 32000) -> bool:
        """Stream raw 16-bit 16 kHz mono PCM audio.

        chunk_size: bytes per DDS message (default 32000 = 1 s).
        """
        if not self._audio:
            return False
        import time
        for i in range(0, len(pcm_bytes), chunk_size):
            code, _ = self._audio.PlayStream(app_name, "stream", pcm_bytes[i : i + chunk_size])
            if code != 0:
                print(f"[unitree] PlayStream chunk failed (code={code})")
                return False
            time.sleep(chunk_size / 32000)  # ~1 s per 32000-byte chunk at 16 kHz
        return True

    def stop_audio(self, app_name: str = "robo") -> bool:
        if not self._audio:
            return False
        return self._audio.PlayStop(app_name) == 0

    def set_volume(self, volume: int) -> bool:
        """Set playback volume (0–100)."""
        if not self._audio:
            return False
        self._volume = max(0, min(100, volume))
        return self._audio.SetVolume(self._volume) == 0

    def set_led(self, r: int, g: int, b: int) -> bool:
        """Set the head LED colour (0–255 per channel)."""
        if not self._audio:
            return False
        return self._audio.LedControl(r, g, b) == 0

    # ── Motion ─────────────────────────────────────────────────────────────

    def wave(self, turn: bool = False) -> bool:
        """Perform a quick wave gesture via the locomotion service.

        turn=True: wave and turn around (greeting while rotating).
        """
        if not self._loco:
            print("[unitree] wave: not connected — stub")
            return False
        code = self._loco.WaveHand(turn_flag=turn)
        return code == 0

    def gesture(self, name: str) -> bool:
        """Execute a named gesture from the G1 gesture library.

        Available names are in hardware.unitree.GESTURES (populated after connect()).

        Common gestures:
            'high wave'      — both arms waving high
            'face wave'      — wave near face
            'shake hand'     — extend hand for handshake
            'clap'           — clapping motion
            'high five'      — raise hand for high five
            'hands up'       — both hands up
            'release arm'    — return arms to neutral (always safe to call)
        """
        if not self._arm:
            print(f"[unitree] gesture '{name}': not connected — stub")
            return False

        try:
            from unitree_sdk2py.g1.arm.g1_arm_action_client import action_map
        except ImportError:
            return False

        action_id = action_map.get(name)
        if action_id is None:
            available = ", ".join(sorted(action_map.keys()))
            print(f"[unitree] unknown gesture '{name}'. Available: {available}")
            return False

        code = self._arm.ExecuteAction(action_id)
        return code == 0

    def release_arm(self) -> bool:
        """Return the arm to its neutral resting position."""
        return self.gesture("release arm")
