"""Robot perception: microphone capture with VAD, camera helpers.

Designed for headless robot operation on Linux / Jetson.
Missing hardware or dependencies degrade gracefully.

Camera notes (Jetson):
  - USB cameras work via cv2.VideoCapture(index)
  - CSI cameras (e.g. IMX219) need a GStreamer pipeline; set
    ACTUM_CAMERA=csi (or pass source="csi") to enable it.
    Requires OpenCV built with GStreamer support (default on JetPack).

Audio notes (Jetson):
  - sounddevice uses ALSA. If the default device is wrong, set
    ACTUM_AUDIO_DEVICE to the device name or index, e.g. "hw:1,0".
"""

import base64
import io
import os
import platform
import threading
import wave
from pathlib import Path
from typing import Callable

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_MS = 30

# GStreamer pipeline for Jetson CSI cameras (IMX219 / IMX477 etc.)
# Adjust width/height/framerate to match your sensor.
_CSI_PIPELINE = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGR ! "
    "appsink drop=1 max-buffers=2 sync=false"
)


# ── Platform helpers ───────────────────────────────────────────────────────────


def _is_jetson() -> bool:
    return platform.machine() == "aarch64" and Path("/etc/nv_tegra_release").exists()


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value is not None else default


def _configure_alsa_default():
    """Configures ALSA default PCM to route through the plug plugin or PulseAudio.

    This enables automatic sample rate conversion and/or host-desktop sound sharing,
    preventing PortAudio / sounddevice / aplay errors when playing/capturing audio.
    """
    import re

    if os.name != "posix" or os.environ.get("ACTUM_AUDIO_DEVICE_CONFIGURED"):
        return

    # Check if we are using containerized PulseAudio
    pulse_socket = os.environ.get("PULSE_SERVER", "")
    if pulse_socket.startswith("unix:"):
        socket_path = pulse_socket[5:]
        if os.path.exists(socket_path):
            asound_content = (
                "pcm.!default {\n"
                "    type pulse\n"
                "}\n"
                "ctl.!default {\n"
                "    type pulse\n"
                "}\n"
            )
            try:
                with open("/etc/asound.conf", "w") as f:
                    f.write(asound_content)
                print(
                    f"[audio] Configured ALSA default PCM -> PulseAudio routing via {pulse_socket}"
                )
                os.environ["ACTUM_AUDIO_DEVICE_CONFIGURED"] = "1"
                os.environ["ACTUM_AUDIO_DEVICE"] = "default"
                return
            except Exception as e:
                print(f"[audio] Failed to write PulseAudio /etc/asound.conf: {e}")

    # Fallback to direct hardware ALSA configuration
    audio_device = os.environ.get("ACTUM_AUDIO_DEVICE", "")
    if not audio_device:
        return

    match = re.match(r"^(?:plug)?hw:(\d+)(?:,(\d+))?$", audio_device)
    if match:
        card = match.group(1)
        device = match.group(2) or "0"
        asound_content = (
            f"pcm.!default {{\n"
            f"    type plug\n"
            f'    slave.pcm "hw:{card},{device}"\n'
            f"}}\n"
            f"ctl.!default {{\n"
            f"    type hw\n"
            f"    card {card}\n"
            f"}}\n"
        )
        try:
            with open("/etc/asound.conf", "w") as f:
                f.write(asound_content)
            print(
                f"[audio] Configured ALSA default PCM -> hw:{card},{device} with plug rate converter"
            )
            os.environ["ACTUM_AUDIO_DEVICE_CONFIGURED"] = "1"
            os.environ["ACTUM_AUDIO_DEVICE"] = "default"
        except Exception as e:
            print(f"[audio] Failed to write /etc/asound.conf: {e}")


_configure_alsa_default()


# ── Camera ─────────────────────────────────────────────────────────────────────


def open_camera(source: str | int | None = None):
    """Open a camera and return a cv2.VideoCapture object, or None on failure.

    source:
        None      — auto-detect (CSI on Jetson, else index 0)
        "csi"     — Jetson CSI camera via GStreamer
        "usb"     — force USB/V4L2 index 0
        int       — explicit device index
        str path  — explicit device path or GStreamer pipeline
    """
    try:
        import cv2
    except ImportError:
        print("[camera] OpenCV not installed.")
        return None

    if source is None:
        source = _env("ACTUM_CAMERA", "auto")

    if source == "auto":
        source = "csi" if _is_jetson() else 0
    if source == "usb":
        source = 0
    if source == "csi":
        source = _CSI_PIPELINE

    if isinstance(source, str) and source != _CSI_PIPELINE:
        # Treat as device path
        cam = cv2.VideoCapture(source)
    elif isinstance(source, str):
        cam = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
    else:
        cam = cv2.VideoCapture(source)

    if cam.isOpened():
        _configure_low_latency_camera(cam)
        label = "CSI (GStreamer)" if source == _CSI_PIPELINE else f"index {source}"
        print(f"[camera] opened: {label}")
        return cam

    # GStreamer CSI failed — try USB fallback
    if source == _CSI_PIPELINE:
        print("[camera] CSI pipeline failed, trying USB index 0")
        cam = cv2.VideoCapture(0)
        if cam.isOpened():
            _configure_low_latency_camera(cam)
            print("[camera] opened: USB index 0 (fallback)")
            return cam

    print("[camera] no camera found.")
    return None


def capture_jpeg(
    cam, width: int = 320, quality: int = 70, drain_frames: int | None = None
) -> str | None:
    """Read one frame and return as base64-encoded JPEG, or None."""
    if cam is None:
        return None
    try:
        import cv2

        if drain_frames is None:
            drain_frames = _env_int("ACTUM_CAMERA_DRAIN_FRAMES", 1, 0, 5)
        for _ in range(max(0, drain_frames)):
            cam.grab()
        ok, frame = cam.read()
        if not ok:
            return None
        h, w = frame.shape[:2]
        if w > width:
            frame = cv2.resize(frame, (width, int(h * width / w)))
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf.tobytes()).decode()
    except Exception as e:
        print(f"[camera] capture error: {e}")
        return None


def _configure_low_latency_camera(cam):
    """Ask OpenCV backends to avoid accumulating stale frames."""
    try:
        import cv2

        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    value = _env(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(minimum, min(maximum, parsed))


# ── VAD ────────────────────────────────────────────────────────────────────────


class EnergyVAD:
    """Adaptive RMS energy-based voice activity detector.

    Estimates the ambient noise floor in real-time and sets speech/silence
    thresholds relative to it — no manual tuning needed across environments.

    speech starts  when rms > noise_floor * speech_ratio   (default 2×)
    speech ends    when rms < noise_floor * silence_ratio  (default 1.3×)
                   for silence_chunks consecutive 30 ms frames

    For very noisy environments (outdoor, factory floor) consider replacing
    with silero-vad: https://github.com/snakers4/silero-vad
    """

    def __init__(
        self,
        speech_ratio: float = 2.0,
        silence_ratio: float = 1.3,
        noise_floor_init: float = 0.030,  # safe starting estimate
        noise_floor_alpha: float = 0.005,  # EMA adaptation speed (~6 s to converge)
        min_speech_chunks: int = 10,  # ~300 ms
        silence_chunks: int = 25,  # ~750 ms
        pre_roll_chunks: int = 5,  # ~150 ms pre-speech padding
    ):
        self.speech_ratio = speech_ratio
        self.silence_ratio = silence_ratio
        self.noise_floor_alpha = noise_floor_alpha
        self.min_speech_chunks = min_speech_chunks
        self.silence_chunks = silence_chunks
        self.pre_roll_chunks = pre_roll_chunks

        self._noise_floor = noise_floor_init
        self._pre_roll: list[np.ndarray] = []
        self._buffer: list[np.ndarray] = []
        self._in_speech = False
        self._silence_count = 0

    def reset(self):
        """Drop any partial capture (e.g. after the mic was muted)."""
        self._pre_roll = []
        self._buffer = []
        self._in_speech = False
        self._silence_count = 0

    def process(self, chunk: np.ndarray) -> np.ndarray | None:
        """Feed one audio chunk. Returns complete utterance PCM when speech ends."""
        rms = float(np.sqrt(np.mean(chunk**2)))

        speech_thr = self._noise_floor * self.speech_ratio
        silence_thr = self._noise_floor * self.silence_ratio

        if _env("ACTUM_VAD_DEBUG"):
            print(
                f"[vad] rms={rms:.4f} floor={self._noise_floor:.4f} "
                f"s_thr={speech_thr:.4f} q_thr={silence_thr:.4f} "
                f"in_speech={self._in_speech} sc={self._silence_count}"
            )

        if not self._in_speech:
            # Adapt noise floor only while not capturing speech
            self._noise_floor = (
                1 - self.noise_floor_alpha
            ) * self._noise_floor + self.noise_floor_alpha * rms
            self._pre_roll.append(chunk)
            if len(self._pre_roll) > self.pre_roll_chunks:
                self._pre_roll.pop(0)
            if rms > speech_thr:
                self._in_speech = True
                self._silence_count = 0
                self._buffer = list(self._pre_roll)
                print(f"[vad] speech started (floor={self._noise_floor:.4f})")
        else:
            self._buffer.append(chunk)
            if rms < silence_thr:
                self._silence_count += 1
                if self._silence_count >= self.silence_chunks:
                    self._in_speech = False
                    result = None
                    if len(self._buffer) >= self.min_speech_chunks:
                        result = np.concatenate(self._buffer)
                        print(f"[vad] speech ended ({len(self._buffer) * CHUNK_MS} ms)")
                    self._buffer = []
                    self._pre_roll = []
                    return result
            else:
                # Decay instead of hard reset so intermittent background noise
                # doesn't prevent silence from accumulating.
                self._silence_count = max(0, self._silence_count - 1)

        return None


# ── Microphone capture ─────────────────────────────────────────────────────────


class AudioCapture:
    """Continuous microphone capture with VAD. Runs in a background thread.

    on_speech is called from the audio thread — use loop.call_soon_threadsafe
    to forward events to an asyncio queue (see agent.py for the pattern).

    Set ACTUM_AUDIO_DEVICE to override the default ALSA device, e.g.:
        ACTUM_AUDIO_DEVICE=hw:1,0 actum
    """

    def __init__(
        self,
        on_speech: Callable[[str], None],
        sample_rate: int = SAMPLE_RATE,
        vad: EnergyVAD | None = None,
    ):
        self._on_speech = on_speech
        self._sample_rate = sample_rate
        self._chunk_frames = int(self._sample_rate * CHUNK_MS / 1000)
        self._vad = vad or EnergyVAD()
        self._stop = threading.Event()
        self._muted = threading.Event()

    def pause(self):
        """Stop feeding audio to the VAD (e.g. while the robot speaks).

        The stream keeps reading so it stays healthy, but chunks are discarded
        and any partial utterance is dropped — otherwise the robot hears its
        own TTS output and triggers itself in a feedback loop.
        """
        self._muted.set()

    def resume(self):
        self._vad.reset()
        self._muted.clear()

    @property
    def muted(self) -> bool:
        return self._muted.is_set()

    def run(self):
        """Blocking mic loop — call in a daemon thread."""
        try:
            import sounddevice as sd
        except Exception as e:
            print(f"[mic] audio input unavailable ({e}) — microphone disabled.")
            hint = _audio_input_setup_hint(e)
            if hint:
                print(f"[mic] {hint}")
            return

        device = _env("ACTUM_AUDIO_DEVICE") or None
        selected_rate = self._sample_rate
        print(
            f"[mic] listening (device={device or 'default'}, requested {selected_rate} Hz)…"
        )

        def _run_stream(sample_rate: int):
            chunk_frames = int(sample_rate * CHUNK_MS / 1000)
            with sd.InputStream(
                device=device,
                samplerate=sample_rate,
                channels=CHANNELS,
                dtype="float32",
                blocksize=chunk_frames,
            ) as stream:
                while not self._stop.is_set():
                    chunk, _ = stream.read(chunk_frames)
                    if self._muted.is_set():
                        continue
                    utterance = self._vad.process(chunk.flatten())
                    if utterance is not None:
                        self._on_speech(pcm_to_wav_b64(utterance, sample_rate))

        try:
            _run_stream(selected_rate)
            return
        except Exception as e:
            msg = str(e)
            if "Invalid sample rate" not in msg:
                print(f"[mic] error: {e}")
                return

            try:
                info = (
                    sd.query_devices(device, "input")
                    if device is not None
                    else sd.query_devices(sd.default.device[0], "input")
                )
                fallback_rate = int(info["default_samplerate"])
            except Exception as qerr:
                print(f"[mic] error: {e} (and fallback lookup failed: {qerr})")
                return

            if fallback_rate == selected_rate:
                print(f"[mic] error: {e}")
                return

            print(f"[mic] requested rate unsupported; retrying at {fallback_rate} Hz")
            self._sample_rate = fallback_rate
            self._chunk_frames = int(self._sample_rate * CHUNK_MS / 1000)
            try:
                _run_stream(fallback_rate)
            except Exception as e2:
                print(f"[mic] error after fallback: {e2}")

    def stop(self):
        self._stop.set()


def _audio_input_setup_hint(error: Exception) -> str:
    message = str(error).lower()
    if isinstance(error, ModuleNotFoundError) or "sounddevice" in message:
        return "Install Python audio dependencies with `pip install -e .` or `pip install sounddevice soundfile`."
    if "portaudio" in message:
        return "Install the native PortAudio library, e.g. `sudo apt install libportaudio2 portaudio19-dev` on Ubuntu/Debian."
    return ""


# ── Audio helpers ──────────────────────────────────────────────────────────────


def pcm_to_wav_b64(pcm: np.ndarray, sample_rate: int) -> str:
    """Convert float32 PCM to base64-encoded WAV (no external deps)."""
    pcm16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
