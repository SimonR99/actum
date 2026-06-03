"""Core robot agent.

Design:
  - Multi-step tool calling: LLM chains navigate → look → speak → done
  - Robot action tools (navigate, rotate, gripper, look, remember, …)
  - Headless operation: sounddevice audio, OpenCV camera
  - Persistent memory across turns
  - Structured action log
  - Optional WebSocket monitoring server (see server.py)

Quickstart (headless):
    MODEL_PATH=/path/to/model.litertlm robo

With monitoring dashboard:
    MODEL_PATH=/path/to/model.litertlm robo-server
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import sys
import time
import threading
from pathlib import Path

import numpy as np

from robo import tts as tts_module
from robo.runtime import RobotRuntime
from robo.tools import RobotTools
from robo.perception import AudioCapture, open_camera, capture_jpeg

try:
    import litert_lm
except ImportError:
    litert_lm = None


# ── Model config ───────────────────────────────────────────────────────────────

HF_REPO = "litert-community/gemma-4-E2B-it-litert-lm"
HF_FILENAME = "gemma-4-E2B-it.litertlm"

SYSTEM_PROMPT_TEMPLATE = """\
You are {robot_name}, an autonomous robot agent. You have a camera, microphone, speaker, \
wheels, and a gripper. You perceive the world through your sensors and act through your tools.

When you receive input (a voice command, a camera frame, or a scheduled event):
1. Call set_plan() for non-trivial tasks so the operator can see your intent.
2. Use look() to observe your surroundings before navigating or picking objects up.
3. Chain tool calls in sequence to accomplish the goal.
4. Use mark_step() as you move through the plan.
5. Use remember() to store facts you will need in future turns.
6. Always finish by calling done() with a one-sentence summary.

Be concise when speaking (1-2 sentences). Prefer action over explanation. \
If you are uncertain, call report_status() to communicate your reasoning."""


# ── Agent ──────────────────────────────────────────────────────────────────────

class RobotAgent:
    """On-device robot agent: multimodal perception + agentic multi-step tool calling."""

    def __init__(self, config_path: str | Path | None = None):
        self.memory: dict[str, str] = {}
        self._config_path = Path(config_path) if config_path else Path(__file__).resolve().parents[2] / "config.json"
        self.config = _load_config(self._config_path)
        self.engine: litert_lm.Engine | None = None
        self.conversation = None
        self.tts_backend: tts_module.TTSBackend | None = None
        self.tools: RobotTools | None = None
        self.runtime = RobotRuntime(self.config, self.get_name())

        # Per-turn state (reset at the start of each process_event call)
        self._pending_speech: list[str] = []
        self._pending_frame: str | None = None  # base64 JPEG

        # Thread-safe event queue: anything that triggers the agent goes here
        self.event_bus: asyncio.Queue = asyncio.Queue()

        # Subscribers receive a status dict after each completed turn (for server.py)
        self._status_subscribers: list[asyncio.Queue] = []

        self._action_log: list[dict] = []
        self._camera = None

    # ── Startup / shutdown ─────────────────────────────────────────────────

    def load_models(self):
        """Load LLM + TTS and wire up tools. Blocking — run via executor."""
        if litert_lm is None:
            raise RuntimeError(
                "litert-lm is not installed. Install project dependencies with "
                "`pip install -e .` before starting the agent."
            )

        model_path = _resolve_model_path()

        print(f"Loading LLM from {model_path}…")
        self.engine = litert_lm.Engine(
            model_path,
            backend=litert_lm.Backend.GPU,
            vision_backend=litert_lm.Backend.GPU,
            audio_backend=litert_lm.Backend.CPU,
        )
        self.engine.__enter__()

        self.tts_backend = tts_module.load()
        self.tools = RobotTools(self)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(robot_name=self.get_name())
        self.conversation = self.engine.create_conversation(
            messages=[{"role": "system", "content": system_prompt}],
            tools=self.tools.get_tools(),
        )
        self.conversation.__enter__()

        self._camera = open_camera()
        self._init_backend()
        print("Robot agent ready.")

    def shutdown(self):
        if self.conversation:
            self.conversation.__exit__(None, None, None)
        if self.engine:
            self.engine.__exit__(None, None, None)
        if self._camera:
            self._camera.release()
        self.runtime.close()

    def _init_backend(self):
        """Initialise the configured robot backend."""
        if self.runtime.connect():
            print(f"[backend] {self.runtime.backend.name} connected")
        else:
            print(f"[backend] {self.runtime.backend.name} unavailable")

    # ── Perception ─────────────────────────────────────────────────────────

    def capture_frame(self) -> str | None:
        """Return a base64 JPEG from the camera, or None."""
        return capture_jpeg(self._camera)

    # ── Core agentic loop ──────────────────────────────────────────────────

    async def process_event(self, event: dict) -> list[dict]:
        """Process one triggering event through the agentic tool-calling loop.

        The LLM receives the event and calls tools in sequence (look, navigate,
        speak, remember, …) until it calls done(). All of this happens within
        a single conversation.send_message() call — litert_lm + Gemma 4 handle
        the multi-step tool loop internally.

        Event dict keys:
            source  : 'voice' | 'vision' | 'text' | 'timer'
            audio   : base64 WAV  (optional)
            image   : base64 JPEG (optional)
            text    : plain-text command (optional)

        Returns a list of action records (one per tool call).
        """
        self.tools._reset()
        self._pending_speech.clear()
        self._pending_frame = None

        content = self._build_content(event)
        t0 = time.time()

        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.conversation.send_message({"role": "user", "content": content}),
        )

        # If the LLM called look(), _pending_frame is now set. We can't inject an
        # image mid-inference, so we do one follow-up message with the actual frame.
        if self._pending_frame:
            frame = self._pending_frame
            self._pending_frame = None
            follow_up = [
                {"type": "image", "blob": frame},
                {"type": "text", "text": "This is the camera frame you requested with look(). Describe what you see and continue your task."},
            ]
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.conversation.send_message({"role": "user", "content": follow_up}),
            )

        elapsed = time.time() - t0
        action_types = [a["type"] for a in self.tools.actions_taken]
        print(f"turn ({elapsed:.2f}s) | {' → '.join(action_types) or 'no actions'}")

        # Execute queued speech after the full tool chain so audio doesn't
        # interleave with ongoing LLM inference.
        for text in self._pending_speech:
            await self._speak(text)

        actions = list(self.tools.actions_taken)
        self._action_log.extend(actions)
        self._broadcast({"source": event.get("source", "?"), "actions": actions, "elapsed": elapsed})
        return actions

    def _build_content(self, event: dict) -> list[dict]:
        content: list[dict] = []

        if event.get("audio"):
            content.append({"type": "audio", "blob": event["audio"]})
        if event.get("image"):
            content.append({"type": "image", "blob": event["image"]})

        src = event.get("source", "")
        if src == "voice" and event.get("image"):
            instruction = (
                "The user spoke to you while showing their camera. "
                "Accomplish what they asked, referencing what you see. "
                "Chain your tools, then call done()."
            )
        elif src == "voice":
            instruction = (
                "The user spoke to you. First call speak() to respond verbally — "
                "acknowledge what they said and give your answer or status. "
                "Then use any other tools needed to accomplish the request. "
                "Always call done() last."
            )
        elif src == "vision":
            instruction = (
                "New camera frame received. Analyse the scene; act if anything requires "
                "attention. Call done() when finished."
            )
        elif src == "timer":
            instruction = (
                f"Scheduled check-in. {event.get('text', 'Perform a brief environment check.')} "
                "Call done() when finished."
            )
        else:
            txt = event.get("text", "Perform a brief environment check.")
            instruction = f"{txt} Use your tools as needed, then call done()."

        if self.memory:
            mem = "\n".join(f"  {k}: {v}" for k, v in self.memory.items())
            instruction += f"\n\nCurrent memory:\n{mem}"

        content.append({"type": "text", "text": instruction})
        return content

    # ── Audio output ───────────────────────────────────────────────────────

    async def _speak(self, text: str):
        if not text.strip():
            return

        mode = self.get_mode()
        if mode == "unitree":
            if self.runtime.backend.name == "unitree_g1" and self.runtime.backend.connected:
                result = await asyncio.get_event_loop().run_in_executor(None, lambda: self.runtime.backend.speak(text))
                if result.ok:
                    return
                print(f"[tts] unitree speak failed; falling back to local ({result.message})")
            else:
                print("[tts] unitree mode selected but Unitree backend is not connected; falling back to local")

        if not self.tts_backend:
            print("[tts] no backend loaded — skipping speech")
            return

        print(f"[tts] generating: {text!r}")
        try:
            pcm = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.tts_backend.generate(text)
            )
            print(f"[tts] playing {len(pcm) / self.tts_backend.sample_rate:.2f}s of audio")
            await self._play_audio(pcm)
        except Exception as e:
            print(f"[tts] error: {e}")

    async def _play_audio(self, pcm: np.ndarray):
        sr = self.tts_backend.sample_rate
        try:
            import sounddevice as sd
            device = os.environ.get("ROBO_AUDIO_DEVICE") or None
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: sd.play(pcm, samplerate=sr, device=device, blocking=True)
            )
        except Exception as e:
            print(f"[tts] sounddevice playback failed ({e}), trying aplay/afplay fallback")
            import soundfile as sf
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                path = f.name
            sf.write(path, pcm, sr)
            cmd = f"aplay '{path}' || afplay '{path}'"
            ret = await asyncio.get_event_loop().run_in_executor(None, lambda: os.system(cmd))
            if ret != 0:
                print(f"[tts] fallback playback also failed (exit {ret})")

    # ── Monitoring ─────────────────────────────────────────────────────────

    def _broadcast(self, payload: dict):
        payload["timestamp"] = time.time()
        for q in list(self._status_subscribers):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    # ── Main loop ──────────────────────────────────────────────────────────

    async def run(self):
        """Drain event_bus and process events. Exits when None is enqueued."""
        while True:
            event = await self.event_bus.get()
            if event is None:
                break
            try:
                await self.process_event(event)
            except Exception as e:
                print(f"[agent] error: {e}")

    # ── Runtime config ────────────────────────────────────────────────────

    def get_mode(self) -> str:
        mode = str(self.config.get("tts", "local")).lower()
        return mode if mode in {"local", "unitree"} else "local"

    def get_name(self) -> str:
        name = str(self.config.get("name", "spacewalker")).strip()
        return name if name else "spacewalker"

    def set_mode(self, mode: str, persist: bool = True) -> tuple[bool, str]:
        mode = str(mode).lower().strip()
        if mode not in {"local", "unitree"}:
            return False, "Invalid mode. Use 'local' or 'unitree'."

        self.config["tts"] = mode

        if persist:
            try:
                _persist_tts_mode(self._config_path, mode)
            except Exception as exc:
                return False, f"Mode changed in memory but failed to save config: {exc}"

        if mode == "unitree" and not (self.runtime.backend.name == "unitree_g1" and self.runtime.backend.connected):
            return True, "Mode set to unitree, but hardware is not connected. Local fallback will be used."

        return True, f"Mode set to {mode}."


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve_model_path() -> str:
    path = os.environ.get("MODEL_PATH", "")
    if path:
        expanded = sorted(glob.glob(os.path.expanduser(path)))
        return expanded[0] if expanded else path
    from huggingface_hub import hf_hub_download
    print(f"Downloading {HF_REPO}/{HF_FILENAME} (first run only)…")
    return hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)


def _load_config(config_path: str | Path | None = None) -> dict:
    """Load runtime config with sensible defaults.

    Search order:
      1) explicit config_path
      2) workspace-root config.json
    """
    default = {
        "name": "spacewalker",
        "tts": "local",
        "robot": {
            "backend": "fake",
            "fake": {},
            "unitree_g1": {
                "network_interface": "eth0",
                "speaker_id": 0,
                "volume": 80,
            },
            "lekiwi": {
                "remote_ip": "127.0.0.1",
                "port": 5555,
                "id": "lekiwi",
            },
        },
        "hardware": {
            "enabled": False,
            "type": "unitree_g1",
            "network_interface": "eth0",
            "speaker_id": 0,
            "volume": 80,
        },
    }

    path = Path(config_path) if config_path else Path(__file__).resolve().parents[2] / "config.json"
    if not path.exists():
        return default

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        print(f"[config] Failed to load {path}: {exc}")
        return default

    cfg = dict(default)
    if isinstance(raw, dict):
        cfg["name"] = raw.get("name", cfg["name"])
        cfg["tts"] = raw.get("tts", cfg["tts"])
        robot = raw.get("robot", {}) if isinstance(raw.get("robot"), dict) else {}
        cfg["robot"] = {**default["robot"], **robot}
        hw = raw.get("hardware", {}) if isinstance(raw.get("hardware"), dict) else {}
        cfg["hardware"] = {**default["hardware"], **hw}
        if "robot" not in raw and cfg["hardware"].get("enabled"):
            cfg["robot"]["backend"] = ""
    return cfg


def _persist_tts_mode(config_path: str | Path, mode: str):
    """Persist TTS mode while preserving any other config keys."""
    path = Path(config_path)
    data: dict = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            data = raw
    data["tts"] = mode
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

def cli():
    """Headless entrypoint: mic → agent → speaker + stdin commands."""
    asyncio.run(_run_headless())


async def _run_headless():
    loop = asyncio.get_event_loop()
    agent = RobotAgent()

    print("Loading models…")
    await loop.run_in_executor(None, agent.load_models)

    def on_speech(wav_b64: str):
        event: dict = {"source": "voice", "audio": wav_b64}
        frame = agent.capture_frame()
        if frame:
            event["image"] = frame
        loop.call_soon_threadsafe(agent.event_bus.put_nowait, event)

    mic = AudioCapture(on_speech)
    mic_thread = threading.Thread(target=mic.run, daemon=True)
    mic_thread.start()

    def stdin_reader():
        for line in sys.stdin:
            line = line.strip()
            if line:
                loop.call_soon_threadsafe(
                    agent.event_bus.put_nowait,
                    {"source": "text", "text": line},
                )

    stdin_thread = threading.Thread(target=stdin_reader, daemon=True)
    stdin_thread.start()

    try:
        await agent.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nShutting down…")
    finally:
        mic.stop()
        agent.shutdown()


if __name__ == "__main__":
    cli()
