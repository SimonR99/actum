"""Core robot agent.

Design:
  - Multi-step tool calling: LLM chains navigate → look → speak → done
  - Robot action tools (navigate, rotate, gripper, look, remember, …)
  - Headless operation: sounddevice audio, OpenCV camera
  - Persistent memory across turns
  - Structured action log
  - Optional WebSocket monitoring server (see server.py)

Quickstart (headless):
    MODEL_PATH=/path/to/model.litertlm actum

With monitoring dashboard:
    MODEL_PATH=/path/to/model.litertlm actum-server
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import sys
import time
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

from actum import tts as tts_module
from actum.runtime import RobotRuntime
from actum.tools import RobotTools
from actum.perception import AudioCapture, open_camera, capture_jpeg

try:
    import litert_lm
except ImportError:
    litert_lm = None


# ── Model config ───────────────────────────────────────────────────────────────

HF_REPO = "litert-community/gemma-4-E2B-it-litert-lm"
HF_FILENAME = "gemma-4-E2B-it.litertlm"

SYSTEM_PROMPT_TEMPLATE = """\
You are {robot_name}, an autonomous companion and robot agent. You perceive the world through \
available sensors, act through registered tools, and keep the human operator informed through \
visible plans, status updates, and tool results.

{personality_block}

Runtime target:
{backend_block}

Operating rules:
1. Call set_plan() for non-trivial tasks so the operator can see your intent.
2. Keep the public behavior tree current with set_behavior_tree() or mark_behavior_node().
3. Use look() before physical navigation, manipulation, or scene-dependent decisions.
4. Chain tool calls in sequence to accomplish the goal.
5. Use mark_step() as you move through the plan.
5. Use memory tools for durable information:
   - remember() for stable facts and preferences.
   - remember_person() for people and relationship context.
   - remember_place() and remember_spatial_note() for landmarks, docking spots, room layout, and mapping notes.
   - record_observation() for meaningful things you saw or events worth recalling.
   - record_map_observation() for landmarks, spatial relations, and map-building observations.
6. Use update_body_perception() when body posture, contacts, held objects, or joint state matters.
7. Use schedule_job() for recurring checks or reminders.
8. Use web_fetch() or configured MCP tools for data tasks when local knowledge is insufficient.
9. Always finish by calling done() with a one-sentence summary.

Always-on companion behavior:
- Direct chat, language/voice, and forced operator commands should be handled.
- Passive camera/timer/cron observations should stay quiet unless there is safety relevance, an explicit request, \
or a useful low-disruption action.
- If there is no active task, keep the behavior tree on a waiting or perception node.
- When uncertain, call report_status() or ask before acting.

Be concise when speaking (1-2 sentences). Prefer useful action over explanation. \
Do not expose private chain-of-thought; expose intent, plans, and status summaries."""


# ── Agent ──────────────────────────────────────────────────────────────────────

class RobotAgent:
    """On-device robot agent: multimodal perception + agentic multi-step tool calling."""

    def __init__(self, config_path: str | Path | None = None):
        self._config_path = Path(config_path) if config_path else Path(__file__).resolve().parents[2] / "config.json"
        self.config = _load_config(self._config_path)
        self.engine: litert_lm.Engine | None = None
        self.conversation = None
        self.tts_backend: tts_module.TTSBackend | None = None
        self.tools: RobotTools | None = None
        self.runtime = RobotRuntime(self.config, self.get_name())
        self.memory: dict[str, str] = self.runtime.memory.facts

        # Per-turn state (reset at the start of each process_event call)
        self._pending_speech: list[str] = []
        self._pending_frame: str | None = None  # base64 JPEG

        # Thread-safe event queue: anything that triggers the agent goes here
        self.event_bus: asyncio.Queue = asyncio.Queue()

        # Subscribers receive a status dict after each completed turn (for server.py)
        self._status_subscribers: list[asyncio.Queue] = []

        self._action_log: list[dict] = []
        self._camera = None
        self._camera_lock = threading.Lock()
        self._background_stop_requested = False

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
        system_prompt = _build_system_prompt(self.config, self.runtime)
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
            with self._camera_lock:
                self._camera.release()
                self._camera = None
        self.runtime.close()

    def _init_backend(self):
        """Initialise the configured robot backend."""
        if self.runtime.connect():
            print(f"[backend] {self.runtime.backend.name} connected")
        else:
            print(f"[backend] {self.runtime.backend.name} unavailable")

    # ── Perception ─────────────────────────────────────────────────────────

    def capture_frame(self, width: int = 320, quality: int = 70, drain_frames: int | None = None) -> str | None:
        """Return a base64 JPEG from the camera, or None."""
        with self._camera_lock:
            return capture_jpeg(self._camera, width=width, quality=quality, drain_frames=drain_frames)

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
        decision = self.runtime.should_process_event(event)
        if not decision.process:
            payload = {
                "source": event.get("source", "?"),
                "actions": [],
                "elapsed": 0.0,
                "ignored": True,
                "reason": decision.reason,
                "companion": decision.to_dict(),
            }
            print(f"turn ignored | {decision.source}: {decision.reason}")
            self._broadcast(payload)
            return []

        self.tools._reset()
        self._pending_speech.clear()
        self._pending_frame = None

        event = dict(event)
        event["_companion_decision"] = decision.to_dict()
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
        self._record_turn_memory(event, actions)
        self._broadcast({"source": event.get("source", "?"), "actions": actions, "elapsed": elapsed})
        return actions

    def _record_turn_memory(self, event: dict, actions: list[dict]):
        source = str(event.get("source", "unknown"))
        done_summary = next((a.get("summary") for a in actions if a.get("type") == "done" and a.get("summary")), "")
        if done_summary:
            summary = str(done_summary)
        elif event.get("text"):
            summary = f"Processed {source} event: {event['text']}"
        else:
            summary = f"Processed {source} event with {len(actions)} action(s)."
        self.runtime.memory.record_episode(
            summary,
            source=source,
            text=str(event.get("text", "")),
            action_types=[str(action.get("type", "")) for action in actions],
        )

    def _build_content(self, event: dict) -> list[dict]:
        content: list[dict] = []

        if event.get("audio"):
            content.append({"type": "audio", "blob": event["audio"]})
        if event.get("image"):
            content.append({"type": "image", "blob": event["image"]})

        src = event.get("source", "")
        companion = event.get("_companion_decision", {})
        companion_note = ""
        if isinstance(companion, dict) and companion.get("reason"):
            companion_note = f"\nCompanion gate: {companion.get('reason')}."

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
        elif src == "language":
            instruction = (
                "A language/voice command was transcribed for you. Respond or act through tools as needed, "
                "then call done()."
            )
        elif src in {"chat", "text"}:
            txt = event.get("text", "Handle the operator command.")
            instruction = f"Operator chat command: {txt} Use your tools as needed, update intent, then call done()."
        elif src == "vision":
            instruction = (
                "A passive camera frame was admitted for review. Analyse the scene and act only "
                "if something needs attention. Update observations, map, body perception, or behavior state if useful. "
                "Stay quiet for routine background observations. "
                "Call done() when finished."
            )
        elif src in {"timer", "cron"}:
            instruction = (
                f"Scheduled check-in. {event.get('text', 'Perform a brief environment check.')} "
                "Act only if useful or safety-relevant. Call done() when finished."
            )
        elif src == "loop":
            instruction = (
                f"Background autonomy loop tick. {event.get('text', 'Keep waiting and maintain situational awareness.')} "
                "Update the behavior tree. Only use disruptive tools if needed. Call done() when finished."
            )
        else:
            txt = event.get("text", "Perform a brief environment check.")
            instruction = f"{txt} Use your tools as needed, then call done()."

        instruction += companion_note

        memory_context = self.runtime.memory.context()
        if memory_context:
            instruction += f"\n\nRelevant memory:\n{memory_context}"

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
            device = (
                os.environ.get("ACTUM_AUDIO_DEVICE")
                or os.environ.get("SENSORIMOTOR_AUDIO_DEVICE")
                or os.environ.get("ROBO_AUDIO_DEVICE")
                or None
            )
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

    async def background_loop(self):
        """Always-on autonomy loop for schedules and idle perception."""
        self._background_stop_requested = False
        while not self._background_stop_requested:
            behavior = self.runtime.behavior
            await asyncio.sleep(max(1.0, behavior.tick_seconds))
            if self._background_stop_requested:
                break

            queued = False
            for job in self.runtime.cron.due():
                self.runtime.cron.mark_run(job.id)
                behavior.tick("scheduled job", job.name, "cron")
                await self.event_bus.put(
                    {
                        "source": "cron",
                        "text": job.instruction,
                        "cron_job_id": job.id,
                        "force": True,
                    }
                )
                queued = True

            idle = self.runtime.intent.status in {"idle", "done", "blocked"} or not self.runtime.intent.goal
            if (
                idle
                and behavior.enabled
                and behavior.idle_review
                and not queued
                and self.event_bus.empty()
                and (time.time() - behavior.last_idle_review_at) >= behavior.idle_review_seconds
            ):
                frame = self.capture_frame()
                if frame:
                    behavior.last_idle_review_at = time.time()
                    behavior.tick("perception", "Reviewing the current camera frame.", "vision")
                    await self.event_bus.put(
                        {
                            "source": "vision",
                            "text": (
                                "Background vision review: update observations, map, body perception, "
                                "or behavior state if useful. Stay quiet unless action is needed."
                            ),
                            "image": frame,
                            "importance": behavior.idle_importance,
                            "force": behavior.force_idle_reviews,
                        }
                    )
                    queued = True
                else:
                    behavior.set_waiting("No camera frame; waiting for chat, language, vision, or schedule trigger.")

            if idle and not queued:
                behavior.set_waiting("Waiting for chat, language, vision, or schedule trigger.")
            self._broadcast({"source": "loop", "actions": [], "elapsed": 0.0, "state_only": True})

    def stop_background_loop(self):
        self._background_stop_requested = True

    # ── Runtime config ────────────────────────────────────────────────────

    def get_mode(self) -> str:
        mode = str(self.config.get("tts", "local")).lower()
        return mode if mode in {"local", "unitree"} else "local"

    def get_name(self) -> str:
        personality = self.config.get("personality", {}) if isinstance(self.config.get("personality"), dict) else {}
        name = str(personality.get("name") or self.config.get("name", "dino")).strip()
        return name if name else "dino"

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

    def set_model_provider(
        self,
        provider: str,
        model: str = "",
        api_key: str = "",
        enabled: bool | None = None,
        persist: bool = False,
        persist_secret: bool = False,
    ) -> tuple[bool, str]:
        try:
            self.runtime.settings.set_model_provider(provider, model=model, api_key=api_key, enabled=enabled)
        except Exception as exc:
            return False, str(exc)
        self.config["models"] = self.runtime.settings.to_config(include_secrets=False)["models"]
        if persist:
            try:
                _persist_config_updates(
                    self._config_path,
                    self.runtime.settings.to_config(include_secrets=persist_secret),
                )
            except Exception as exc:
                return False, f"Model settings changed in memory but failed to save config: {exc}"
        return True, f"Model provider set to {provider}."

    def set_tool_enabled(self, tool: str, enabled: bool, persist: bool = False) -> tuple[bool, str]:
        try:
            self.runtime.settings.set_tool_enabled(tool, enabled)
        except Exception as exc:
            return False, str(exc)
        self.config["tools"] = self.runtime.settings.to_config()["tools"]
        if persist:
            try:
                _persist_config_updates(self._config_path, {"tools": self.config["tools"]})
            except Exception as exc:
                return False, f"Tool setting changed in memory but failed to save config: {exc}"
        return True, f"Tool {tool} {'enabled' if enabled else 'disabled'}."

    def set_robot_identity(self, name: str, persist: bool = True) -> tuple[bool, str]:
        clean_name = str(name).strip()
        if not clean_name:
            return False, "Robot name cannot be empty."

        personality = self.config.setdefault("personality", {})
        if not isinstance(personality, dict):
            personality = {}
            self.config["personality"] = personality
        self.config["name"] = clean_name
        personality["name"] = clean_name
        self.runtime.robot_name = clean_name
        self.runtime.personality = personality
        self.runtime.events.append("robot.identity", "operator", name=clean_name)

        if persist:
            try:
                _persist_config_updates(self._config_path, {"name": clean_name, "personality": personality})
            except Exception as exc:
                return False, f"Robot name changed in memory but failed to save config: {exc}"
        return True, f"Robot name set to {clean_name}."

    def set_robot_config(self, robot_config: dict, persist: bool = True) -> tuple[bool, str]:
        try:
            clean_config = _normalize_robot_config(robot_config, self.config.get("robot", {}))
        except Exception as exc:
            return False, str(exc)

        connected, message = self.runtime.configure_robot(clean_config)
        self.config["robot"] = clean_config
        if persist:
            try:
                _persist_config_updates(self._config_path, {"robot": clean_config})
            except Exception as exc:
                return False, f"Robot config changed in memory but failed to save config: {exc}"
        return True, message if connected else f"{message} Config was still applied."

    def set_robot_settings(
        self,
        name: str | None = None,
        robot_config: dict | None = None,
        persist: bool = True,
    ) -> tuple[bool, str]:
        clean_name: str | None = None
        clean_robot_config: dict | None = None
        if name is not None:
            clean_name = str(name).strip()
            if not clean_name:
                return False, "Robot name cannot be empty."
        if robot_config is not None:
            try:
                clean_robot_config = _normalize_robot_config(robot_config, self.config.get("robot", {}))
            except Exception as exc:
                return False, str(exc)

        messages: list[str] = []
        if clean_name is not None:
            ok, message = self.set_robot_identity(clean_name, persist=False)
            if not ok:
                return False, message
            messages.append(message)

        if clean_robot_config is not None:
            connected, message = self.runtime.configure_robot(clean_robot_config)
            self.config["robot"] = clean_robot_config
            if not connected:
                message = f"{message} Config was still applied."
            messages.append(message)

        if persist:
            try:
                updates: dict[str, Any] = {}
                if name is not None:
                    updates["name"] = self.config["name"]
                    updates["personality"] = self.config["personality"]
                if robot_config is not None:
                    updates["robot"] = self.config["robot"]
                if updates:
                    _persist_config_updates(self._config_path, updates)
            except Exception as exc:
                return False, f"Robot settings changed in memory but failed to save config: {exc}"

        return True, " ".join(messages) if messages else "No robot settings changed."


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve_model_path() -> str:
    path = os.environ.get("MODEL_PATH", "")
    if path:
        expanded = sorted(glob.glob(os.path.expanduser(path)))
        return expanded[0] if expanded else path
    from huggingface_hub import hf_hub_download
    print(f"Downloading {HF_REPO}/{HF_FILENAME} (first run only)…")
    return hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)


def _build_system_prompt(config: dict, runtime: RobotRuntime) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        robot_name=runtime.robot_name,
        personality_block=_format_personality_block(config),
        backend_block=_format_backend_block(runtime),
    )


def _format_personality_block(config: dict) -> str:
    personality = config.get("personality", {}) if isinstance(config.get("personality"), dict) else {}
    persona = str(personality.get("persona", "")).strip()
    likes = _format_string_list(personality.get("likes"))
    principles = _format_string_list(personality.get("principles"))
    speaking_style = str(personality.get("speaking_style", "")).strip()

    lines = ["Personality:"]
    if persona:
        lines.append(f"- Persona: {persona}")
    if likes:
        lines.append(f"- Likes: {likes}")
    if principles:
        lines.append(f"- Principles: {principles}")
    if speaking_style:
        lines.append(f"- Speaking style: {speaking_style}")
    if len(lines) == 1:
        lines.append("- Warm, useful, careful, and transparent about intent.")
    return "\n".join(lines)


def _format_backend_block(runtime: RobotRuntime) -> str:
    state = runtime.backend.get_state()
    metadata = state.metadata or {}
    if runtime.backend.name == "laptop":
        sensors = []
        if metadata.get("webcam", True):
            sensors.append("webcam")
        if metadata.get("microphone", True):
            sensors.append("microphone")
        if metadata.get("speaker", True):
            sensors.append("speaker")
        sensor_text = ", ".join(sensors) if sensors else "software-only"
        return (
            f"- Backend: laptop companion.\n"
            f"- Available local I/O: {sensor_text}.\n"
            "- Physical movement, gestures, and manipulation are unavailable on this backend.\n"
            "- Software/data capabilities may include web_fetch() and configured MCP tools."
        )
    if runtime.backend.name == "fake":
        return "- Backend: fake simulation with bounded locomotion, rotation, gripper, gestures, and speech."
    if runtime.backend.name == "unitree_g1":
        return "- Backend: Unitree G1 humanoid. Use bounded movement and gestures cautiously."
    if runtime.backend.name == "lekiwi":
        return "- Backend: LeKiwi / LeRobot-compatible robot. Use backend results to verify action success."
    return f"- Backend: {runtime.backend.name}. Respect capability results and avoid assuming unsupported hardware."


def _format_string_list(value) -> str:
    if not isinstance(value, list):
        return ""
    items = [str(item).strip() for item in value if str(item).strip()]
    return ", ".join(items)


def _load_config(config_path: str | Path | None = None) -> dict:
    """Load runtime config with sensible defaults.

    Search order:
      1) explicit config_path
      2) workspace-root config.json
    """
    default = _default_config()

    path = Path(config_path) if config_path else Path(__file__).resolve().parents[2] / "config.json"
    if not path.exists():
        return default

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        print(f"[config] Failed to load {path}: {exc}")
        return default

    if not isinstance(raw, dict):
        return default

    cfg = _merge_dict(default, raw)
    if "robot" not in raw and cfg.get("hardware", {}).get("enabled"):
        cfg["robot"]["backend"] = ""
    return cfg


def _default_config() -> dict:
    return {
        "name": "dino",
        "tts": "local",
        "personality": {
            "name": "dino",
            "persona": "A warm, curious, practical robot companion that likes helping with robotics and everyday reasoning.",
            "likes": ["clear plans", "useful autonomy", "learning from the environment"],
            "principles": [
                "be helpful without being intrusive",
                "ask before disruptive or risky actions",
                "make intent visible to the operator",
            ],
            "speaking_style": "concise, warm, and calm",
        },
        "companion": {
            "always_on": True,
            "proactive_mode": "conservative",
            "direct_sources": ["chat", "language", "text", "voice"],
            "passive_sources": ["cron", "loop", "timer", "vision"],
            "min_seconds_between_proactive_actions": 60.0,
            "vision_importance_threshold": 0.75,
            "curious_importance_threshold": 0.55,
        },
        "triggers": {
            "vision": True,
            "chat": True,
            "language": True,
        },
        "behavior_loop": {
            "enabled": True,
            "tick_seconds": 15,
            "idle_review": True,
            "idle_review_seconds": 45,
            "idle_importance": 0.2,
            "force_idle_reviews": False,
        },
        "cron": [],
        "models": {
            "active_provider": "local",
            "providers": {
                "local": {
                    "enabled": True,
                    "model": HF_REPO,
                },
                "openai": {
                    "enabled": False,
                    "model": "",
                    "api_key_env": "OPENAI_API_KEY",
                },
                "anthropic": {
                    "enabled": False,
                    "model": "",
                    "api_key_env": "ANTHROPIC_API_KEY",
                },
            },
        },
        "tools": {
            "enabled": [],
            "_enabled_comment": "Empty means all registered tools are available.",
        },
        "memory": {
            "enabled": True,
            "path": "data/memory.json",
            "max_episodes": 1000,
        },
        "robot": {
            "backend": "laptop",
            "laptop": {
                "webcam": True,
                "microphone": True,
                "speaker": True,
            },
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
        "mcp": {
            "enabled": False,
            "servers": {},
        },
        "hardware": {
            "enabled": False,
            "type": "unitree_g1",
            "network_interface": "eth0",
            "speaker_id": 0,
            "volume": 80,
        },
    }


SUPPORTED_ROBOT_BACKENDS = {"laptop", "fake", "unitree_g1", "lekiwi"}


def _normalize_robot_config(robot_config: dict, current_robot_config: dict | None = None) -> dict:
    if not isinstance(robot_config, dict):
        raise ValueError("robot config must be an object")

    base = _merge_dict(_default_config()["robot"], current_robot_config or {})
    merged = _merge_dict(base, robot_config)
    backend = str(merged.get("backend") or "laptop").lower().strip()
    if backend in {"local", "companion"}:
        backend = "laptop"
    if backend in {"sim", "simulation"}:
        backend = "fake"
    if backend not in SUPPORTED_ROBOT_BACKENDS:
        raise ValueError(f"Unsupported robot backend: {backend!r}")
    merged["backend"] = backend

    laptop = merged.get("laptop", {}) if isinstance(merged.get("laptop"), dict) else {}
    merged["laptop"] = {
        "webcam": bool(laptop.get("webcam", True)),
        "microphone": bool(laptop.get("microphone", True)),
        "speaker": bool(laptop.get("speaker", True)),
    }

    fake = merged.get("fake", {}) if isinstance(merged.get("fake"), dict) else {}
    merged["fake"] = dict(fake)

    unitree = merged.get("unitree_g1", {}) if isinstance(merged.get("unitree_g1"), dict) else {}
    merged["unitree_g1"] = {
        "network_interface": str(unitree.get("network_interface", "eth0")).strip() or "eth0",
        "speaker_id": _coerce_int(unitree.get("speaker_id", 0), 0, 0, 1),
        "volume": _coerce_int(unitree.get("volume", 80), 80, 0, 100),
    }

    lekiwi = merged.get("lekiwi", {}) if isinstance(merged.get("lekiwi"), dict) else {}
    merged["lekiwi"] = {
        "remote_ip": str(lekiwi.get("remote_ip", "127.0.0.1")).strip() or "127.0.0.1",
        "port": _coerce_int(lekiwi.get("port", 5555), 5555, 1, 65535),
        "id": str(lekiwi.get("id", "lekiwi")).strip() or "lekiwi",
    }
    if "_backend_options" in merged:
        merged["_backend_options"] = str(merged["_backend_options"])
    return merged


def _coerce_int(value, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))



def _merge_dict(default: dict, override: dict) -> dict:
    merged = dict(default)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _persist_tts_mode(config_path: str | Path, mode: str):
    """Persist TTS mode while preserving any other config keys."""
    _persist_config_updates(config_path, {"tts": mode})


def _persist_config_updates(config_path: str | Path, updates: dict):
    """Persist a shallow/deep config patch while preserving other keys."""
    path = Path(config_path)
    data: dict = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            data = raw
    data = _merge_dict(data, updates)
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
        event: dict = {"source": "language", "audio": wav_b64}
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
                    {"source": "chat", "text": line},
                )

    stdin_thread = threading.Thread(target=stdin_reader, daemon=True)
    stdin_thread.start()
    background_task = asyncio.create_task(agent.background_loop())

    try:
        await agent.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nShutting down…")
    finally:
        mic.stop()
        agent.stop_background_loop()
        background_task.cancel()
        with suppress(asyncio.CancelledError):
            await background_task
        agent.shutdown()


if __name__ == "__main__":
    cli()
