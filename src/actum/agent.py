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
from actum.inference import InferenceProvider, build_provider
from actum.inference.stt import build_stt
from actum.runtime import RobotRuntime
from actum.tools import RobotTools
from actum.perception import AudioCapture, open_camera, capture_jpeg

# ── Model config ───────────────────────────────────────────────────────────────

HF_REPO = "litert-community/gemma-4-E2B-it-litert-lm"
HF_FILENAME = "gemma-4-E2B-it.litertlm"

# How many times the background loop re-prompts a stalled active task before
# marking it blocked instead of looping forever.
_MAX_TASK_CONTINUATIONS = 3

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
6. Use memory tools for durable information:
   - remember() for stable facts and preferences.
   - remember_person() for people and relationship context.
   - remember_place() and remember_spatial_note() for landmarks, docking spots, room layout, and mapping notes.
   - record_observation() for meaningful things you saw or events worth recalling.
   - record_map_observation() for landmarks, spatial relations, and map-building observations.
7. Use update_body_perception() when body posture, contacts, held objects, or joint state matters.
8. Use schedule_job() for recurring checks or reminders.
9. Use web_fetch() or configured MCP tools for data tasks when local knowledge is insufficient.
10. Always finish by calling done() with a one-sentence summary. Any final plain-text reply is shown to the operator in chat.

Always-on companion behavior:
- Direct chat, language/voice, and forced operator commands should be handled.
- Passive camera/timer/cron observations should stay quiet unless there is safety relevance, an explicit request, \
or a useful low-disruption action.
- If there is no active task, keep the behavior tree on a waiting or perception node.
- When uncertain, call report_status() or ask before acting.

Be concise when speaking (1-2 sentences). Prefer useful action over explanation. \
Do not expose private chain-of-thought; expose intent, plans, and status summaries.
{language_block}"""


# ── Agent ──────────────────────────────────────────────────────────────────────


class RobotAgent:
    """On-device robot agent: multimodal perception + agentic multi-step tool calling."""

    def __init__(self, config_path: str | Path | None = None):
        self._config_path = (
            Path(config_path)
            if config_path
            else Path(__file__).resolve().parents[2] / "config.json"
        )
        _load_dotenv(self._config_path.parent)
        self.config = _load_config(self._config_path)
        self.provider: InferenceProvider | None = None
        self.stt = None
        self.tts_backend: tts_module.TTSBackend | None = None
        self.tools: RobotTools | None = None
        self.mic: AudioCapture | None = None  # set by server.py / _run_headless
        self.runtime = RobotRuntime(self.config, self.get_name())

        # Per-turn state (reset at the start of each process_event call)
        self._pending_speech: list[str] = []
        self._pending_frame: str | None = None  # base64 JPEG
        # Speech is started the moment the model calls speak(), overlapping the
        # rest of the tool loop. These track the in-flight playback coroutines
        # and the event loop they run on (set per turn in process_event).
        self._loop: asyncio.AbstractEventLoop | None = None
        self._speech_futures: list = []
        self._speech_lock: asyncio.Lock | None = None
        self._first_speech_at: float = 0.0

        # Thread-safe event queue: anything that triggers the agent goes here
        self.event_bus: asyncio.Queue = asyncio.Queue()

        # Subscribers receive a status dict after each completed turn (for server.py)
        self._status_subscribers: list[asyncio.Queue] = []

        self._action_log: list[dict] = []
        self._camera = None
        self._camera_lock = threading.Lock()
        self._background_stop_requested = False
        self._last_deliberate_at = 0.0
        # Task-continuation state: how many times the loop nudged the current
        # goal forward without the model finishing it.
        self._continued_goal = ""
        self._continuation_attempts = 0
        self._color_trigger_watcher = None

    # ── Startup / shutdown ─────────────────────────────────────────────────

    def load_models(self):
        """Load the inference provider + TTS and wire up tools.

        Blocking — run via executor. The active provider (local LiteRT, OpenAI,
        …) and its compute backend come from the active speed profile.
        """
        self.tts_backend = tts_module.load(self.runtime.settings)
        self.tools = RobotTools(self)
        self._rebuild_provider()

        self.stt = build_stt(self.runtime.settings)
        if self.stt is not None and hasattr(self.stt, "_ensure_model"):
            try:
                self.stt._ensure_model()
            except Exception as e:
                print(f"[stt] warning: failed to preload model: {e}")

        self._camera = open_camera()
        self._init_backend()
        self._start_color_triggers()
        print("Robot agent ready.")

    def _rebuild_provider(self):
        """(Re)build the inference provider from current settings.

        Closes any live provider first (releases LiteRT engines / GPU memory),
        then starts a fresh conversation with the current system prompt and
        tool registry. Raises on failure; callers decide how to report it.
        """
        if self.tools is None:
            raise RuntimeError("Agent tools are not initialised yet.")
        if self.provider is not None:
            self.provider.close()
            self.provider = None
        self.provider = build_provider(
            self.runtime.settings,
            compute=self.runtime.compute_backend,
            pop_pending_frame=self._pop_pending_frame,
            resolve_model_path=_resolve_model_path,
        )
        print(f"Starting inference provider: {self.provider.name}…")
        self.provider.start(
            _build_system_prompt(self.config, self.runtime), self.tools.get_tools()
        )

    def _pop_pending_frame(self) -> str | None:
        """Hand the most recent look() frame to the provider, then clear it."""
        frame = self._pending_frame
        self._pending_frame = None
        return frame

    def shutdown(self):
        if self._color_trigger_watcher is not None:
            self._color_trigger_watcher.stop()
            self._color_trigger_watcher = None
        if self.provider:
            self.provider.close()
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

    def capture_frame(
        self, width: int = 320, quality: int = 70, drain_frames: int | None = None
    ) -> str | None:
        """Return a base64 JPEG from the camera, or None."""
        flip = self.config.get("camera_flip")
        if flip is not None:
            flip = int(flip)
        with self._camera_lock:
            return capture_jpeg(
                self._camera,
                width=width,
                quality=quality,
                drain_frames=drain_frames,
                flip=flip,
            )

    def read_frame_bgr(self):
        """Return a raw BGR numpy frame from the camera, or None."""
        if self._camera is None:
            return None
        try:
            import cv2
        except ImportError:
            return None
        flip = self.config.get("camera_flip")
        with self._camera_lock:
            ok, frame = self._camera.read()
            if not ok:
                return None
            if flip is not None:
                frame = cv2.flip(frame, int(flip))
            return frame

    def _start_color_triggers(self):
        """Start the color-group trigger watcher when enabled in config."""
        trigger_cfg = self.config.get("color_triggers")
        if not isinstance(trigger_cfg, dict) or not trigger_cfg.get("enabled"):
            return
        try:
            from actum.color_triggers.watcher import ColorTriggerWatcher
        except ImportError:
            print("[color_trigger] module unavailable (install actum[camera])")
            return

        watcher = ColorTriggerWatcher.from_agent_config(
            self.config,
            frame_reader=self.read_frame_bgr,
            backend=self.runtime.backend,
            config_path=self._config_path,
        )
        if watcher is None:
            return
        watcher.start()
        self._color_trigger_watcher = watcher

    # ── Core agentic loop ──────────────────────────────────────────────────

    async def process_event(self, event: dict) -> list[dict]:
        """Process one triggering event through the agentic tool-calling loop.

        The LLM receives the event and calls tools in sequence (look, navigate,
        speak, remember, …) until it calls done(). The active InferenceProvider
        runs the multi-step tool loop (internally for LiteRT/Gemma, or by driving
        the API loop for cloud providers like OpenAI).

        Event dict keys:
            source  : 'voice' | 'vision' | 'text' | 'timer'
            audio   : base64 WAV  (optional)
            image   : base64 JPEG (optional)
            text    : plain-text command (optional)

        Returns a list of action records (one per tool call).
        """
        if self.provider is None or self.tools is None:
            payload = {
                "source": event.get("source", "?"),
                "actions": [],
                "elapsed": 0.0,
                "ignored": True,
                "reason": "Agent models are not loaded yet.",
            }
            print(f"turn ignored | {event.get('source', '?')}: models not loaded")
            self._broadcast(payload)
            return []

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
        self._speech_futures = []
        self._first_speech_at = 0.0
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        event = dict(event)
        event["_companion_decision"] = decision.to_dict()
        content = self._build_content(event)
        t0 = time.time()

        # The provider runs the full tool-calling loop, including any look()
        # image follow-ups (it pops frames via self._pop_pending_frame).
        error = ""
        final_text = ""
        try:
            final_text = (
                await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: self.provider.send(content),
                )
                or ""
            )
        except Exception as exc:
            error = str(exc)
            self.runtime.events.append(
                "turn.error",
                "agent",
                message=error,
                event_source=event.get("source", "?"),
            )
            print(f"[agent] turn failed: {error}")

        elapsed = time.time() - t0
        action_types = [a["type"] for a in self.tools.actions_taken]
        print(f"turn ({elapsed:.2f}s) | {' → '.join(action_types) or 'no actions'}")
        if os.environ.get("ACTUM_TIMING"):
            spoke_at = self._first_speech_at - t0 if self._first_speech_at else None
            print(
                f"[timing] llm+tools={elapsed:.2f}s"
                + (
                    f" | first speak() queued at +{spoke_at:.2f}s"
                    if spoke_at is not None
                    else " | no speech this turn"
                )
            )

        actions = list(self.tools.actions_taken)
        # A final plain-text reply (no tool call) is still an answer for the
        # operator — keep it instead of dropping it on the floor.
        if final_text.strip():
            actions.append(
                {"type": "reply", "text": final_text.strip(), "time": time.time()}
            )
            self.runtime.events.append("agent.reply", "agent", text=final_text.strip())
        # Vision turns leave a visible "what I last saw" summary for the operator.
        if str(event.get("source", "")) == "vision":
            scene = (
                next(
                    (
                        a.get("summary")
                        for a in actions
                        if a.get("type") == "done" and a.get("summary")
                    ),
                    "",
                )
                or final_text.strip()
            )
            self.runtime.set_scene(scene, source="vision")
        self._action_log.extend(actions)
        self._record_turn_memory(event, actions)

        # Broadcast immediately so text appears in the discussion tab before
        # audio playback begins (which can take several seconds).
        self._broadcast(
            {
                "source": event.get("source", "?"),
                "actions": actions,
                "elapsed": elapsed,
                "error": error,
            }
        )

        # Speech for this turn was already kicked off the instant the model
        # called speak() (see queue_speech), overlapping the rest of the tool
        # loop. Wait for those playbacks to finish so the turn doesn't return
        # while the robot is still talking. The _pending_speech list is only
        # populated on the fallback path (no bound event loop, e.g. tests).
        for text in self._pending_speech:
            await self._speak(text)
        for fut in self._speech_futures:
            with suppress(Exception):
                await asyncio.wrap_future(fut)

        return actions

    def queue_speech(self, text: str) -> None:
        """Start speaking `text` immediately, overlapping the running tool loop.

        Called from the provider's tool thread when the model invokes speak().
        Scheduling onto the agent event loop lets playback begin before the
        model finishes the turn with done(), removing seconds of dead air.
        Falls back to the deferred queue when no loop is bound (unit tests).
        """
        if not text.strip():
            return
        if not self._first_speech_at:
            self._first_speech_at = time.time()
        if self._loop is None:
            self._pending_speech.append(text)
            return
        fut = asyncio.run_coroutine_threadsafe(self._speak(text), self._loop)
        self._speech_futures.append(fut)

    def _record_turn_memory(self, event: dict, actions: list[dict]):
        source = str(event.get("source", "unknown"))
        done_summary = next(
            (
                a.get("summary")
                for a in actions
                if a.get("type") == "done" and a.get("summary")
            ),
            "",
        )
        reply = next(
            (
                a.get("text")
                for a in actions
                if a.get("type") == "reply" and a.get("text")
            ),
            "",
        )
        if done_summary:
            summary = str(done_summary)
        elif reply:
            summary = str(reply)
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
            # Transcribe through the selected STT engine; if it yields nothing
            # (engine is "model", disabled, or failed), pass raw audio to the model.
            stt_t0 = time.time()
            transcript = (
                self.stt.transcribe(event["audio"]) if self.stt is not None else ""
            )
            if os.environ.get("ACTUM_TIMING") and self.stt is not None:
                print(f"[timing] stt={time.time() - stt_t0:.2f}s")
            if transcript:
                content.append({"type": "text", "text": f"(voice) {transcript}"})
                event["text"] = transcript
                self._broadcast({"type": "mic_transcript", "text": transcript})
            else:
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
        elif src == "language" and event.get("audio"):
            instruction = (
                "The user spoke to you through a microphone. Listen to the attached audio, "
                "respond or act through tools as needed, then call done()."
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
        elif src == "continue":
            goal = event.get("text") or self.runtime.intent.goal or "your current task"
            active = self.runtime.intent.active_step_id or "unknown"
            instruction = (
                f"Your task is not finished. Goal: {goal} (active step: {active}). "
                "Continue with the next concrete step using your tools, and mark progress with mark_step(). "
                "If the goal is complete, call done() with a summary. "
                "If you are blocked, call report_status() explaining why, then done()."
            )
        elif src == "deliberate":
            instruction = (
                "Self-directed planning tick. Review your standing goals, recent memory, and what you know "
                "about the environment. Use search_memory() to recall anything relevant. "
                "If there is something useful and safe you could proactively do or prepare, set a plan with "
                "set_plan() and pursue it. If a recurring check would help, create it with schedule_job(). "
                "If nothing is needed right now, note your reasoning briefly with report_status(). "
                "Always call done() when finished."
            )
        else:
            txt = event.get("text", "Perform a brief environment check.")
            instruction = f"{txt} Use your tools as needed, then call done()."

        instruction += companion_note

        # Retrieve memory relevant to this turn instead of dumping the whole
        # store, so the prompt stays focused as memory grows.
        query = str(event.get("text", "")).strip()
        memory_context = self.runtime.memory.context(query=query or None)
        if memory_context:
            instruction += f"\n\n{memory_context}"

        content.append({"type": "text", "text": instruction})
        return content

    # ── Audio output ───────────────────────────────────────────────────────

    async def _speak(self, text: str):
        if not text.strip():
            return
        # speak() can be called several times in one turn; each schedules its
        # own _speak coroutine, so serialize them to keep utterances in order
        # and stop two sd.play() calls from overlapping on the same device.
        if self._speech_lock is None:
            self._speech_lock = asyncio.Lock()
        async with self._speech_lock:
            # Mute the mic while the robot talks so it doesn't transcribe its own
            # voice and trigger itself in a feedback loop.
            if self.mic is not None:
                self.mic.pause()
            try:
                await self._speak_unmuted(text)
            finally:
                if self.mic is not None:
                    self.mic.resume()

    async def _speak_unmuted(self, text: str):
        # Strip emoji and other non-readable characters so the speech engine
        # doesn't vocalise them as noise (and Unitree TTS stays clean too).
        text = tts_module.strip_unspeakable(text)
        if not text:
            return

        mode = self.get_mode()
        if mode == "unitree":
            if (
                self.runtime.backend.name == "unitree_g1"
                and self.runtime.backend.connected
            ):
                result = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: self.runtime.backend.speak(text)
                )
                if result.ok:
                    return
                print(
                    f"[tts] unitree speak failed; falling back to local ({result.message})"
                )
            else:
                print(
                    "[tts] unitree mode selected but Unitree backend is not connected; falling back to local"
                )

        if not self.tts_backend:
            print("[tts] no backend loaded — skipping speech")
            return

        language = self.config.get("language", "en")
        voice = "ff_siwis" if language == "fr" else "af_heart"
        # Phonemizer language code for kokoro-onnx — without it French is spoken
        # with an English accent.
        lang_code = "fr-fr" if language == "fr" else "en-us"
        sentences = tts_module.split_sentences(text)
        backend_name = type(self.tts_backend).__name__
        print(f"[tts] generating {len(sentences)} sentence(s): {text!r}")
        print(
            f"[tts] backend={backend_name} language={language} lang={lang_code} "
            f"voice={voice}"
        )

        loop = asyncio.get_running_loop()
        timing = bool(os.environ.get("ACTUM_TIMING"))
        turn_t0 = time.time()

        def generate(sentence: str):
            return self.tts_backend.generate(sentence, voice=voice, lang=lang_code)

        try:
            # Pipeline: generate sentence i+1 while sentence i is playing, so the
            # first audio starts after only one sentence is synthesised instead
            # of the whole reply, and gaps between sentences stay tight.
            next_pcm = await loop.run_in_executor(None, generate, sentences[0])
            if timing:
                print(
                    f"[timing] tts first-sentence={time.time() - turn_t0:.2f}s "
                    f"({len(sentences)} sentence(s))"
                )
            for i, sentence in enumerate(sentences):
                pcm = next_pcm
                gen_fut = (
                    loop.run_in_executor(None, generate, sentences[i + 1])
                    if i + 1 < len(sentences)
                    else None
                )
                await self._play_audio(pcm)
                if gen_fut is not None:
                    next_pcm = await gen_fut
        except Exception as e:
            print(f"[tts] error: {e}")

    async def _play_audio(self, pcm: np.ndarray):
        sr = self.tts_backend.sample_rate
        try:
            import sounddevice as sd

            device = os.environ.get("ACTUM_AUDIO_DEVICE") or None
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: sd.play(pcm, samplerate=sr, device=device, blocking=True)
            )
        except Exception as e:
            print(
                f"[tts] sounddevice playback failed ({e}), trying aplay/afplay fallback"
            )
            import soundfile as sf
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                path = f.name
            sf.write(path, pcm, sr)
            if device:
                cmd = f"aplay -D {device} '{path}'"
            else:
                cmd = f"aplay '{path}' || afplay '{path}'"
            ret = await asyncio.get_running_loop().run_in_executor(
                None, lambda: os.system(cmd)
            )
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
        """Always-on autonomy loop for schedules, task continuation, and idle perception."""
        self._background_stop_requested = False
        # Start the deliberation clock now so the robot doesn't self-task
        # seconds after boot, before the operator has said anything.
        self._last_deliberate_at = time.time()
        while not self._background_stop_requested:
            await asyncio.sleep(max(1.0, self.runtime.behavior.tick_seconds))
            if self._background_stop_requested:
                break
            await self._background_tick()

    async def _background_tick(self):
        """One pass of the autonomy loop. Split out so tests can drive it."""
        behavior = self.runtime.behavior

        # Self-maintenance: keep memory tidy without operator involvement.
        self.runtime.maybe_consolidate_memory()

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

        intent = self.runtime.intent
        idle = intent.status in {"idle", "done", "blocked"} or not intent.goal

        # Task continuation: if a plan is active but the model stopped calling
        # tools (no done(), no progress), nudge it forward instead of letting
        # the task hang. Capped so a stuck goal fails visibly instead of
        # looping forever.
        if not idle and behavior.enabled:
            if intent.goal != self._continued_goal:
                self._continued_goal = intent.goal
                self._continuation_attempts = 0
            stalled_s = time.time() - intent.updated_at
            if (
                not queued
                and self.event_bus.empty()
                and stalled_s >= max(2.0 * behavior.tick_seconds, 20.0)
            ):
                if self._continuation_attempts >= _MAX_TASK_CONTINUATIONS:
                    self.runtime.fail_task(
                        "Task stalled: no progress after repeated continuation prompts."
                    )
                else:
                    self._continuation_attempts += 1
                    behavior.tick("continue task", intent.goal, "continue")
                    await self.event_bus.put(
                        {"source": "continue", "text": intent.goal, "force": True}
                    )
                    queued = True
        elif idle:
            self._continued_goal = ""
            self._continuation_attempts = 0

        # Self-directed deliberation: when idle, periodically let the robot
        # review its goals/memory and set its own task.
        deliberate_seconds = self.runtime.deliberate_seconds
        if (
            idle
            and behavior.enabled
            and not queued
            and deliberate_seconds > 0
            and self.event_bus.empty()
            and (time.time() - self._last_deliberate_at) >= deliberate_seconds
        ):
            self._last_deliberate_at = time.time()
            behavior.tick(
                "deliberate",
                "Reviewing goals and memory for useful next steps.",
                "deliberate",
            )
            await self.event_bus.put(
                {
                    "source": "deliberate",
                    "text": "Decide if there is something useful and safe to do or prepare right now.",
                    "force": True,
                }
            )
            queued = True
        if (
            idle
            and behavior.enabled
            and behavior.idle_review
            and not queued
            and self.event_bus.empty()
            and (time.time() - behavior.last_idle_review_at)
            >= behavior.idle_review_seconds
        ):
            frame = self.capture_frame()
            if frame:
                behavior.last_idle_review_at = time.time()
                behavior.tick(
                    "perception", "Reviewing the current camera frame.", "vision"
                )
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
                behavior.set_waiting(
                    "No camera frame; waiting for chat, language, vision, or schedule trigger."
                )

        if idle and not queued:
            behavior.set_waiting(
                "Waiting for chat, language, vision, or schedule trigger."
            )
        self._broadcast(
            {"source": "loop", "actions": [], "elapsed": 0.0, "state_only": True}
        )

    def stop_background_loop(self):
        self._background_stop_requested = True

    # ── Conversation lifecycle ────────────────────────────────────────────

    def reset_conversation(self) -> tuple[bool, str]:
        """Forget the LLM conversation and clear the live task state.

        Durable memory and the event log survive; only the dialogue history,
        intent, and behavior tree are reset.
        """
        if self.provider is None:
            return False, "Agent models are not loaded yet."
        try:
            self.provider.reset()
        except Exception as exc:
            return False, f"Failed to reset conversation: {exc}"
        self.runtime.reset_session()
        self._continued_goal = ""
        self._continuation_attempts = 0
        self._broadcast(
            {"source": "operator", "actions": [], "elapsed": 0.0, "state_only": True}
        )
        return True, "Conversation reset."

    # ── Runtime config ────────────────────────────────────────────────────

    def get_mode(self) -> str:
        mode = str(self.config.get("tts", "local")).lower()
        return mode if mode in {"local", "unitree"} else "local"

    def get_name(self) -> str:
        personality = (
            self.config.get("personality", {})
            if isinstance(self.config.get("personality"), dict)
            else {}
        )
        name = str(personality.get("name") or self.config.get("name", "dino")).strip()
        return name if name else "dino"

    def set_mode(self, mode: str, persist: bool = True) -> tuple[bool, str]:
        mode = str(mode).lower().strip()
        if mode not in {"local", "unitree"}:
            return False, "Invalid mode. Use 'local' or 'unitree'."

        self.config["tts"] = mode

        if persist:
            try:
                _persist_config_updates(self._config_path, {"tts": mode})
            except Exception as exc:
                return False, f"Mode changed in memory but failed to save config: {exc}"

        if mode == "unitree" and not (
            self.runtime.backend.name == "unitree_g1" and self.runtime.backend.connected
        ):
            return (
                True,
                "Mode set to unitree, but hardware is not connected. Local fallback will be used.",
            )

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
            self.runtime.settings.set_model_provider(
                provider, model=model, api_key=api_key, enabled=enabled
            )
        except Exception as exc:
            return False, str(exc)
        self.config["models"] = self.runtime.settings.to_config(include_secrets=False)[
            "models"
        ]

        if self.tools is not None:
            try:
                self._rebuild_provider()
            except Exception as exc:
                return False, f"Failed to switch to provider {provider}: {exc}"

        if persist:
            try:
                _persist_config_updates(
                    self._config_path,
                    self.runtime.settings.to_config(include_secrets=persist_secret),
                )
            except Exception as exc:
                return (
                    False,
                    f"Model settings changed in memory but failed to save config: {exc}",
                )
        return True, f"Model provider set to {provider}."

    def set_profile(self, profile: str, persist: bool = True) -> tuple[bool, str]:
        ok, message = self.runtime.set_profile(profile)
        if not ok:
            return False, message
        self.config["active_profile"] = self.runtime.profiles.active_name

        if self.tools is not None:
            try:
                self._rebuild_provider()
            except Exception as exc:
                return (
                    False,
                    f"Profile updated, but failed to reload model provider: {exc}",
                )

        if persist:
            try:
                _persist_config_updates(
                    self._config_path,
                    {"active_profile": self.runtime.profiles.active_name},
                )
            except Exception as exc:
                return (
                    False,
                    f"Profile changed in memory but failed to save config: {exc}",
                )
        return True, message

    def set_stt_engine(self, engine: str, persist: bool = True) -> tuple[bool, str]:
        return self.update_speech_settings(stt_engine=engine, persist=persist)

    def update_speech_settings(
        self,
        stt_engine: str | None = None,
        whisper_model: str | None = None,
        persist: bool = True,
    ) -> tuple[bool, str]:
        try:
            self.runtime.settings.update_speech(
                stt_engine=stt_engine, whisper_model=whisper_model
            )
        except Exception as exc:
            return False, str(exc)
        # STT can be swapped live (unlike the LLM provider); rebuild now.
        self.stt = build_stt(self.runtime.settings)
        self.config["speech"] = dict(self.runtime.settings.speech)
        if persist:
            try:
                _persist_config_updates(
                    self._config_path, {"speech": self.config["speech"]}
                )
            except Exception as exc:
                return (
                    False,
                    f"Speech setting changed in memory but failed to save config: {exc}",
                )
        return True, "Speech settings updated successfully."

    def set_language(self, language: str, persist: bool = True) -> tuple[bool, str]:
        language = str(language).lower().strip()
        if language not in {"en", "fr"}:
            return False, "Invalid language. Use 'en' or 'fr'."

        self.config["language"] = language
        self.runtime.settings.language = language

        # Re-build/re-initialize speech settings (STT)
        self.stt = build_stt(self.runtime.settings)

        if self.provider is not None:
            try:
                self._rebuild_provider()
            except Exception as exc:
                return False, f"Language set but failed to reload model provider: {exc}"

        if persist:
            try:
                _persist_config_updates(self._config_path, {"language": language})
            except Exception as exc:
                return (
                    False,
                    f"Language changed in memory but failed to save config: {exc}",
                )

        return True, f"Language set to {language}."

    def set_tool_enabled(
        self, tool: str, enabled: bool, persist: bool = False
    ) -> tuple[bool, str]:
        try:
            self.runtime.settings.set_tool_enabled(tool, enabled)
        except Exception as exc:
            return False, str(exc)
        self.config["tools"] = self.runtime.settings.to_config()["tools"]
        if persist:
            try:
                _persist_config_updates(
                    self._config_path, {"tools": self.config["tools"]}
                )
            except Exception as exc:
                return (
                    False,
                    f"Tool setting changed in memory but failed to save config: {exc}",
                )
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
                _persist_config_updates(
                    self._config_path, {"name": clean_name, "personality": personality}
                )
            except Exception as exc:
                return (
                    False,
                    f"Robot name changed in memory but failed to save config: {exc}",
                )
        return True, f"Robot name set to {clean_name}."

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
                clean_robot_config = _normalize_robot_config(
                    robot_config, self.config.get("robot", {})
                )
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
                return (
                    False,
                    f"Robot settings changed in memory but failed to save config: {exc}",
                )

        return True, " ".join(messages) if messages else "No robot settings changed."


# ── Helpers ────────────────────────────────────────────────────────────────────


def _resolve_model_path() -> str:
    path = os.environ.get("MODEL_PATH", "")
    if path:
        expanded = sorted(glob.glob(os.path.expanduser(path)))
        if expanded:
            return expanded[0]
    from huggingface_hub import hf_hub_download

    print(f"Downloading {HF_REPO}/{HF_FILENAME} (first run only)…")
    return hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)


def _build_system_prompt(config: dict, runtime: RobotRuntime) -> str:
    language = config.get("language", "en")
    if language == "fr":
        language_block = "\nIMPORTANT: Always understand and respond to the user in French. Write your speech, plan, status updates, and done summaries in French."
    else:
        language_block = "\nIMPORTANT: Always understand and respond to the user in English. Write your speech, plan, status updates, and done summaries in English."

    return SYSTEM_PROMPT_TEMPLATE.format(
        robot_name=runtime.robot_name,
        personality_block=_format_personality_block(config),
        backend_block=_format_backend_block(runtime),
        language_block=language_block,
    )


def _format_personality_block(config: dict) -> str:
    personality = (
        config.get("personality", {})
        if isinstance(config.get("personality"), dict)
        else {}
    )
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
    if runtime.backend.name == "ros2":
        return (
            f"- Backend: ROS 2 node ({metadata.get('node_name')}).\n"
            f"- Subscribed topics: {metadata.get('odom_topic')} (odom), {metadata.get('joint_states_topic')} (joints).\n"
            f"- Published topics: {metadata.get('cmd_vel_topic')} (velocity), {metadata.get('gripper_topic')} (gripper)."
        )
    return f"- Backend: {runtime.backend.name}. Respect capability results and avoid assuming unsupported hardware."


def _format_string_list(value) -> str:
    if not isinstance(value, list):
        return ""
    items = [str(item).strip() for item in value if str(item).strip()]
    return ", ".join(items)


def _load_dotenv(start_dir: Path) -> None:
    """Load KEY=VALUE pairs from a .env file into the environment.

    Searches the given directory and its parents for a .env (so API keys like
    OPENAI_API_KEY work without exporting them). Existing environment variables
    are never overwritten.
    """
    for directory in [start_dir, *start_dir.parents]:
        env_path = directory / ".env"
        if not env_path.exists():
            continue
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except Exception as exc:
            print(f"[config] Failed to read {env_path}: {exc}")
        return


def _load_config(config_path: str | Path | None = None) -> dict:
    """Load runtime config with sensible defaults.

    Search order:
      1) explicit config_path
      2) workspace-root config.json
    """
    default = _default_config()

    path = (
        Path(config_path)
        if config_path
        else Path(__file__).resolve().parents[2] / "config.json"
    )
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

    return _merge_dict(default, raw)


def _default_config() -> dict:
    return {
        "name": "dino",
        "tts": "local",
        "language": "en",
        "personality": {
            "name": "dino",
            "persona": "A warm, curious, practical robot companion that likes helping with robotics and everyday reasoning.",
            "likes": [
                "clear plans",
                "useful autonomy",
                "learning from the environment",
            ],
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
        "active_profile": "fast",
        "profiles": {
            "fast": {
                "provider": "openai",
                "compute": "gpu",
                "tick_seconds": 6,
                "idle_review_seconds": 20,
                "camera_fps": 10,
                "deliberate_seconds": 120,
            },
            "balanced": {
                "provider": "local",
                "compute": "gpu",
                "tick_seconds": 15,
                "idle_review_seconds": 45,
                "camera_fps": 6.7,
                "deliberate_seconds": 240,
            },
            "power_saver": {
                "provider": "local",
                "compute": "cpu",
                "tick_seconds": 30,
                "idle_review_seconds": 120,
                "camera_fps": 3,
                "deliberate_seconds": 600,
            },
        },
        "cron": [],
        "models": {
            "active_provider": "openai",
            "providers": {
                "local": {
                    "enabled": False,
                    "model": HF_REPO,
                },
                "openai": {
                    "enabled": True,
                    "model": "gpt-5.4-nano-2026-03-17",
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
        "speech": {
            "stt_engine": "whisper",
            "whisper_model": "base",
            "whisper_compute": "auto",
        },
        "memory": {
            "enabled": True,
            "path": "data/memory.json",
            "max_episodes": 1000,
            "consolidate_seconds": 300,
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
            "ros2": {
                "node_name": "actum_node",
                "cmd_vel_topic": "/cmd_vel",
                "odom_topic": "/odom",
                "joint_states_topic": "/joint_states",
                "gripper_topic": "/gripper_cmd",
                "linear_speed": 0.25,
                "angular_speed": 0.5,
            },
        },
        "mcp": {
            "enabled": False,
            "servers": {},
        },
    }


SUPPORTED_ROBOT_BACKENDS = {"laptop", "fake", "unitree_g1", "lekiwi", "ros2"}


def _normalize_robot_config(
    robot_config: dict, current_robot_config: dict | None = None
) -> dict:
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

    unitree = (
        merged.get("unitree_g1", {})
        if isinstance(merged.get("unitree_g1"), dict)
        else {}
    )
    merged["unitree_g1"] = {
        "network_interface": str(unitree.get("network_interface", "eth0")).strip()
        or "eth0",
        "speaker_id": _coerce_int(unitree.get("speaker_id", 0), 0, 0, 1),
        "volume": _coerce_int(unitree.get("volume", 80), 80, 0, 100),
    }

    lekiwi = merged.get("lekiwi", {}) if isinstance(merged.get("lekiwi"), dict) else {}
    merged["lekiwi"] = {
        "remote_ip": str(lekiwi.get("remote_ip", "127.0.0.1")).strip() or "127.0.0.1",
        "port": _coerce_int(lekiwi.get("port", 5555), 5555, 1, 65535),
        "id": str(lekiwi.get("id", "lekiwi")).strip() or "lekiwi",
    }

    ros2 = merged.get("ros2", {}) if isinstance(merged.get("ros2"), dict) else {}
    merged["ros2"] = {
        "node_name": str(ros2.get("node_name", "actum_node")).strip() or "actum_node",
        "cmd_vel_topic": str(ros2.get("cmd_vel_topic", "/cmd_vel")).strip()
        or "/cmd_vel",
        "odom_topic": str(ros2.get("odom_topic", "/odom")).strip() or "/odom",
        "joint_states_topic": str(
            ros2.get("joint_states_topic", "/joint_states")
        ).strip()
        or "/joint_states",
        "gripper_topic": str(ros2.get("gripper_topic", "/gripper_cmd")).strip()
        or "/gripper_cmd",
        "linear_speed": float(ros2.get("linear_speed", 0.25)),
        "angular_speed": float(ros2.get("angular_speed", 0.5)),
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
    loop = asyncio.get_running_loop()
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
    agent.mic = mic
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
