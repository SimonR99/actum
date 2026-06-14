"""Optional monitoring + remote-control server.

Exposes:
  GET  /          — web dashboard (intent, perception, behavior tree, settings)
  WS   /events    — live stream of turn summaries as JSON
  POST /command   — inject a chat command into the agent's event bus

Run:
    actum-server                        # agent + server together
    PORT=8080 actum-server              # custom port
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import os
import threading
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from actum.agent import RobotAgent
from actum.perception import AudioCapture

# ── Dashboard assets ────────────────────────────────────────────────────────

_DASHBOARD_PATH = Path(__file__).with_name("dashboard.html")
_FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
_FRONTEND_INDEX = _FRONTEND_DIST / "index.html"


def _load_dashboard_html() -> str:
    try:
        return _DASHBOARD_PATH.read_text(encoding="utf-8")
    except Exception as exc:
        # Fallback keeps server usable even if the HTML file is missing.
        return (
            f"<html><body><h1>Dashboard unavailable</h1><pre>{exc}</pre></body></html>"
        )


# ── Camera streaming ───────────────────────────────────────────────────────────


class CameraFrameStream:
    """Continuously keep the freshest dashboard frame outside the WebSocket loop."""

    def __init__(self, agent: RobotAgent):
        self.agent = agent
        self.interval_s = _camera_stream_interval(agent)
        self.width = _camera_stream_width()
        self.quality = _camera_stream_quality()
        self.drain_frames = _camera_stream_drain_frames()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: dict | None = None
        self._frame_id = 0

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="actum-camera-stream", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)

    def latest_after(self, frame_id: int) -> dict | None:
        with self._lock:
            if not self._latest or self._latest["frame_id"] <= frame_id:
                return None
            return dict(self._latest)

    def _run(self):
        while not self._stop.is_set():
            started_at = time.monotonic()
            frame = self.agent.capture_frame(
                width=self.width,
                quality=self.quality,
                drain_frames=self.drain_frames,
            )
            if frame:
                with self._lock:
                    self._frame_id += 1
                    self._latest = {
                        "type": "frame",
                        "jpeg": frame,
                        "frame_id": self._frame_id,
                        "sent_at": time.time(),
                    }
            elapsed = time.monotonic() - started_at
            self._stop.wait(max(0.01, self.interval_s - elapsed))


# ── Route attachment ───────────────────────────────────────────────────────────


def attach_server(agent: RobotAgent, app: FastAPI):
    """Wire monitoring routes onto a FastAPI app."""
    _init_server_state(app)
    if not hasattr(app.state, "camera_stream"):
        app.state.camera_stream = CameraFrameStream(agent)
    camera_stream: CameraFrameStream = app.state.camera_stream

    assets_dir = _FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

    @app.get("/")
    async def dashboard():
        if _FRONTEND_INDEX.exists():
            return FileResponse(_FRONTEND_INDEX)
        return HTMLResponse(_load_dashboard_html())

    @app.post("/command")
    async def command(body: dict):
        not_ready = _not_ready_response(app)
        if not_ready:
            return not_ready
        text = (body.get("text") or "").strip()
        if not text:
            return JSONResponse({"error": "empty command"}, status_code=400)
        await agent.event_bus.put(
            {"source": "chat", "text": text, "force": bool(body.get("force", True))}
        )
        return {"queued": text}

    @app.post("/trigger/{source}")
    async def trigger(source: str, body: dict | None = None):
        not_ready = _not_ready_response(app)
        if not_ready:
            return not_ready
        body = body or {}
        source = source.strip().lower()
        if source not in {"vision", "chat", "language", "voice", "timer", "cron"}:
            return JSONResponse(
                {"error": "unsupported trigger source"}, status_code=400
            )
        event = {
            "source": "language" if source == "voice" else source,
            "text": str(body.get("text") or ""),
            "importance": body.get("importance", 0.0),
            "force": bool(body.get("force", source in {"chat", "language", "voice"})),
        }
        image = body.get("image")
        if source == "vision" and not image:
            image = agent.capture_frame()
        if image:
            event["image"] = image
        audio = body.get("audio")
        if audio:
            event["audio"] = str(audio)
        await agent.event_bus.put(event)
        return {"queued": event["source"]}

    @app.post("/debug/transcribe")
    async def debug_transcribe(body: dict | None = None):
        """Transcribe a base64 WAV blob through the configured STT engine only.

        Deliberately *not* gated behind the runtime-ready check: it exercises the
        speech-to-text stage in isolation, so voice transcription can be tested
        even when model loading / hardware bring-up has failed. Returns an
        actionable message when the selected engine's dependency is missing
        instead of a silent empty transcript.
        """
        body = body or {}
        audio = str(body.get("audio") or "")
        settings = agent.runtime.settings
        speech = getattr(settings, "speech", {}) or {}
        engine_name = str(speech.get("stt_engine", "whisper"))
        info: dict = {"engine": engine_name}

        if not audio:
            return JSONResponse(
                {**info, "ok": False, "error": "No audio provided."}, status_code=400
            )
        try:
            info["audio_bytes"] = len(base64.b64decode(audio))
        except Exception:
            return JSONResponse(
                {**info, "ok": False, "error": "Audio is not valid base64."},
                status_code=400,
            )

        # Pre-flight dependency checks → clear guidance instead of empty output.
        if (
            engine_name == "whisper"
            and importlib.util.find_spec("faster_whisper") is None
        ):
            return {
                **info,
                "ok": False,
                "transcript": "",
                "error": "faster-whisper is not installed. Install with  pip install -e '.[whisper]'  "
                "or pick OpenAI / Multimodal model as the speech engine.",
            }
        if engine_name == "openai" and importlib.util.find_spec("openai") is None:
            return {
                **info,
                "ok": False,
                "transcript": "",
                "error": "openai is not installed. Install with  pip install -e '.[openai]'.",
            }

        from actum.inference.stt import build_stt

        engine = build_stt(settings)
        if engine is None:
            return {
                **info,
                "ok": True,
                "transcript": "",
                "note": "Engine 'model' sends audio straight to the multimodal LLM, so there is no "
                "standalone transcript to show. Switch to Whisper or OpenAI to test STT.",
            }

        started = time.time()
        try:
            text = await asyncio.get_running_loop().run_in_executor(
                None, engine.transcribe, audio
            )
        except Exception as exc:  # surface engine/runtime errors to the operator
            return {**info, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        info["elapsed"] = round(time.time() - started, 2)
        info["transcript"] = text
        info["ok"] = bool(text)
        if not text:
            info["note"] = (
                "Engine ran but produced no text — audio may be too short/quiet, or transcription failed (check server logs)."
            )
        return info

    @app.get("/settings")
    async def settings_get():
        return {
            "name": agent.get_name(),
            "mode": agent.get_mode(),
            "language": agent.config.get("language", "en"),
            "server": _server_status(app),
            "backend": agent.runtime.backend.name,
            "robot_config": agent.config.get("robot", {}),
            "hardware_connected": agent.runtime.backend.connected,
            "runtime_settings": agent.runtime.settings.to_dict(),
            "state": agent.runtime.snapshot(),
        }

    @app.get("/capabilities")
    async def capabilities_get():
        return {"capabilities": agent.runtime.capabilities.list()}

    @app.get("/health")
    async def health_get():
        status = _server_status(app)
        status_code = 200 if status["ready"] else 503
        return JSONResponse(status, status_code=status_code)

    @app.get("/state")
    async def state_get():
        return _state_snapshot(agent, app)

    @app.post("/settings/mode")
    async def settings_mode(body: dict):
        mode = (body.get("mode") or "").strip().lower()
        persist = bool(body.get("persist", True))
        ok, message = agent.set_mode(mode, persist=persist)
        status = 200 if ok else 400
        return JSONResponse(
            {
                "ok": ok,
                "mode": agent.get_mode(),
                "backend": agent.runtime.backend.name,
                "hardware_connected": agent.runtime.backend.connected,
                "message": message,
            },
            status_code=status,
        )

    @app.get("/settings/robot")
    async def settings_robot_get():
        return {
            "name": agent.get_name(),
            "robot": agent.config.get("robot", {}),
            "backend": agent.runtime.backend.name,
            "hardware_connected": agent.runtime.backend.connected,
            "state": agent.runtime.snapshot(),
        }

    @app.post("/settings/robot")
    async def settings_robot_post(body: dict):
        persist = bool(body.get("persist", True))
        name = body.get("name") if "name" in body else None
        robot_config = body.get("robot") if "robot" in body else None
        ok, message = agent.set_robot_settings(
            name=name, robot_config=robot_config, persist=persist
        )
        return JSONResponse(
            {
                "ok": ok,
                "message": message,
                "name": agent.get_name(),
                "robot": agent.config.get("robot", {}),
                "backend": agent.runtime.backend.name,
                "hardware_connected": agent.runtime.backend.connected,
                "state": agent.runtime.snapshot(),
            },
            status_code=200 if ok else 400,
        )

    @app.post("/settings/model")
    async def settings_model(body: dict):
        loop = asyncio.get_running_loop()
        ok, message = await loop.run_in_executor(
            None,
            lambda: agent.set_model_provider(
                body.get("provider", ""),
                model=body.get("model", ""),
                api_key=body.get("api_key", ""),
                enabled=body.get("enabled"),
                persist=bool(body.get("persist", False)),
                persist_secret=bool(body.get("persist_secret", False)),
            ),
        )
        return JSONResponse(
            {
                "ok": ok,
                "message": message,
                "settings": agent.runtime.settings.to_dict(),
            },
            status_code=200 if ok else 400,
        )

    @app.post("/settings/profile")
    async def settings_profile(body: dict):
        loop = asyncio.get_running_loop()
        ok, message = await loop.run_in_executor(
            None,
            lambda: agent.set_profile(
                body.get("profile", ""), persist=bool(body.get("persist", True))
            ),
        )
        return JSONResponse(
            {"ok": ok, "message": message, "profile": agent.runtime.profiles.to_dict()},
            status_code=200 if ok else 400,
        )

    @app.post("/settings/speech")
    async def settings_speech(body: dict):
        ok, message = agent.update_speech_settings(
            stt_engine=body.get("stt_engine"),
            whisper_model=body.get("whisper_model"),
            persist=bool(body.get("persist", True)),
        )
        return JSONResponse(
            {
                "ok": ok,
                "message": message,
                "settings": agent.runtime.settings.to_dict(),
            },
            status_code=200 if ok else 400,
        )

    @app.post("/settings/language")
    async def settings_language(body: dict):
        language = (body.get("language") or "").strip().lower()
        persist = bool(body.get("persist", True))
        ok, message = agent.set_language(language, persist=persist)
        status = 200 if ok else 400
        return JSONResponse(
            {
                "ok": ok,
                "language": agent.config.get("language", "en"),
                "message": message,
            },
            status_code=status,
        )

    @app.post("/settings/tool")
    async def settings_tool(body: dict):
        ok, message = agent.set_tool_enabled(
            body.get("tool", ""),
            bool(body.get("enabled", True)),
            persist=bool(body.get("persist", False)),
        )
        return JSONResponse(
            {
                "ok": ok,
                "message": message,
                "settings": agent.runtime.settings.to_dict(),
            },
            status_code=200 if ok else 400,
        )

    @app.post("/conversation/reset")
    async def conversation_reset():
        not_ready = _not_ready_response(app)
        if not_ready:
            return not_ready
        ok, message = await asyncio.get_running_loop().run_in_executor(
            None, agent.reset_conversation
        )
        return JSONResponse(
            {"ok": ok, "message": message, "state": agent.runtime.snapshot()},
            status_code=200 if ok else 400,
        )

    @app.post("/cron")
    async def cron_add(body: dict):
        name = str(body.get("name") or "").strip()
        instruction = str(body.get("instruction") or "").strip()
        if not name or not instruction:
            return JSONResponse(
                {"error": "name and instruction are required"}, status_code=400
            )
        job = agent.runtime.add_cron_job(
            name, float(body.get("every_seconds", 60)), instruction
        )
        return {"job": job, "state": agent.runtime.snapshot()}

    @app.websocket("/events")
    async def events(ws: WebSocket):
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        agent._status_subscribers.append(q)
        camera_stream.start()
        frame_interval_s = camera_stream.interval_s
        last_frame_id = 0

        async def send_latest_frame():
            nonlocal last_frame_id
            payload = camera_stream.latest_after(last_frame_id)
            if not payload:
                return
            last_frame_id = int(payload["frame_id"])
            await ws.send_text(json.dumps(payload))

        # Send current state on connect
        await ws.send_text(
            json.dumps({"type": "memory", "data": agent.runtime.memory.snapshot()})
        )
        await ws.send_text(
            json.dumps({"type": "state", "data": _state_snapshot(agent, app)})
        )
        await ws.send_text(
            json.dumps(
                {
                    "type": "settings",
                    "name": agent.get_name(),
                    "mode": agent.get_mode(),
                    "backend": agent.runtime.backend.name,
                    "robot_config": agent.config.get("robot", {}),
                    "hardware_connected": agent.runtime.backend.connected,
                }
            )
        )
        await send_latest_frame()

        try:
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=frame_interval_s)
                except asyncio.TimeoutError:
                    payload = None

                if payload is not None:
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "turn",
                                "source": payload.get("source", ""),
                                "actions": payload.get("actions", []),
                                "elapsed": payload.get("elapsed", 0.0),
                                "ignored": bool(payload.get("ignored")),
                                "state_only": bool(payload.get("state_only")),
                                "reason": payload.get("reason", ""),
                                "error": payload.get("error", ""),
                                "companion": payload.get("companion", {}),
                            }
                        )
                    )
                    await ws.send_text(
                        json.dumps(
                            {"type": "memory", "data": agent.runtime.memory.snapshot()}
                        )
                    )
                    await ws.send_text(
                        json.dumps(
                            {"type": "state", "data": _state_snapshot(agent, app)}
                        )
                    )

                await send_latest_frame()
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            # Sends race against client disconnects; don't spam tracebacks.
            print(f"[server] events stream closed: {exc}")
        finally:
            if q in agent._status_subscribers:
                agent._status_subscribers.remove(q)


def _init_server_state(app: FastAPI):
    if not hasattr(app.state, "agent_status"):
        app.state.agent_status = "starting"
    if not hasattr(app.state, "agent_ready"):
        app.state.agent_ready = False
    if not hasattr(app.state, "agent_error"):
        app.state.agent_error = ""


def _set_server_status(app: FastAPI, status: str, ready: bool, error: str = ""):
    app.state.agent_status = status
    app.state.agent_ready = ready
    app.state.agent_error = error


def _server_status(app: FastAPI) -> dict:
    _init_server_state(app)
    port = int(os.environ.get("PORT", "8000"))
    return {
        "ready": bool(app.state.agent_ready),
        "status": str(app.state.agent_status),
        "error": str(app.state.agent_error),
        "port": port,
    }


def _state_snapshot(agent: RobotAgent, app: FastAPI) -> dict:
    snapshot = agent.runtime.snapshot()
    snapshot["server"] = _server_status(app)
    return snapshot


def _not_ready_response(app: FastAPI) -> JSONResponse | None:
    status = _server_status(app)
    if status["ready"]:
        return None
    message = (
        "Agent runtime is still loading."
        if status["status"] != "failed"
        else f"Agent startup failed: {status['error']}"
    )
    return JSONResponse({"error": message, "server": status}, status_code=503)


# ── Lifespan ───────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    agent: RobotAgent = app.state.agent
    camera_stream: CameraFrameStream | None = getattr(app.state, "camera_stream", None)
    _init_server_state(app)

    async def start_runtime():
        _set_server_status(app, "loading", ready=False)
        agent.runtime.events.append(
            "server.loading", "server", message="Loading models and hardware."
        )
        print("Loading models…")
        try:
            await loop.run_in_executor(None, agent.load_models)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            message = str(exc)
            _set_server_status(app, "failed", ready=False, error=message)
            agent.runtime.events.append(
                "server.failed", "server", message=message, error=message
            )
            print(f"[server] startup failed: {message}")
            agent._broadcast(
                {"source": "server", "actions": [], "elapsed": 0.0, "state_only": True}
            )
            return

        if camera_stream:
            camera_stream.start()

        def on_speech(wav_b64: str):
            event: dict = {"source": "language", "audio": wav_b64}
            frame = agent.capture_frame()
            if frame:
                event["image"] = frame
            loop.call_soon_threadsafe(agent.event_bus.put_nowait, event)

        mic = AudioCapture(on_speech)
        app.state.mic = mic
        agent.mic = mic
        threading.Thread(target=mic.run, daemon=True).start()

        app.state.agent_task = asyncio.create_task(agent.run())
        app.state.background_task = asyncio.create_task(agent.background_loop())
        _set_server_status(app, "ready", ready=True, error="")
        agent.runtime.events.append(
            "server.ready", "server", message="Agent runtime ready."
        )
        agent._broadcast(
            {"source": "server", "actions": [], "elapsed": 0.0, "state_only": True}
        )

    app.state.startup_task = asyncio.create_task(start_runtime())
    yield

    if camera_stream:
        camera_stream.stop()
    mic: AudioCapture | None = getattr(app.state, "mic", None)
    if mic:
        mic.stop()
    agent.stop_background_loop()
    background_task: asyncio.Task | None = getattr(app.state, "background_task", None)
    if background_task:
        background_task.cancel()
        with suppress(asyncio.CancelledError):
            await background_task
    agent_task: asyncio.Task | None = getattr(app.state, "agent_task", None)
    if agent_task:
        await agent.event_bus.put(None)
        await agent_task
    startup_task: asyncio.Task | None = getattr(app.state, "startup_task", None)
    if startup_task and not startup_task.done():
        startup_task.cancel()
        with suppress(asyncio.CancelledError):
            await startup_task
    agent.shutdown()


# ── CLI entrypoint ─────────────────────────────────────────────────────────────


def cli():
    """Agent + monitoring server entrypoint."""
    agent = RobotAgent()
    app = FastAPI(lifespan=lifespan)
    app.state.agent = agent
    attach_server(agent, app)

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


def _camera_stream_interval(agent: RobotAgent) -> float:
    """Frame interval for the dashboard stream.

    Defaults to the active speed profile's camera_fps; ACTUM_CAMERA_STREAM_*
    env vars override it for one-off tuning.
    """
    raw = os.environ.get("ACTUM_CAMERA_STREAM_INTERVAL_S")
    if raw is not None:
        try:
            return max(0.05, min(1.0, float(raw)))
        except ValueError:
            pass
    profile_fps = getattr(getattr(agent, "runtime", None), "camera_fps", 6.7)
    fps = _env_float("ACTUM_CAMERA_STREAM_FPS", float(profile_fps), 1.0, 20.0)
    return 1.0 / fps


def _camera_stream_width() -> int:
    return _env_int("ACTUM_CAMERA_STREAM_WIDTH", 320, 160, 1280)


def _camera_stream_quality() -> int:
    return _env_int("ACTUM_CAMERA_STREAM_QUALITY", 60, 30, 95)


def _camera_stream_drain_frames() -> int:
    return _env_int("ACTUM_CAMERA_STREAM_DRAIN_FRAMES", 2, 0, 8)


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


if __name__ == "__main__":
    cli()
