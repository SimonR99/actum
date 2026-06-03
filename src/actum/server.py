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
import json
import os
import threading
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from actum.agent import RobotAgent
from actum.perception import AudioCapture


# ── Dashboard HTML (external file) ───────────────────────────────────────────

_DASHBOARD_PATH = Path(__file__).with_name("dashboard.html")


def _load_dashboard_html() -> str:
    try:
        return _DASHBOARD_PATH.read_text(encoding="utf-8")
    except Exception as exc:
        # Fallback keeps server usable even if the HTML file is missing.
        return f"<html><body><h1>Dashboard unavailable</h1><pre>{exc}</pre></body></html>"


# ── Route attachment ───────────────────────────────────────────────────────────

def attach_server(agent: RobotAgent, app: FastAPI):
    """Wire monitoring routes onto a FastAPI app."""

    @app.get("/")
    async def dashboard():
        return HTMLResponse(_load_dashboard_html())

    @app.post("/command")
    async def command(body: dict):
        text = (body.get("text") or "").strip()
        if not text:
            return JSONResponse({"error": "empty command"}, status_code=400)
        await agent.event_bus.put({"source": "chat", "text": text, "force": bool(body.get("force", True))})
        return {"queued": text}

    @app.post("/trigger/{source}")
    async def trigger(source: str, body: dict | None = None):
        body = body or {}
        source = source.strip().lower()
        if source not in {"vision", "chat", "language", "voice", "timer", "cron"}:
            return JSONResponse({"error": "unsupported trigger source"}, status_code=400)
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
        await agent.event_bus.put(event)
        return {"queued": event["source"]}

    @app.get("/settings")
    async def settings_get():
        return {
            "name": agent.get_name(),
            "mode": agent.get_mode(),
            "backend": agent.runtime.backend.name,
            "robot_config": agent.config.get("robot", {}),
            "hardware_connected": agent.runtime.backend.connected,
            "runtime_settings": agent.runtime.settings.to_dict(),
            "state": agent.runtime.snapshot(),
        }

    @app.get("/capabilities")
    async def capabilities_get():
        return {"capabilities": agent.runtime.capabilities.list()}

    @app.get("/state")
    async def state_get():
        return agent.runtime.snapshot()

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
        ok, message = agent.set_robot_settings(name=name, robot_config=robot_config, persist=persist)
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
        ok, message = agent.set_model_provider(
            body.get("provider", ""),
            model=body.get("model", ""),
            api_key=body.get("api_key", ""),
            enabled=body.get("enabled"),
            persist=bool(body.get("persist", False)),
            persist_secret=bool(body.get("persist_secret", False)),
        )
        return JSONResponse(
            {"ok": ok, "message": message, "settings": agent.runtime.settings.to_dict()},
            status_code=200 if ok else 400,
        )

    @app.post("/settings/tool")
    async def settings_tool(body: dict):
        ok, message = agent.set_tool_enabled(
            body.get("tool", ""),
            bool(body.get("enabled", True)),
            persist=bool(body.get("persist", False)),
        )
        return JSONResponse(
            {"ok": ok, "message": message, "settings": agent.runtime.settings.to_dict()},
            status_code=200 if ok else 400,
        )

    @app.post("/cron")
    async def cron_add(body: dict):
        name = str(body.get("name") or "").strip()
        instruction = str(body.get("instruction") or "").strip()
        if not name or not instruction:
            return JSONResponse({"error": "name and instruction are required"}, status_code=400)
        job = agent.runtime.add_cron_job(name, float(body.get("every_seconds", 60)), instruction)
        return {"job": job, "state": agent.runtime.snapshot()}

    @app.websocket("/events")
    async def events(ws: WebSocket):
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        agent._status_subscribers.append(q)

        # Send current state on connect
        await ws.send_text(json.dumps({"type": "memory", "data": agent.runtime.memory.snapshot()}))
        await ws.send_text(json.dumps({"type": "state", "data": agent.runtime.snapshot()}))
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
        frame = agent.capture_frame()
        if frame:
            await ws.send_text(json.dumps({"type": "frame", "jpeg": frame}))

        try:
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=0.35)
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
                                "companion": payload.get("companion", {}),
                            }
                        )
                    )
                    await ws.send_text(json.dumps({"type": "memory", "data": agent.runtime.memory.snapshot()}))
                    await ws.send_text(json.dumps({"type": "state", "data": agent.runtime.snapshot()}))

                frame = agent.capture_frame()
                if frame:
                    await ws.send_text(json.dumps({"type": "frame", "jpeg": frame}))
        except WebSocketDisconnect:
            pass
        finally:
            agent._status_subscribers.remove(q)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    agent: RobotAgent = app.state.agent

    print("Loading models…")
    await loop.run_in_executor(None, agent.load_models)

    def on_speech(wav_b64: str):
        event: dict = {"source": "language", "audio": wav_b64}
        frame = agent.capture_frame()
        if frame:
            event["image"] = frame
        loop.call_soon_threadsafe(agent.event_bus.put_nowait, event)

    mic = AudioCapture(on_speech)
    threading.Thread(target=mic.run, daemon=True).start()

    agent_task = asyncio.create_task(agent.run())
    background_task = asyncio.create_task(agent.background_loop())
    yield

    mic.stop()
    agent.stop_background_loop()
    background_task.cancel()
    with suppress(asyncio.CancelledError):
        await background_task
    await agent.event_bus.put(None)
    await agent_task
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


if __name__ == "__main__":
    cli()
