"""Optional monitoring + remote-control server.

Exposes:
  GET  /          — web dashboard (action log, camera, memory, send commands)
  WS   /events    — live stream of turn summaries as JSON
  POST /command   — inject a text command into the agent's event bus

Run:
    robo-server                        # agent + server together
    PORT=8080 robo-server              # custom port
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from robo.agent import RobotAgent
from robo.perception import AudioCapture


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
        await agent.event_bus.put({"source": "text", "text": text})
        return {"queued": text}

    @app.get("/settings")
    async def settings_get():
        return {
        "name": agent.get_name(),
            "mode": agent.get_mode(),
            "hardware_connected": agent._hardware is not None,
        }

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
                "hardware_connected": agent._hardware is not None,
                "message": message,
            },
            status_code=status,
        )

    @app.websocket("/events")
    async def events(ws: WebSocket):
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        agent._status_subscribers.append(q)

        # Send current state on connect
        await ws.send_text(json.dumps({"type": "memory", "data": agent.memory}))
        await ws.send_text(
            json.dumps(
                {
                    "type": "settings",
                    "mode": agent.get_mode(),
                    "hardware_connected": agent._hardware is not None,
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
                    await ws.send_text(json.dumps({"type": "turn", "actions": payload.get("actions", [])}))
                    await ws.send_text(json.dumps({"type": "memory", "data": agent.memory}))

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
        event: dict = {"source": "voice", "audio": wav_b64}
        frame = agent.capture_frame()
        if frame:
            event["image"] = frame
        loop.call_soon_threadsafe(agent.event_bus.put_nowait, event)

    mic = AudioCapture(on_speech)
    threading.Thread(target=mic.run, daemon=True).start()

    agent_task = asyncio.create_task(agent.run())
    yield

    mic.stop()
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
