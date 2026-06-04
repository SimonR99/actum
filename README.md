# Actum

Agentic robotics runtime for local multimodal agents, robot backends, and operator observability.

The project is being shaped as a reference architecture for robots that need to answer questions, follow instructions, plan, use tools, run learned policies, and expose their current intent to a human operator.
The name points to the core loop: sense the world, form intent, act through an embodiment, observe the result, and adapt.

## Design Goals

- Robot-agnostic core: Unitree G1, LeKiwi, simulated robots, and future ROS 2/MCP adapters share one backend interface.
- Explicit intent: the robot maintains a live goal, plan, active step, tool graph, and event log.
- Safety-first extensibility: high-level agent tools call typed backend actions instead of directly touching hardware.
- Local-first operation: camera, microphone, TTS, and model inference can run without cloud services.
- Always-on companion mode: passive observations are filtered before the LLM so the robot can stay present without constantly acting.
- MCP-ready software tools: trusted external data/tool servers can be attached without owning the robot kernel.
- Policy-ready: LeRobot/VLA policies are treated as capabilities that the planner can invoke, not as the whole robot brain.

## Architecture

```
operator vision/chat/language
    -> RobotAgent
        -> CompanionPolicy   # direct input vs passive observation gate
        -> IntentState       # goal, plan, active step, done/blocked state
        -> BehaviorTreeState # waiting/perception/action nodes
        -> MemoryStore       # facts, people, places, observations, spatial notes
        -> RobotTools        # LLM-visible capabilities
        -> RobotRuntime      # event log, tool graph, backend state
        -> RobotBackend      # laptop | fake | unitree_g1 | lekiwi | future ros2
    -> dashboard             # intent, behavior tree, perception, map, settings, tool graph
```

The core runtime is framework-agnostic on purpose. MCP, ROS 2, LangGraph, RAI, or other agent frameworks should attach through adapters. The robot's safety, state, and action contract stay inside this repo.

## Install

Linux host dependencies:

```bash
sudo apt install libportaudio2 portaudio19-dev python3-opencv
```

```bash
pip install -e ".[dev,camera]"
```

Jetson note: install OpenCV with GStreamer from apt instead of PyPI:

```bash
sudo apt install libportaudio2 portaudio19-dev python3-opencv
pip install -e ".[dev]"
```

Optional robot stacks:

```bash
# MCP client bridge for trusted external tool servers
pip install -e ".[mcp]"

# Unitree G1 SDK2
pip install git+https://github.com/unitreerobotics/unitree_sdk2_python.git

# LeKiwi / LeRobot
pip install "lerobot[lekiwi]"
```

## Configure

`config.json` selects the active robot backend:

```json
{
  "name": "dino",
  "tts": "local",
  "personality": {
    "name": "dino",
    "persona": "A warm, curious, practical robot companion.",
    "likes": ["clear plans", "useful autonomy"],
    "principles": ["be helpful without being intrusive"],
    "speaking_style": "concise, warm, and calm"
  },
  "companion": {
    "always_on": true,
    "proactive_mode": "conservative",
    "direct_sources": ["chat", "language", "voice", "text"],
    "passive_sources": ["vision", "timer", "cron", "loop"],
    "min_seconds_between_proactive_actions": 60
  },
  "behavior_loop": {
    "enabled": true,
    "tick_seconds": 15,
    "idle_review": true,
    "idle_review_seconds": 45
  },
  "active_profile": "balanced",
  "profiles": {
    "fast":        {"provider": "openai", "compute": "gpu", "tick_seconds": 6,  "idle_review_seconds": 20,  "camera_fps": 10,  "deliberate_seconds": 120},
    "balanced":    {"provider": "local",  "compute": "gpu", "tick_seconds": 15, "idle_review_seconds": 45,  "camera_fps": 6.7, "deliberate_seconds": 240},
    "power_saver": {"provider": "local",  "compute": "cpu", "tick_seconds": 30, "idle_review_seconds": 120, "camera_fps": 3,   "deliberate_seconds": 600}
  },
  "models": {
    "active_provider": "local",
    "providers": {
      "local": {"enabled": true, "model": "litert-community/gemma-4-E2B-it-litert-lm"},
      "openai": {"enabled": false, "model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
      "anthropic": {"enabled": false, "model": "", "api_key_env": "ANTHROPIC_API_KEY"}
    }
  },
  "memory": {
    "enabled": true,
    "path": "data/memory.json",
    "max_episodes": 1000,
    "consolidate_seconds": 300
  },
  "robot": {
    "backend": "laptop",
    "laptop": {
      "webcam": true,
      "microphone": true,
      "speaker": true
    },
    "unitree_g1": {
      "network_interface": "eth0",
      "speaker_id": 0,
      "volume": 80
    },
    "lekiwi": {
      "remote_ip": "127.0.0.1",
      "port": 5555,
      "id": "lekiwi"
    }
  },
  "mcp": {
    "enabled": false,
    "servers": {}
  }
}
```

Backends:

- `laptop`: local companion backend using webcam, microphone, speaker, web/MCP software tools, and no physical motion.
- `fake`: deterministic local backend for development and tests.
- `unitree_g1`: Unitree SDK2 adapter for G1 speech, locomotion, gestures, and LEDs.
- `lekiwi`: LeRobot client adapter for LeKiwi.

The dashboard Context panel includes a Robot editor for changing the active name,
backend, laptop I/O flags, Unitree G1 network/audio settings, and LeKiwi
connection settings. Changes can be applied in memory or saved back to
`config.json`.

Speed profiles:

`active_profile` is one switch for the speed/cost/power tradeoff. Each profile bundles
the inference provider, the local compute backend, the autonomy loop tick rate, the idle
review cadence, the dashboard camera frame rate, and how often the robot sets its own
tasks. The active profile is authoritative for those knobs and overrides
`models.active_provider`.

- `fast`: cloud inference (OpenAI), GPU, snappy loops and camera — lowest latency, needs an API key.
- `balanced`: local on-device model on GPU at moderate cadence (default).
- `power_saver`: local model on CPU with slow loops and low camera fps — for thermally or battery constrained hardware.

Define your own profiles by adding entries under `profiles`. Switch at runtime from the
dashboard or with `POST /settings/profile {"profile": "fast"}`. Provider/compute changes
take effect on the next model load; loop-rate changes apply immediately.

Inference providers (the swappable "brain"):

- `local`: on-device LiteRT/Gemma multimodal model. `compute` selects the LiteRT backend (`gpu`, `cpu`, or `npu`).
- `openai`: OpenAI chat models (multimodal + tool calling), e.g. `gpt-4o-mini`. Install with `pip install -e ".[openai]"`.

API keys are read from the environment. A `.env` file in the project root is loaded
automatically (it is gitignored), so `OPENAI_API_KEY=...` in `.env` is enough.

Speech-to-text (selectable in the dashboard, `speech.stt_engine`):

- `whisper`: local [faster-whisper](https://github.com/SYSTRAN/faster-whisper), fully on-device (default). Install with `pip install -e ".[whisper]"`; the model downloads on first use.
- `openai`: OpenAI cloud transcription — no local model, needs an API key.
- `model`: no separate STT; raw audio is passed to the multimodal model (local Gemma only).

The robot's voice (text-to-speech) remains local Kokoro regardless of the STT choice.

Companion modes:

- `off`: direct voice/text only; passive camera/timer events are ignored.
- `conservative`: passive events are admitted for safety, explicit requests, or high-importance observations.
- `curious`: allows moderately important passive observations after the configured cooldown.

Memory:

- Semantic facts: stable preferences and facts from `remember(key, value)`.
- People: relationship/context notes from `remember_person(name, note)`.
- Places: named rooms, landmarks, dock locations, and work areas from `remember_place(name, note)`.
- Spatial notes: lightweight mapping/navigation facts from `remember_spatial_note(summary, place)`.
- Episodes/observations: completed turns and meaningful things the robot saw or did.

The default store is JSON-backed at `data/memory.json` so it works on a laptop without a database. The interface is intentionally ready for SQLite, vector search, and richer spatial maps later.

The robot organises its own memory:

- Relevant-memory retrieval: each turn pulls only the entries related to the request (lexical
  relevance ranking) instead of dumping the whole store, so prompts stay focused as memory grows.
  The agent can also recall on demand with `search_memory(query)`.
- Self-consolidation: every `memory.consolidate_seconds` the background loop dedupes and trims
  memory automatically (set to `0` to disable). The agent can also trigger it with `consolidate_memory()`.

## Run

```bash
# Dashboard at http://localhost:8000
actum-server

# Headless mic/text loop
actum

# Use a local LiteRT model
MODEL_PATH=/path/to/model.litertlm actum-server
```

`actum` is the headless terminal entrypoint and does not open the web UI. If
`http://localhost:8000` returns connection refused, check that `actum-server`
is the process you started and that the active Python environment has the
project installed:

```bash
which actum-server
pip install -e ".[dev,camera]"
```

Docker:

```bash
./docker/build.sh
./docker/run-server.sh
./docker/run-headless.sh
```

Frontend:

```bash
# Served by actum-server after a build
npm install --prefix frontend
npm run build --prefix frontend

# UI development server; keep actum-server running on port 8000
npm run dev --prefix frontend
```

Hardware environment variables:

- `ACTUM_CAMERA`: `auto`, `usb`, `csi`, device index, device path, or GStreamer pipeline.
- `ACTUM_CAMERA_STREAM_INTERVAL_S`: dashboard camera send interval in seconds, default `0.15`.
- `ACTUM_CAMERA_STREAM_FPS`: dashboard camera frame rate if interval is unset, default `6.7`.
- `ACTUM_CAMERA_STREAM_WIDTH`: dashboard JPEG width, default `320`.
- `ACTUM_CAMERA_STREAM_QUALITY`: dashboard JPEG quality, default `60`.
- `ACTUM_CAMERA_STREAM_DRAIN_FRAMES`: stale frames to drop for the dashboard stream, default `2`.
- `ACTUM_CAMERA_DRAIN_FRAMES`: stale frames to drop for one-off captures such as `look()`, default `1`.
- `ACTUM_AUDIO_DEVICE`: sounddevice/ALSA device such as `hw:1,0`.
- `ACTUM_VAD_DEBUG`: set to any value to print voice activity detector diagnostics.

If server-side microphone capture logs `PortAudio library not found`, install `libportaudio2`/`portaudio19-dev` on the host. The dashboard Language button can also record through the browser microphone on `localhost`; press it once to start recording and again to send the audio.

## Agent Tools

The LLM can call:

| Tool | Purpose |
|---|---|
| `set_plan(goal, steps)` | Publish current intent and a newline-separated plan |
| `mark_step(step)` | Mark the active plan step |
| `set_behavior_tree(goal, nodes_json)` | Publish a behavior-tree-like autonomy state |
| `mark_behavior_node(node, status, detail)` | Update a behavior node |
| `look()` | Capture a fresh camera frame |
| `navigate(direction, distance_m)` | Move through the active backend |
| `rotate(degrees)` | Rotate through the active backend |
| `gripper(action)` | Control end-effector if supported |
| `wave(gesture)` | Run backend gesture |
| `speak(text)` | Queue speech |
| `remember(key, value)` / `recall(key)` | Store and retrieve memory |
| `remember_person(name, note)` | Store a person profile note |
| `remember_place(name, note)` | Store a place or landmark note |
| `record_observation(summary, tags)` | Store an episodic observation |
| `remember_spatial_note(summary, place)` | Store a mapping/navigation note |
| `record_map_observation(summary, place, x, y, yaw_deg, confidence)` | Add a live map observation |
| `update_body_perception(summary, posture, holding, contacts, joints_json)` | Update body/self perception |
| `recent_memories(limit, kind)` | Read recent memory records |
| `search_memory(query, limit)` | Retrieve memory entries relevant to a query |
| `consolidate_memory()` | Remove duplicate/stale memory records |
| `schedule_job(name, every_seconds, instruction)` | Create a recurring background instruction |
| `web_fetch(url, max_chars)` | Fetch readable text from an HTTP(S) URL |
| `list_mcp_servers()` | Show configured MCP servers |
| `list_mcp_tools(server)` | Inspect tools exposed by a configured MCP server |
| `call_mcp_tool(server, tool, arguments_json)` | Call a configured MCP tool |
| `report_status(message)` | Publish operator-visible status |
| `done(summary)` | Finish the task |

## Dashboard

The dashboard includes a combined **chat + voice console** (right rail, alongside Settings):
type a message or hold the mic to talk, and the robot's replies render as Markdown in a
transcript. The UI is bilingual — **French by default**, switchable to English with the FR/EN
toggle in the header (the choice is remembered per browser).

The dashboard streams:

- camera frames
- action trace
- live intent/plan state
- behavior tree state
- tool graph nodes with results
- live map and body perception
- robot/backend, model/provider, and tool settings
- memory
- backend state
- companion/personality state

It intentionally shows structured reasoning artifacts and summaries, not private chain-of-thought.

## Project Layout

```text
src/actum/
  agent.py              # LLM loop and CLI
  runtime.py            # shared runtime state
  tools.py              # LLM-visible tools
  server.py             # FastAPI/WebSocket dashboard
  perception.py         # camera and microphone helpers
  tts.py                # local TTS backends
  core/                 # events, intent, memory, companion policy, profiles, schemas
  inference/            # swappable brain (LiteRT/OpenAI) + selectable speech-to-text
  backends/             # laptop, fake, Unitree G1, LeKiwi
  integrations/         # optional MCP and web adapters
frontend/
  src/                  # React/Tailwind dashboard
  dist/                 # generated dashboard served by actum-server
```

## Development

A `Makefile` wraps the common tasks — run `make` to list them:

```bash
make install-all   # editable install with every stack (dev, camera, openai, whisper, mcp)
make frontend      # build the dashboard into frontend/dist
make server        # run the agent + dashboard
make test          # run the test suite
make up            # build and run everything in Docker
```

Override the interpreter with `make <target> PYTHON=~/miniconda3/bin/python`.

Equivalent raw commands:

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src tests
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the architecture stance and [docs/REBOOT_PLAN.md](docs/REBOOT_PLAN.md) for the roadmap.
