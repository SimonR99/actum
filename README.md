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

```bash
pip install -e ".[dev,camera]"
```

Jetson note: install OpenCV with GStreamer from apt instead of PyPI:

```bash
sudo apt install python3-opencv
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
  "models": {
    "active_provider": "local",
    "providers": {
      "local": {"enabled": true, "model": "litert-community/gemma-4-E2B-it-litert-lm"},
      "openai": {"enabled": false, "model": "", "api_key_env": "OPENAI_API_KEY"},
      "anthropic": {"enabled": false, "model": "", "api_key_env": "ANTHROPIC_API_KEY"}
    }
  },
  "memory": {
    "enabled": true,
    "path": "data/memory.json",
    "max_episodes": 1000
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

## Run

```bash
# Dashboard at http://localhost:8000
actum-server

# Headless mic/text loop
actum

# Use a local LiteRT model
MODEL_PATH=/path/to/model.litertlm actum-server
```

Docker:

```bash
./docker/build.sh
./docker/run-server.sh
./docker/run-headless.sh
```

Hardware environment variables:

- `ACTUM_CAMERA`: `auto`, `usb`, `csi`, device index, device path, or GStreamer pipeline.
- `ACTUM_AUDIO_DEVICE`: sounddevice/ALSA device such as `hw:1,0`.
- `ACTUM_VAD_DEBUG`: set to any value to print voice activity detector diagnostics.

Legacy `SENSORIMOTOR_*` and `ROBO_*` names are still accepted as fallbacks for older local configs.

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
| `schedule_job(name, every_seconds, instruction)` | Create a recurring background instruction |
| `web_fetch(url, max_chars)` | Fetch readable text from an HTTP(S) URL |
| `list_mcp_servers()` | Show configured MCP servers |
| `list_mcp_tools(server)` | Inspect tools exposed by a configured MCP server |
| `call_mcp_tool(server, tool, arguments_json)` | Call a configured MCP tool |
| `report_status(message)` | Publish operator-visible status |
| `done(summary)` | Finish the task |

## Dashboard

The dashboard streams:

- camera frames
- action trace
- live intent/plan state
- behavior tree state
- tool graph nodes with results
- live map and body perception
- model/provider/tool settings
- memory
- backend state
- companion/personality state

It intentionally shows structured reasoning artifacts and summaries, not private chain-of-thought.

## Project Layout

```
src/actum/
  agent.py              # LLM loop and CLI
  runtime.py            # shared runtime state
  tools.py              # LLM-visible tools
  server.py             # FastAPI/WebSocket dashboard
  perception.py         # camera and microphone helpers
  tts.py                # local TTS backends
  core/                 # events, intent, memory, companion policy, schemas
  backends/             # laptop, fake, Unitree G1, LeKiwi
  integrations/         # optional MCP and web adapters
```

## Development

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src tests
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the architecture stance and [docs/REBOOT_PLAN.md](docs/REBOOT_PLAN.md) for the roadmap.
