# Robo

Agentic robotics runtime for local multimodal agents, robot backends, and operator observability.

The project is being shaped as a reference architecture for robots that need to answer questions, follow instructions, plan, use tools, run learned policies, and expose their current intent to a human operator.

## Design Goals

- Robot-agnostic core: Unitree G1, LeKiwi, simulated robots, and future ROS 2/MCP adapters share one backend interface.
- Explicit intent: the robot maintains a live goal, plan, active step, tool graph, and event log.
- Safety-first extensibility: high-level agent tools call typed backend actions instead of directly touching hardware.
- Local-first operation: camera, microphone, TTS, and model inference can run without cloud services.
- Policy-ready: LeRobot/VLA policies are treated as capabilities that the planner can invoke, not as the whole robot brain.

## Architecture

```
operator voice/text/camera
    -> RobotAgent
        -> IntentState       # goal, plan, active step, done/blocked state
        -> RobotTools        # LLM-visible capabilities
        -> RobotRuntime      # event log, tool graph, backend state
        -> RobotBackend      # fake | unitree_g1 | lekiwi | future ros2/mcp
    -> dashboard             # chat, camera, action trace, intent, tool graph
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
# Unitree G1 SDK2
pip install git+https://github.com/unitreerobotics/unitree_sdk2_python.git

# LeKiwi / LeRobot
pip install "lerobot[lekiwi]"
```

## Configure

`config.json` selects the active robot backend:

```json
{
  "name": "spacewalker",
  "tts": "local",
  "robot": {
    "backend": "fake",
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
  }
}
```

Backends:

- `fake`: deterministic local backend for development and tests.
- `unitree_g1`: Unitree SDK2 adapter for G1 speech, locomotion, gestures, and LEDs.
- `lekiwi`: LeRobot client adapter for LeKiwi.

## Run

```bash
# Dashboard at http://localhost:8000
robo-server

# Headless mic/text loop
robo

# Use a local LiteRT model
MODEL_PATH=/path/to/model.litertlm robo-server
```

Docker:

```bash
./docker/build.sh
./docker/run-server.sh
./docker/run-headless.sh
```

## Agent Tools

The LLM can call:

| Tool | Purpose |
|---|---|
| `set_plan(goal, steps)` | Publish current intent and a newline-separated plan |
| `mark_step(step)` | Mark the active plan step |
| `look()` | Capture a fresh camera frame |
| `navigate(direction, distance_m)` | Move through the active backend |
| `rotate(degrees)` | Rotate through the active backend |
| `gripper(action)` | Control end-effector if supported |
| `wave(gesture)` | Run backend gesture |
| `speak(text)` | Queue speech |
| `remember(key, value)` / `recall(key)` | Store and retrieve memory |
| `report_status(message)` | Publish operator-visible status |
| `done(summary)` | Finish the task |

## Dashboard

The dashboard streams:

- chat responses
- camera frames
- action trace
- live intent/plan state
- tool graph nodes with results
- memory
- backend state

It intentionally shows structured reasoning artifacts and summaries, not private chain-of-thought.

## Project Layout

```
src/robo/
  agent.py              # LLM loop and CLI
  runtime.py            # shared runtime state
  tools.py              # LLM-visible tools
  server.py             # FastAPI/WebSocket dashboard
  perception.py         # camera and microphone helpers
  tts.py                # local TTS backends
  core/                 # events, intent, schemas
  backends/             # fake, Unitree G1, LeKiwi
```

## Development

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src tests
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the architecture stance and [docs/REBOOT_PLAN.md](docs/REBOOT_PLAN.md) for the roadmap.
