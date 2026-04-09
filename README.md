# robo

On-device multimodal robot agent. The robot listens through its microphone, sees through its camera, and acts through a set of tools — navigating, manipulating objects, speaking, and remembering — all driven by an on-device LLM (Gemma 4 2B via litert-lm).

Designed to run on an **NVIDIA Jetson** (Orin / AGX), also works on Linux x86 and macOS Apple Silicon.

## How it works

Each time the robot hears a voice command (or receives a text/timer event), the LLM is given the audio + camera frame and a set of tools. It chains tool calls in sequence until it calls `done()`:

```
voice input + camera frame
    → LLM (Gemma 4 2B, on-device GPU)
        → look()             # capture camera frame
        → navigate("forward", 1.0)
        → speak("I'm on my way")
        → done("Moved toward the door")
    → TTS (Kokoro-82M) → speaker
```

This multi-step loop runs entirely on-device — no cloud, no network required.

## Install

```bash
# Jetson: install OpenCV with GStreamer support via apt, not pip
sudo apt install python3-opencv

# Install the package
pip install -e .

# Or with uv
uv pip install -e .
```

> **Python**: requires 3.10+ (JetPack 6.x ships 3.10).

## Run

```bash
# Headless — mic → agent → speaker, stdin also accepted
robo

# With monitoring dashboard at http://localhost:8000
robo-server
PORT=8080 robo-server

# Point to a local model instead of auto-downloading
MODEL_PATH=/path/to/gemma-4-E2B-it.litertlm robo
```

On first run without `MODEL_PATH`, the Gemma 4 2B model (~3.6 GB) is downloaded automatically from Hugging Face.

## Tools

The LLM can call any of these during a turn:

| Tool | Description |
|---|---|
| `done(summary)` | Mark task complete |
| `speak(text)` | Say something aloud |
| `navigate(direction, distance_m)` | Move forward / backward / left / right |
| `rotate(degrees)` | Rotate in place |
| `gripper(action)` | open / close / grab / release |
| `look()` | Capture a camera frame for analysis |
| `remember(key, value)` | Persist a fact across turns |
| `recall(key)` | Retrieve a stored fact |
| `list_memories()` | List all stored keys |
| `report_status(message)` | Log reasoning or ask for clarification |

To add a new capability, add a method to `RobotTools` in `src/robo/tools.py` and add it to `get_tools()`. Hardware integration points are marked with `# --- hardware hook ---`.

## Monitoring dashboard

`robo-server` starts a web UI at `http://localhost:8000` that shows:
- Live action log (colour-coded by tool type)
- Camera feed (updated after each turn)
- Memory contents
- Text command input

## Jetson setup notes

**Camera:**
- CSI cameras (IMX219 etc.) are auto-detected via a GStreamer pipeline (`nvarguscamerasrc`). Falls back to USB index 0 if unavailable.
- Force USB: `ROBO_CAMERA=usb robo`
- OpenCV must be the `apt` version — the PyPI build lacks GStreamer support.

**Audio:**
- Uses ALSA via `sounddevice`. If the default device is wrong: `ROBO_AUDIO_DEVICE=hw:1,0 robo`

**TTS GPU acceleration:**
- `kokoro-onnx` uses ONNX Runtime. The PyPI `onnxruntime-gpu` package does not have an aarch64+CUDA wheel.
- To enable GPU TTS, install NVIDIA's build from [Jetson Zoo](https://elinux.org/Jetson_Zoo#ONNX_Runtime). Without it, TTS runs on CPU (~0.3 s/sentence on Orin — fast enough).

**LLM GPU backend:**
- litert-lm uses CUDA/OpenCL delegates automatically when `Backend.GPU` is selected. Requires `libOpenCL.so` (part of JetPack).

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | _(auto-download)_ | Path to a local `.litertlm` model file |
| `PORT` | `8000` | Monitoring server port |
| `ROBO_CAMERA` | `auto` | Camera source: `auto`, `csi`, `usb`, or device index |
| `ROBO_AUDIO_DEVICE` | _(system default)_ | ALSA device name or index |
| `KOKORO_ONNX` | _(unset)_ | Set to `1` on macOS to force the ONNX backend instead of MLX |

## Project layout

```
pyproject.toml
src/
└── robo/
    ├── __init__.py
    ├── agent.py       # RobotAgent — core event loop and LLM interaction
    ├── tools.py       # RobotTools — all tool definitions
    ├── tts.py         # TTS backends (MLX on macOS, ONNX on Linux/Jetson)
    ├── perception.py  # AudioCapture (VAD), camera helpers
    └── server.py      # Optional monitoring WebSocket server
```
