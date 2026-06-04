# Docker

Docker images are for local development, dashboard operation, and Jetson deployment.
The image build compiles the React/Tailwind dashboard from `frontend/` and serves the generated bundle through `actum-server`.

The image is **self-contained** — it bundles every stack so there are no host
environment conflicts: the local LiteRT/Gemma model, the OpenAI cloud provider,
local Whisper speech-to-text, camera/audio support, and the MCP bridge. The default
local Gemma and Whisper `base` models are baked in so first use is instant.

API keys come from the project `.env` automatically (the compose files load it when
present), so `OPENAI_API_KEY=...` in `.env` is enough to use the OpenAI provider or
cloud STT. Voice from the dashboard works without any host audio device — the browser
records and the container transcribes with Whisper; host `/dev/snd` is only needed for
the headless server-side microphone.

## Build

```bash
./docker/build.sh
```

or:

```bash
docker build -t actum:latest .
```

Jetson:

```bash
docker build -f Dockerfile.jetson -t actum:jetson .
```

## Run

Dashboard:

```bash
./docker/run-server.sh
```

Headless:

```bash
./docker/run-headless.sh
```

Shell:

```bash
./docker/shell.sh
```

## Configuration

The container mounts `config.json` into `/workspace/config.json`.

Select a backend in `config.json`:

```json
{
  "robot": {
    "backend": "fake"
  }
}
```

Use `unitree_g1` or `lekiwi` only after installing the relevant optional stack in your image or development environment.

## Hardware Devices

The helper scripts mount `/dev/snd` and `/dev/video0` only when they exist.

For Unitree G1, use host networking or the correct routed interface if DDS traffic cannot cross the container network boundary:

```bash
docker run --rm -it --network host ...
```

Install the Unitree SDK2 package in a derived image or interactive shell:

```bash
pip install git+https://github.com/unitreerobotics/unitree_sdk2_python.git
```

For LeKiwi:

```bash
pip install "lerobot[lekiwi]"
```

## Model Cache

Hugging Face model downloads are cached in the `actum-huggingface` Docker volume.

```bash
docker volume ls | grep actum
docker volume rm actum-huggingface
```

## Troubleshooting

GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-runtime nvidia-smi
```

Audio:

```bash
docker exec actum-agent arecord -L
docker exec actum-agent speaker-test -t wav -c 2 -l 1
```

Camera:

```bash
docker exec actum-agent ls /dev/video*
```
