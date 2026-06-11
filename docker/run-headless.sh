#!/usr/bin/env bash
# Run Actum in headless mode (no web server)
# Speech input via microphone → agent → speaker output
# Also accepts stdin commands
# Usage: ./docker/run-headless.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Running Actum in headless mode..."
echo "  Mic -> Agent -> Speaker (no web dashboard)"
echo "  Type commands in terminal or speak:"
echo ""

ENV_ARGS=()
if [[ -f "$PROJECT_ROOT/.env.docker" ]]; then
    ENV_ARGS=(--env-file "$PROJECT_ROOT/.env.docker")
fi

DEVICE_ARGS=()
if [[ -d /dev/snd ]]; then
    DEVICE_ARGS+=(-v /dev/snd:/dev/snd --device /dev/snd)
fi
if [[ -e /dev/video0 ]]; then
    DEVICE_ARGS+=(-v /dev/video0:/dev/video0 --device /dev/video0)
fi

mkdir -p "$PROJECT_ROOT/logs"

docker run --rm -it \
    --gpus all \
    --name actum-headless \
    "${ENV_ARGS[@]}" \
    -v "$PROJECT_ROOT/config.json:/workspace/config.json" \
    -v actum-huggingface:/root/.cache/huggingface \
    -v "$PROJECT_ROOT/logs:/workspace/logs" \
    "${DEVICE_ARGS[@]}" \
    actum:latest \
    actum

echo ""
echo "Headless mode ended."
