#!/usr/bin/env bash
# Open interactive shell in vlm-g1 container
# Useful for debugging, running commands, or manual testing
# Usage: ./docker/shell.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Opening interactive shell in vlm-g1 container..."
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
    --name vlm-g1-shell \
    "${ENV_ARGS[@]}" \
    -v "$PROJECT_ROOT/config.json:/workspace/config.json" \
    -v vlm-g1-huggingface:/root/.cache/huggingface \
    -v "$PROJECT_ROOT/logs:/workspace/logs" \
    "${DEVICE_ARGS[@]}" \
    vlm-g1:latest \
    /bin/bash

echo ""
echo "Shell session ended."
