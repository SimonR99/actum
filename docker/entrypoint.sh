#!/usr/bin/env bash
# Container entrypoint — installs lerobot from /lerobot if mounted and not yet installed.
set -e

# The container is recreated on every launch, so the image's venv resets and
# lerobot is reinstalled each time. Some of its deps (e.g. blosc) have no
# prebuilt aarch64 wheel and compile from C source — minutes on a Pi. The image
# sets PIP_NO_CACHE_DIR=1, so without this that compile repeats every start.
# Persist a wheel cache (mounted as a named volume at /root/.cache/pip) and
# enable caching here so the slow build happens once, then is reused.
export PIP_NO_CACHE_DIR=0
PIP_CACHE_DIR="/root/.cache/pip"
PIP_INSTALL="pip install --cache-dir ${PIP_CACHE_DIR}"

if [ -d "/lerobot" ]; then
    if ! python -c "import lerobot" 2>/dev/null; then
        echo "[entrypoint] Installing lerobot from /lerobot (first build compiles native deps; cached afterwards)..."
        $PIP_INSTALL -e /lerobot
        echo "[entrypoint] lerobot installed."
    fi
    # scservo_sdk (feetech motor SDK) is not declared in lerobot's pyproject.toml
    if ! python -c "import scservo_sdk" 2>/dev/null; then
        echo "[entrypoint] Installing feetech-servo-sdk..."
        $PIP_INSTALL -q feetech-servo-sdk
        echo "[entrypoint] feetech-servo-sdk installed."
    fi
fi

exec "$@"
