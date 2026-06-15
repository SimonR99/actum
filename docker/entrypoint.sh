#!/usr/bin/env bash
# Container entrypoint — installs lerobot from /lerobot if mounted and not yet installed.
set -e

if [ -d "/lerobot" ]; then
    if ! python -c "import lerobot" 2>/dev/null; then
        echo "[entrypoint] Installing lerobot from /lerobot..."
        pip install -q -e /lerobot
        echo "[entrypoint] lerobot installed."
    fi
    # scservo_sdk (feetech motor SDK) is not declared in lerobot's pyproject.toml
    if ! python -c "import scservo_sdk" 2>/dev/null; then
        echo "[entrypoint] Installing feetech-servo-sdk..."
        pip install -q feetech-servo-sdk
        echo "[entrypoint] feetech-servo-sdk installed."
    fi
fi

exec "$@"
