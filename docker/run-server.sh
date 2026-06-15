#!/usr/bin/env bash
# Run Actum with monitoring dashboard (web server)
# Dashboard available at: http://localhost:8000
# Usage: ./docker/run-server.sh           — fast restart (no rebuild, src mounted live)
#        ./docker/run-server.sh --build   — force full image rebuild (after dep changes)
# Stop with: Ctrl+C or docker compose down

set -e

BUILD_FLAG=""
for arg in "$@"; do
    case "$arg" in
        --build|-b) BUILD_FLAG="--build" ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

HOST_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if ($i=="src") print $(i+1)}' | head -1)
HOST_IP=${HOST_IP:-localhost}

echo "Starting Actum with monitoring dashboard..."
echo ""
echo "Dashboard (local):   http://localhost:8000"
echo "Dashboard (network): http://${HOST_IP}:8000"
echo "Stop with: Ctrl+C"
echo ""

cd "$PROJECT_ROOT"

# Build a runtime compose override with only the hardware that actually exists
# on this host.  This makes the same image work on any robot without editing
# docker-compose.yml.
OVERRIDE_FILE=$(mktemp /tmp/actum-override-XXXXXX.yml)
trap 'rm -f "$OVERRIDE_FILE"' EXIT

DEVICES_SECTION=""
VOLUMES_SECTION=""
ENV_SECTION=""
DEPLOY_SECTION=""
CGROUP_RULES_SECTION=""
NETWORK_SECTION=""

# Network mode — use host networking only for Unitree robots (CycloneDDS multicast).
# Host networking breaks /sys/devices/system/cpu/possible on some kernels
# (e.g. Raspberry Pi 6.8.x), which crashes onnxruntime via cpuinfo.
ROBOT_BACKEND=$(python3 -c "
import json, sys
try:
    c = json.load(open('config.json'))
    print(c.get('robot', {}).get('backend', ''))
except Exception:
    pass
" 2>/dev/null)
if [ "$ROBOT_BACKEND" = "unitree_g1" ]; then
    NETWORK_SECTION="    network_mode: host"
    echo "Network: host (Unitree DDS requires multicast)"
else
    echo "Network: bridge (port 8000 mapped)"
fi

# GPU (NVIDIA)
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    DEPLOY_SECTION='    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]'
    echo "GPU:    NVIDIA (nvidia-smi found)"
else
    echo "GPU:    not found — running CPU-only"
fi

# Audio — bind-mount /dev/snd as a volume (not a device) to avoid the kernel
# restriction (errno 524) that prevents Docker from creating ALSA device nodes.
# Grant cgroup access via device_cgroup_rules using the snd major number.
if [ -e /dev/snd ]; then
    SND_MAJOR=$(stat -c '%t' /dev/snd/pcmC0D0p 2>/dev/null \
                || stat -c '%t' "$(ls /dev/snd/pcm* 2>/dev/null | head -1)" 2>/dev/null \
                || echo "116")
    SND_MAJOR=$((16#${SND_MAJOR:-116}))
    VOLUMES_SECTION="${VOLUMES_SECTION}      - /dev/snd:/dev/snd\n"
    CGROUP_RULES_SECTION="${CGROUP_RULES_SECTION}      - 'c ${SND_MAJOR}:* rmw'\n"
    echo "Audio:  /dev/snd (major $SND_MAJOR)"
else
    echo "Audio:  not found — running without audio"
fi

# Device permissions — rootless Docker uses user namespace remapping, so devices
# owned by root:dialout or root:video appear as nobody:nogroup inside the container.
# A one-time udev rule sets MODE="0666" (world-accessible) so no privileged mode
# is needed.  The rule persists across reboots.
_UDEV_RULE=/etc/udev/rules.d/99-actum-devices.rules
_NEED_UDEV=0
{ ls /dev/ttyACM* &>/dev/null || ls /dev/ttyUSB* &>/dev/null; } 2>/dev/null && _NEED_UDEV=1
ls /dev/video* &>/dev/null 2>&1 && _NEED_UDEV=1
if [ "$_NEED_UDEV" = "1" ] && [ ! -f "$_UDEV_RULE" ]; then
    echo "Creating udev rules for device access (one-time sudo)..."
    {
        echo 'SUBSYSTEM=="tty", KERNEL=="ttyACM*|ttyUSB*", MODE="0666"'
        echo 'SUBSYSTEM=="video4linux", MODE="0666"'
    } | sudo tee "$_UDEV_RULE" >/dev/null \
    && sudo udevadm control --reload-rules \
    && sudo udevadm trigger \
    && echo "Udev rules created: $_UDEV_RULE" \
    || echo "WARNING: udev rule creation failed — run manually:
  sudo tee $_UDEV_RULE <<'EOF'
SUBSYSTEM==\"tty\", KERNEL==\"ttyACM*|ttyUSB*\", MODE=\"0666\"
SUBSYSTEM==\"video4linux\", MODE=\"0666\"
EOF
  sudo udevadm control --reload-rules && sudo udevadm trigger"
fi

# Serial ports — Feetech motor bus (LeKiwi arm + wheels on /dev/ttyACM0)
if { ls /dev/ttyACM* &>/dev/null || ls /dev/ttyUSB* &>/dev/null; } 2>/dev/null; then
    for dev in /dev/ttyACM* /dev/ttyUSB*; do
        [ -e "$dev" ] || continue
        DEVICES_SECTION="${DEVICES_SECTION}      - \"${dev}:${dev}\"\n"
        echo "Serial: $dev"
    done
fi

# Python source — mounted live so code changes don't require an image rebuild.
# pip install -e . in the image already points to /workspace/src, so the
# volume mount makes the container read directly from the host tree.
VOLUMES_SECTION="${VOLUMES_SECTION}      - ${PROJECT_ROOT}/src:/workspace/src\n"

# lerobot source — bind-mounted so the entrypoint can pip-install it on startup
LEROBOT_SRC="/home/simon/lerobot"
if [ -d "$LEROBOT_SRC" ]; then
    VOLUMES_SECTION="${VOLUMES_SECTION}      - ${LEROBOT_SRC}:/lerobot:ro\n"
    echo "LeRobot: $LEROBOT_SRC → /lerobot"
fi

# Camera — bind-mount all /dev/videoN nodes as volumes + grant cgroup access.
# Uses the same pattern as audio: volumes avoids kernel device-node creation
# issues, and device_cgroup_rules grants access for the whole major number so
# every video node (capture, metadata, ISP) is accessible.
if ls /dev/video* &>/dev/null; then
    VIDEO_FIRST=$(ls /dev/video* | head -1)
    VIDEO_MAJOR_HEX=$(stat -c '%t' "$VIDEO_FIRST" 2>/dev/null || echo "51")
    VIDEO_MAJOR=$((16#${VIDEO_MAJOR_HEX}))
    for dev in /dev/video*; do
        VOLUMES_SECTION="${VOLUMES_SECTION}      - ${dev}:${dev}\n"
    done
    CGROUP_RULES_SECTION="${CGROUP_RULES_SECTION}      - 'c ${VIDEO_MAJOR}:* rmw'\n"
    VIDEO_LIST=$(ls /dev/video* | tr '\n' ' ' | sed 's/ $//')
    echo "Camera: $VIDEO_LIST (major $VIDEO_MAJOR)"
else
    echo "Camera: not found — running without camera"
fi

# PulseAudio / PipeWire socket (desktop/embedded systems with a sound daemon)
PULSE_SOCKET=$(ls /run/user/*/pulse/native 2>/dev/null | head -1)
if [ -n "$PULSE_SOCKET" ]; then
    VOLUMES_SECTION="${VOLUMES_SECTION}      - ${PULSE_SOCKET}:/tmp/pulse-socket\n"
    ENV_SECTION="${ENV_SECTION}      - PULSE_SERVER=unix:/tmp/pulse-socket\n"
    echo "Pulse:  $PULSE_SOCKET"
fi

echo ""

# Write the override (omit empty sections so compose doesn't complain)
{
    echo "services:"
    echo "  actum:"
    if [ -n "$NETWORK_SECTION" ]; then
        printf "%s\n" "$NETWORK_SECTION"
    fi
    if [ -n "$DEPLOY_SECTION" ]; then
        printf "%s\n" "$DEPLOY_SECTION"
    fi
    if [ -n "$CGROUP_RULES_SECTION" ]; then
        echo "    device_cgroup_rules:"
        printf "%b" "$CGROUP_RULES_SECTION"
    fi
    if [ -n "$DEVICES_SECTION" ]; then
        echo "    devices:"
        printf "%b" "$DEVICES_SECTION"
    fi
    if [ -n "$VOLUMES_SECTION" ]; then
        echo "    volumes:"
        printf "%b" "$VOLUMES_SECTION"
    fi
    if [ -n "$ENV_SECTION" ]; then
        echo "    environment:"
        printf "%b" "$ENV_SECTION"
    fi
} > "$OVERRIDE_FILE"

# Run — src is mounted live so Python changes never require a rebuild.
# Pass --build (or -b) to force a full image rebuild after dep/Dockerfile changes.
docker compose -f docker-compose.yml -f "$OVERRIDE_FILE" up $BUILD_FLAG actum

echo ""
echo "Dashboard stopped."
