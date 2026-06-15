# Multi-stage build for Actum robot agent
# Supports GPU via NVIDIA CUDA (x86) and Jetson (aarch64)
# Build: docker build -t actum:latest .
# Run:   docker compose up

ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04

# ── Frontend Stage: React dashboard ───────────────────────────────────────
FROM node:22-bookworm-slim as frontend-builder

WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend ./
RUN npm run build

# ── Base Stage: NVIDIA CUDA + Python 3.10 ────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} as base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install Python 3.10 + system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-venv \
    # Audio capture + playback
    portaudio19-dev \
    libportaudio2 \
    alsa-utils \
    libasound2-plugins \
    # OpenCV dependencies (headless mode with codec support)
    libopencv-dev \
    python3-opencv \
    # Build essentials
    build-essential \
    cmake \
    git \
    wget \
    curl \
    # Other dependencies
    libffi-dev \
    libssl-dev \
    ffmpeg \
    libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

# Install CycloneDDS C library (required by unitree_sdk2py → cyclonedds Python binding)
RUN git clone --depth 1 --branch 0.10.2 https://github.com/eclipse-cyclonedds/cyclonedds /tmp/cyclonedds \
    && cmake -B /tmp/cyclonedds/build /tmp/cyclonedds -DCMAKE_INSTALL_PREFIX=/opt/cyclonedds \
    && cmake --build /tmp/cyclonedds/build --target install -j$(nproc) \
    && rm -rf /tmp/cyclonedds
ENV CYCLONEDDS_HOME=/opt/cyclonedds

# Create virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV=/opt/venv

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# ── Builder Stage: Install Python dependencies ────────────────────────────
FROM base as builder

WORKDIR /build

# Copy dependency files
COPY pyproject.toml README.md ./
COPY src ./src

# Install project dependencies.
# lerobot (LeKiwi motor bus) is installed at container startup from /lerobot
# if that volume is mounted — see docker/entrypoint.sh.
RUN pip install -e ".[camera,openai,whisper,mcp,unitree]"

# Ensure the cache directory exists even if no models are pre-downloaded
RUN mkdir -p /root/.cache/huggingface

# Pre-download models to avoid runtime delays (optional but recommended).
# Comment out if you want a smaller image; models will download on first run.
# RUN python3 -c "from huggingface_hub import snapshot_download; \
#     snapshot_download('litert-community/gemma-4-E2B-it-litert-lm', \
#     allow_patterns='gemma-4-E2B-it.litertlm', \
#     cache_dir='/root/.cache/huggingface')" || true

# Pre-download the default local Whisper model so first voice input is instant.
# RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('base')" || true

# ── Runtime Stage: Minimal production image ──────────────────────────────
FROM base as runtime

WORKDIR /workspace

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Copy project source
COPY pyproject.toml README.md ./
COPY src ./src
COPY --from=frontend-builder /frontend/dist ./frontend/dist

# Install project in editable mode (already in venv, but ensure sys.path is correct)
RUN pip install -e .

# Create directories for persistent data
RUN mkdir -p /workspace/models /workspace/config /workspace/logs

# Audio device access (for container)
RUN echo "pcm.!default { type hw card 0 }" > /etc/asound.conf

# Default environment variables
ENV MODEL_PATH=/root/.cache/huggingface/models--litert-community--gemma-4-E2B-it-litert-lm/snapshots/*/gemma-4-E2B-it.litertlm \
    PORT=8000 \
    PYTHONPATH=/workspace/src:${PYTHONPATH}

# Expose ports
EXPOSE 8000

# Entrypoint installs lerobot from /lerobot if mounted (LeKiwi motor bus)
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["actum-server"]
