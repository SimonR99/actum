# Multi-stage build for Actum robot agent
# Supports GPU via NVIDIA CUDA (x86) and Jetson (aarch64)
# Build: docker build -t actum:latest .
# Run:   docker compose up

ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04

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
    && rm -rf /var/lib/apt/lists/*

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

# Install project dependencies
# Note: kokoro-onnx is Linux-specific; camera is optional
RUN pip install -e ".[camera]"

# Pre-download models to avoid runtime delays (optional but recommended)
# Comment out if you want smaller image; models will download on first run
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('litert-community/gemma-4-E2B-it-litert-lm', \
    allow_patterns='gemma-4-E2B-it.litertlm', \
    cache_dir='/root/.cache/huggingface')" || true

# ── Runtime Stage: Minimal production image ──────────────────────────────
FROM base as runtime

WORKDIR /workspace

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Copy project source
COPY pyproject.toml README.md ./
COPY src ./src

# Install project in editable mode (already in venv, but ensure sys.path is correct)
RUN pip install -e .

# Create directories for persistent data
RUN mkdir -p /workspace/models /workspace/config /workspace/logs

# Audio device access (for container)
RUN echo "pcm.!default { type hw card 0 }" > /etc/asound.conf

# Default environment variables
ENV MODEL_PATH=/root/.cache/huggingface/hub/litert-community/gemma-4-E2B-it-litert-lm/snapshots/*/gemma-4-E2B-it.litertlm \
    PORT=8000 \
    PYTHONPATH=/workspace/src:${PYTHONPATH}

# Expose ports
EXPOSE 8000

# Default: run the monitoring server
CMD ["actum-server"]
