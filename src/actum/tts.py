"""Platform-aware Kokoro TTS.

Backend selection:
  - macOS Apple Silicon → mlx-audio (GPU via MLX)
  - Linux / Jetson      → kokoro-onnx with best available ONNX Runtime provider
                          (CUDAExecutionProvider if onnxruntime-gpu is installed,
                           CPUExecutionProvider otherwise)

On Jetson, onnxruntime-gpu is not on PyPI — install the NVIDIA-provided wheel:
    pip install https://nvidia.box.com/shared/static/<wheel>.whl  # see Jetson AI Lab
Or use the CPU backend; Kokoro-82M is fast enough at ~0.3s/sentence on Orin.
"""

import os
import platform
import sys
from pathlib import Path

import numpy as np


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _is_jetson() -> bool:
    return platform.machine() == "aarch64" and Path("/etc/nv_tegra_release").exists()


def _best_onnx_providers() -> list[str]:
    """Return available ONNX execution providers, preferring GPU."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        return providers or ["CPUExecutionProvider"]
    except ImportError:
        return ["CPUExecutionProvider"]


class TTSBackend:
    sample_rate: int = 24000

    def generate(self, text: str, voice: str = "af_heart", speed: float = 1.1) -> np.ndarray:
        raise NotImplementedError


class MLXBackend(TTSBackend):
    """mlx-audio backend — Apple Silicon only."""

    def __init__(self):
        from mlx_audio.tts.generate import load_model
        self._model = load_model("mlx-community/Kokoro-82M-bf16")
        self.sample_rate = self._model.sample_rate
        list(self._model.generate(text="Hello", voice="af_heart", speed=1.0))  # warmup

    def generate(self, text: str, voice: str = "af_heart", speed: float = 1.1) -> np.ndarray:
        results = list(self._model.generate(text=text, voice=voice, speed=speed))
        return np.concatenate([np.array(r.audio) for r in results])


class ONNXBackend(TTSBackend):
    """kokoro-onnx backend — Linux / Jetson (CPU or CUDA)."""

    def __init__(self):
        import kokoro_onnx
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download("fastrtc/kokoro-onnx", "kokoro-v1.0.onnx")
        voices_path = hf_hub_download("fastrtc/kokoro-onnx", "voices-v1.0.bin")
        providers = _best_onnx_providers()
        # Backward compatibility: older kokoro-onnx releases don't accept
        # the providers keyword and default to their internal provider order.
        try:
            self._model = kokoro_onnx.Kokoro(model_path, voices_path, providers=providers)
            self._providers = providers
        except TypeError as exc:
            if "providers" not in str(exc):
                raise
            self._model = kokoro_onnx.Kokoro(model_path, voices_path)
            self._providers = ["default"]
        self.sample_rate = 24000

    def generate(self, text: str, voice: str = "af_heart", speed: float = 1.1) -> np.ndarray:
        pcm, _sr = self._model.create(text, voice=voice, speed=speed)
        return pcm


def load() -> TTSBackend:
    """Load the best available TTS backend for this platform."""
    if _is_apple_silicon() and not os.environ.get("KOKORO_ONNX"):
        try:
            backend = MLXBackend()
            print(f"TTS: mlx-audio (Apple GPU, {backend.sample_rate} Hz)")
            return backend
        except ImportError:
            print("TTS: mlx-audio not installed, falling back to kokoro-onnx")

    backend = ONNXBackend()
    label = "+".join(p.replace("ExecutionProvider", "") for p in backend._providers)
    tag = "Jetson" if _is_jetson() else "Linux"
    print(f"TTS: kokoro-onnx [{label}] ({tag}, {backend.sample_rate} Hz)")
    return backend
