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
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

# Split on sentence-ending punctuation (keeping the punctuation), so TTS can
# generate and play one sentence at a time instead of the whole reply at once.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…。！？])\s+")


def strip_unspeakable(text: str) -> str:
    """Drop emoji and other non-readable characters from TTS input.

    Keeps letters (including accented French ones), numbers, punctuation and
    whitespace; removes Unicode *symbol* code points (category ``S*`` — emoji,
    pictographs, arrows, …) and *control/other* code points (``C*``) that the
    speech model would otherwise mispronounce or vocalise as noise.
    """
    kept: list[str] = []
    for ch in text:
        if ch in "\n\t":
            kept.append(" ")
            continue
        category = unicodedata.category(ch)
        if category and category[0] in ("L", "N", "P", "Z", "M"):
            kept.append(ch)
    return re.sub(r"\s+", " ", "".join(kept)).strip()


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-sized chunks for incremental TTS playback."""
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]
    if parts:
        return parts
    stripped = text.strip()
    return [stripped] if stripped else []


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _is_jetson() -> bool:
    return platform.machine() == "aarch64" and Path("/etc/nv_tegra_release").exists()


def _best_onnx_providers() -> list[str]:
    """Return available ONNX execution providers, preferring GPU."""
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
        providers = [
            p
            for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if p in available
        ]
        return providers or ["CPUExecutionProvider"]
    except ImportError:
        return ["CPUExecutionProvider"]


class TTSBackend:
    sample_rate: int = 24000

    def generate(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.1,
        lang: str = "en-us",
    ) -> np.ndarray:
        raise NotImplementedError


class MLXBackend(TTSBackend):
    """mlx-audio backend — Apple Silicon only."""

    def __init__(self):
        from mlx_audio.tts.generate import load_model

        self._model = load_model("mlx-community/Kokoro-82M-bf16")
        self.sample_rate = self._model.sample_rate
        list(self._model.generate(text="Hello", voice="af_heart", speed=1.0))  # warmup

    def generate(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.1,
        lang: str = "en-us",
    ) -> np.ndarray:
        # mlx-audio's Kokoro pipeline infers the phonemizer language from the
        # voice prefix (e.g. "ff_" → French), so lang is accepted for interface
        # parity but not forwarded.
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
            self._model = kokoro_onnx.Kokoro(
                model_path, voices_path, providers=providers
            )
            self._providers = providers
        except TypeError as exc:
            if "providers" not in str(exc):
                raise
            self._model = kokoro_onnx.Kokoro(model_path, voices_path)
            self._providers = ["default"]
        self.sample_rate = 24000

    def generate(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.1,
        lang: str = "en-us",
    ) -> np.ndarray:
        # kokoro-onnx defaults lang to "en-us"; without this French text is
        # phonemized with English rules and comes out with an English accent.
        pcm, _sr = self._model.create(text, voice=voice, speed=speed, lang=lang)
        return pcm


DEFAULT_OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_OPENAI_TTS_VOICE = "alloy"


class OpenAITTSBackend(TTSBackend):
    """Cloud synthesis via the OpenAI audio.speech API.

    Streams 24 kHz 16-bit PCM and is language-agnostic — the model speaks the
    input text in whatever language it is written, so the same configured voice
    serves both English and French. Offloads synthesis from a CPU-bound Pi.
    """

    sample_rate = 24000

    def __init__(
        self,
        api_key: str = "",
        model: str = DEFAULT_OPENAI_TTS_MODEL,
        voice: str = DEFAULT_OPENAI_TTS_VOICE,
    ):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self._model = model or DEFAULT_OPENAI_TTS_MODEL
        self._voice = voice or DEFAULT_OPENAI_TTS_VOICE
        # The first request pays a cold TLS/connection + model warm-up cost
        # (~5 s on a Pi); do it now at startup so the first real utterance is
        # fast (~0.8 s to first audio) instead of stalling the first reply.
        try:
            self.generate("Ready.")
        except Exception as exc:
            print(f"[tts] OpenAI warmup skipped: {exc}")

    @classmethod
    def from_settings(cls, settings) -> "OpenAITTSBackend":
        from actum.inference import resolve_api_key

        speech = getattr(settings, "speech", {}) or {}
        providers = settings.to_config(include_secrets=True)["models"].get(
            "providers", {}
        )
        api_key = resolve_api_key(providers.get("openai", {}))
        return cls(
            api_key=api_key,
            model=str(speech.get("openai_tts_model", DEFAULT_OPENAI_TTS_MODEL)),
            voice=str(speech.get("openai_tts_voice", DEFAULT_OPENAI_TTS_VOICE)),
        )

    def generate(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.1,
        lang: str = "en-us",
    ) -> np.ndarray:
        # voice/lang are ignored: the OpenAI voice is fixed by config and the
        # model auto-detects the input language. Raw PCM is already 24 kHz mono
        # 16-bit LE, matching sample_rate.
        buffer = bytearray()
        with self._client.audio.speech.with_streaming_response.create(
            model=self._model,
            voice=self._voice,
            input=text,
            response_format="pcm",
        ) as response:
            for chunk in response.iter_bytes():
                buffer.extend(chunk)
        if not buffer:
            return np.zeros(0, dtype=np.float32)
        return np.frombuffer(bytes(buffer), dtype=np.int16).astype(np.float32) / 32768.0


def load(settings=None) -> TTSBackend:
    """Load the configured TTS backend.

    When ``speech.tts_engine == "openai"`` synthesis runs in the cloud; otherwise
    the best local Kokoro backend for this platform is used. The local backend is
    also used as a fallback if the OpenAI backend can't be constructed.
    """
    engine = ""
    if settings is not None:
        speech = getattr(settings, "speech", {}) or {}
        engine = str(speech.get("tts_engine", "")).lower().strip()

    if engine == "openai":
        try:
            backend = OpenAITTSBackend.from_settings(settings)
            print(f"TTS: OpenAI cloud ({backend._model}, voice={backend._voice})")
            return backend
        except Exception as exc:
            print(f"TTS: OpenAI backend unavailable ({exc}); falling back to local.")

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
