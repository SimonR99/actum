"""Selectable speech-to-text (STT) engines.

STT is a stage independent of the inference "brain", so the operator can choose
how voice is transcribed regardless of which LLM is active:

- ``whisper``: local faster-whisper, fully on-device (default).
- ``openai``: OpenAI cloud transcription.
- ``model``: no separate STT — pass raw audio to a multimodal model (Gemma).

``build_stt`` returns ``None`` for the ``model`` engine, signalling the agent to
forward the audio blob to the model instead of transcribing it first.
"""

from __future__ import annotations

import base64
import io
from abc import ABC, abstractmethod
from typing import Any

STT_ENGINES = ("whisper", "openai", "model")
DEFAULT_ENGINE = "whisper"
DEFAULT_WHISPER_MODEL = "base"
DEFAULT_OPENAI_TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"


class STTEngine(ABC):
    name = "base"

    @abstractmethod
    def transcribe(self, audio_b64: str) -> str:
        """Transcribe a base64-encoded WAV blob to text (best-effort, '' on failure)."""


class WhisperSTT(STTEngine):
    """Local transcription via faster-whisper. Model loads lazily on first use."""

    name = "whisper"

    def __init__(self, model_size: str = DEFAULT_WHISPER_MODEL, compute_type: str = "auto", device: str = "auto"):
        self._model_size = model_size or DEFAULT_WHISPER_MODEL
        self._compute_type = "default" if compute_type in ("", "auto") else compute_type
        self._device = device or "auto"
        self._model: Any = None
        self._warned_missing = False

    def _ensure_model(self) -> Any:
        if self._model is None:
            from faster_whisper import WhisperModel

            print(f"[stt] loading local Whisper model '{self._model_size}' (first run downloads it)…")
            self._model = WhisperModel(self._model_size, device=self._device, compute_type=self._compute_type)
            print("[stt] Whisper model ready.")
        return self._model

    def transcribe(self, audio_b64: str) -> str:
        try:
            model = self._ensure_model()
        except ModuleNotFoundError:
            if not self._warned_missing:
                print(
                    "[stt] faster-whisper is not installed, so voice cannot be transcribed. "
                    "Install it with:  pip install -e '.[whisper]'   "
                    "— or pick 'OpenAI' / 'Multimodal model' as the speech engine in the dashboard."
                )
                self._warned_missing = True
            return ""
        except Exception as exc:
            print(f"[stt] whisper model failed to load: {exc}")
            return ""

        try:
            buffer = io.BytesIO(base64.b64decode(audio_b64))
            buffer.name = "audio.wav"
            segments, _ = model.transcribe(buffer)
            text = " ".join(segment.text.strip() for segment in segments).strip()
            if text:
                print(f"[stt] transcript: {text!r}")
            else:
                print("[stt] whisper produced no text (audio too quiet or empty?)")
            return text
        except Exception as exc:  # best-effort; agent falls back to passthrough
            print(f"[stt] whisper transcription failed: {exc}")
            return ""


class OpenAISTT(STTEngine):
    """Cloud transcription via the OpenAI audio API."""

    name = "openai"

    def __init__(self, model: str = DEFAULT_OPENAI_TRANSCRIBE_MODEL, api_key: str = ""):
        self._model = model or DEFAULT_OPENAI_TRANSCRIBE_MODEL
        self._api_key = api_key
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key) if self._api_key else OpenAI()
        return self._client

    def transcribe(self, audio_b64: str) -> str:
        try:
            client = self._ensure_client()
            buffer = io.BytesIO(base64.b64decode(audio_b64))
            buffer.name = "audio.wav"
            result = client.audio.transcriptions.create(model=self._model, file=buffer)
            return (getattr(result, "text", "") or "").strip()
        except Exception as exc:
            print(f"[stt] openai transcription failed: {exc}")
            return ""


def build_stt(settings: Any) -> STTEngine | None:
    """Construct the selected STT engine, or None for multimodal passthrough."""
    speech = getattr(settings, "speech", {}) or {}
    engine = str(speech.get("stt_engine", DEFAULT_ENGINE)).lower().strip()

    if engine in {"model", "none", ""}:
        return None
    if engine == "whisper":
        return WhisperSTT(
            model_size=str(speech.get("whisper_model", DEFAULT_WHISPER_MODEL)),
            compute_type=str(speech.get("whisper_compute", "auto")),
            device=str(speech.get("whisper_device", "auto")),
        )
    if engine == "openai":
        models = settings.to_config(include_secrets=True)["models"]
        api_key = str(models.get("providers", {}).get("openai", {}).get("api_key", ""))
        return OpenAISTT(
            model=str(speech.get("openai_transcribe_model", DEFAULT_OPENAI_TRANSCRIBE_MODEL)),
            api_key=api_key,
        )
    return None
