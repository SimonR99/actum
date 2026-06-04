"""On-device LiteRT inference provider (Gemma multimodal)."""

from __future__ import annotations

from typing import Any, Callable

try:
    import litert_lm
except ImportError:  # pragma: no cover - exercised only without the runtime
    litert_lm = None

from actum.inference.base import InferenceProvider

# Maximum follow-up rounds for chained look() calls in a single turn.
_MAX_FRAME_FOLLOWUPS = 4


def _resolve_compute_backend(name: str):
    """Map a config string to a litert_lm.Backend, falling back to CPU."""
    if litert_lm is None:
        return None
    requested = str(name or "gpu").strip().upper()
    backend = getattr(litert_lm.Backend, requested, None)
    if backend is None:
        print(f"[litert] compute backend {requested!r} unavailable; using CPU")
        backend = litert_lm.Backend.CPU
    return backend


class LiteRTProvider(InferenceProvider):
    """Wraps a local litert_lm Engine + Conversation.

    The Gemma model handles the multi-step tool loop internally, so a turn is a
    single ``send_message``. ``look()`` cannot inject an image mid-inference, so
    a captured frame is delivered as a follow-up message.
    """

    name = "litert"
    supports_audio = True
    supports_vision = True

    def __init__(
        self,
        model: str,
        compute: str,
        pop_pending_frame: Callable[[], str | None],
        resolve_model_path: Callable[[], str],
    ):
        self._model = model
        self._compute = compute
        self._pop_pending_frame = pop_pending_frame
        self._resolve_model_path = resolve_model_path
        self._engine: Any = None
        self._conversation: Any = None

    def start(self, system_prompt: str, tools: list[Callable[..., Any]]) -> None:
        if litert_lm is None:
            raise RuntimeError(
                "litert-lm is not installed. Install project dependencies with "
                "`pip install -e .`, or select a cloud provider in config.json."
            )
        model_path = self._resolve_model_path()
        compute = _resolve_compute_backend(self._compute)
        print(f"Loading LiteRT model from {model_path} on {self._compute.upper()}…")
        self._engine = litert_lm.Engine(
            model_path,
            backend=compute,
            vision_backend=compute,
            audio_backend=litert_lm.Backend.CPU,
        )
        self._engine.__enter__()
        self._conversation = self._engine.create_conversation(
            messages=[{"role": "system", "content": system_prompt}],
            tools=tools,
        )
        self._conversation.__enter__()

    def send(self, content: list[dict[str, Any]]) -> str:
        self._conversation.send_message({"role": "user", "content": content})
        for _ in range(_MAX_FRAME_FOLLOWUPS):
            frame = self._pop_pending_frame()
            if not frame:
                break
            follow_up = [
                {"type": "image", "blob": frame},
                {
                    "type": "text",
                    "text": "This is the camera frame you requested with look(). "
                    "Describe what you see and continue your task.",
                },
            ]
            self._conversation.send_message({"role": "user", "content": follow_up})
        return ""

    def close(self) -> None:
        if self._conversation is not None:
            self._conversation.__exit__(None, None, None)
            self._conversation = None
        if self._engine is not None:
            self._engine.__exit__(None, None, None)
            self._engine = None
