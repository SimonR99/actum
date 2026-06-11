"""Inference provider abstraction.

The robot's *body* is abstracted behind ``RobotBackend``; this abstracts the
robot's *brain*. A provider owns the conversation and the multi-step tool-calling
loop, so the agent can switch between a local on-device model and a cloud model
without changing the rest of the runtime.

Content blocks use the agent's internal format::

    {"type": "text",  "text": "..."}
    {"type": "image", "blob": "<base64 jpeg>"}
    {"type": "audio", "blob": "<base64 wav>"}
"""

from __future__ import annotations

import inspect
import types
import typing
from abc import ABC, abstractmethod
from typing import Any, Callable


class InferenceProvider(ABC):
    """Owns a conversation and runs the agent's tool-calling loop."""

    name = "base"
    supports_audio = False
    supports_vision = True

    @abstractmethod
    def start(self, system_prompt: str, tools: list[Callable[..., Any]]) -> None:
        """Load the model and register the system prompt + tools."""

    @abstractmethod
    def send(self, content: list[dict[str, Any]]) -> str:
        """Run one user turn through the full tool-calling loop.

        Blocking. Tools (including ``look()``) are executed internally. Returns
        the final assistant text, which may be empty when the turn ended on a
        tool call such as ``done()``.
        """

    def reset(self) -> None:
        """Forget the conversation history, keeping the system prompt and tools."""
        return None

    def close(self) -> None:  # pragma: no cover - trivial default
        return None


def build_tool_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Build an OpenAI-style function schema from a Python callable.

    Uses the signature for parameter names/types and the first docstring
    paragraph as the description. Good enough for the robot tools, which take
    flat string/number/bool arguments.
    """
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    description = (doc.split("\n\n", 1)[0].strip() or func.__name__)[:1024]

    properties: dict[str, Any] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if pname == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        properties[pname] = {"type": _json_type(param.annotation)}
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _json_type(annotation: Any) -> str:
    base = annotation
    # Unwrap Optional/Union (e.g. ``float | None``) to its first non-None member.
    if isinstance(annotation, types.UnionType) or typing.get_origin(annotation) is typing.Union:
        members = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        base = members[0] if members else str
    if base is bool:
        return "boolean"
    if base is int:
        return "integer"
    if base is float:
        return "number"
    return "string"
