"""Cloud inference provider backed by the OpenAI chat completions API.

Unlike the local LiteRT engine, the OpenAI API does not run the tool loop for
us, so this provider implements it: call the model, execute any requested tools,
feed the results back, and repeat until the model stops calling tools or invokes
``done()``. ``look()`` frames are injected as a follow-up image message inside
the same loop.
"""

from __future__ import annotations

import base64
import inspect
import io
import json
import types
import typing
from typing import Any, Callable, Union

from actum.inference.base import InferenceProvider, build_tool_schema

DEFAULT_MODEL = "gpt-5.4-nano-2026-03-17"
DEFAULT_TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"
_MAX_TOOL_ROUNDS = 12
_MAX_HISTORY = 40  # messages kept after the system prompt


class OpenAIProvider(InferenceProvider):
    name = "openai"
    supports_audio = True
    supports_vision = True

    def __init__(
        self,
        model: str,
        pop_pending_frame: Callable[[], str | None],
        api_key: str = "",
        transcribe_model: str = DEFAULT_TRANSCRIBE_MODEL,
    ):
        self._model = model or DEFAULT_MODEL
        self._transcribe_model = transcribe_model or DEFAULT_TRANSCRIBE_MODEL
        self._pop_pending_frame = pop_pending_frame
        self._api_key = api_key
        self._client: Any = None
        self._tool_schemas: list[dict[str, Any]] = []
        self._tool_map: dict[str, Callable[..., Any]] = {}
        self._messages: list[dict[str, Any]] = []

    def start(self, system_prompt: str, tools: list[Callable[..., Any]]) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without the SDK
            raise RuntimeError(
                "openai is not installed. Install it with `pip install -e '.[openai]'` "
                "to use the OpenAI provider."
            ) from exc
        self._client = OpenAI(api_key=self._api_key) if self._api_key else OpenAI()
        self._tool_schemas = [build_tool_schema(tool) for tool in tools]
        self._tool_map = {tool.__name__: tool for tool in tools}
        self._messages = [{"role": "system", "content": system_prompt}]
        print(f"OpenAI provider ready ({self._model}).")

    def send(self, content: list[dict[str, Any]]) -> str:
        self._messages.append(
            {"role": "user", "content": self._to_openai_content(content)}
        )
        final_text = ""
        for _ in range(_MAX_TOOL_ROUNDS):
            response = self._client.chat.completions.create(
                model=self._model,
                messages=self._messages,
                tools=self._tool_schemas or None,
                tool_choice="auto" if self._tool_schemas else None,
            )
            message = response.choices[0].message
            self._messages.append(message.model_dump(exclude_none=True))

            if not message.tool_calls:
                final_text = message.content or ""
                break

            done_called = False
            for call in message.tool_calls:
                result = self._run_tool(call.function.name, call.function.arguments)
                self._messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
                if call.function.name == "done":
                    done_called = True

            frame = self._pop_pending_frame()
            if frame:
                self._messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Camera frame from look():"},
                            _image_block(frame),
                        ],
                    }
                )
            if done_called:
                break

        self._trim_history()
        return final_text

    def _run_tool(self, name: str, raw_arguments: str) -> str:
        tool = self._tool_map.get(name)
        if tool is None:
            return f"Unknown tool: {name}"
        try:
            arguments = json.loads(raw_arguments or "{}")
            if not isinstance(arguments, dict):
                arguments = {}
            arguments = _coerce_tool_args(tool, arguments)
            return str(tool(**arguments))
        except Exception as exc:  # surface tool errors back to the model
            return f"Tool {name} failed: {exc}"

    def _to_openai_content(self, content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for item in content:
            kind = item.get("type")
            if kind == "text":
                blocks.append({"type": "text", "text": item.get("text", "")})
            elif kind == "image" and item.get("blob"):
                blocks.append(_image_block(item["blob"]))
            elif kind == "audio" and item.get("blob"):
                transcript = self._transcribe(item["blob"])
                blocks.append(
                    {
                        "type": "text",
                        "text": (
                            f"(voice transcript) {transcript}"
                            if transcript
                            else "[voice message could not be transcribed]"
                        ),
                    }
                )
        return blocks or [{"type": "text", "text": ""}]

    def _transcribe(self, blob: str) -> str:
        """Transcribe a base64 WAV blob to text via the OpenAI audio API."""
        try:
            buffer = io.BytesIO(base64.b64decode(blob))
            buffer.name = "audio.wav"
            result = self._client.audio.transcriptions.create(
                model=self._transcribe_model,
                file=buffer,
            )
            return (getattr(result, "text", "") or "").strip()
        except Exception as exc:  # transcription is best-effort
            print(f"[openai] transcription failed: {exc}")
            return ""

    def reset(self) -> None:
        """Drop all turns, keeping only the system prompt."""
        if self._messages:
            self._messages = self._messages[:1]

    def _trim_history(self) -> None:
        if len(self._messages) <= _MAX_HISTORY + 1:
            return
        tail = self._messages[-_MAX_HISTORY:]
        # Tool replies must always follow the assistant message that requested
        # them, or the API rejects the request. Drop leading orphans.
        while tail and tail[0].get("role") == "tool":
            tail.pop(0)
        self._messages = [self._messages[0], *tail]


def _image_block(blob: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{blob}"}}


def _coerce_tool_args(tool: Callable, arguments: dict) -> dict:
    """Coerce LLM-supplied argument values to their annotated Python types.

    LLMs occasionally send numbers as JSON strings (e.g. "0.5" instead of 0.5).
    Silently cast float/int annotations so tools don't crash on type mismatches.
    """
    try:
        sig = inspect.signature(tool)
    except (ValueError, TypeError):
        return arguments

    out = {}
    for key, val in arguments.items():
        param = sig.parameters.get(key)
        if param is None or param.annotation is inspect.Parameter.empty:
            out[key] = val
            continue
        ann = param.annotation
        # Unwrap Optional / Union → first non-None member
        origin = typing.get_origin(ann)
        if origin is Union or isinstance(ann, types.UnionType):
            members = [a for a in typing.get_args(ann) if a is not type(None)]
            ann = members[0] if members else ann
        try:
            if ann is float and not isinstance(val, float):
                val = float(val)
            elif ann is int and not isinstance(val, int):
                val = int(float(val))  # handle "2" and "2.0" both
        except (ValueError, TypeError):
            pass
        out[key] = val
    return out
