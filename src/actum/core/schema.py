"""Shared data structures for robot state, actions, and operator telemetry."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


def now() -> float:
    return time.time()


@dataclass
class Event:
    type: str
    source: str
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActionResult:
    action: str
    ok: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=now)
    ended_at: float = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["duration_s"] = max(0.0, self.ended_at - self.started_at)
        return out


@dataclass
class RobotState:
    backend: str
    connected: bool = False
    mode: str = "idle"
    battery_percent: float | None = None
    pose: dict[str, float] = field(default_factory=dict)
    joints: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlanStep:
    id: str
    label: str
    status: str = "pending"
    detail: str = ""
    tool: str | None = None
    updated_at: float = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
