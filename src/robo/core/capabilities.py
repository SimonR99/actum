"""Serializable capability registry for tools, MCP, and dashboard contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Capability:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    safety_level: str = "normal"
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CapabilityRegistry:
    def __init__(self):
        self._items: dict[str, Capability] = {}

    def register(self, capability: Capability):
        self._items[capability.name] = capability

    def list(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._items.values()]

    def get(self, name: str) -> Capability | None:
        return self._items.get(name)


def default_capabilities() -> CapabilityRegistry:
    registry = CapabilityRegistry()
    registry.register(Capability("set_plan", "Publish current goal and plan.", safety_level="safe"))
    registry.register(Capability("mark_step", "Mark a plan step active.", safety_level="safe"))
    registry.register(Capability("look", "Capture a camera frame.", safety_level="safe"))
    registry.register(Capability("speak", "Queue speech output.", safety_level="safe"))
    registry.register(Capability("navigate", "Move the robot base a short bounded distance.", safety_level="motion"))
    registry.register(Capability("rotate", "Rotate the robot base by a bounded angle.", safety_level="motion"))
    registry.register(Capability("gripper", "Open or close an end-effector.", safety_level="manipulation"))
    registry.register(Capability("wave", "Run a named gesture.", safety_level="motion"))
    registry.register(Capability("remember", "Store a memory key/value.", safety_level="safe"))
    registry.register(Capability("recall", "Read a memory key.", safety_level="safe"))
    registry.register(Capability("report_status", "Publish an operator-visible status.", safety_level="safe"))
    registry.register(Capability("done", "Mark the current task complete.", safety_level="safe"))
    return registry
