"""Structured task intent and behavior-tree-like plan state."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from robo.core.schema import PlanStep, now


_LIST_PREFIX = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*")


def parse_plan_lines(plan_text: str) -> list[str]:
    lines: list[str] = []
    for raw in plan_text.splitlines():
        label = _LIST_PREFIX.sub("", raw).strip()
        if label:
            lines.append(label)
    return lines


@dataclass
class IntentState:
    goal: str = ""
    status: str = "idle"
    active_step_id: str | None = None
    summary: str = ""
    risk: str = "normal"
    steps: list[PlanStep] = field(default_factory=list)
    updated_at: float = field(default_factory=now)

    def set_plan(self, goal: str, plan_text: str):
        labels = parse_plan_lines(plan_text)
        self.goal = goal.strip()
        self.status = "planning" if labels else "active"
        self.summary = ""
        self.steps = [
            PlanStep(id=f"step-{idx + 1}", label=label)
            for idx, label in enumerate(labels)
        ]
        self.active_step_id = self.steps[0].id if self.steps else None
        if self.steps:
            self.steps[0].status = "active"
        self.updated_at = now()

    def mark_active(self, label_or_id: str):
        target = label_or_id.strip().lower()
        if not target:
            return
        for step in self.steps:
            if step.id.lower() == target or step.label.lower() == target:
                if step.status == "pending":
                    step.status = "active"
                step.updated_at = now()
                self.active_step_id = step.id
                self.status = "active"
                self.updated_at = now()
                return

    def complete_active(self, detail: str = ""):
        if not self.steps or self.active_step_id is None:
            return
        active_index = None
        for idx, step in enumerate(self.steps):
            if step.id == self.active_step_id:
                active_index = idx
                step.status = "done"
                step.detail = detail
                step.updated_at = now()
                break
        if active_index is None:
            return
        for step in self.steps[active_index + 1 :]:
            if step.status == "pending":
                step.status = "active"
                step.updated_at = now()
                self.active_step_id = step.id
                self.status = "active"
                self.updated_at = now()
                return
        self.active_step_id = None
        self.status = "done"
        self.updated_at = now()

    def finish(self, summary: str):
        self.summary = summary.strip()
        self.status = "done"
        self.active_step_id = None
        for step in self.steps:
            if step.status in {"pending", "active"}:
                step.status = "done"
                step.updated_at = now()
        self.updated_at = now()

    def fail(self, message: str):
        self.summary = message.strip()
        self.status = "blocked"
        self.updated_at = now()

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["steps"] = [step.to_dict() for step in self.steps]
        return out
