"""Autonomy state for behavior loops, schedules, maps, and body perception."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from actum.core.schema import RobotState, now


@dataclass
class BehaviorNode:
    id: str
    label: str
    kind: str = "action"
    status: str = "waiting"
    detail: str = ""
    children: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BehaviorTreeState:
    """Behavior-tree-like public state for the operator dashboard.

    This is not a full BT executor yet. It is the stable telemetry contract the
    agent can update while a richer executor grows underneath it.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", True))
        self.tick_seconds = float(self.config.get("tick_seconds", 15.0))
        self.idle_review_seconds = float(self.config.get("idle_review_seconds", 45.0))
        self.idle_review = bool(self.config.get("idle_review", True))
        self.force_idle_reviews = bool(self.config.get("force_idle_reviews", False))
        self.idle_importance = float(self.config.get("idle_importance", 0.2))
        self.tick_count = 0
        self.last_tick_at = 0.0
        self.last_idle_review_at = 0.0
        self.goal = ""
        self.active_node_id = "wait"
        self.nodes: list[BehaviorNode] = [
            BehaviorNode(
                "wait",
                "Wait for vision, chat, language, or schedule trigger",
                "wait",
                "active",
            )
        ]

    def set_tree(self, goal: str, nodes: list[dict[str, Any]]):
        self.goal = str(goal).strip()
        parsed: list[BehaviorNode] = []
        for idx, raw in enumerate(nodes):
            label = str(
                raw.get("label") or raw.get("name") or raw.get("id") or ""
            ).strip()
            if not label:
                continue
            node_id = str(raw.get("id") or _slug(label) or f"node-{idx + 1}")
            parsed.append(
                BehaviorNode(
                    id=node_id,
                    label=label,
                    kind=str(raw.get("kind") or raw.get("type") or "action"),
                    status=str(
                        raw.get("status") or ("active" if not parsed else "waiting")
                    ),
                    detail=str(raw.get("detail") or ""),
                    children=(
                        [
                            str(item)
                            for item in raw.get("children", [])
                            if str(item).strip()
                        ]
                        if isinstance(raw.get("children"), list)
                        else []
                    ),
                )
            )
        self.nodes = parsed or [
            BehaviorNode("wait", "Wait for next trigger", "wait", "active")
        ]
        self.active_node_id = self.nodes[0].id
        self.last_tick_at = now()

    def set_waiting(self, detail: str = ""):
        self.goal = self.goal or "Maintain situational awareness"
        self.nodes = [
            BehaviorNode("wait", "Wait for next trigger", "wait", "active", detail)
        ]
        self.active_node_id = "wait"
        self.last_tick_at = now()

    def mark_node(self, node_id: str, status: str = "active", detail: str = "") -> bool:
        target = str(node_id).strip().lower()
        if not target:
            return False
        found = False
        for node in self.nodes:
            if node.id.lower() == target or node.label.lower() == target:
                node.status = _status(status)
                node.detail = str(detail)
                node.updated_at = now()
                self.active_node_id = node.id
                found = True
            elif status == "active" and node.status == "active":
                node.status = "waiting"
                node.updated_at = now()
        return found

    def tick(self, label: str, detail: str = "", kind: str = "loop"):
        self.tick_count += 1
        self.last_tick_at = now()
        node_id = _slug(label) or "loop"
        for node in self.nodes:
            if node.id == node_id:
                node.status = "active"
                node.detail = detail
                node.updated_at = self.last_tick_at
                self.active_node_id = node.id
                return
        self.nodes.append(BehaviorNode(node_id, label, kind, "active", detail))
        self.active_node_id = node_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "goal": self.goal,
            "active_node_id": self.active_node_id,
            "tick_seconds": self.tick_seconds,
            "idle_review_seconds": self.idle_review_seconds,
            "idle_review": self.idle_review,
            "force_idle_reviews": self.force_idle_reviews,
            "idle_importance": self.idle_importance,
            "tick_count": self.tick_count,
            "last_tick_at": self.last_tick_at,
            "last_idle_review_at": self.last_idle_review_at,
            "nodes": [node.to_dict() for node in self.nodes],
        }


@dataclass
class CronJob:
    id: str
    name: str
    instruction: str
    every_seconds: float
    enabled: bool = True
    next_run_at: float = field(default_factory=now)
    last_run_at: float = 0.0
    run_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CronRegistry:
    def __init__(self, config: list[dict[str, Any]] | None = None):
        self.jobs: list[CronJob] = []
        for idx, raw in enumerate(config or []):
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or f"job-{idx + 1}")
            instruction = str(raw.get("instruction") or raw.get("text") or "").strip()
            every = _positive_float(raw.get("every_seconds"), 60.0)
            self.add(
                name,
                every,
                instruction,
                enabled=bool(raw.get("enabled", True)),
                job_id=str(raw.get("id") or ""),
            )

    def add(
        self,
        name: str,
        every_seconds: float,
        instruction: str,
        enabled: bool = True,
        job_id: str = "",
    ) -> CronJob:
        clean_name = str(name).strip() or "scheduled job"
        clean_id = job_id.strip() if job_id else _slug(clean_name)
        if not clean_id:
            clean_id = f"job-{len(self.jobs) + 1}"
        ids = {job.id for job in self.jobs}
        base = clean_id
        suffix = 2
        while clean_id in ids:
            clean_id = f"{base}-{suffix}"
            suffix += 1
        job = CronJob(
            id=clean_id,
            name=clean_name,
            instruction=str(instruction).strip() or f"Run scheduled job: {clean_name}",
            every_seconds=max(1.0, float(every_seconds)),
            enabled=enabled,
            next_run_at=now() + max(1.0, float(every_seconds)),
        )
        self.jobs.append(job)
        return job

    def due(self, timestamp: float | None = None) -> list[CronJob]:
        t = now() if timestamp is None else timestamp
        return [job for job in self.jobs if job.enabled and job.next_run_at <= t]

    def mark_run(self, job_id: str):
        t = now()
        for job in self.jobs:
            if job.id == job_id:
                job.last_run_at = t
                job.run_count += 1
                job.next_run_at = t + job.every_seconds
                return

    def to_dict(self) -> dict[str, Any]:
        return {"jobs": [job.to_dict() for job in self.jobs]}


@dataclass
class MapObservation:
    id: str
    summary: str
    place: str = ""
    x: float | None = None
    y: float | None = None
    yaw_deg: float | None = None
    confidence: float = 1.0
    timestamp: float = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SpatialMap:
    def __init__(self):
        self.observations: list[MapObservation] = []

    def record(
        self,
        summary: str,
        place: str = "",
        x: float | None = None,
        y: float | None = None,
        yaw_deg: float | None = None,
        confidence: float = 1.0,
    ) -> MapObservation:
        obs = MapObservation(
            id=f"map-{len(self.observations) + 1}",
            summary=str(summary).strip(),
            place=str(place).strip(),
            x=_optional_float(x),
            y=_optional_float(y),
            yaw_deg=_optional_float(yaw_deg),
            confidence=max(0.0, min(1.0, float(confidence))),
        )
        self.observations.append(obs)
        return obs

    def to_dict(self) -> dict[str, Any]:
        places: dict[str, list[dict[str, Any]]] = {}
        for obs in self.observations:
            key = obs.place or "unknown"
            places.setdefault(key, []).append(obs.to_dict())
        return {
            "observations": [obs.to_dict() for obs in self.observations[-100:]],
            "places": places,
        }


class BodyPerception:
    def __init__(self):
        self.summary = "Body state unavailable"
        self.posture = "unknown"
        self.holding = ""
        self.contacts: list[str] = []
        self.base_pose: dict[str, float] = {}
        self.joints: dict[str, float] = {}
        self.updated_at = now()

    def update_from_robot_state(self, state: RobotState):
        self.base_pose = dict(state.pose or {})
        self.joints = dict(state.joints or {})
        if self.posture in {"unknown", ""}:
            self.posture = str(state.mode or "unknown")
        if self.summary == "Body state unavailable":
            self.summary = f"{state.backend} body is {self.posture}"
        self.updated_at = now()

    def update(
        self,
        summary: str,
        posture: str = "",
        holding: str = "",
        contacts: list[str] | None = None,
        joints: dict[str, float] | None = None,
    ):
        if summary:
            self.summary = str(summary)
        if posture:
            self.posture = str(posture)
        if holding:
            self.holding = str(holding)
        if contacts is not None:
            self.contacts = [str(item) for item in contacts if str(item).strip()]
        if joints is not None:
            self.joints = {str(key): float(value) for key, value in joints.items()}
        self.updated_at = now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "posture": self.posture,
            "holding": self.holding,
            "contacts": self.contacts,
            "base_pose": self.base_pose,
            "joints": self.joints,
            "updated_at": self.updated_at,
        }


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", str(value).strip().lower()).strip("-")
    return text[:48]


def _status(value: str) -> str:
    clean = str(value).strip().lower()
    return (
        clean
        if clean in {"waiting", "active", "done", "blocked", "failed"}
        else "active"
    )


def _positive_float(value: Any, default: float) -> float:
    try:
        return max(1.0, float(value))
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
