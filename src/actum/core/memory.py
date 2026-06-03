"""Persistent memory primitives for the robot companion."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from actum.core.schema import now


@dataclass
class MemoryRecord:
    id: str
    kind: str
    summary: str
    source: str = "agent"
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryStore:
    """Small JSON-backed memory store.

    This gives Actum durable memory without introducing a database dependency.
    The contract is intentionally shaped so the backend can later become SQLite,
    LanceDB, Qdrant, or another vector/spatial store without changing tools.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", True))
        self.path = _memory_path(self.config.get("path", "data/memory.json"))
        self.max_episodes = int(self.config.get("max_episodes", 1000))
        self.facts: dict[str, str] = {}
        self.people: dict[str, dict[str, Any]] = {}
        self.places: dict[str, dict[str, Any]] = {}
        self.spatial_notes: list[MemoryRecord] = []
        self.episodes: list[MemoryRecord] = []
        self._next_id = 1
        self.load()

    def remember_fact(self, key: str, value: str):
        key = str(key).strip()
        if not key:
            raise ValueError("Memory key cannot be empty.")
        self.facts[key] = str(value)
        self.flush()

    def recall_fact(self, key: str) -> str | None:
        return self.facts.get(str(key).strip())

    def remember_person(self, name: str, note: str, **data: Any):
        person = _named_entry(self.people, name)
        person["notes"].append({"text": str(note), "timestamp": now(), **data})
        person["updated_at"] = now()
        self.flush()

    def remember_place(self, name: str, note: str, **data: Any):
        place = _named_entry(self.places, name)
        place["notes"].append({"text": str(note), "timestamp": now(), **data})
        place["updated_at"] = now()
        self.flush()

    def record_observation(self, summary: str, source: str = "vision", **data: Any) -> MemoryRecord:
        return self._append_record("observation", summary, source, data)

    def record_episode(self, summary: str, source: str = "agent", **data: Any) -> MemoryRecord:
        return self._append_record("episode", summary, source, data)

    def remember_spatial_note(self, summary: str, source: str = "agent", **data: Any) -> MemoryRecord:
        record = self._new_record("spatial", summary, source, data)
        self.spatial_notes.append(record)
        self.flush()
        return record

    def recent(self, limit: int = 8, kind: str | None = None) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 50))
        records = self.episodes
        if kind:
            records = [record for record in records if record.kind == kind]
        return [record.to_dict() for record in records[-limit:]]

    def context(self, limit: int = 6) -> str:
        lines: list[str] = []
        if self.facts:
            facts = list(self.facts.items())[:12]
            lines.append("Facts:")
            lines.extend(f"  {key}: {value}" for key, value in facts)
        if self.people:
            lines.append("People:")
            for name, person in list(self.people.items())[:6]:
                latest = person.get("notes", [])[-1]["text"] if person.get("notes") else ""
                lines.append(f"  {name}: {latest}")
        if self.places:
            lines.append("Places:")
            for name, place in list(self.places.items())[:6]:
                latest = place.get("notes", [])[-1]["text"] if place.get("notes") else ""
                lines.append(f"  {name}: {latest}")
        recent = self.episodes[-max(1, min(int(limit), 12)) :]
        if recent:
            lines.append("Recent episodes:")
            lines.extend(f"  {record.kind}: {record.summary}" for record in recent)
        if self.spatial_notes:
            lines.append("Spatial notes:")
            lines.extend(f"  {record.summary}" for record in self.spatial_notes[-6:])
        return "\n".join(lines)

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "path": str(self.path),
            "facts": dict(self.facts),
            "people": self.people,
            "places": self.places,
            "spatial_notes": [record.to_dict() for record in self.spatial_notes[-100:]],
            "recent": self.recent(25),
            "counts": {
                "facts": len(self.facts),
                "people": len(self.people),
                "places": len(self.places),
                "spatial_notes": len(self.spatial_notes),
                "episodes": len(self.episodes),
            },
        }

    def load(self):
        if not self.enabled or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(raw, dict):
            return
        self.facts = {str(key): str(value) for key, value in raw.get("facts", {}).items()} if isinstance(raw.get("facts"), dict) else {}
        self.people = raw.get("people", {}) if isinstance(raw.get("people"), dict) else {}
        self.places = raw.get("places", {}) if isinstance(raw.get("places"), dict) else {}
        self.spatial_notes = [_record_from_dict(item) for item in raw.get("spatial_notes", []) if isinstance(item, dict)]
        self.episodes = [_record_from_dict(item) for item in raw.get("episodes", []) if isinstance(item, dict)]
        all_ids = [
            _numeric_id(record.id)
            for record in [*self.spatial_notes, *self.episodes]
            if _numeric_id(record.id) is not None
        ]
        self._next_id = max(all_ids, default=0) + 1

    def flush(self):
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "facts": self.facts,
            "people": self.people,
            "places": self.places,
            "spatial_notes": [record.to_dict() for record in self.spatial_notes],
            "episodes": [record.to_dict() for record in self.episodes[-self.max_episodes :]],
        }
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)

    def _append_record(self, kind: str, summary: str, source: str, data: dict[str, Any]) -> MemoryRecord:
        record = self._new_record(kind, summary, source, data)
        self.episodes.append(record)
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes :]
        self.flush()
        return record

    def _new_record(self, kind: str, summary: str, source: str, data: dict[str, Any]) -> MemoryRecord:
        record = MemoryRecord(
            id=f"mem-{self._next_id}",
            kind=kind,
            summary=str(summary).strip(),
            source=str(source),
            data=data,
        )
        self._next_id += 1
        return record


def _memory_path(path: Any) -> Path:
    value = Path(str(path)).expanduser()
    return value if value.is_absolute() else Path.cwd() / value


def _named_entry(collection: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    key = str(name).strip()
    if not key:
        raise ValueError("Name cannot be empty.")
    if key not in collection:
        collection[key] = {"name": key, "notes": [], "created_at": now(), "updated_at": now()}
    return collection[key]


def _record_from_dict(raw: dict[str, Any]) -> MemoryRecord:
    return MemoryRecord(
        id=str(raw.get("id", "")),
        kind=str(raw.get("kind", "episode")),
        summary=str(raw.get("summary", "")),
        source=str(raw.get("source", "agent")),
        data=raw.get("data", {}) if isinstance(raw.get("data"), dict) else {},
        timestamp=float(raw.get("timestamp", now())),
    )


def _numeric_id(value: str) -> int | None:
    if value.startswith("mem-") and value[4:].isdigit():
        return int(value[4:])
    return None
