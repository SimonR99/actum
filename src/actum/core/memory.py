"""Persistent memory primitives for the robot companion."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from actum.core.schema import now

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "the a an and or of to in on at is are was were be been it this that with for from your you i my".split()
)


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
        self.consolidate_seconds = float(self.config.get("consolidate_seconds", 300.0))
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

    def context(self, limit: int = 6, query: str | None = None) -> str:
        if query and query.strip():
            relevant = self.search(query, limit=max(limit, 8))
            if relevant:
                body = "\n".join(f"  {kind}: {text}" for kind, text in relevant)
                return f"Relevant memory for this turn:\n{body}"
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

    def search(self, query: str, limit: int = 8) -> list[tuple[str, str]]:
        """Return the most relevant memory entries for a query.

        Dependency-free lexical retrieval: token overlap scored against every
        fact, person, place, spatial note, and recent episode, with a small
        recency bonus for episodes. This keeps prompts focused on relevant
        memory instead of dumping the whole store, so it scales as memory grows.
        """
        terms = _tokenize(query)
        if not terms:
            return []
        scored: list[tuple[float, float, str, str]] = []
        for key, value in self.facts.items():
            scored.append((_score(terms, f"{key} {value}"), 0.0, "fact", f"{key}: {value}"))
        for name, person in self.people.items():
            note = person.get("notes", [])[-1]["text"] if person.get("notes") else ""
            scored.append((_score(terms, f"{name} {note}"), 0.0, "person", f"{name}: {note}"))
        for name, place in self.places.items():
            note = place.get("notes", [])[-1]["text"] if place.get("notes") else ""
            scored.append((_score(terms, f"{name} {note}"), 0.0, "place", f"{name}: {note}"))
        for record in self.spatial_notes:
            scored.append((_score(terms, record.summary), record.timestamp, "spatial", record.summary))
        total = len(self.episodes)
        for index, record in enumerate(self.episodes):
            recency = (index + 1) / total if total else 0.0
            base = _score(terms, record.summary)
            score = base + 0.15 * recency if base > 0 else 0.0
            scored.append((score, record.timestamp, record.kind, record.summary))
        ranked = sorted(
            (item for item in scored if item[0] > 0),
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )
        return [(kind, text) for _, _, kind, text in ranked[: max(1, int(limit))]]

    def consolidate(self) -> dict[str, Any]:
        """Self-maintenance: drop duplicate/stale records and enforce limits.

        Deterministic and safe — collapses exact-duplicate episode and spatial
        summaries (keeping the most recent), then trims episodes to the cap.
        """
        before_episodes = len(self.episodes)
        before_spatial = len(self.spatial_notes)

        self.episodes = _dedupe_records(self.episodes)
        self.spatial_notes = _dedupe_records(self.spatial_notes)
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes :]

        report = {
            "removed_episodes": before_episodes - len(self.episodes),
            "removed_spatial_notes": before_spatial - len(self.spatial_notes),
            "episodes": len(self.episodes),
            "spatial_notes": len(self.spatial_notes),
        }
        if report["removed_episodes"] or report["removed_spatial_notes"]:
            self.flush()
        return report

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


def _tokenize(text: str) -> set[str]:
    return {tok for tok in _TOKEN_RE.findall(str(text).lower()) if tok not in _STOPWORDS}


def _score(terms: set[str], text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    overlap = terms & tokens
    return len(overlap) / len(terms) if overlap else 0.0


def _dedupe_records(records: list[MemoryRecord]) -> list[MemoryRecord]:
    """Keep the most recent record for each distinct summary, preserving order."""
    last_index: dict[str, int] = {}
    for index, record in enumerate(records):
        key = record.summary.strip().lower()
        if key:
            last_index[key] = index
    keep = set(last_index.values())
    return [record for index, record in enumerate(records) if index in keep or not record.summary.strip()]
