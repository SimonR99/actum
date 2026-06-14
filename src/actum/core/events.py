"""Append-only event log used by the dashboard and tests."""

from __future__ import annotations

from collections import deque
from typing import Any

from actum.core.schema import Event


class EventLog:
    def __init__(self, maxlen: int = 1000):
        self._events: deque[Event] = deque(maxlen=maxlen)

    def append(
        self, event_type: str, source: str, message: str = "", **data: Any
    ) -> Event:
        event = Event(type=event_type, source=source, message=message, data=data)
        self._events.append(event)
        return event

    def tail(self, limit: int = 200) -> list[dict[str, Any]]:
        return [event.to_dict() for event in list(self._events)[-limit:]]

    def clear(self):
        self._events.clear()
