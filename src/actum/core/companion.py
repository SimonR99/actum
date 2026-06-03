"""Always-on companion policy for deciding when passive events should run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from actum.core.schema import now


DEFAULT_SAFETY_KEYWORDS = (
    "danger",
    "unsafe",
    "emergency",
    "help",
    "hurt",
    "injured",
    "fall",
    "fire",
    "smoke",
    "blocked",
    "stuck",
)

DEFAULT_REQUEST_KEYWORDS = (
    "hey",
    "hello",
    "come here",
    "look at this",
    "watch this",
    "can you",
    "could you",
    "please",
    "need you",
)


@dataclass
class CompanionDecision:
    process: bool
    reason: str
    source: str
    mode: str
    matched: str = ""
    importance: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CompanionPolicy:
    """Conservative gate for always-on camera/timer events.

    Direct user inputs are always admitted. Passive observations only reach the
    LLM when they are likely to matter, which keeps the robot present without
    making it noisy or impulsive.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.always_on = bool(self.config.get("always_on", True))
        self.proactive_mode = str(self.config.get("proactive_mode", "conservative")).lower().strip()
        self.direct_sources = set(self.config.get("direct_sources", ["chat", "language", "text", "voice"]))
        self.passive_sources = set(self.config.get("passive_sources", ["cron", "loop", "timer", "vision"]))
        self.min_seconds_between_proactive_actions = float(
            self.config.get("min_seconds_between_proactive_actions", 60.0)
        )
        self.vision_importance_threshold = float(self.config.get("vision_importance_threshold", 0.75))
        self.curious_importance_threshold = float(self.config.get("curious_importance_threshold", 0.55))
        self.safety_keywords = tuple(
            str(item).lower() for item in self.config.get("safety_keywords", DEFAULT_SAFETY_KEYWORDS)
        )
        self.request_keywords = tuple(
            str(item).lower() for item in self.config.get("request_keywords", DEFAULT_REQUEST_KEYWORDS)
        )
        self._last_proactive_at = 0.0

    def decide(self, event: dict[str, Any]) -> CompanionDecision:
        timestamp = now()
        source = str(event.get("source", "unknown")).lower().strip()
        mode = self.proactive_mode or "conservative"
        text = str(event.get("text", "")).lower()
        importance = _importance(event)

        if bool(event.get("force")):
            return self._process("forced by caller", source, mode, "force", importance, timestamp)

        if source in self.direct_sources:
            return self._process("direct user input", source, mode, source, importance, timestamp)

        if not self.always_on:
            return self._ignore("always-on companion mode is disabled", source, mode, importance, timestamp)

        if source not in self.passive_sources:
            return self._ignore("source is not configured for passive processing", source, mode, importance, timestamp)

        safety_match = _first_match(text, self.safety_keywords)
        if safety_match:
            return self._process("passive event matched safety language", source, mode, safety_match, importance, timestamp)

        request_match = _first_match(text, self.request_keywords)
        if request_match:
            return self._process("passive event looks like an explicit request", source, mode, request_match, importance, timestamp)

        if importance >= self.vision_importance_threshold:
            return self._process("passive event crossed importance threshold", source, mode, "importance", importance, timestamp)

        if mode in {"off", "disabled", "quiet"}:
            return self._ignore("proactive mode is off", source, mode, importance, timestamp)

        if source == "timer" and self._cooldown_ready(timestamp):
            return self._process("scheduled companion check-in", source, mode, "timer", importance, timestamp)

        if mode in {"curious", "high"} and importance >= self.curious_importance_threshold and self._cooldown_ready(timestamp):
            return self._process("curious mode admitted a moderately important event", source, mode, "curious", importance, timestamp)

        return self._ignore("passive event did not require action", source, mode, importance, timestamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "always_on": self.always_on,
            "proactive_mode": self.proactive_mode,
            "direct_sources": sorted(self.direct_sources),
            "passive_sources": sorted(self.passive_sources),
            "min_seconds_between_proactive_actions": self.min_seconds_between_proactive_actions,
            "vision_importance_threshold": self.vision_importance_threshold,
            "curious_importance_threshold": self.curious_importance_threshold,
            "last_proactive_at": self._last_proactive_at,
        }

    def _cooldown_ready(self, timestamp: float) -> bool:
        return (timestamp - self._last_proactive_at) >= self.min_seconds_between_proactive_actions

    def _process(
        self,
        reason: str,
        source: str,
        mode: str,
        matched: str,
        importance: float,
        timestamp: float,
    ) -> CompanionDecision:
        if source in self.passive_sources:
            self._last_proactive_at = timestamp
        return CompanionDecision(True, reason, source, mode, matched, importance, timestamp)

    def _ignore(
        self,
        reason: str,
        source: str,
        mode: str,
        importance: float,
        timestamp: float,
    ) -> CompanionDecision:
        return CompanionDecision(False, reason, source, mode, "", importance, timestamp)


def _importance(event: dict[str, Any]) -> float:
    try:
        return max(0.0, min(1.0, float(event.get("importance", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def _first_match(text: str, keywords: tuple[str, ...]) -> str:
    if not text:
        return ""
    for keyword in keywords:
        if keyword and keyword in text:
            return keyword
    return ""
