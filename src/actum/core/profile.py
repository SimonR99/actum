"""Speed profiles: one switch for compute vs. responsiveness vs. power.

A profile bundles every knob that trades speed against cost/power into a single
named preset the operator can select:

- ``provider``           which inference brain (``local`` / ``openai``)
- ``compute``            local model compute backend (``gpu`` / ``cpu`` / ``npu``)
- ``tick_seconds``       background autonomy loop period
- ``idle_review_seconds`` how often idle vision review fires
- ``camera_fps``         dashboard camera stream rate
- ``deliberate_seconds`` how often the robot sets its own tasks (0 disables)

The active profile is authoritative for these knobs; the matching values in the
raw ``models`` / ``behavior_loop`` config act only as fallbacks.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

PROFILE_KEYS = (
    "provider",
    "compute",
    "tick_seconds",
    "idle_review_seconds",
    "camera_fps",
    "deliberate_seconds",
)

DEFAULT_PROFILES: dict[str, dict[str, Any]] = {
    "fast": {
        "provider": "openai",
        "compute": "gpu",
        "tick_seconds": 6.0,
        "idle_review_seconds": 20.0,
        "camera_fps": 10.0,
        "deliberate_seconds": 120.0,
    },
    "balanced": {
        "provider": "local",
        "compute": "gpu",
        "tick_seconds": 15.0,
        "idle_review_seconds": 45.0,
        "camera_fps": 6.7,
        "deliberate_seconds": 240.0,
    },
    "power_saver": {
        "provider": "local",
        "compute": "cpu",
        "tick_seconds": 30.0,
        "idle_review_seconds": 120.0,
        "camera_fps": 3.0,
        "deliberate_seconds": 600.0,
    },
}

DEFAULT_ACTIVE = "balanced"


class ProfileManager:
    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.profiles: dict[str, dict[str, Any]] = deepcopy(DEFAULT_PROFILES)
        raw = config.get("profiles")
        if isinstance(raw, dict):
            for name, values in raw.items():
                if isinstance(values, dict):
                    base = self.profiles.get(str(name), {})
                    self.profiles[str(name)] = {**base, **_clean(values)}
        active = str(config.get("active_profile") or DEFAULT_ACTIVE).strip()
        self.active_name = active if active in self.profiles else DEFAULT_ACTIVE

    @property
    def resolved(self) -> dict[str, Any]:
        merged = deepcopy(DEFAULT_PROFILES[DEFAULT_ACTIVE])
        merged.update(self.profiles.get(self.active_name, {}))
        return merged

    def set_active(self, name: str) -> dict[str, Any]:
        clean = str(name).strip()
        if clean not in self.profiles:
            raise ValueError(
                f"Unknown profile {clean!r}. Available: {', '.join(sorted(self.profiles))}"
            )
        self.active_name = clean
        return self.resolved

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self.active_name,
            "resolved": self.resolved,
            "available": sorted(self.profiles),
            "profiles": deepcopy(self.profiles),
        }

    def to_config(self) -> dict[str, Any]:
        return {"active_profile": self.active_name, "profiles": deepcopy(self.profiles)}


def _clean(values: dict[str, Any]) -> dict[str, Any]:
    return {key: values[key] for key in PROFILE_KEYS if key in values}
