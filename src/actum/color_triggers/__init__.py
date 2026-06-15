"""Camera color-group detection mapped to robot or text actions."""

from actum.color_triggers.actions import (
    ActionExecutor,
    ColorTriggerConfig,
    TriggerAction,
    load_trigger_config,
)
from actum.color_triggers.detector import BandDetectionResult, BandDetector, DetectionParams
from actum.color_triggers.library import Combination, ColorLibrary, load_color_library, match_combinations
from actum.color_triggers.watcher import ColorTriggerWatcher

__all__ = [
    "ActionExecutor",
    "BandDetectionResult",
    "BandDetector",
    "ColorLibrary",
    "ColorTriggerConfig",
    "ColorTriggerWatcher",
    "Combination",
    "DetectionParams",
    "TriggerAction",
    "load_color_library",
    "load_trigger_config",
    "match_combinations",
]
