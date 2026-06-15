"""HSV color classification for tape/band detection."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

SEPARATOR_NAME = "black"
UNKNOWN_NAME = "unknown"


@dataclass(frozen=True)
class HSVRange:
    """Inclusive HSV range. Hue wraps at 180 in OpenCV."""

    h_min: int
    h_max: int
    s_min: int
    s_max: int
    v_min: int
    v_max: int

    def contains(self, h: int, s: int, v: int) -> bool:
        if s < self.s_min or s > self.s_max or v < self.v_min or v > self.v_max:
            return False
        if self.h_min <= self.h_max:
            return self.h_min <= h <= self.h_max
        return h >= self.h_min or h <= self.h_max


def default_tape_colors() -> Dict[str, HSVRange]:
    """Default HSV ranges for pink, blue, orange, purple (+ black separator)."""
    return {
        "blue": HSVRange(93, 117, 120, 255, 110, 255),
        "orange": HSVRange(0, 18, 100, 255, 175, 255),
        "purple": HSVRange(140, 156, 100, 255, 100, 255),
        "pink": HSVRange(156, 173, 100, 255, 150, 255),
        "black": HSVRange(0, 180, 0, 255, 0, 50),
    }


def extract_label_runs(labels: List[str], min_width: int) -> List[Tuple[str, int, int]]:
    """Contiguous runs of equal labels as (label, start, end), dropping short runs."""
    if not labels:
        return []

    segments: List[Tuple[str, int, int]] = []
    current = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != current:
            segments.append((current, start, i))
            current = labels[i]
            start = i
    segments.append((current, start, len(labels)))

    merged: List[Tuple[str, int, int]] = []
    for label, seg_start, seg_end in segments:
        if seg_end - seg_start < min_width:
            continue
        if merged and merged[-1][0] == label:
            prev_label, prev_start, _ = merged[-1]
            merged[-1] = (prev_label, prev_start, seg_end)
        else:
            merged.append((label, seg_start, seg_end))
    return merged


def _range_hue_center(range_: HSVRange) -> float:
    if range_.h_min <= range_.h_max:
        return (range_.h_min + range_.h_max) / 2.0
    span = (180 - range_.h_min) + range_.h_max
    return (range_.h_min + span / 2.0) % 180.0


def _hue_distance(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def classify_pixel_named(
    h: int,
    s: int,
    v: int,
    ranges: Dict[str, HSVRange],
) -> Optional[str]:
    matches = [name for name, range_ in ranges.items() if range_.contains(h, s, v)]
    if not matches:
        return None
    pool = [name for name in matches if name != SEPARATOR_NAME] or matches
    if len(pool) == 1:
        return pool[0]

    best_name = pool[0]
    best_dist = float("inf")
    for name in pool:
        range_ = ranges[name]
        ch = _range_hue_center(range_)
        cs = (range_.s_min + range_.s_max) / 2.0
        cv = (range_.v_min + range_.v_max) / 2.0
        dist = _hue_distance(h, ch) * 2.0 + abs(s - cs) / 8.0 + abs(v - cv) / 8.0
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def classify_named_columns(
    hsv_strip: np.ndarray,
    ranges: Dict[str, HSVRange],
) -> List[str]:
    if hsv_strip.size == 0:
        return []

    labels: List[str] = []
    for col in range(hsv_strip.shape[1]):
        column = hsv_strip[:, col, :]
        h = int(np.median(column[:, 0]))
        s = int(np.median(column[:, 1]))
        v = int(np.median(column[:, 2]))
        name = classify_pixel_named(h, s, v, ranges)
        labels.append(name if name is not None else UNKNOWN_NAME)
    return labels
