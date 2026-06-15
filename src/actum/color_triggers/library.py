"""Named color library, combinations (groups), and calibration I/O."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from actum.color_triggers.colors import HSVRange

LIBRARY_VERSION = 3


@dataclass
class NamedColor:
    name: str
    hsv_range: HSVRange
    sample_point: Optional[Tuple[int, int]] = None
    reference_bgr: Optional[Tuple[int, int, int]] = None


@dataclass
class Combination:
    """A named group of colors seen together on the floor."""

    name: str
    colors: List[str]

    def color_set(self) -> FrozenSet[str]:
        return frozenset(self.colors)


@dataclass
class ColorLibrary:
    colors: Dict[str, NamedColor] = field(default_factory=dict)
    combinations: Dict[str, Combination] = field(default_factory=dict)
    detection: dict = field(default_factory=dict)

    def get(self, name: str) -> Optional[NamedColor]:
        return self.colors.get(name)

    def upsert(self, entry: NamedColor) -> None:
        self.colors[entry.name] = entry

    def upsert_combination(self, combination: Combination) -> None:
        self.combinations[combination.name] = combination

    def remove_combination(self, name: str) -> bool:
        if name in self.combinations:
            del self.combinations[name]
            return True
        return False

    def names(self) -> List[str]:
        return sorted(self.colors.keys())

    def combination_names(self) -> List[str]:
        return sorted(self.combinations.keys())


def match_combinations(
    detected_colors: Set[str],
    combinations: Dict[str, Combination],
) -> List[str]:
    """Names of combinations whose color set equals the detected color set."""
    detected = frozenset(detected_colors)
    return sorted(
        name
        for name, combination in combinations.items()
        if combination.color_set() == detected
    )


def hsv_range_to_dict(range_: HSVRange) -> dict:
    return {
        "h_min": range_.h_min,
        "h_max": range_.h_max,
        "s_min": range_.s_min,
        "s_max": range_.s_max,
        "v_min": range_.v_min,
        "v_max": range_.v_max,
    }


def hsv_range_from_dict(data: dict) -> HSVRange:
    return HSVRange(
        data["h_min"],
        data["h_max"],
        data["s_min"],
        data["s_max"],
        data["v_min"],
        data["v_max"],
    )


def named_color_to_dict(entry: NamedColor) -> dict:
    payload = {"hsv_range": hsv_range_to_dict(entry.hsv_range)}
    if entry.sample_point is not None:
        payload["sample_point"] = [
            int(entry.sample_point[0]),
            int(entry.sample_point[1]),
        ]
    if entry.reference_bgr is not None:
        payload["reference_bgr"] = list(entry.reference_bgr)
    return payload


def named_color_from_dict(name: str, data: dict) -> NamedColor:
    sample_point = None
    if "sample_point" in data:
        sample_point = (int(data["sample_point"][0]), int(data["sample_point"][1]))
    reference_bgr = None
    if "reference_bgr" in data:
        bgr = data["reference_bgr"]
        reference_bgr = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    return NamedColor(
        name=name,
        hsv_range=hsv_range_from_dict(data["hsv_range"]),
        sample_point=sample_point,
        reference_bgr=reference_bgr,
    )


def save_color_library(path: Path, library: ColorLibrary) -> None:
    payload = {
        "version": LIBRARY_VERSION,
        "colors": {name: named_color_to_dict(c) for name, c in library.colors.items()},
        "combinations": {
            name: list(combo.colors) for name, combo in library.combinations.items()
        },
        "detection": library.detection,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _combinations_from_payload(payload: dict) -> Dict[str, Combination]:
    return {
        name: Combination(name=name, colors=list(colors))
        for name, colors in payload.get("combinations", {}).items()
    }


def load_color_library(path: Path) -> ColorLibrary:
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    payload = json.loads(path.read_text())
    version = payload.get("version", 1)

    if version >= 2:
        colors = {
            name: named_color_from_dict(name, data)
            for name, data in payload.get("colors", {}).items()
        }
        return ColorLibrary(
            colors=colors,
            combinations=_combinations_from_payload(payload),
            detection=payload.get("detection", {}),
        )

    library = ColorLibrary(detection=payload.get("detection", {}))
    for name, params in payload.get("hsv_ranges", {}).items():
        library.upsert(NamedColor(name=name, hsv_range=hsv_range_from_dict(params)))
    return library


def preview_bgr_for_entry(entry: NamedColor) -> Tuple[int, int, int]:
    if entry.reference_bgr is not None:
        return entry.reference_bgr
    v = (entry.hsv_range.v_min + entry.hsv_range.v_max) // 2
    return (v, v, v)
