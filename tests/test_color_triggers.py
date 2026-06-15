"""Tests for color group detection and action mapping."""

import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from actum.color_triggers.actions import ActionExecutor, ColorTriggerConfig, TriggerAction
from actum.color_triggers.colors import default_tape_colors, extract_label_runs
from actum.color_triggers.detector import BandDetector, DetectionParams
from actum.color_triggers.library import (
    ColorLibrary,
    Combination,
    NamedColor,
    load_color_library,
    match_combinations,
    save_color_library,
)
from actum.color_triggers.synthetic import generate_tag_image
from actum.color_triggers.watcher import ColorTriggerWatcher


def test_match_is_order_independent():
    combos = {"go": Combination("go", ["purple", "blue", "pink"])}
    assert match_combinations({"pink", "purple", "blue"}, combos) == ["go"]


def test_match_requires_exact_set():
    combos = {"rg": Combination("rg", ["purple", "blue"])}
    assert match_combinations({"purple", "blue", "pink"}, combos) == []
    assert match_combinations({"purple"}, combos) == []
    assert match_combinations({"purple", "blue"}, combos) == ["rg"]


def test_extract_label_runs_merges_short_blips():
    labels = ["pink", "pink", "pink", "x", "blue", "blue"]
    runs = extract_label_runs(labels, min_width=2)
    assert runs == [("pink", 0, 3), ("blue", 4, 6)]


def test_save_load_calibration_roundtrip(tmp_path):
    path = tmp_path / "cal.json"
    library = ColorLibrary()
    from actum.color_triggers.colors import HSVRange

    library.upsert(NamedColor(name="pink", hsv_range=HSVRange(156, 173, 100, 255, 150, 255)))
    library.upsert_combination(Combination("group-1", ["purple", "blue", "pink"]))
    save_color_library(path, library)

    loaded = load_color_library(path)
    assert "group-1" in loaded.combinations
    assert loaded.combinations["group-1"].color_set() == frozenset({"purple", "blue", "pink"})


def test_detector_finds_synthetic_group(tmp_path):
    cal_path = Path(__file__).resolve().parents[1] / "config" / "color_triggers_calibration.json"
    detector = BandDetector(calibration_path=cal_path)
    frame, _ = generate_tag_image(
        seed=7,
        color_order=["purple", "blue", "pink"],
        angle_deg=12.0,
    )
    result = detector.detect(frame)
    assert result.combination_detected
    assert "group-1" in result.matched_combinations


def test_action_executor_speak(capsys):
    calls = []

    class FakeBackend:
        name = "fake"

        def speak(self, text):
            calls.append(text)
            from actum.core.schema import ActionResult, now

            return ActionResult(action="speak", ok=True, message=text, started_at=now())

    executor = ActionExecutor(backend=FakeBackend())
    executor.execute("group-1", TriggerAction(type="speak", params={"text": "hello"}))
    assert calls == ["hello"]


def test_watcher_fires_once_with_cooldown():
    cfg = ColorTriggerConfig(
        enabled=True,
        calibration_path=str(
            Path(__file__).resolve().parents[1] / "config" / "color_triggers_calibration.json"
        ),
        detect_every_frames=1,
        cooldown_seconds=60.0,
        actions={
            "group-1": TriggerAction(type="log", params={"text": "seen group 1"}),
        },
    )
    frame, _ = generate_tag_image(seed=3, color_order=["purple", "blue", "pink"])
    fired = []

    class FakeBackend:
        name = "fake"

    def on_action(group, action, context):
        fired.append(group)

    watcher = ColorTriggerWatcher(
        cfg,
        frame_reader=lambda: frame,
        backend=FakeBackend(),
        on_action=on_action,
    )
    watcher.process_frame(frame)
    watcher.process_frame(frame)
    assert fired == ["group-1"]


def test_trigger_config_from_dict(tmp_path):
    path = tmp_path / "triggers.json"
    path.write_text(
        json.dumps(
            {
                "enabled": True,
                "actions": {
                    "group-2": {"type": "drive", "direction": "forward", "distance_m": 0.5}
                },
            }
        )
    )
    cfg = ColorTriggerConfig.from_dict(json.loads(path.read_text()), base_dir=tmp_path)
    assert cfg.enabled
    assert cfg.actions["group-2"].type == "drive"
