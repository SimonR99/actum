"""Background camera watcher: detect color groups and fire mapped actions."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from actum.backends.base import RobotBackend
from actum.color_triggers.actions import ActionCallback, ActionExecutor, ColorTriggerConfig
from actum.color_triggers.detector import BandDetectionResult, BandDetector


FrameReader = Callable[[], np.ndarray | None]


class ColorTriggerWatcher:
    """Poll the camera, match tape color groups, and execute bound actions."""

    def __init__(
        self,
        config: ColorTriggerConfig,
        frame_reader: FrameReader,
        backend: RobotBackend | None = None,
        on_detection: Callable[[BandDetectionResult], None] | None = None,
        on_action: ActionCallback | None = None,
    ):
        self.config = config
        self.frame_reader = frame_reader
        self.detector = BandDetector(calibration_path=config.calibration_path)
        self.executor = ActionExecutor(backend=backend, on_action=on_action)
        self.on_detection = on_detection
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_idx = 0
        self._last_group = ""
        self._last_fired_at = 0.0
        self.last_result: BandDetectionResult | None = None

    def process_frame(self, frame_bgr: np.ndarray) -> BandDetectionResult | None:
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        self._frame_idx += 1
        if self._frame_idx % self.config.detect_every_frames != 0:
            return self.last_result

        result = self.detector.detect(frame_bgr)
        self.last_result = result
        if self.on_detection is not None:
            self.on_detection(result)

        if not result.combination_detected or not result.matched_combinations:
            return result

        group = result.matched_combinations[0]
        now = time.time()
        if group == self._last_group and (now - self._last_fired_at) < self.config.cooldown_seconds:
            return result

        action = self.config.actions.get(group)
        if action is None:
            print(f"[color_trigger] matched {group} but no action configured")
            return result

        self._last_group = group
        self._last_fired_at = now
        context = {
            "group": group,
            "colors": list(result.colors_sequence),
            "matched_combinations": list(result.matched_combinations),
        }
        try:
            self.executor.execute(group, action, context)
        except Exception as exc:
            print(f"[color_trigger] failed to run {group}: {exc}")
        return result

    def _run(self):
        while not self._stop.is_set():
            try:
                frame = self.frame_reader()
                if frame is not None:
                    self.process_frame(frame)
            except Exception as exc:
                print(f"[color_trigger] watcher error: {exc}")
            time.sleep(0.05)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="actum-color-triggers",
            daemon=True,
        )
        self._thread.start()
        print("[color_trigger] watcher started")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    @classmethod
    def from_agent_config(
        cls,
        agent_config: dict[str, Any],
        frame_reader: FrameReader,
        backend: RobotBackend | None = None,
        config_path: Path | None = None,
    ) -> ColorTriggerWatcher | None:
        base_dir = config_path.parent if config_path else Path.cwd()
        trigger_data = agent_config.get("color_triggers")
        if isinstance(trigger_data, dict) and trigger_data.get("actions_file"):
            actions_file = Path(str(trigger_data["actions_file"]))
            if not actions_file.is_absolute():
                actions_file = (base_dir / actions_file).resolve()
            cfg = load_merged_config(actions_file, trigger_data, base_dir)
        else:
            cfg = ColorTriggerConfig.from_dict(
                trigger_data if isinstance(trigger_data, dict) else None,
                base_dir=base_dir,
            )
        if not cfg.enabled:
            return None
        return cls(cfg, frame_reader, backend=backend)


def load_merged_config(
    actions_path: Path,
    inline: dict[str, Any],
    base_dir: Path,
) -> ColorTriggerConfig:
    file_cfg = ColorTriggerConfig.from_dict(
        json_load(actions_path) if actions_path.exists() else {},
        base_dir=base_dir,
    )
    inline_cfg = ColorTriggerConfig.from_dict(inline, base_dir=base_dir)
    merged = ColorTriggerConfig(
        enabled=inline_cfg.enabled or file_cfg.enabled,
        calibration_path=inline_cfg.calibration_path or file_cfg.calibration_path,
        detect_every_frames=inline_cfg.detect_every_frames or file_cfg.detect_every_frames,
        cooldown_seconds=inline_cfg.cooldown_seconds or file_cfg.cooldown_seconds,
        show_debug=inline_cfg.show_debug or file_cfg.show_debug,
        actions={**file_cfg.actions, **inline_cfg.actions},
    )
    return merged


def json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
