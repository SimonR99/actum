"""CLI entry point for live color-group trigger demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Color group trigger demo")
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "config" / "color_triggers.json",
        help="Trigger config (groups -> actions)",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Override calibration file (defaults to config value)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show debug overlay window",
    )
    parser.add_argument(
        "--no-robot",
        action="store_true",
        help="Do not connect a robot backend (log actions only)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    try:
        import cv2
    except ImportError:
        raise SystemExit("OpenCV required: pip install -e '.[camera]'") from None

    from actum.backends.factory import create_backend
    from actum.color_triggers.actions import load_trigger_config
    from actum.color_triggers.detector import BandDetector
    from actum.color_triggers.watcher import ColorTriggerWatcher
    from actum.perception import open_camera

    cfg = load_trigger_config(args.config)
    cfg.enabled = True
    if args.calibration is not None:
        cfg.calibration_path = str(args.calibration.resolve())

    backend = None
    if not args.no_robot:
        try:
            config_path = _project_root() / "config.json"
            agent_config = json.loads(config_path.read_text()) if config_path.exists() else {}
            backend = create_backend(agent_config)
            if backend.connect():
                print(f"[robot] connected: {backend.name}")
            else:
                print(f"[robot] unavailable: {backend.name}")
                backend = None
        except Exception as exc:
            print(f"[robot] skipped ({exc})")

    cap = open_camera(args.camera)
    if cap is None:
        raise SystemExit("Could not open camera")

    detector = BandDetector(calibration_path=cfg.calibration_path)

    def read_frame():
        ok, frame = cap.read()
        return frame if ok else None

    watcher = ColorTriggerWatcher(cfg, read_frame, backend=backend)
    window = "Color Triggers"
    if args.show:
        cv2.namedWindow(window)

    print("Watching for color groups. Press q to quit.")
    try:
        while True:
            frame = read_frame()
            if frame is None:
                break
            result = watcher.process_frame(frame)
            if args.show and result is not None:
                debug = detector.draw_debug(frame, result)
                cv2.imshow(window, debug)
            key = cv2.waitKey(1) & 0xFF if args.show else 255
            if key == ord("q"):
                break
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    main()
