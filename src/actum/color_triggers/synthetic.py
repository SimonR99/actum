"""Generate synthetic colored tape tags for tests."""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import cv2
import numpy as np


def generate_tag_image(
    width: int = 640,
    height: int = 480,
    tag_height: int = 14,
    color_width_range: Tuple[int, int] = (14, 22),
    gap_max_px: int = 3,
    seed: Optional[int] = None,
    color_order: Optional[List[str]] = None,
    color_bgr: Optional[dict[str, Tuple[int, int, int]]] = None,
    angle_deg: float = 0.0,
    tag_center: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, dict]:
    rng = random.Random(seed)
    palette = color_bgr or {
        "pink": (172, 91, 242),
        "blue": (207, 119, 31),
        "orange": (117, 139, 252),
        "purple": (171, 59, 164),
    }
    order = color_order or ["purple", "blue", "pink"]
    frame = np.full((height, width, 3), 240, dtype=np.uint8)

    cx, cy = tag_center or (width // 2, height // 2)
    segments: List[Tuple[str, int]] = []
    total_w = 0
    for name in order:
        seg_w = rng.randint(*color_width_range)
        segments.append((name, seg_w))
        total_w += seg_w
        if gap_max_px > 0 and name != order[-1]:
            total_w += rng.randint(0, gap_max_px)

    tag = np.full((tag_height, total_w, 3), 240, dtype=np.uint8)
    x = 0
    for name, seg_w in segments:
        bgr = palette.get(name, (128, 128, 128))
        tag[:, x : x + seg_w] = bgr
        x += seg_w
        if gap_max_px > 0 and x < total_w:
            gap = min(rng.randint(0, gap_max_px), total_w - x)
            x += gap

    matrix = cv2.getRotationMatrix2D((total_w / 2, tag_height / 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(
        tag,
        matrix,
        (total_w, tag_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(240, 240, 240),
    )
    rh, rw = rotated.shape[:2]
    x0 = max(0, cx - rw // 2)
    y0 = max(0, cy - rh // 2)
    x1 = min(width, x0 + rw)
    y1 = min(height, y0 + rh)
    crop = rotated[: y1 - y0, : x1 - x0]
    frame[y0:y1, x0:x1] = crop
    return frame, {"color_order": order, "tag_center": (cx, cy), "angle_deg": angle_deg}
