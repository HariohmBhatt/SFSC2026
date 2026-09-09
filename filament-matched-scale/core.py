"""Pure geometry and selection logic for matched-scale detector training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def crop_origins(height: int, width: int, window: int) -> list[tuple[int, int]]:
    if window <= 0 or height < window or width < window:
        raise ValueError("Crop window must fit inside the image")
    return [(x, y) for y in (0, height - window) for x in (0, width - window)]


def training_views(height: int, width: int, window: int) -> list[tuple[str, int, int, int, int]]:
    result = [("whole", 0, 0, width, height)]
    result.extend(
        (f"crop:{x0}:{y0}", x0, y0, x0 + window, y0 + window)
        for x0, y0 in crop_origins(height, width, window)
    )
    return result


def crop_instance(
    mask: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    minimum_pixels: int = 64,
) -> np.ndarray | None:
    if mask.ndim != 2 or not (0 <= x0 < x1 <= mask.shape[1]) or not (0 <= y0 < y1 <= mask.shape[0]):
        raise ValueError("Invalid mask or crop bounds")
    visible = np.asarray(mask[y0:y1, x0:x1], dtype=bool)
    return visible.copy() if int(visible.sum()) >= minimum_pixels else None


def select_configuration(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        raise ValueError("No configurations to select")
    return max(
        rows,
        key=lambda row: (
            float(row["pq"]),
            -int(row["fp"]),
            float(row["threshold"]),
        ),
    )
