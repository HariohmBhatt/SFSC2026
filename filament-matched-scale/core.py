"""Pure geometry and selection logic for matched-scale detector training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import ndimage


def crop_origins(height: int, width: int, window: int) -> list[tuple[int, int]]:
    if window <= 0 or height < window or width < window:
        raise ValueError("Crop window must fit inside the image")
    return [(x, y) for y in (0, height - window) for x in (0, width - window)]


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


def connected_components(mask: np.ndarray, minimum_pixels: int) -> list[np.ndarray]:
    """Return 8-connected components without scanning empty space around the mask."""
    if mask.ndim != 2 or minimum_pixels <= 0:
        raise ValueError("Invalid component configuration")
    ys, xs = np.where(mask)
    if not len(xs):
        return []
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    local = np.asarray(mask[y0:y1, x0:x1], dtype=bool)
    labels, count = ndimage.label(local, structure=np.ones((3, 3), dtype=np.uint8))
    result = []
    for component_id in range(1, count + 1):
        component = labels == component_id
        if int(component.sum()) < minimum_pixels:
            continue
        full = np.zeros_like(mask, dtype=bool)
        full[y0:y1, x0:x1] = component
        result.append(full)
    return result


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.logical_and(first, second).sum())
    union = int(np.logical_or(first, second).sum())
    return intersection / union if union else 0.0


def additive_reconcile(
    base_masks: Sequence[np.ndarray],
    additions: Sequence[Mapping[str, Any]],
    minimum_area: int = 64,
    duplicate_iou: float = 0.7,
    base_duplicate_iou: float | None = None,
) -> list[np.ndarray]:
    """Preserve every base pixel, then add deduplicated masks around it."""
    base_duplicate_iou = duplicate_iou if base_duplicate_iou is None else base_duplicate_iou
    if minimum_area <= 0 or not 0 < duplicate_iou < 1 or not 0 < base_duplicate_iou < 1:
        raise ValueError("Invalid additive reconciliation configuration")
    output = [np.asarray(mask, dtype=bool).copy() for mask in base_masks]
    immutable_base = list(output)
    if output:
        occupied = np.logical_or.reduce(output)
        shape = output[0].shape
    elif additions:
        shape = np.asarray(additions[0]["mask"]).shape
        occupied = np.zeros(shape, bool)
    else:
        return []
    accepted_additions = []
    for addition in sorted(additions, key=lambda row: -float(row["score"])):
        mask = np.asarray(addition["mask"], dtype=bool)
        if mask.shape != shape or int(mask.sum()) < minimum_area:
            continue
        if any(mask_iou(mask, previous) >= base_duplicate_iou for previous in immutable_base):
            continue
        if any(mask_iou(mask, previous) >= duplicate_iou for previous in accepted_additions):
            continue
        clean = mask & ~occupied
        if int(clean.sum()) < minimum_area:
            continue
        output.append(clean)
        accepted_additions.append(mask)
        occupied |= clean
    return output


def select_configuration(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        raise ValueError("No configurations to select")
    return max(
        rows,
        key=lambda row: (
            float(row["pq"]),
            -int(row["fp"]),
            row.get("policy") == "control",
            float(row["threshold"]),
        ),
    )
