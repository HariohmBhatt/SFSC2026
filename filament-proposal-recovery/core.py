"""Pure geometry and cross-view reconciliation for sliced proposal recovery."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def crop_origins(height: int, width: int, window: int) -> list[tuple[int, int]]:
    """Return four corner-aligned native-scale views for a square-like image."""
    if height < window or width < window or window <= 0:
        raise ValueError("Crop window must fit inside the image")
    xs = [0, width - window]
    ys = [0, height - window]
    return [(x, y) for y in ys for x in xs]


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.logical_and(first, second).sum())
    union = int(np.logical_or(first, second).sum())
    return intersection / union if union else 0.0


def reconcile(
    candidates: Sequence[Mapping],
    minimum_area: int,
    duplicate_iou: float = 0.6,
) -> tuple[list[np.ndarray], list[dict]]:
    """Keep the control masks intact, then add nonduplicate crop masks."""
    if minimum_area <= 0 or not 0 < duplicate_iou < 1:
        raise ValueError("Invalid reconciliation configuration")
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: (
            item[1]["source"] != "whole",
            -float(item[1]["score"]),
            item[0],
        ),
    )
    accepted: list[tuple[Mapping, np.ndarray]] = []
    for _, candidate in ranked:
        mask = np.asarray(candidate["mask"], dtype=bool)
        if int(mask.sum()) < minimum_area:
            continue
        duplicate = any(
            (candidate["source"] != previous["source"])
            and mask_iou(mask, previous_mask) >= duplicate_iou
            for previous, previous_mask in accepted
        )
        if not duplicate:
            accepted.append((candidate, mask))
    if not accepted:
        return [], []
    occupied = np.zeros_like(accepted[0][1], bool)
    masks, provenance = [], []
    for candidate, mask in accepted:
        clean = mask & ~occupied
        if int(clean.sum()) < minimum_area:
            continue
        masks.append(clean)
        provenance.append(
            {key: value for key, value in candidate.items() if key not in {"mask", "seed"}}
        )
        occupied |= clean
    return masks, provenance
