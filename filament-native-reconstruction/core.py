"""Pure geometry and label-association primitives for native mask reconstruction."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

WINDOW = 512
STRIDE = 384


def axis_starts(length: int, window: int = WINDOW, stride: int = STRIDE) -> list[int]:
    """Return deterministic starts that cover an axis and anchor its final edge."""
    if length <= 0:
        raise ValueError("Axis length must be positive")
    if window <= 0 or stride <= 0 or stride > window:
        raise ValueError("Window and stride must be positive, with stride <= window")
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    final = length - window
    if starts[-1] != final:
        starts.append(final)
    return starts


def roi_windows(
    bounds: tuple[int, int, int, int], window: int = WINDOW, stride: int = STRIDE
) -> list[tuple[int, int, int, int]]:
    """Cover an ROI in image coordinates without uncovered strips."""
    x0, y0, x1, y1 = bounds
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        raise ValueError("ROI bounds must have positive area")
    result = []
    for dy in axis_starts(height, window, stride):
        for dx in axis_starts(width, window, stride):
            result.append((x0 + dx, y0 + dy, min(x0 + dx + window, x1), min(y0 + dy + window, y1)))
    return result


def expanded_square(
    mask: np.ndarray, context: float = 1.6, minimum: int = 128
) -> tuple[int, int, int, int]:
    """Return a centered square ROI shifted inside the image at boundaries."""
    if mask.ndim != 2:
        raise ValueError("Mask must be two-dimensional")
    height, width = mask.shape
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("Cannot construct an ROI from an empty prompt")
    center_x = (int(xs.min()) + int(xs.max()) + 1) / 2
    center_y = (int(ys.min()) + int(ys.max()) + 1) / 2
    object_side = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    side = min(max(int(math.ceil(object_side * context)), minimum), min(height, width))
    x0, y0 = int(round(center_x - side / 2)), int(round(center_y - side / 2))
    x0 = min(max(x0, 0), width - side)
    y0 = min(max(y0, 0), height - side)
    return x0, y0, x0 + side, y0 + side


def taper(window: int = WINDOW, floor: float = 0.05) -> np.ndarray:
    """Positive separable cosine weights for overlap blending."""
    if window <= 0 or not 0 < floor <= 1:
        raise ValueError("Invalid taper configuration")
    axis = np.hanning(window).astype(np.float32)
    axis = floor + (1 - floor) * axis
    return np.outer(axis, axis).astype(np.float32)


def blend_tiles(
    shape: tuple[int, int],
    tiles: Sequence[tuple[np.ndarray, int, int]],
    floor: float = 0.05,
) -> np.ndarray:
    """Blend probability tiles in ROI coordinates before thresholding."""
    height, width = shape
    if height <= 0 or width <= 0 or not tiles:
        raise ValueError("A positive shape and at least one tile are required")
    values = np.zeros(shape, np.float32)
    weights = np.zeros(shape, np.float32)
    cached_weights: dict[int, np.ndarray] = {}
    for tile, x0, y0 in tiles:
        if tile.ndim != 2:
            raise ValueError("Each probability tile must be two-dimensional")
        tile_height = min(tile.shape[0], height - y0)
        tile_width = min(tile.shape[1], width - x0)
        if x0 < 0 or y0 < 0 or tile_height <= 0 or tile_width <= 0:
            raise ValueError("Tile lies outside the reconstruction canvas")
        size = tile.shape[0]
        if tile.shape[1] != size:
            raise ValueError("Tiles must be square")
        weight = cached_weights.setdefault(size, taper(size, floor))[:tile_height, :tile_width]
        values[y0 : y0 + tile_height, x0 : x0 + tile_width] += (
            tile[:tile_height, :tile_width] * weight
        )
        weights[y0 : y0 + tile_height, x0 : x0 + tile_width] += weight
    if np.any(weights == 0):
        raise ValueError("Tiles do not cover the requested shape")
    return values / weights


def pad_to_window(
    array: np.ndarray, window: int = WINDOW, mode: str = "constant"
) -> tuple[np.ndarray, np.ndarray]:
    """Pad the bottom/right of a 2-D ROI and return its valid-pixel mask."""
    if array.ndim != 2 or array.shape[0] > window or array.shape[1] > window:
        raise ValueError("Expected a two-dimensional array no larger than the window")
    valid = np.zeros((window, window), bool)
    valid[: array.shape[0], : array.shape[1]] = True
    pad = ((0, window - array.shape[0]), (0, window - array.shape[1]))
    if mode == "reflect" and min(array.shape) == 1:
        mode = "edge"
    padded = np.pad(array, pad, mode=mode)
    return padded, valid


def associate_prompt(
    seed: np.ndarray,
    selected_truths: Sequence[np.ndarray],
    all_truth_sets: Sequence[Sequence[np.ndarray]],
    minimum_purity: float = 0.5,
    maximum_other_coverage: float = 0.1,
) -> dict[str, int | str | None]:
    """Classify a prompt as one recoverable instance, hard negative, or ambiguous."""
    if seed.ndim != 2 or not seed.any():
        return {"status": "ambiguous", "target_index": None}
    seed = seed.astype(bool)
    intersections = np.array([np.logical_and(seed, truth).sum() for truth in selected_truths])
    if len(intersections) and intersections.max() > 0:
        target_index = int(intersections.argmax())
        purity = float(intersections[target_index] / seed.sum())
        other_coverage = [
            float(intersections[index] / max(truth.sum(), 1))
            for index, truth in enumerate(selected_truths)
            if index != target_index
        ]
        if purity >= minimum_purity and max(other_coverage, default=0.0) < maximum_other_coverage:
            return {"status": "positive", "target_index": target_index}
        return {"status": "ambiguous", "target_index": None}
    overlaps_any_record = any(
        np.logical_and(seed, truth).any() for truths in all_truth_sets for truth in truths
    )
    return {
        "status": "ambiguous" if overlaps_any_record else "negative",
        "target_index": None,
    }
