"""Panoptic Quality evaluation for filament instance masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class PQResult:
    """Additive PQ sufficient statistics."""

    matched_iou: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def denominator(self) -> float:
        return self.true_positives + 0.5 * (self.false_positives + self.false_negatives)

    @property
    def pq(self) -> float:
        return self.matched_iou / self.denominator if self.denominator else 1.0

    @property
    def segmentation_quality(self) -> float:
        return self.matched_iou / self.true_positives if self.true_positives else 0.0

    @property
    def recognition_quality(self) -> float:
        return self.true_positives / self.denominator if self.denominator else 1.0

    def __add__(self, other: Self) -> Self:
        if not isinstance(other, PQResult):
            return NotImplemented
        return PQResult(
            matched_iou=self.matched_iou + other.matched_iou,
            true_positives=self.true_positives + other.true_positives,
            false_positives=self.false_positives + other.false_positives,
            false_negatives=self.false_negatives + other.false_negatives,
        )


def _validate_masks(masks: list[np.ndarray], label: str) -> tuple[int, ...] | None:
    shape: tuple[int, ...] | None = None
    for index, mask in enumerate(masks):
        array = np.asarray(mask)
        if array.ndim != 2:
            raise ValueError(f"{label}[{index}] must be a two-dimensional mask")
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            raise ValueError(f"All {label} masks must have the same shape")
        if not np.any(array):
            raise ValueError(f"{label}[{index}] is empty")
    return shape


def pairwise_iou(truth: list[np.ndarray], prediction: list[np.ndarray]) -> np.ndarray:
    """Return an `[n_truth, n_prediction]` IoU matrix."""
    truth_shape = _validate_masks(truth, "truth")
    prediction_shape = _validate_masks(prediction, "prediction")
    if truth_shape is not None and prediction_shape is not None and truth_shape != prediction_shape:
        raise ValueError("Truth and prediction masks must have the same shape")

    matrix = np.zeros((len(truth), len(prediction)), dtype=np.float64)
    truth_bool = [np.asarray(mask, dtype=bool) for mask in truth]
    prediction_bool = [np.asarray(mask, dtype=bool) for mask in prediction]
    truth_area = [int(mask.sum()) for mask in truth_bool]
    prediction_area = [int(mask.sum()) for mask in prediction_bool]

    for truth_index, truth_mask in enumerate(truth_bool):
        for prediction_index, prediction_mask in enumerate(prediction_bool):
            intersection = int(np.logical_and(truth_mask, prediction_mask).sum())
            union = truth_area[truth_index] + prediction_area[prediction_index] - intersection
            matrix[truth_index, prediction_index] = intersection / union
    return matrix


def panoptic_quality(
    truth: list[np.ndarray],
    prediction: list[np.ndarray],
    *,
    match_iou: float = 0.5,
) -> PQResult:
    """Compute one image-record's PQ totals using strict `IoU > match_iou`."""
    if not 0.0 <= match_iou < 1.0:
        raise ValueError("match_iou must be in [0, 1)")
    if not truth and not prediction:
        return PQResult()
    if not truth:
        _validate_masks(prediction, "prediction")
        return PQResult(false_positives=len(prediction))
    if not prediction:
        _validate_masks(truth, "truth")
        return PQResult(false_negatives=len(truth))

    ious = pairwise_iou(truth, prediction)
    eligible = np.where(ious > match_iou, ious, 0.0)
    truth_indices, prediction_indices = linear_sum_assignment(eligible, maximize=True)
    matched = [
        eligible[truth_index, prediction_index]
        for truth_index, prediction_index in zip(truth_indices, prediction_indices, strict=True)
        if eligible[truth_index, prediction_index] > 0
    ]
    true_positives = len(matched)
    return PQResult(
        matched_iou=float(sum(matched)),
        true_positives=true_positives,
        false_positives=len(prediction) - true_positives,
        false_negatives=len(truth) - true_positives,
    )
