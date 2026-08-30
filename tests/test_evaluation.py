import numpy as np
import pytest

from sfsc.evaluation import PQResult, panoptic_quality


def mask(*indices: int, size: int = 8) -> np.ndarray:
    result = np.zeros(size, dtype=bool)
    result[list(indices)] = True
    return result.reshape(2, size // 2)


def test_perfect_prediction_scores_one() -> None:
    result = panoptic_quality([mask(0, 1)], [mask(0, 1)])

    assert result == PQResult(1.0, 1, 0, 0)
    assert result.pq == 1.0
    assert result.segmentation_quality == 1.0
    assert result.recognition_quality == 1.0


def test_iou_equal_to_threshold_is_not_a_match() -> None:
    result = panoptic_quality([mask(0, 1)], [mask(0, 1, 2, 3)])

    assert result == PQResult(0.0, 0, 1, 1)
    assert result.pq == 0.0


def test_false_positive_and_false_negative_are_penalized() -> None:
    truth = [mask(0, 1), mask(6, 7)]
    prediction = [mask(0, 1), mask(3)]

    result = panoptic_quality(truth, prediction)

    assert result == PQResult(1.0, 1, 1, 1)
    assert result.pq == 0.5


def test_global_totals_are_added_before_computing_pq() -> None:
    perfect = panoptic_quality([mask(0)], [mask(0)])
    missed = panoptic_quality([mask(1), mask(2), mask(3)], [])

    aggregate = perfect + missed

    assert aggregate.pq == pytest.approx(1 / 2.5)
    assert aggregate.pq != pytest.approx((perfect.pq + missed.pq) / 2)


def test_empty_image_record_scores_one() -> None:
    assert panoptic_quality([], []).pq == 1.0


def test_empty_instance_mask_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        panoptic_quality([np.zeros((2, 2), dtype=bool)], [])
