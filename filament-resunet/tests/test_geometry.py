"""Instance geometry and metric parity tests; no local model execution."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geometry import (  # noqa: E402
    blend_window,
    crop_targets,
    input_channels,
    instances,
    polygon_bounds,
    pq_score,
    pq_totals,
    tile_starts,
)

from sfsc.evaluation import panoptic_quality  # noqa: E402


def annotation(points):
    # No bbox/spine/category metadata: all targets must use the allowed polygons alone.
    return {"segmentation": [points]}


def test_crop_halo_does_not_invent_boundary_at_patch_edge():
    ann = annotation([0, 10, 40, 10, 40, 20, 0, 20])
    targets = crop_targets([ann], 10, 0, 24)
    assert targets.shape == (24, 24, 3)
    assert targets[15, 0, 0] == 1
    assert targets[15, 0, 1] == 1
    assert targets[15, 0, 2] == 0


def test_tiny_thin_masks_retain_seed_targets():
    targets = crop_targets([annotation([2, 3, 15, 3, 15, 4, 2, 4])], 0, 0, 20)
    assert targets[..., 0].sum() == 13
    assert np.array_equal(targets[..., 0], targets[..., 1])


def test_polygon_bounds_ignore_forbidden_metadata():
    ann = annotation([2, 3, 15, 3, 15, 5, 2, 5])
    ann["bbox"] = [999, 999, 1, 1]
    assert polygon_bounds(ann) == (2, 3, 13, 2)


def test_full_resolution_tiles_cover_every_pixel_with_positive_weights():
    coverage = np.zeros((2048, 2048), np.float32)
    for y in tile_starts(2048, 768, 512):
        for x in tile_starts(2048, 768, 512):
            coverage[y : y + 768, x : x + 768] += blend_window(768)[..., 0]
    assert np.all(coverage > 0)
    assert tile_starts(20, 768, 512) == [0]


def test_touching_filaments_split_by_predicted_contours():
    probabilities = np.zeros((40, 40, 3), np.float32)
    probabilities[5:30, 8:25, 0] = 0.9
    probabilities[6:29, 9:16, 1] = 0.9
    probabilities[6:29, 18:24, 1] = 0.9
    probabilities[5:30, 16:18, 2] = 0.9
    labels = instances(probabilities, {"foreground": 0.5, "interior": 0.5, "minimum_area": 8})
    assert labels.max() == 2
    assert labels[15, 10] != labels[15, 22]
    assert np.all(labels[probabilities[..., 0] == 0] == 0)
    assert np.all(labels[probabilities[..., 0] > 0] > 0)


def test_seedless_component_and_empty_images():
    probabilities = np.zeros((20, 20, 3), np.float32)
    parameters = {"foreground": 0.5, "interior": 0.5, "minimum_area": 8}
    assert instances(probabilities, parameters).max() == 0
    probabilities[3:4, 3:18, 0] = 0.9
    labels = instances(probabilities, parameters)
    assert labels.max() == 1 and np.count_nonzero(labels) == 15


@pytest.mark.parametrize("seed", range(8))
def test_fast_pq_matches_fixed_evaluator_even_with_overlapping_truths(seed):
    rng = np.random.default_rng(seed)
    labels = np.zeros((30, 30), np.int32)
    labels[2:10, 2:10] = 1
    labels[15:25, 15:25] = 7
    truth = [labels == 1, labels == 7]
    truth.append(rng.random(labels.shape) < 0.02)
    if seed % 2:
        truth[0] = np.roll(truth[0], 4, axis=0)
    predictions = [labels == i for i in (1, 7)]
    fixed = panoptic_quality(truth, predictions)
    totals = pq_totals(truth, labels)
    assert pq_score(totals) == pytest.approx(fixed.pq)
    assert totals.tolist() == pytest.approx(
        [
            fixed.matched_iou,
            fixed.true_positives,
            fixed.false_positives,
            fixed.false_negatives,
        ]
    )


def test_fast_pq_strict_threshold_and_empty_cases():
    labels = np.array([[1, 1], [0, 0]], dtype=np.int32)
    assert pq_score(pq_totals([np.ones((2, 2), bool)], labels)) == 0
    assert pq_score(pq_totals([], np.zeros((2, 2), np.int32))) == 1


def test_input_channels_scale_dark_uint8_images_correctly():
    channels = input_channels(np.ones((20, 20), dtype=np.uint8))
    assert np.allclose(channels[..., 0], 2 / 255 - 1)
    assert channels.dtype == np.float32 and np.allclose(channels[..., 1], 0)
