"""Regression tests for the precision experiment's instance decisions."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sfsc.evaluation import panoptic_quality

path = Path(__file__).resolve().parents[1] / "filament-precision/geometry.py"
spec = importlib.util.spec_from_file_location("precision_geometry", path)
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


def parameters(**overrides):
    return (
        dict(
            mode="hysteresis",
            foreground=0.4,
            confidence=0.8,
            interior=0.5,
            minimum_area=8,
            minimum_seed_area=8,
        )
        | overrides
    )


def test_weak_object_rejected_but_faint_connected_tail_retained():
    p = np.zeros((32, 32, 3), np.float32)
    p[3:6, 2:25, 0] = 0.5
    p[3:6, 2:5, 0] = 0.9
    p[15:18, 2:25, 0] = 0.5
    result = geometry.instances(p, parameters())
    assert result.max() == 1
    assert np.all(result[3:6, 2:25] == 1)
    assert not result[15:18].any()


def test_multiple_confident_patches_do_not_fragment_one_filament():
    p = np.zeros((24, 40, 3), np.float32)
    p[8:12, 2:35, 0] = 0.5
    for left in [3, 20]:
        p[8:12, left : left + 4, :2] = 0.9
    result = geometry.instances(p, parameters())
    assert result.max() == 1
    assert np.count_nonzero(result) == 132


def test_restrained_watershed_can_still_separate_touching_objects():
    p = np.zeros((40, 40, 3), np.float32)
    p[5:30, 8:25, 0] = 0.9
    p[6:29, 9:16, 1] = 0.9
    p[6:29, 18:24, 1] = 0.9
    p[5:30, 16:18, 2] = 0.9
    labels = geometry.instances(p, parameters(mode="watershed"))
    assert labels.max() == 2
    assert labels[15, 10] != labels[15, 22]


def test_filtered_labels_are_compact_and_match_reference_pq():
    p = np.zeros((32, 32, 3), np.float32)
    p[2:6, 2:8, 0] = 0.9
    p[10:12, 10:12, 0] = 0.9
    p[20:25, 20:25, 0] = 0.9
    labels = geometry.instances(p, parameters())
    assert set(np.unique(labels)) == {0, 1, 2}
    truth = [labels == 1, np.roll(labels == 2, 1, axis=0)]
    reference = panoptic_quality(truth, [labels == 1, labels == 2])
    assert geometry.pq_score(geometry.pq_totals(truth, labels)) == pytest.approx(reference.pq)


def test_legacy_recipe_is_exactly_preserved():
    original_path = path.parent.parent / "filament-resunet/geometry.py"
    original_spec = importlib.util.spec_from_file_location("original_geometry", original_path)
    original = importlib.util.module_from_spec(original_spec)
    original_spec.loader.exec_module(original)
    rng = np.random.default_rng(2026)
    p = rng.random((80, 80, 3), dtype=np.float32)
    recipe = {"mode": "legacy", "foreground": 0.65, "interior": 0.5, "minimum_area": 64}
    assert np.array_equal(geometry.instances(p, recipe), original.instances(p, recipe))
