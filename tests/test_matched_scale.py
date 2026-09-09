"""Behavioral tests for matched-scale detector training and selection."""

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "filament-matched-scale/core.py"
RUN_MODULE = ROOT / "filament-matched-scale/run.py"


def load_core():
    spec = importlib.util.spec_from_file_location("matched_scale_core", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_run():
    spec = importlib.util.spec_from_file_location("matched_scale_run", RUN_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_uses_four_native_crops():
    core = load_core()
    assert core.crop_origins(2048, 2048, 1280) == [
        (0, 0),
        (768, 0),
        (0, 768),
        (768, 768),
    ]


def test_crop_instance_translates_and_clips_visible_mask():
    core = load_core()
    mask = np.zeros((12, 12), bool)
    mask[4:10, 5:11] = True
    cropped = core.crop_instance(mask, 6, 3, 12, 9, minimum_pixels=4)
    expected = np.zeros((6, 6), bool)
    expected[1:6, 0:5] = True
    assert np.array_equal(cropped, expected)


def test_crop_instance_rejects_tiny_visible_fragment():
    core = load_core()
    mask = np.zeros((12, 12), bool)
    mask[5:7, 5:7] = True
    assert core.crop_instance(mask, 6, 6, 12, 12, minimum_pixels=1) is not None
    assert core.crop_instance(mask, 6, 6, 12, 12, minimum_pixels=2) is None


def test_component_filter_applies_minimum_to_each_disconnected_fragment():
    core = load_core()
    mask = np.zeros((12, 12), bool)
    mask[1:4, 1:4] = True
    mask[10, 10] = True
    components = core.connected_components(mask, minimum_pixels=4)
    assert len(components) == 1
    assert int(components[0].sum()) == 9


def test_selection_uses_calibration_pq_then_false_positives_then_threshold():
    core = load_core()
    rows = [
        {"policy": "crop", "threshold": 0.2, "pq": 0.41, "fp": 20},
        {"policy": "crop", "threshold": 0.3, "pq": 0.42, "fp": 21},
        {"policy": "crop", "threshold": 0.4, "pq": 0.42, "fp": 19},
        {"policy": "crop", "threshold": 0.5, "pq": 0.42, "fp": 19},
    ]
    assert core.select_configuration(rows)["threshold"] == 0.5


def test_selection_prefers_immutable_control_on_an_exact_tie():
    core = load_core()
    rows = [
        {"policy": "crop", "threshold": 0.5, "pq": 0.42, "fp": 19},
        {"policy": "control", "threshold": 0.5, "pq": 0.42, "fp": 19},
    ]
    assert core.select_configuration(rows)["policy"] == "control"


def test_additive_reconciliation_preserves_control_and_deduplicates_additions():
    core = load_core()
    base = np.zeros((16, 16), bool)
    base[1:5, 1:5] = True
    duplicate = base.copy()
    distinct = np.zeros_like(base)
    distinct[10:14, 10:14] = True
    masks = core.additive_reconcile(
        [base],
        [
            {"mask": duplicate, "score": 0.9},
            {"mask": distinct, "score": 0.8},
        ],
        minimum_area=4,
    )
    assert len(masks) == 2
    assert np.array_equal(masks[0], base)
    assert np.array_equal(masks[1], distinct)


def test_additive_reconciliation_uses_stricter_cross_scale_duplicate_iou():
    core = load_core()
    base = np.zeros((16, 16), bool)
    base[2:10, 2:10] = True
    shifted_duplicate = np.zeros_like(base)
    shifted_duplicate[5:13, 5:13] = True
    masks = core.additive_reconcile(
        [base],
        [{"mask": shifted_duplicate, "score": 0.9}],
        minimum_area=4,
        duplicate_iou=0.7,
        base_duplicate_iou=0.2,
    )
    assert len(masks) == 1
    assert np.array_equal(masks[0], base)


def test_additions_use_their_own_duplicate_threshold_after_acceptance():
    core = load_core()
    first = np.zeros((20, 20), bool)
    first[2:10, 2:10] = True
    second = np.zeros_like(first)
    second[5:13, 5:13] = True
    masks = core.additive_reconcile(
        [],
        [{"mask": first, "score": 0.9}, {"mask": second, "score": 0.8}],
        minimum_area=4,
        duplicate_iou=0.7,
        base_duplicate_iou=0.2,
    )
    assert len(masks) == 2


def test_controller_source_bundle_is_complete_and_rejects_staleness():
    run = load_run()
    bundle = run.source_bundle()
    assert set(bundle) == {
        "matched_core.py",
        "remote.py",
        "recovery_core.py",
        "metric_core.py",
        "legacy_runtime.py",
    }
    current = run.source_digest(bundle)
    run.assert_source_fresh({"source_sha256": current})
    with np.testing.assert_raises_regex(ValueError, "Prepared source is stale"):
        run.assert_source_fresh({"source_sha256": "0" * 64})
