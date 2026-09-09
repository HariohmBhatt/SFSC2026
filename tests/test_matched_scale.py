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


def test_training_views_include_whole_image_and_four_native_crops():
    core = load_core()
    assert core.training_views(2048, 2048, 1280) == [
        ("whole", 0, 0, 2048, 2048),
        ("crop:0:0", 0, 0, 1280, 1280),
        ("crop:768:0", 768, 0, 2048, 1280),
        ("crop:0:768", 0, 768, 1280, 2048),
        ("crop:768:768", 768, 768, 2048, 2048),
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


def test_selection_uses_calibration_pq_then_false_positives_then_threshold():
    core = load_core()
    rows = [
        {"threshold": 0.2, "pq": 0.41, "fp": 20},
        {"threshold": 0.3, "pq": 0.42, "fp": 21},
        {"threshold": 0.4, "pq": 0.42, "fp": 19},
        {"threshold": 0.5, "pq": 0.42, "fp": 19},
    ]
    assert core.select_configuration(rows)["threshold"] == 0.5


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
