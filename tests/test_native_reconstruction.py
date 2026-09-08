"""Behavioral tests for native ROI geometry, association, and stitching."""

import ast
import base64
import importlib.util
import json
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "filament-native-reconstruction/core.py"
spec = importlib.util.spec_from_file_location("native_core", MODULE)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
AUDIT_MODULE = ROOT / "filament-native-reconstruction/audit.py"
audit_spec = importlib.util.spec_from_file_location("native_audit", AUDIT_MODULE)
audit = importlib.util.module_from_spec(audit_spec)
audit_spec.loader.exec_module(audit)


def test_windows_cover_roi_once_at_edges_and_with_overlap_inside():
    assert native.axis_starts(500) == [0]
    assert native.axis_starts(512) == [0]
    assert native.axis_starts(513) == [0, 1]
    assert native.axis_starts(1024) == [0, 384, 512]

    windows = native.roi_windows((10, 20, 1034, 1044))
    coverage = np.zeros((1024, 1024), dtype=np.uint8)
    for x0, y0, x1, y1 in windows:
        coverage[y0 - 20 : y1 - 20, x0 - 10 : x1 - 10] += 1
    assert coverage.min() == 1
    assert coverage.max() > 1


def test_blending_constant_tiles_reconstructs_roi_without_seams():
    shape = (700, 900)
    tiles = []
    for y0 in native.axis_starts(shape[0]):
        for x0 in native.axis_starts(shape[1]):
            tiles.append((np.full((512, 512), 0.73, np.float32), x0, y0))
    reconstructed = native.blend_tiles(shape, tiles)
    assert np.allclose(reconstructed, 0.73, atol=1e-6)


def test_expanded_roi_preserves_context_at_image_boundary():
    mask = np.zeros((1000, 1000), bool)
    mask[10:110, 0:80] = True
    assert native.expanded_square(mask, context=1.6, minimum=128) == (0, 0, 160, 160)


def test_conditioning_roi_covers_seed_and_coarse_global_extent():
    seed = np.zeros((1000, 1000), bool)
    seed[400:450, 400:450] = True
    coarse = np.zeros_like(seed)
    coarse[300:650, 350:550] = True
    x0, y0, x1, y1 = native.conditioning_bounds(seed, coarse)
    assert x0 <= 350 and y0 <= 300
    assert x1 >= 550 and y1 >= 650


def test_threshold_selection_uses_calibration_pq_and_deterministic_tie_break():
    assert native.choose_threshold([(0.3, 0.4), (0.5, 0.42), (0.7, 0.41)]) == 0.5
    assert native.choose_threshold([(0.4, 0.42), (0.6, 0.42)], default=0.5) == 0.6


def test_first_blocked_pilot_split_is_frozen_and_purged():
    tree = ast.parse((ROOT / "kaggle/yolo-refiner/train_validation.py").read_text())
    encoded = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "SPLIT_B64" for target in node.targets
        )
    )
    names = json.loads(zlib.decompress(base64.b64decode(encoded)))["train"]
    split = native.make_pilot_split(names)
    assert {key: len(value) for key, value in split.items()} == {
        "fitting": 289,
        "quality_fit": 69,
        "quality_calibration": 34,
        "assessment": 87,
        "purged": 16,
    }
    days = {name: datetime.strptime(name[:14], "%Y%m%d%H%M%S").date().toordinal() for name in names}
    held_back = split["quality_fit"] + split["quality_calibration"] + split["assessment"]
    assert min(abs(days[a] - days[b]) for a in split["fitting"] for b in held_back) > 7
    assert (
        min(
            abs(days[a] - days[b])
            for a in split["quality_fit"]
            for b in split["quality_calibration"]
        )
        > 7
    )


def test_prompt_association_keeps_partial_object_and_rejects_merges():
    first = np.zeros((64, 64), bool)
    first[8:40, 8:24] = True
    second = np.zeros_like(first)
    second[8:40, 28:44] = True

    partial = np.zeros_like(first)
    partial[8:24, 8:18] = True
    match = native.associate_prompt(partial, [first, second], [[first, second]])
    assert match == {"status": "positive", "target_index": 0}

    merged = partial.copy()
    merged[8:24, 28:36] = True
    match = native.associate_prompt(merged, [first, second], [[first, second]])
    assert match == {"status": "ambiguous", "target_index": None}


def test_prompt_is_negative_only_when_every_annotation_record_has_no_overlap():
    truth = np.zeros((32, 32), bool)
    truth[4:12, 4:12] = True
    seed = np.zeros_like(truth)
    seed[20:25, 20:25] = True
    assert native.associate_prompt(seed, [truth], [[truth]])["status"] == "negative"

    alternate = seed.copy()
    assert native.associate_prompt(seed, [truth], [[truth], [alternate]])["status"] == "ambiguous"


def test_padded_crop_marks_only_real_roi_pixels_valid():
    image = np.arange(12, dtype=np.float32).reshape(3, 4)
    crop, valid = native.pad_to_window(image, window=6, mode="reflect")
    assert crop.shape == valid.shape == (6, 6)
    assert valid.sum() == 12
    assert np.array_equal(crop[:3, :4], image)
    assert not valid[3:, :].any()
    assert not valid[:, 4:].any()


def test_artifact_audit_rejects_complete_run_with_all_empty_masks():
    report = {
        "status": "complete",
        "training": {
            "nonfinite_batches": 1,
            "last_loss": 0.2,
            "minimum_loss": 0.2,
            "maximum_loss": 0.8,
        },
        "assessment": {"native": {"pq": 0.0}, "native82": {"pq": 0.0}},
    }
    candidates = {"image.jpeg": [{"native_area": 0}, {"native_area": 0}]}
    assert audit.audit_payload(report, candidates) == [
        "training recorded 1 non-finite batches",
        "all reconstructed candidate masks are empty",
        "run was marked complete despite integrity failures",
    ]


def test_artifact_audit_accepts_finite_noncollapsed_run():
    report = {
        "status": "complete",
        "training": {
            "nonfinite_batches": 0,
            "last_loss": 0.2,
            "minimum_loss": 0.2,
            "maximum_loss": 0.8,
        },
        "assessment": {"native": {"pq": 0.4}, "native82": {"pq": 0.4}},
    }
    candidates = {"image.jpeg": [{"native_area": 100}, {"native_area": 0}]}
    assert audit.audit_payload(report, candidates) == []


def test_artifact_audit_reports_malformed_numbers_instead_of_crashing():
    report = {
        "status": "complete",
        "training": {"last_loss": "not-a-number"},
        "assessment": {"native": {"pq": None}, "native82": {"pq": "bad"}},
    }
    candidates = {"image.jpeg": [{"native_area": None}, {"native_area": "12.5"}]}
    issues = audit.audit_payload(report, candidates)
    assert "training last_loss is missing or non-finite" in issues
    assert "candidate 0 native_area is invalid" in issues
    assert "candidate 1 native_area is invalid" in issues
    assert "assessment native PQ is missing or non-finite" in issues
    assert "assessment native82 PQ is missing or non-finite" in issues
    assert "run was marked complete despite integrity failures" in issues


def test_artifact_audit_requires_complete_report_metrics():
    report = {"status": "complete", "training": {}, "assessment": {}}
    candidates = {"image.jpeg": [{"native_area": 10}]}
    issues = audit.audit_payload(report, candidates)
    assert "training nonfinite_batches is missing or invalid" in issues
    assert "training last_loss is missing or non-finite" in issues
    assert "assessment native PQ is missing or non-finite" in issues
    assert "assessment native82 PQ is missing or non-finite" in issues


def test_artifact_audit_rejects_non_object_report():
    assert audit.audit_payload([], {}) == ["report payload is not an object"]
