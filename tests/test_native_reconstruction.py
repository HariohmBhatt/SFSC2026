"""Behavioral tests for native ROI geometry, association, and stitching."""

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "filament-native-reconstruction/core.py"
spec = importlib.util.spec_from_file_location("native_core", MODULE)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


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
