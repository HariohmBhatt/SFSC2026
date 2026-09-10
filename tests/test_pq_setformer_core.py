"""Behavioral tests for PQ-SetFormer's CPU experiment seam."""

import ast
import base64
import importlib.util
import json
import sys
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "filament-pq-setformer"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


config = load_module("pq_setformer_config_core_test", "config.py")
core = load_module("pq_setformer_core", "core.py")


def frozen_train_names() -> list[str]:
    tree = ast.parse((ROOT / "kaggle/yolo-refiner/train_validation.py").read_text())
    encoded = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "SPLIT_B64" for target in node.targets
        )
    )
    return json.loads(zlib.decompress(base64.b64decode(encoded)))["train"]


def test_two_blocked_pilots_are_frozen_purged_and_assessment_disjoint() -> None:
    names = frozen_train_names()
    first = core.make_experiment_split(names, pilot=1)
    second = core.make_experiment_split(names, pilot=2)

    assert {key: len(value) for key, value in first.members.items()} == {
        "fitting": 289,
        "quality_fit": 69,
        "quality_calibration": 34,
        "assessment": 87,
        "purged": 16,
    }
    assert {key: len(value) for key, value in second.members.items()} == {
        "fitting": 313,
        "quality_fit": 50,
        "quality_calibration": 33,
        "assessment": 88,
        "purged": 11,
    }
    assert set(first.assessment_blocks).isdisjoint(second.assessment_blocks)
    assert first.sha256 != second.sha256

    days = {name: datetime.strptime(name[:14], "%Y%m%d%H%M%S").date().toordinal() for name in names}
    for split in (first, second):
        held_back = split.quality_fit + split.quality_calibration + split.assessment
        assert min(abs(days[a] - days[b]) for a in split.fitting for b in held_back) > 7
        assert (
            min(
                abs(days[a] - days[b]) for a in split.quality_fit for b in split.quality_calibration
            )
            > 7
        )


def test_official_match_targets_make_a_losing_duplicate_negative() -> None:
    truth = np.zeros((8, 8), bool)
    truth[1:5, 1:5] = True
    candidates = [
        core.Candidate("q0", truth, 0.9, 0.95),
        core.Candidate("q1", truth, 0.8, 0.95),
    ]

    targets = core.official_match_targets(candidates, [[truth]])

    assert targets.match_probability.tolist() == [1.0, 0.0]
    assert targets.matched_iou.tolist() == [1.0, 0.0]


def test_calibrator_learns_match_probability_and_round_trips() -> None:
    caches = {}
    truths = {}
    for index, presence in enumerate((0.05, 0.10, 0.20, 0.80, 0.90, 0.95)):
        mask = np.zeros((8, 8), bool)
        mask[1:5, 1:5] = True
        candidate = core.Candidate(f"q{index}", mask, presence, 0.8)
        name = f"20110{index + 1}01000000Bh.jpeg"
        caches[name] = [candidate]
        truths[name] = [[mask]] if presence > 0.5 else [[np.roll(mask, 3, axis=0)]]

    fitted = core.fit_calibrator(caches, truths, regularization=0.01)
    restored = core.Calibrator.from_dict(fitted.to_dict())

    low_name = next(iter(caches))
    low = restored.predict(caches[low_name], core.candidate_features(caches[low_name]))
    high_name = list(caches)[-1]
    high = restored.predict(caches[high_name], core.candidate_features(caches[high_name]))
    assert fitted.sha256 == restored.sha256
    assert high[0] > low[0]


def test_assembly_preserves_disconnected_identity_and_assigns_each_pixel_once() -> None:
    first = np.zeros((8, 8), bool)
    first[1:3, 1:3] = True
    first[5:7, 1:3] = True
    second = np.zeros((8, 8), bool)
    second[2:6, 2:5] = True
    first_probability = np.where(first, 0.9, 0.01).astype(np.float32)
    second_probability = np.where(second, 0.8, 0.01).astype(np.float32)
    candidates = [
        core.Candidate("q0", first, 0.9, 0.9, first_probability),
        core.Candidate("q1", second, 0.8, 0.8, second_probability),
    ]
    recipe = config.AssemblyConfig(
        utility_threshold=0.0,
        minimum_area=0,
        duplicate_iou=0.7,
        containment=0.95,
    )

    masks = core.assemble_instances(candidates, recipe=recipe)

    assert len(masks) == 2
    assert sum(core.component_count(mask) for mask in masks) >= 3
    assert not np.logical_and(masks[0], masks[1]).any()
    assert np.logical_or.reduce(masks).sum() == np.logical_or(first, second).sum()


def test_evaluation_aggregates_every_annotation_record_globally() -> None:
    mask = np.zeros((4, 4), bool)
    mask[0:2, 0:2] = True
    other = np.zeros_like(mask)
    other[2:4, 2:4] = True

    report = core.evaluate_run(
        {"image.jpeg": [mask]},
        {"image.jpeg": [[mask], [mask, other]]},
    )

    assert report["pq"] == pytest.approx(0.8)
    assert report["tp"] == 2
    assert report["fn"] == 1
    assert report["fp"] == 0


def test_rle_round_trip_rejects_empty_masks() -> None:
    mask = np.eye(8, dtype=bool)
    assert np.array_equal(core.decode_rle(core.encode_rle(mask)), mask)
    with pytest.raises(ValueError, match="empty"):
        core.encode_rle(np.zeros((8, 8), bool))
