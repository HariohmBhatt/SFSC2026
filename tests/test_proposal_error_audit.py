"""Behavioral tests for proposal-stage error attribution."""

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "filament-proposal-recovery/error_audit.py"
spec = importlib.util.spec_from_file_location("proposal_error_audit", MODULE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
QUALITY_MODULE = ROOT / "filament-proposal-recovery/quality_experiment.py"
quality_spec = importlib.util.spec_from_file_location("proposal_quality", QUALITY_MODULE)
quality = importlib.util.module_from_spec(quality_spec)
quality_spec.loader.exec_module(quality)
RECOVERY_MODULE = ROOT / "filament-proposal-recovery/core.py"
recovery_spec = importlib.util.spec_from_file_location("proposal_recovery_core", RECOVERY_MODULE)
recovery = importlib.util.module_from_spec(recovery_spec)
recovery_spec.loader.exec_module(recovery)


def rectangle(y0, y1, x0, x1, shape=(32, 32)):
    mask = np.zeros(shape, bool)
    mask[y0:y1, x0:x1] = True
    return mask


def candidate(seed, refined, score=0.9):
    return {"seed": seed, "refined": refined, "score": score}


def test_classifies_missing_proposal_and_refinement_damage():
    first = rectangle(2, 10, 2, 10)
    second = rectangle(20, 28, 20, 28)
    damaged_seed = rectangle(2, 10, 2, 10)
    damaged_refined = rectangle(2, 6, 2, 6)
    rows = audit.classify_record(
        [first, second],
        [candidate(damaged_seed, damaged_refined)],
        score_threshold=0.3,
        minimum_area=4,
        minimum_prompt_intersection=4,
    )
    assert [row["stage"] for row in rows] == ["refinement_damage", "no_detector_proposal"]


def test_separates_score_rejection_from_overlap_painting_loss():
    rejected_truth = rectangle(2, 10, 2, 10)
    painted_truth = rectangle(16, 24, 16, 24)
    rejected = candidate(rejected_truth, rejected_truth, score=0.1)
    covering = rectangle(14, 26, 14, 26)
    painted = candidate(painted_truth, painted_truth, score=0.8)
    occluder = candidate(covering, covering, score=0.9)
    rows = audit.classify_record(
        [rejected_truth, painted_truth],
        [rejected, occluder, painted],
        score_threshold=0.3,
        minimum_area=4,
        minimum_prompt_intersection=4,
    )
    assert rows[0]["stage"] == "score_rejection"
    assert rows[1]["stage"] in {"identity_collision", "assembly_loss"}


def test_viable_partial_prompt_is_mask_quality_failure():
    truth = rectangle(4, 20, 4, 20)
    partial = rectangle(4, 12, 4, 12)
    rows = audit.classify_record(
        [truth],
        [candidate(partial, partial)],
        score_threshold=0.3,
        minimum_area=4,
        minimum_prompt_intersection=4,
    )
    assert rows[0]["stage"] == "insufficient_mask_quality"


def test_one_to_one_matching_prioritizes_number_of_matches():
    iou = np.array([[0.99, 0.51], [0.51, 0.0]])
    matches = audit.one_to_one_matches(iou, threshold=0.5)
    assert matches == {0: 1, 1: 0}


def test_assignment_targets_keep_one_alternative_and_remove_duplicate_credit():
    first = rectangle(2, 10, 2, 10)
    second = rectangle(18, 26, 18, 26)
    items = [
        {"proposal_id": 0, "kind": "seed", "mask": rectangle(2, 9, 2, 9)},
        {"proposal_id": 0, "kind": "refined", "mask": first},
        {"proposal_id": 1, "kind": "refined", "mask": first},
        {"proposal_id": 2, "kind": "refined", "mask": second},
    ]
    targets = quality.assignment_targets(items, [[first, second]])
    assert targets[1] == 1.0
    assert targets[3] == 1.0
    assert targets[0] == targets[2] == 0.0


def test_four_native_crop_views_cover_the_image_and_touch_expected_borders():
    assert recovery.crop_origins(2048, 2048, 1280) == [
        (0, 0),
        (768, 0),
        (0, 768),
        (768, 768),
    ]


def test_cross_view_reconciliation_removes_duplicate_but_keeps_distinct_masks():
    first = rectangle(2, 12, 2, 12)
    duplicate = rectangle(2, 12, 2, 12)
    distinct = rectangle(18, 28, 18, 28)
    rows = [
        {"mask": first, "score": 0.8, "source": "whole"},
        {"mask": duplicate, "score": 0.7, "source": "crop:0:0"},
        {"mask": distinct, "score": 0.6, "source": "crop:768:768"},
    ]
    masks, provenance = recovery.reconcile(rows, minimum_area=4, duplicate_iou=0.6)
    assert len(masks) == 2
    assert [row["source"] for row in provenance] == ["whole", "crop:768:768"]
