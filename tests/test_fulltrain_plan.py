"""Protect training lineage, the held-out split, and the frozen LR schedule."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from pycocotools import mask as coco_mask

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "fulltrain_plan", ROOT / "kaggle/precision-fulltrain/plan.py"
)
plan_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan_module)


def recipes():
    baseline = {
        "best_epoch": 60,
        "config": json.loads((ROOT / "filament-resunet/config.json").read_text()),
    }
    precision = {
        "best_epoch": 15,
        "config": json.loads((ROOT / "filament-precision/config.json").read_text()),
        "postprocess": {"foreground": 0.55, "confidence": 0.75, "minimum_area": 256},
    }
    return precision, baseline


def test_selected_epoch_preserves_original_schedule_and_frozen_thresholds():
    precision, baseline = recipes()
    plan = plan_module.training_plan(precision, baseline, list(range(601)), [999])
    assert plan["optimizer_steps"] == 3000
    assert plan["schedule_steps"] == 8000
    assert plan["postprocess"] == precision["postprocess"]


def test_cannot_silently_train_from_incomplete_baseline():
    precision, baseline = recipes()
    baseline["best_epoch"] = 15
    with pytest.raises(ValueError, match="60-epoch"):
        plan_module.training_plan(precision, baseline, list(range(601)), [999])


def test_internal_holdout_cannot_be_mixed_into_training():
    precision, baseline = recipes()
    with pytest.raises(ValueError, match="holdout"):
        plan_module.training_plan(precision, baseline, list(range(601)), [600])


def test_duplicate_sources_do_not_count_as_additional_training_images():
    precision, baseline = recipes()
    with pytest.raises(ValueError, match="unique"):
        plan_module.training_plan(precision, baseline, [*range(600), 0], [999])


def verifier():
    file = ROOT / "kaggle/precision-fulltrain/finish.py"
    loaded = importlib.util.spec_from_file_location("fulltrain_finish", file)
    module = importlib.util.module_from_spec(loaded)
    loaded.loader.exec_module(module)
    return module.verify_csv


def write_submission(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["filament_id", "segmentation_rle"])
        writer.writerows(rows)


def test_preupload_rle_verification_accepts_valid_masks_and_empty_image_counts(tmp_path):
    mask = np.zeros((2048, 2048), np.uint8, order="F")
    mask[10:20, 10:20] = 1
    rle = coco_mask.encode(mask)["counts"].decode()
    path = tmp_path / "submission.csv"
    write_submission(path, [("solar_1", rle)])
    assert verifier()(path, {"solar.jpeg": 1, "empty.jpeg": 0}) == 1


def test_preupload_rle_verification_rejects_overlapping_instances(tmp_path):
    mask = np.zeros((2048, 2048), np.uint8, order="F")
    mask[10:20, 10:20] = 1
    rle = coco_mask.encode(mask)["counts"].decode()
    path = tmp_path / "submission.csv"
    write_submission(path, [("solar_1", rle), ("solar_2", rle)])
    with pytest.raises(ValueError, match="overlapping"):
        verifier()(path, {"solar.jpeg": 2})
