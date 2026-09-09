"""Fast retrospective probe for crop-score calibration failure."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CROP_CACHE = next(
    (ROOT / "outputs/sliced-recovery-20260909-063156/results").rglob(
        "sliced-validation-candidates.json"
    )
)
WHOLE_CACHE = (
    ROOT / "outputs/yolo-refiner-validation/sfsc-yolo-refiner-validation/validation-candidates.json"
)
ANNOTATIONS = ROOT / "SFSCDataset/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
RUN_RECORD = ROOT / "reports/sliced_recovery_run.json"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = load("threshold_probe_core", HERE / "core.py")
QUALITY = load("threshold_probe_quality", HERE / "quality_experiment.py")
METRIC = load("threshold_probe_metric", ROOT / "filament-oof-consensus/core.py")


def whole_score(row) -> float:
    detector = float(row["detector_score"])
    if row.get("long_object_path", False):
        return detector
    return detector * math.sqrt(max(float(row["refiner_score"]), 0.0))


def main() -> None:
    crop_cache = json.loads(CROP_CACHE.read_text())
    whole_cache = json.loads(WHOLE_CACHE.read_text())
    config = json.loads(RUN_RECORD.read_text())["config"]
    calibration = set(config["calibration_names"])
    assessment = set(config["validation_names"]) - calibration
    policies = {
        "score-0.30": lambda row: row["score"] >= 0.3,
        "score-0.40": lambda row: row["score"] >= 0.4,
        "score-0.50": lambda row: row["score"] >= 0.5,
        "score-0.60": lambda row: row["score"] >= 0.6,
        "score-0.70": lambda row: row["score"] >= 0.7,
        "score-0.80": lambda row: row["score"] >= 0.8,
        "score-0.90": lambda row: row["score"] >= 0.9,
        "support-0.30": lambda row: row["score"] >= 0.3 and row["cross_view_iou"] >= 0.3,
        "support-0.50": lambda row: row["score"] >= 0.3 and row["cross_view_iou"] >= 0.5,
        "score-0.40-support-0.30": lambda row: row["score"] >= 0.4 and row["cross_view_iou"] >= 0.3,
        "score-0.50-support-0.30": lambda row: row["score"] >= 0.5 and row["cross_view_iou"] >= 0.3,
        "score-0.40-interior": lambda row: row["score"] >= 0.4 and not row["border_touching"],
        "score-0.40-novel": lambda row: row["score"] >= 0.4 and row["whole_iou"] < 0.2,
        "supported-or-strong": lambda row: (
            row["score"] >= 0.7 or (row["score"] >= 0.3 and row["cross_view_iou"] >= 0.5)
        ),
    }
    index = QUALITY.CocoIndex(ANNOTATIONS)
    stats = {name: {} for name in policies}
    baseline_stats = {}
    for number, filename in enumerate(config["validation_names"], 1):
        whole = [
            {
                "mask": METRIC.decode(row["refined"]),
                "score": whole_score(row),
                "source": "whole",
            }
            for row in whole_cache[filename]
            if whole_score(row) >= 0.3 and int(row["refined_area"]) >= 64
        ]
        crops = [
            {
                "mask": METRIC.decode(row["refined"]),
                "score": float(row["score"]),
                "source": row["source"],
                "border_touching": bool(row["border_touching"]),
                "rle": row["refined"],
            }
            for row in crop_cache[filename]
            if int(row["refined_area"]) >= 64
        ]
        truth = index.records(filename)
        baseline_masks, _ = CORE.reconcile(whole, minimum_area=64)
        baseline_stats[filename] = METRIC.score(truth, baseline_masks)
        crop_rles = [CORE.rle_for_decode(row["rle"]) for row in crops]
        whole_rles = [
            mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in baseline_masks
        ]
        crop_iou = (
            np.asarray(mask_utils.iou(crop_rles, crop_rles, [0] * len(crop_rles)))
            if crop_rles
            else np.zeros((0, 0))
        )
        whole_iou = (
            np.asarray(mask_utils.iou(crop_rles, whole_rles, [0] * len(whole_rles)))
            if crop_rles and whole_rles
            else np.zeros((len(crop_rles), len(whole_rles)))
        )
        for candidate_index, row in enumerate(crops):
            other_view = np.array([other["source"] != row["source"] for other in crops], dtype=bool)
            row["cross_view_iou"] = float(crop_iou[candidate_index, other_view].max(initial=0))
            row["whole_iou"] = float(whole_iou[candidate_index].max(initial=0))
        for policy_name, policy in policies.items():
            selected = [row for row in crops if policy(row)]
            masks, _ = CORE.reconcile(whole + selected, minimum_area=64)
            stats[policy_name][filename] = METRIC.score(truth, masks)
        if number % 20 == 0:
            print(f"images={number}", file=sys.stderr, flush=True)

    def aggregate(values, names):
        return METRIC.metrics(np.sum([values[name] for name in names], axis=0))

    baseline_cal = aggregate(baseline_stats, calibration)
    baseline_assess = aggregate(baseline_stats, assessment)
    grid = []
    for policy_name in policies:
        cal = aggregate(stats[policy_name], calibration)
        assess = aggregate(stats[policy_name], assessment)
        grid.append(
            {
                "policy": policy_name,
                "calibration_pq": cal["pq"],
                "calibration_delta": cal["pq"] - baseline_cal["pq"],
                "assessment_pq": assess["pq"],
                "assessment_delta": assess["pq"] - baseline_assess["pq"],
                "assessment_tp": assess["tp"],
                "assessment_fp": assess["fp"],
            }
        )
    selected = max(grid, key=lambda row: row["calibration_pq"])
    result = {
        "scope": "retrospective threshold-only diagnostic",
        "baseline_calibration": baseline_cal,
        "baseline_assessment": baseline_assess,
        "grid": grid,
        "selected_on_calibration": selected,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
