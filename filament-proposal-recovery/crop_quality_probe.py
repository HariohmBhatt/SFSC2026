"""Test whether inference-time features can rescue sliced proposals.

This is a retrospective diagnostic. It fits on 16 historical calibration images,
selects one threshold on eight disjoint calibration images, and reports the
already-used 82-image assessment separately. It cannot promote a model.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import minimize
from scipy.special import expit

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
REPORT = ROOT / "reports/crop_quality_probe.json"
MODEL = ROOT / "reports/crop_quality_probe_model.json"

FEATURES = [
    "detector_logit",
    "refiner_logit",
    "combined_logit",
    "log_refined_area",
    "log_seed_area",
    "log_area_ratio",
    "seed_refined_iou",
    "log_aspect",
    "fill",
    "border_touching",
    "long_object",
    "cross_view_max_iou",
    "cross_view_support",
    "cluster_size",
    "cluster_score_mean",
    "whole_max_iou",
    "whole_overlap_count",
    "center_radius",
    "edge_distance",
]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = load("crop_quality_core", HERE / "core.py")
QUALITY = load("crop_quality_legacy", HERE / "quality_experiment.py")
METRIC = load("crop_quality_metric", ROOT / "filament-oof-consensus/core.py")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bounded_logit(value: float) -> float:
    value = float(np.clip(value, 1e-5, 1 - 1e-5))
    return math.log(value / (1 - value))


def whole_score(row) -> float:
    detector = float(row["detector_score"])
    if row.get("long_object_path", False):
        return detector
    return detector * math.sqrt(max(float(row["refiner_score"]), 0.0))


def calibration_split(names: Sequence[str]) -> tuple[list[str], list[str]]:
    ordered = sorted(names)
    threshold = ordered[2::3]
    fit = sorted(set(ordered) - set(threshold))
    if len(fit) != 16 or len(threshold) != 8:
        raise ValueError(f"Unexpected diagnostic split: {len(fit)}/{len(threshold)}")
    return fit, threshold


def cluster_indices(rows, pairwise_iou, threshold=0.3) -> list[list[int]]:
    """Greedily group repeated cross-view predictions around the best score."""
    remaining = set(range(len(rows)))
    clusters = []
    for representative in sorted(remaining, key=lambda i: (-rows[i]["score"], i)):
        if representative not in remaining:
            continue
        members = [representative]
        for other in sorted(remaining - {representative}):
            cross_view = rows[other]["source"] != rows[representative]["source"]
            if cross_view and pairwise_iou[representative, other] >= threshold:
                members.append(other)
        remaining.difference_update(members)
        clusters.append(members)
    return clusters


def candidate_features(row, members, rows, crop_iou, whole_iou) -> list[float]:
    seed = CORE.rle_for_decode(row["seed"])
    refined = CORE.rle_for_decode(row["refined"])
    seed_refined_iou = float(mask_utils.iou([seed], [refined], [0])[0, 0])
    x, y, width, height = map(float, mask_utils.toBbox(refined))
    refined_area = float(row["refined_area"])
    seed_area = float(row["seed_area"])
    member_scores = [float(rows[index]["score"]) for index in members]
    cross_values = [
        crop_iou[members[0], index]
        for index in members[1:]
        if rows[index]["source"] != row["source"]
    ]
    center_x = (x + width / 2) / 2048
    center_y = (y + height / 2) / 2048
    edge_distance = min(x, y, 2048 - x - width, 2048 - y - height) / 2048
    return [
        bounded_logit(row["detector_score"]),
        bounded_logit(row["refiner_score"]),
        bounded_logit(row["score"]),
        math.log1p(refined_area),
        math.log1p(seed_area),
        math.log(max(refined_area / max(seed_area, 1), 1e-5)),
        seed_refined_iou,
        math.log(max(width, height) / max(min(width, height), 1)),
        refined_area / max(width * height, 1),
        float(row["border_touching"]),
        float(row["long_object_path"]),
        float(max(cross_values, default=0.0)),
        float(sum(value >= 0.3 for value in cross_values)),
        float(len(members)),
        float(np.mean(member_scores)),
        float(whole_iou[members[0]].max(initial=0)),
        float((whole_iou[members[0]] >= 0.1).sum()),
        math.hypot(center_x - 0.5, center_y - 0.5),
        float(edge_distance),
    ]


def build_examples(crop_cache, whole_cache, names, index):
    examples, baseline_by_name = {}, {}
    for number, filename in enumerate(names, 1):
        whole = [
            {
                "mask": METRIC.decode(row["refined"]),
                "score": whole_score(row),
                "source": "whole",
            }
            for row in whole_cache[filename]
            if whole_score(row) >= 0.3 and int(row["refined_area"]) >= 64
        ]
        baseline_masks, _ = CORE.reconcile(whole, minimum_area=64)
        baseline_rles = [
            mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in baseline_masks
        ]
        baseline_by_name[filename] = {
            "rles": baseline_rles,
            "stats": METRIC.score(index.records(filename), baseline_masks).tolist(),
        }
        raw = [row for row in crop_cache[filename] if int(row["refined_area"]) >= 64]
        crop_rles = [CORE.rle_for_decode(row["refined"]) for row in raw]
        crop_iou = (
            np.asarray(mask_utils.iou(crop_rles, crop_rles, [0] * len(crop_rles)))
            if crop_rles
            else np.zeros((0, 0))
        )
        whole_iou = (
            np.asarray(mask_utils.iou(crop_rles, baseline_rles, [0] * len(baseline_rles)))
            if crop_rles and baseline_rles
            else np.zeros((len(crop_rles), len(baseline_rles)))
        )
        clusters = cluster_indices(raw, crop_iou)
        items = []
        for members in clusters:
            representative = members[0]
            row = raw[representative]
            items.append(
                {
                    "rle": crop_rles[representative],
                    "score": float(row["score"]),
                    "source": row["source"],
                    "features": candidate_features(row, members, raw, crop_iou, whole_iou),
                    "target": 0.0,
                }
            )
        for truth_record in index.records(filename):
            baseline_iou = (
                np.asarray(mask_utils.iou(truth_record, baseline_rles, [0] * len(baseline_rles)))
                if truth_record and baseline_rles
                else np.zeros((len(truth_record), len(baseline_rles)))
            )
            matched_truth = set(QUALITY.one_to_one_matches(baseline_iou))
            unmatched = [
                truth
                for truth_index, truth in enumerate(truth_record)
                if truth_index not in matched_truth
            ]
            proposal_iou = (
                np.asarray(
                    mask_utils.iou(
                        unmatched,
                        [item["rle"] for item in items],
                        [0] * len(items),
                    )
                )
                if unmatched and items
                else np.zeros((len(unmatched), len(items)))
            )
            for truth_index, candidate_index in QUALITY.one_to_one_matches(proposal_iou).items():
                items[candidate_index]["target"] += float(
                    proposal_iou[truth_index, candidate_index]
                )
        records = len(index.records(filename))
        for item in items:
            item["target"] /= records
        examples[filename] = items
        if number % 20 == 0:
            print(f"features={number}/{len(names)}", file=sys.stderr, flush=True)
    return examples, baseline_by_name


def basis(values: np.ndarray) -> np.ndarray:
    return np.column_stack([values, values**2])


def fit_model(examples, names, regularization=0.05):
    rows = [item for name in names for item in examples[name]]
    x = np.asarray([row["features"] for row in rows], dtype=float)
    y = np.asarray([row["target"] for row in rows], dtype=float)
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 0.05)
    z = basis((x - mean) / scale)
    z = np.column_stack([np.ones(len(z)), z])
    weights = np.ones(len(y)) / len(y)

    def objective(beta):
        logits = z @ beta
        loss = np.dot(weights, np.logaddexp(0, logits) - y * logits)
        loss += regularization * np.dot(beta[1:], beta[1:]) / 2
        gradient = z.T @ (weights * (expit(logits) - y))
        gradient[1:] += regularization * beta[1:]
        return loss, gradient

    fitted = minimize(objective, np.zeros(z.shape[1]), jac=True, method="L-BFGS-B")
    if not fitted.success:
        raise ValueError(f"Quality fit failed: {fitted.message}")
    return {
        "features": FEATURES,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": fitted.x.tolist(),
        "regularization": regularization,
        "fit_images": list(names),
        "examples": len(rows),
        "positive_target_sum": float(y.sum()),
        "loss": float(fitted.fun),
    }


def predict(item, model) -> float:
    x = (np.asarray(item["features"]) - model["mean"]) / model["scale"]
    z = np.concatenate(([1.0], basis(x[None])[0]))
    return float(expit(np.dot(z, model["coefficients"])))


def evaluate(examples, baseline, index, names, model, threshold):
    result = {}
    for filename in names:
        whole = [
            {
                "mask": mask_utils.decode(rle).astype(bool),
                "score": 1.0,
                "source": "whole",
            }
            for rle in baseline[filename]["rles"]
        ]
        selected = []
        for item in examples[filename]:
            quality = predict(item, model)
            if quality >= threshold:
                selected.append(
                    {
                        "mask": mask_utils.decode(item["rle"]).astype(bool),
                        "score": quality,
                        "source": item["source"],
                    }
                )
        masks, _ = CORE.reconcile(
            whole + selected,
            minimum_area=64,
            duplicate_iou=0.3,
        )
        result[filename] = METRIC.score(index.records(filename), masks)
    return result


def aggregate(stats, names):
    return METRIC.metrics(np.sum([stats[name] for name in names], axis=0))


def main() -> None:
    crop_cache = json.loads(CROP_CACHE.read_text())
    whole_cache = json.loads(WHOLE_CACHE.read_text())
    run = json.loads(RUN_RECORD.read_text())
    names = run["config"]["validation_names"]
    calibration = run["config"]["calibration_names"]
    fit_names, threshold_names = calibration_split(calibration)
    assessment_names = sorted(set(names) - set(calibration))
    index = QUALITY.CocoIndex(ANNOTATIONS)
    examples, baseline = build_examples(crop_cache, whole_cache, names, index)
    model = fit_model(examples, fit_names)
    baseline_stats = {name: np.asarray(baseline[name]["stats"]) for name in names}
    grid = []
    thresholds = np.linspace(0.02, 0.50, 25)
    for threshold in thresholds:
        stats = evaluate(examples, baseline, index, threshold_names, model, float(threshold))
        metrics = aggregate(stats, threshold_names)
        grid.append({"threshold": float(threshold), **metrics})
    selected = max(grid, key=lambda row: (row["pq"], -row["fp"], row["threshold"]))
    assessment_stats = evaluate(
        examples,
        baseline,
        index,
        assessment_names,
        model,
        selected["threshold"],
    )
    assessment = aggregate(assessment_stats, assessment_names)
    baseline_assessment = aggregate(baseline_stats, assessment_names)
    threshold_baseline = aggregate(baseline_stats, threshold_names)
    report = {
        "status": "complete",
        "scope": "retrospective diagnostic; cannot promote",
        "input_hashes": {
            "crop_cache": digest(CROP_CACHE),
            "whole_cache": digest(WHOLE_CACHE),
            "annotations": digest(ANNOTATIONS),
            "run_record": digest(RUN_RECORD),
            "implementation": digest(Path(__file__)),
        },
        "fit_images": fit_names,
        "threshold_images": threshold_names,
        "assessment_images": assessment_names,
        "model": model,
        "threshold_baseline": threshold_baseline,
        "threshold_grid": grid,
        "selected": selected,
        "baseline_assessment": baseline_assessment,
        "assessment": assessment,
        "delta_assessment": assessment["pq"] - baseline_assessment["pq"],
        "decision": "descriptive_only",
    }
    MODEL.write_text(json.dumps(model, indent=2) + "\n")
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
