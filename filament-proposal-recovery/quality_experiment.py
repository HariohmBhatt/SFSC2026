"""Evaluate duplicate-aware contextual quality selection on frozen candidates."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
CONSENSUS = ROOT / "filament-oof-consensus/core.py"
OOF_ROOT = ROOT / "outputs/consensus-20260907-125349"
VALIDATION = (
    ROOT / "outputs/yolo-refiner-validation/sfsc-yolo-refiner-validation/validation-candidates.json"
)
ANNOTATIONS = ROOT / "SFSCDataset/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMAGES = ROOT / "SFSCDataset/train/train_images"
MEMBERSHIP = ROOT / "reports/native_training_20260908-162230.json"
REPORT = ROOT / "reports/contextual_quality_experiment.json"
MODEL = ROOT / "reports/contextual_quality_model.json"
VALIDATION_FEATURES = ROOT / "reports/contextual_quality_validation_candidates.json"

BASE_FEATURES = [
    "detector_logit",
    "refiner_score",
    "log_area",
    "log_seed_area",
    "area_ratio",
    "seed_iou",
    "log_aspect",
    "fill",
    "components",
    "inside_intensity",
    "ring_contrast",
    "kind_refined",
]
CONTEXT_FEATURES = [
    "baseline_logit",
    "log_proposal_count",
    "detector_rank_fraction",
    "detector_gap_to_best",
    "max_other_mask_iou",
    "other_overlap_count_010",
    "other_overlap_count_050",
    "relative_log_area",
]
FEATURES = BASE_FEATURES + CONTEXT_FEATURES
INTERACTIONS = [
    ("detector_logit", "refiner_score"),
    ("baseline_logit", "max_other_mask_iou"),
    ("max_other_mask_iou", "other_overlap_count_050"),
    ("log_area", "fill"),
    ("seed_iou", "kind_refined"),
    ("inside_intensity", "ring_contrast"),
    ("detector_rank_fraction", "log_proposal_count"),
]


def load_consensus_core():
    spec = importlib.util.spec_from_file_location("quality_consensus_core", CONSENSUS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = load_consensus_core()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_rle(mask: np.ndarray) -> dict[str, Any]:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": list(mask.shape), "counts": encoded["counts"].decode("ascii")}


def native_rle(item: dict[str, Any]):
    if "mask" in item:
        return mask_utils.encode(np.asfortranarray(item["mask"].astype(np.uint8)))
    rle = item["rle"]
    return {"size": rle["size"], "counts": rle["counts"].encode("ascii")}


def decode(item: dict[str, Any]) -> np.ndarray:
    if "mask" in item:
        return item["mask"].copy()
    return mask_utils.decode(native_rle(item)).astype(bool)


class CocoIndex:
    def __init__(self, path: Path):
        coco = json.loads(path.read_text())
        self.images = {str(row["id"]): row for row in coco["images"]}
        self.annotations = defaultdict(list)
        for row in coco["annotations"]:
            if not row.get("iscrowd", 0):
                self.annotations[str(row["image_id"])].append(row)
        self.image_ids = defaultdict(list)
        for image_id, image in self.images.items():
            self.image_ids[image["file_name"]].append(image_id)

    @staticmethod
    def annotation_rle(annotation, height, width):
        segmentation = annotation.get("segmentation")
        if isinstance(segmentation, dict):
            result = dict(segmentation)
            if isinstance(result.get("counts"), list):
                result = mask_utils.frPyObjects(result, height, width)
            return result
        polygons = segmentation or []
        if not polygons:
            return mask_utils.encode(np.asfortranarray(np.zeros((height, width), np.uint8)))
        return mask_utils.merge(mask_utils.frPyObjects(polygons, height, width))

    def records(self, filename: str):
        result = []
        for image_id in sorted(self.image_ids[filename]):
            image = self.images[image_id]
            result.append(
                [
                    self.annotation_rle(row, int(image["height"]), int(image["width"]))
                    for row in self.annotations[image_id]
                ]
            )
        if not result:
            raise ValueError(f"Missing truth for {filename}")
        return result


def one_to_one_matches(iou: np.ndarray, threshold: float = 0.5) -> dict[int, int]:
    if not iou.size:
        return {}
    eligible = iou > threshold
    cardinality_bonus = min(iou.shape) + 1
    utility = eligible.astype(float) * cardinality_bonus + np.where(eligible, iou, 0)
    rows, columns = linear_sum_assignment(utility, maximize=True)
    return {
        int(row): int(column)
        for row, column in zip(rows, columns, strict=True)
        if eligible[row, column]
    }


def assignment_targets(items: list[dict[str, Any]], truth_records) -> np.ndarray:
    """Assign credit once per proposal and truth, removing duplicate candidate credit."""
    targets = np.zeros(len(items), dtype=float)
    if not items:
        return targets
    proposal_ids = sorted({int(item["proposal_id"]) for item in items})
    by_proposal = {
        proposal_id: [
            index for index, item in enumerate(items) if int(item["proposal_id"]) == proposal_id
        ]
        for proposal_id in proposal_ids
    }
    candidate_rles = [native_rle(item) for item in items]
    for record in truth_records:
        truth_rles = [
            mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            if isinstance(mask, np.ndarray)
            else mask
            for mask in record
        ]
        if not truth_rles:
            continue
        iou = np.asarray(mask_utils.iou(truth_rles, candidate_rles, [0] * len(items)), dtype=float)
        proposal_iou = np.column_stack(
            [iou[:, by_proposal[proposal_id]].max(axis=1) for proposal_id in proposal_ids]
        )
        for truth_index, proposal_column in one_to_one_matches(proposal_iou).items():
            alternatives = by_proposal[proposal_ids[proposal_column]]
            selected = max(alternatives, key=lambda index: (iou[truth_index, index], index))
            targets[selected] += iou[truth_index, selected]
    return targets / max(len(truth_records), 1)


def baseline_score(item: dict[str, Any]) -> float:
    detector = float(item["detector_score"])
    if item["kind"] == "seed":
        return detector
    return detector * math.sqrt(max(float(item["features"][1]), 0.0))


def contextualize(items: list[dict[str, Any]]) -> None:
    if not items:
        return
    rles = [native_rle(item) for item in items]
    pairwise = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)), dtype=float)
    proposal_ids = np.array([int(item["proposal_id"]) for item in items])
    proposal_scores = {
        proposal_id: max(
            float(item["detector_score"])
            for item in items
            if int(item["proposal_id"]) == proposal_id
        )
        for proposal_id in sorted(set(proposal_ids))
    }
    ranked = sorted(proposal_scores, key=lambda key: (-proposal_scores[key], key))
    ranks = {proposal_id: rank for rank, proposal_id in enumerate(ranked)}
    best_score = max(proposal_scores.values())
    median_log_area = float(np.median([math.log1p(int(item["area"])) for item in items]))
    for index, item in enumerate(items):
        other = proposal_ids != proposal_ids[index]
        overlaps = pairwise[index, other]
        score = float(np.clip(baseline_score(item), 1e-5, 1 - 1e-5))
        detector = float(item["detector_score"])
        context = [
            math.log(score / (1 - score)),
            math.log1p(len(proposal_scores)),
            ranks[int(item["proposal_id"])] / max(len(proposal_scores) - 1, 1),
            best_score - detector,
            float(overlaps.max(initial=0)),
            float((overlaps >= 0.1).sum()),
            float((overlaps >= 0.5).sum()),
            math.log1p(int(item["area"])) - median_log_area,
        ]
        item["extended_features"] = [*map(float, item["features"]), *context]


def expanded_design(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = np.clip((x - mean) / scale, -5, 5)
    pieces = [z, z * z - 1]
    indices = {name: index for index, name in enumerate(FEATURES)}
    pieces.extend(
        (z[:, indices[first]] * z[:, indices[second]])[:, None] for first, second in INTERACTIONS
    )
    return np.column_stack(pieces)


def fit_model(items: list[dict[str, Any]], regularization=0.05) -> dict[str, Any]:
    x = np.asarray([item["extended_features"] for item in items], dtype=float)
    y = np.asarray([item["assignment_target"] for item in items], dtype=float)
    weights = np.asarray([item["record_weight"] for item in items], dtype=float)
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 0.05)
    design = expanded_design(x, mean, scale)
    design = np.column_stack([np.ones(len(design)), design])
    weights /= weights.sum()

    def objective(beta):
        logits = design @ beta
        penalty = regularization * np.dot(beta[1:], beta[1:]) / 2
        loss = np.dot(weights, np.logaddexp(0, logits) - y * logits) + penalty
        gradient = design.T @ (weights * (expit(logits) - y))
        gradient[1:] += regularization * beta[1:]
        return loss, gradient

    fitted = minimize(objective, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B")
    if not fitted.success or not np.isfinite(fitted.fun):
        raise RuntimeError(f"Quality fit failed: {fitted.message}")
    return {
        "feature_names": FEATURES,
        "interactions": INTERACTIONS,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": fitted.x.tolist(),
        "regularization": regularization,
        "examples": len(items),
        "positive_examples": int((y > 0).sum()),
        "mean_assignment_target": float(np.average(y, weights=weights)),
        "loss": float(fitted.fun),
        "target": "duplicate-aware one-to-one E[IoU * 1(IoU > 0.5)]",
    }


def predict_quality(item: dict[str, Any], model: dict[str, Any]) -> float:
    x = np.asarray(item["extended_features"], dtype=float)[None]
    design = expanded_design(x, np.asarray(model["mean"]), np.asarray(model["scale"]))[0]
    return float(expit(model["coefficients"][0] + np.dot(design, model["coefficients"][1:])))


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.logical_and(first, second).sum())
    union = int(np.logical_or(first, second).sum())
    return intersection / union if union else 0.0


def assemble(items, model, threshold, minimum_area=64):
    best = {}
    for item in items:
        value = predict_quality(item, model)
        if int(item["area"]) < minimum_area or value < threshold:
            continue
        proposal_id = int(item["proposal_id"])
        if proposal_id not in best or value > best[proposal_id][0]:
            best[proposal_id] = (value, item)
    accepted = []
    occupied = None
    for value, item in sorted(best.values(), key=lambda row: row[0], reverse=True):
        mask = decode(item)
        if any(mask_iou(mask, previous) >= 0.6 for _, previous in accepted):
            continue
        if occupied is None:
            occupied = np.zeros_like(mask)
        mask &= ~occupied
        if int(mask.sum()) >= minimum_area:
            accepted.append((value, mask))
            occupied |= mask
    return [mask for _, mask in accepted]


def baseline_masks(raw_items):
    candidates = []
    for raw in raw_items:
        score = float(raw["detector_score"])
        if not raw.get("long_object_path", False):
            score *= math.sqrt(max(float(raw["refiner_score"]), 0.0))
        if score >= 0.3 and int(raw["refined_area"]) >= 64:
            candidates.append((score, decode({"rle": raw["refined"]})))
    occupied = np.zeros((2048, 2048), bool)
    result = []
    for _, mask in sorted(candidates, reverse=True, key=lambda row: row[0]):
        clean = mask & ~occupied
        if int(clean.sum()) >= 64:
            result.append(clean)
            occupied |= clean
    return result


def validation_alternatives(raw_cache) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for number, (filename, raw_items) in enumerate(sorted(raw_cache.items()), 1):
        image = np.asarray(Image.open(IMAGES / filename).convert("L"))
        items = []
        for proposal_id, raw in enumerate(raw_items):
            seed = decode({"rle": raw["seed"]})
            for kind in ("seed", "refined"):
                mask = seed if kind == "seed" else decode({"rle": raw["refined"]})
                if int(mask.sum()) < 32:
                    continue
                items.append(
                    {
                        "proposal_id": proposal_id,
                        "kind": kind,
                        "rle": json_rle(mask),
                        "area": int(mask.sum()),
                        "detector_score": float(raw["detector_score"]),
                        "features": CORE.mask_features(
                            image,
                            seed,
                            mask,
                            float(raw["detector_score"]),
                            float(raw["refiner_score"]),
                            kind,
                        ),
                    }
                )
        contextualize(items)
        result[filename] = items
        if number % 20 == 0:
            print(f"validation features {number}/{len(raw_cache)}", flush=True)
    return result


def load_oof(index: CocoIndex):
    combined, training_rows, lineage = {}, [], []
    for fold in range(3):
        path = next((OOF_ROOT / f"fold-{fold}/results").rglob(f"fold-{fold}-candidates.json"))
        report_path = next((OOF_ROOT / f"fold-{fold}/results").rglob(f"fold-{fold}-report.json"))
        fold_report = json.loads(report_path.read_text())
        if fold_report.get("status") != "complete":
            raise ValueError(f"Fold {fold} report is incomplete")
        if digest(path) != fold_report.get("candidates_sha256"):
            raise ValueError(f"Fold {fold} candidate lineage mismatch")
        lineage.append(
            {
                "fold": fold,
                "candidates": str(path.relative_to(ROOT)),
                "candidates_sha256": digest(path),
                "report": str(report_path.relative_to(ROOT)),
                "report_sha256": digest(report_path),
                "source_sha256": fold_report["source_sha256"],
                "detector_sha256": fold_report["detector_sha256"],
                "refiner_sha256": fold_report["refiner_sha256"],
                "train_names_sha256": digest_bytes(
                    json.dumps(fold_report["train_names"], sort_keys=True).encode()
                ),
                "held_names_sha256": digest_bytes(
                    json.dumps(fold_report["held_names"], sort_keys=True).encode()
                ),
            }
        )
        cache = json.loads(path.read_text())
        if set(cache) & set(combined):
            raise ValueError("OOF memberships overlap")
        combined.update(cache)
    if len(combined) != 495:
        raise ValueError(f"Expected 495 OOF images, found {len(combined)}")
    for number, (filename, items) in enumerate(sorted(combined.items()), 1):
        contextualize(items)
        targets = assignment_targets(items, index.records(filename))
        record_weight = len(index.records(filename))
        for item, target in zip(items, targets, strict=True):
            item["assignment_target"] = float(target)
            item["record_weight"] = record_weight
            training_rows.append(item)
        if number % 100 == 0:
            print(f"OOF targets {number}/{len(combined)}", flush=True)
    return training_rows, lineage


def main() -> None:
    index = CocoIndex(ANNOTATIONS)
    training_rows, oof_lineage = load_oof(index)
    model = fit_model(training_rows)
    MODEL.write_text(json.dumps(model, indent=2) + "\n")
    raw_cache = json.loads(VALIDATION.read_text())
    validation = validation_alternatives(raw_cache)
    VALIDATION_FEATURES.write_text(json.dumps(validation) + "\n")
    membership = json.loads(MEMBERSHIP.read_text())["config"]
    names = membership["validation_names"]
    calibration = membership["calibration_names"]
    assessment = sorted(set(names) - set(calibration))
    if set(raw_cache) != set(names) or not set(calibration) < set(names):
        raise ValueError("Validation/calibration membership mismatch")
    truths = {name: index.records(name) for name in names}
    baseline_stats = {
        name: CORE.score(truths[name], baseline_masks(raw_cache[name])).tolist() for name in names
    }
    baseline = CORE.metrics(np.sum(list(baseline_stats.values()), axis=0))
    if abs(baseline["pq"] - 0.4351790286733054) > 1e-8:
        raise ValueError(f"Baseline parity failed: {baseline}")
    thresholds = [0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.24, 0.28]
    grid = []
    for threshold in thresholds:
        total = np.zeros(8)
        for name in calibration:
            total += CORE.score(truths[name], assemble(validation[name], model, threshold))
        grid.append({"threshold": threshold, **CORE.metrics(total)})
    selected = max(grid, key=lambda row: (row["pq"], -row["fp"], row["threshold"]))
    candidate_stats = {
        name: CORE.score(
            truths[name], assemble(validation[name], model, selected["threshold"])
        ).tolist()
        for name in names
    }
    full = CORE.metrics(np.sum(list(candidate_stats.values()), axis=0))
    candidate82 = CORE.metrics(np.sum([candidate_stats[name] for name in assessment], axis=0))
    baseline82 = CORE.metrics(np.sum([baseline_stats[name] for name in assessment], axis=0))
    interval = CORE.bootstrap_delta(candidate_stats, baseline_stats, assessment)
    report = {
        "status": "complete",
        "scope": "retrospective historical contextual-quality diagnostic; cannot promote",
        "input_hashes": {
            "annotations": digest(ANNOTATIONS),
            "validation_candidates": digest(VALIDATION),
            "membership": digest(MEMBERSHIP),
            "model": digest(MODEL),
            "validation_features": digest(VALIDATION_FEATURES),
            "implementation": digest(Path(__file__)),
        },
        "oof_lineage": oof_lineage,
        "model": model,
        "selected": selected,
        "calibration_grid": grid,
        "baseline": baseline,
        "full_validation": full,
        "control82": baseline82,
        "assessment82": candidate82,
        "delta82": candidate82["pq"] - baseline82["pq"],
        "paired_block_bootstrap82": interval,
        "decision": "descriptive_only; blocked R/Q/H evidence is required for promotion",
        "train_images": 495,
        "calibration_images": len(calibration),
        "assessment_images": len(assessment),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
