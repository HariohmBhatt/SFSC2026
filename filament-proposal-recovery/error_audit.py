"""Attribute control-model misses to proposal, mask, selection, and assembly stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = (
    ROOT / "outputs/yolo-refiner-validation/sfsc-yolo-refiner-validation/validation-candidates.json"
)
DEFAULT_ANNOTATIONS = ROOT / "SFSCDataset/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
DEFAULT_REPORT = ROOT / "reports/proposal_error_audit.json"
DEFAULT_MARKDOWN = ROOT / "reports/proposal_error_audit.md"
EXPECTED_CANDIDATES_SHA256 = "d47bb86b19359ce00e92fb32540856e5581bcbadd7d1f112b8a1f05e0abd56b6"
EXPECTED_DETECTOR_SHA256 = "68016483e5ff218fa445d04ee10e1e54921e061981578fa96685b32a26662379"
EXPECTED_REFINER_SHA256 = "d208b8fb4a2ef557898a8cae9b458daded6cfdb1bf979353c2b5c9224900009a"
EXPECTED_CONTROL_PQ = 0.4351790286733054


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def decode_rle(encoded: Mapping[str, Any]) -> np.ndarray:
    payload = {"size": encoded["size"], "counts": encoded["counts"].encode("ascii")}
    return mask_utils.decode(payload).astype(bool)


def annotation_rle(annotation: Mapping[str, Any], height: int, width: int):
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


def load_truth_records(path: Path, filenames: set[str]) -> dict[str, list[list[np.ndarray]]]:
    coco = json.loads(path.read_text())
    images = {str(row["id"]): row for row in coco["images"] if row["file_name"] in filenames}
    annotations = defaultdict(list)
    for row in coco["annotations"]:
        if str(row["image_id"]) in images and not row.get("iscrowd", 0):
            annotations[str(row["image_id"])].append(row)
    image_ids = defaultdict(list)
    for image_id, image in images.items():
        image_ids[image["file_name"]].append(image_id)
    result = {}
    for filename in sorted(filenames):
        records = []
        for image_id in sorted(image_ids[filename]):
            image = images[image_id]
            records.append(
                [
                    mask_utils.decode(
                        annotation_rle(row, int(image["height"]), int(image["width"]))
                    ).astype(bool)
                    for row in annotations[image_id]
                ]
            )
        if not records:
            raise ValueError(f"No annotation record for {filename}")
        result[filename] = records
    return result


def pairwise_iou(first: Sequence[np.ndarray], second: Sequence[np.ndarray]) -> np.ndarray:
    if not first or not second:
        return np.zeros((len(first), len(second)), dtype=float)
    first_rle = [mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in first]
    second_rle = [mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in second]
    return np.asarray(mask_utils.iou(first_rle, second_rle, [0] * len(second)), dtype=float)


def one_to_one_matches(iou: np.ndarray, threshold: float = 0.5) -> dict[int, int]:
    """Return truth-to-prediction matches, prioritizing cardinality then IoU."""
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


def assemble_trace(candidates, score_threshold: float, minimum_area: int):
    eligible = []
    statuses = {}
    for index, candidate in enumerate(candidates):
        area = int(candidate["refined"].sum())
        if area < minimum_area:
            statuses[index] = "area_rejection"
        elif float(candidate["score"]) < score_threshold:
            statuses[index] = "score_rejection"
        else:
            statuses[index] = "eligible"
            eligible.append((float(candidate["score"]), index, candidate["refined"]))
    output, origins = [], []
    occupied = np.zeros_like(candidates[0]["refined"], bool) if candidates else None
    for _, index, mask in sorted(eligible, reverse=True):
        clean = mask & ~occupied
        if int(clean.sum()) < minimum_area:
            statuses[index] = "painting_rejection"
            continue
        statuses[index] = "assembled"
        output.append(clean)
        origins.append(index)
        occupied |= clean
    return output, origins, statuses


def viable_prompt(
    seed: np.ndarray,
    truth_index: int,
    truths: Sequence[np.ndarray],
    minimum_intersection: int,
) -> bool:
    truth = truths[truth_index]
    intersection = int(np.logical_and(seed, truth).sum())
    if intersection < minimum_intersection or intersection / max(int(seed.sum()), 1) < 0.5:
        return False
    other_coverage = [
        np.logical_and(seed, other).sum() / max(int(other.sum()), 1)
        for index, other in enumerate(truths)
        if index != truth_index
    ]
    return max(other_coverage, default=0.0) < 0.1


def classify_record(
    truths: Sequence[np.ndarray],
    candidates: Sequence[Mapping[str, Any]],
    score_threshold: float,
    minimum_area: int,
    minimum_prompt_intersection: int = 16,
) -> list[dict[str, Any]]:
    seeds = [candidate["seed"] for candidate in candidates]
    refined = [candidate["refined"] for candidate in candidates]
    final, origins, statuses = assemble_trace(candidates, score_threshold, minimum_area)
    seed_iou = pairwise_iou(truths, seeds)
    refined_iou = pairwise_iou(truths, refined)
    final_iou = pairwise_iou(truths, final)
    final_matches = one_to_one_matches(final_iou)
    rows = []
    for truth_index, truth in enumerate(truths):
        row = {
            "truth_index": truth_index,
            "truth_area": int(truth.sum()),
            "max_seed_iou": float(seed_iou[truth_index].max(initial=0)),
            "max_refined_iou": float(refined_iou[truth_index].max(initial=0)),
            "max_final_iou": float(final_iou[truth_index].max(initial=0)),
        }
        if truth_index in final_matches:
            row.update(stage="matched", candidate_index=origins[final_matches[truth_index]])
            rows.append(row)
            continue
        if row["max_final_iou"] > 0.5:
            row["stage"] = "identity_collision"
            rows.append(row)
            continue
        good_refined = np.flatnonzero(refined_iou[truth_index] > 0.5).tolist()
        if good_refined:
            good_statuses = {statuses[index] for index in good_refined}
            if "assembled" in good_statuses or "painting_rejection" in good_statuses:
                row["stage"] = "assembly_loss"
            elif good_statuses == {"area_rejection"}:
                row["stage"] = "area_rejection"
            else:
                row["stage"] = "score_rejection"
            rows.append(row)
            continue
        if row["max_seed_iou"] > 0.5:
            row["stage"] = "refinement_damage"
            rows.append(row)
            continue
        viable = [
            index
            for index, seed in enumerate(seeds)
            if viable_prompt(seed, truth_index, truths, minimum_prompt_intersection)
        ]
        if viable:
            row["stage"] = "insufficient_mask_quality"
        elif any(
            np.logical_and(seed, truth).sum() >= minimum_prompt_intersection for seed in seeds
        ):
            row["stage"] = "ambiguous_or_merged_proposal"
        else:
            row["stage"] = "no_detector_proposal"
        rows.append(row)
    return rows


def candidate_score(candidate: Mapping[str, Any]) -> float:
    detector = float(candidate["detector_score"])
    if candidate.get("long_object_path", False):
        return detector
    return detector * math.sqrt(max(float(candidate["refiner_score"]), 0.0))


def truth_size(area: int) -> str:
    if area < 256:
        return "tiny:<256"
    if area < 1024:
        return "small:256-1023"
    if area < 4096:
        return "medium:1024-4095"
    return "large:>=4096"


def metrics_for_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stages = Counter(row["stage"] for row in rows)
    misses = len(rows) - stages["matched"]
    return {
        "truths": len(rows),
        "matched": stages["matched"],
        "misses": misses,
        "stage_counts": dict(sorted(stages.items())),
        "miss_stage_percent": {
            stage: count / misses if misses else 0.0
            for stage, count in sorted(stages.items())
            if stage != "matched"
        },
    }


def audit(candidates_path: Path, annotations_path: Path, score_threshold=0.3, minimum_area=64):
    raw_cache = json.loads(candidates_path.read_text())
    truths = load_truth_records(annotations_path, set(raw_cache))
    rows = []
    control_totals = np.zeros(4, dtype=float)
    for filename in sorted(raw_cache):
        candidates = [
            {
                "seed": decode_rle(candidate["seed"]),
                "refined": decode_rle(candidate["refined"]),
                "score": candidate_score(candidate),
            }
            for candidate in raw_cache[filename]
        ]
        for record_index, record in enumerate(truths[filename]):
            final, _, _ = assemble_trace(candidates, score_threshold, minimum_area)
            final_iou = pairwise_iou(record, final)
            final_matches = one_to_one_matches(final_iou)
            matched_iou = sum(final_iou[row, column] for row, column in final_matches.items())
            matched = len(final_matches)
            control_totals += [
                matched_iou,
                matched,
                len(final) - matched,
                len(record) - matched,
            ]
            classified = classify_record(record, candidates, score_threshold, minimum_area)
            for row in classified:
                row.update(
                    filename=filename,
                    record_index=record_index,
                    site=Path(filename).stem[-2:],
                    year=int(filename[:4]),
                    size=truth_size(row["truth_area"]),
                )
                rows.append(row)
    sum_iou, tp, fp, fn = control_totals
    denominator = tp + (fp + fn) / 2
    control = {
        "pq": float(sum_iou / denominator) if denominator else 0.0,
        "sq": float(sum_iou / tp) if tp else 0.0,
        "rq": float(tp / denominator) if denominator else 0.0,
        "sum_iou": float(sum_iou),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }
    candidate_hash = digest(candidates_path)
    is_default_input = candidates_path.resolve() == DEFAULT_CANDIDATES.resolve()
    if is_default_input and candidate_hash != EXPECTED_CANDIDATES_SHA256:
        raise ValueError(f"Default control cache lineage mismatch: {candidate_hash}")
    default_lineage = is_default_input
    model_root = candidates_path.parent / "models/fold_0"
    model_hashes = {
        name: digest(model_root / name) if (model_root / name).exists() else None
        for name in ("detector.pt", "refiner.pt")
    }
    if default_lineage:
        if model_hashes != {
            "detector.pt": EXPECTED_DETECTOR_SHA256,
            "refiner.pt": EXPECTED_REFINER_SHA256,
        }:
            raise ValueError(f"Control model lineage mismatch: {model_hashes}")
        if abs(control["pq"] - EXPECTED_CONTROL_PQ) > 1e-8:
            raise ValueError(f"Control metric parity failed: {control}")
    by_size = {
        group: metrics_for_counts([row for row in rows if row["size"] == group])
        for group in sorted({row["size"] for row in rows})
    }
    by_site = {
        group: metrics_for_counts([row for row in rows if row["site"] == group])
        for group in sorted({row["site"] for row in rows})
    }
    result = {
        "status": "complete",
        "scope": "truth-assisted historical validation diagnostic; never deployment logic",
        "generated_at": datetime.now().astimezone().isoformat(),
        "inputs": {
            "candidates": str(candidates_path.relative_to(ROOT)),
            "candidates_sha256": candidate_hash,
            "annotations": str(annotations_path.relative_to(ROOT)),
            "annotations_sha256": digest(annotations_path),
            "images": len(raw_cache),
            "annotation_records": sum(len(value) for value in truths.values()),
            "detector_sha256": model_hashes["detector.pt"],
            "refiner_sha256": model_hashes["refiner.pt"],
            "implementation_sha256": digest(Path(__file__)),
            "lineage_verified": default_lineage,
        },
        "configuration": {
            "score_threshold": score_threshold,
            "minimum_area": minimum_area,
            "match_iou": "strictly greater than 0.5",
            "minimum_prompt_intersection": 16,
            "prompt_purity": 0.5,
            "maximum_other_truth_coverage": 0.1,
        },
        "overall": metrics_for_counts(rows),
        "control_metric": control,
        "by_size": by_size,
        "by_site": by_site,
        "rows": rows,
        "limitations": [
            "The cache begins after detector box NMS and seed-mask deduplication, so losses "
            "inside those operations are included in no_detector_proposal.",
            "Every annotation record is evaluated independently, matching the competition "
            "aggregation.",
            "Truth-assisted attribution and oracle quantities cannot be used during test "
            "inference.",
        ],
    }
    return result


def render_markdown(result: Mapping[str, Any]) -> str:
    overall = result["overall"]
    lines = [
        "# SFSC control error-stage audit",
        "",
        f"Generated: **{result['generated_at']}**.",
        "",
        "This is a truth-assisted historical validation diagnostic. It is not deployment "
        "logic and cannot promote a model.",
        "",
        "## Overall attribution",
        "",
        f"Across {result['inputs']['annotation_records']} annotation records and "
        f"{overall['truths']} truths, the control matched {overall['matched']} and missed "
        f"{overall['misses']} under one-to-one strict IoU > 0.5 matching.",
        "",
        "| Stage | Count | Share of misses |",
        "|---|---:|---:|",
    ]
    for stage, count in sorted(
        overall["stage_counts"].items(), key=lambda item: (item[0] == "matched", -item[1])
    ):
        share = "—" if stage == "matched" else f"{overall['miss_stage_percent'][stage]:.1%}"
        lines.append(f"| {stage} | {count} | {share} |")
    lines.extend(["", "## Misses by truth size", ""])
    all_stages = sorted(
        {stage for group in result["by_size"].values() for stage in group["stage_counts"]}
        - {"matched"}
    )
    lines.append("| Size | Truths | Matched | Misses | " + " | ".join(all_stages) + " |")
    lines.append("|---|---:|---:|---:|" + "---:|" * len(all_stages))
    for size, group in result["by_size"].items():
        values = [str(group["stage_counts"].get(stage, 0)) for stage in all_stages]
        lines.append(
            f"| {size} | {group['truths']} | {group['matched']} | {group['misses']} | "
            + " | ".join(values)
            + " |"
        )
    lines.extend(["", "## Decision rule", ""])
    stages = overall["stage_counts"]
    proposal = stages.get("no_detector_proposal", 0) + stages.get("ambiguous_or_merged_proposal", 0)
    selection = stages.get("score_rejection", 0) + stages.get("area_rejection", 0)
    mask = stages.get("refinement_damage", 0) + stages.get("insufficient_mask_quality", 0)
    assembly = stages.get("assembly_loss", 0) + stages.get("identity_collision", 0)
    choices = {
        "sliced proposal recovery": proposal,
        "quality/threshold selection": selection,
        "realistic-prompt mask refinement": mask,
        "assembly repair": assembly,
    }
    chosen, count = max(choices.items(), key=lambda item: item[1])
    lines.append(
        f"The largest actionable bucket is **{chosen}** ({count} attributed misses). "
        "This selects the next isolated experiment; it does not estimate leaderboard gain."
    )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {value}" for value in result["limitations"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    result = audit(args.candidates, args.annotations)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown.write_text(render_markdown(result))
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
