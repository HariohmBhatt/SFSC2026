"""Run fixed four-view sliced proposal recovery on historical validation."""

from __future__ import annotations

import hashlib
import json
import math
import time
import traceback
from pathlib import Path

import core
import cv2
import legacy_runtime as base
import metric_core
import numpy as np
import torch

CONFIG = json.loads(Path("/kaggle/working/task-config.json").read_text())
WORK = Path("/kaggle/working/sliced-recovery")
WORK.mkdir(parents=True, exist_ok=True)
STATUS = {"status": "starting", "run_id": CONFIG["run_id"]}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def progress(event: str, **values) -> None:
    STATUS.update(event=event, **values)
    write_json(WORK / "progress.json", STATUS)
    print(json.dumps({"event": event, **values}), flush=True)


def source_file(filename: str, expected_hash: str | None = None) -> Path:
    matches = list(Path("/kaggle/input").glob(f"**/{filename}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {filename}, found {matches}")
    if expected_hash is not None and digest(matches[0]) != expected_hash:
        raise ValueError(f"Hash mismatch for {filename}")
    return matches[0]


def decode(value) -> np.ndarray:
    return base.mask_utils.decode(core.rle_for_decode(value)).astype(bool)


def encode(mask: np.ndarray) -> dict:
    return metric_core.encode(mask)


def truth_rles(filename: str, indices) -> list[list[dict]]:
    result = []
    for image_id in indices["image_ids_by_file"][filename]:
        info = indices["images_by_id"][image_id]
        result.append(
            [
                base.annotation_to_rle(row, int(info["height"]), int(info["width"]))
                for row in indices["annotations_by_image"].get(image_id, [])
                if not row.get("iscrowd", 0)
            ]
        )
    return result


def candidate_score(candidate) -> float:
    detector = float(candidate["detector_score"])
    if candidate.get("long_object_path", False):
        return detector
    return detector * math.sqrt(max(float(candidate["refiner_score"]), 0.0))


def control_masks(candidates) -> list[np.ndarray]:
    masks, scores = [], []
    for candidate in candidates:
        score = candidate_score(candidate)
        if score < CONFIG["score_threshold"] or int(candidate["refined_area"]) < 64:
            continue
        masks.append(decode(candidate["refined"]))
        scores.append(score)
    return base.resolve_overlaps(masks, scores, CONFIG["minimum_area"])[0]


def whole_rows(candidates):
    return [
        {
            "mask": decode(candidate["refined"]),
            "score": candidate_score(candidate),
            "source": "whole",
            "detector_score": float(candidate["detector_score"]),
            "refiner_score": float(candidate["refiner_score"]),
            "border_touching": False,
        }
        for candidate in candidates
        if candidate_score(candidate) >= CONFIG["score_threshold"]
        and int(candidate["refined_area"]) >= CONFIG["minimum_area"]
    ]


@torch.inference_mode()
def crop_rows(image, detector, refiner, cfg, device):
    rows = []
    window = CONFIG["crop_size"]
    for x0, y0 in core.crop_origins(*image.shape, window):
        crop = image[y0 : y0 + window, x0 : x0 + window]
        detector_input = core.detector_input(crop)
        prediction = detector.predict(
            detector_input,
            imgsz=window,
            conf=CONFIG["detector_confidence"],
            iou=CONFIG["detector_iou"],
            agnostic_nms=True,
            retina_masks=True,
            max_det=CONFIG["maximum_detections_per_view"],
            verbose=False,
        )[0]
        if prediction.masks is None:
            continue
        masks = [mask.cpu().numpy().astype(bool) for mask in prediction.masks.data]
        if masks and masks[0].shape != crop.shape:
            masks = [
                cv2.resize(
                    mask.astype(np.uint8), crop.shape[::-1], interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                for mask in masks
            ]
        scores = prediction.boxes.conf.detach().cpu().numpy().astype(float).tolist()
        masks, scores = base.remove_duplicate_seeds(masks, scores)
        for view_proposal, (crop_seed, detector_score) in enumerate(
            zip(masks, scores, strict=True)
        ):
            if int(crop_seed.sum()) < 32:
                continue
            border_touching = bool(
                crop_seed[:4].any()
                or crop_seed[-4:].any()
                or crop_seed[:, :4].any()
                or crop_seed[:, -4:].any()
            )
            seed = np.zeros_like(image, bool)
            seed[y0 : y0 + window, x0 : x0 + window] = crop_seed
            refined, refiner_score = base.refine_seed(image, seed, refiner, device, cfg)
            ys, xs = np.where(seed)
            width = int(xs.max() - xs.min() + 1)
            height = int(ys.max() - ys.min() + 1)
            long_object = (
                max(width, height) >= 720 or max(width, height) / max(min(width, height), 1) >= 4.5
            )
            row = {
                "mask": refined,
                "seed": seed,
                "score": float(detector_score)
                if long_object
                else float(detector_score * math.sqrt(max(refiner_score, 0.0))),
                "source": f"crop:{x0}:{y0}",
                "view_proposal": view_proposal,
                "detector_score": float(detector_score),
                "refiner_score": float(refiner_score),
                "border_touching": border_touching,
                "long_object_path": long_object,
            }
            rows.append(row)
    return rows


def serialized(rows):
    return [
        {
            **{key: value for key, value in row.items() if key not in {"mask", "seed"}},
            "seed": encode(row["seed"]),
            "refined": encode(row["mask"]),
            "seed_area": int(row["seed"].sum()),
            "refined_area": int(row["mask"].sum()),
        }
        for row in rows
    ]


def truth_size(area: int) -> str:
    if area < 256:
        return "tiny:<256"
    if area < 1024:
        return "small:256-1023"
    if area < 4096:
        return "medium:1024-4095"
    return "large:>=4096"


def update_size_recall(totals, truth_records, masks) -> None:
    for record in truth_records:
        grouped = {}
        for truth in record:
            grouped.setdefault(truth_size(int(decode(truth).sum())), []).append(truth)
        for group, subset in grouped.items():
            stats = metric_core.score([subset], masks)
            totals.setdefault(group, np.zeros(2, dtype=float))
            totals[group] += [stats[5], stats[7]]


def size_recall(totals) -> dict:
    return {
        group: {
            "matched": int(values[0]),
            "missed": int(values[1]),
            "recall": float(values[0] / values.sum()) if values.sum() else 0.0,
        }
        for group, values in sorted(totals.items())
    }


def grouped_comparison(baseline_stats, candidate_stats, key) -> dict:
    groups = {}
    for filename in baseline_stats:
        groups.setdefault(key(filename), []).append(filename)
    result = {}
    for group, names in sorted(groups.items()):
        baseline = metric_core.metrics(np.sum([baseline_stats[name] for name in names], axis=0))
        candidate = metric_core.metrics(np.sum([candidate_stats[name] for name in names], axis=0))
        result[str(group)] = {
            "images": len(names),
            "baseline": baseline,
            "sliced": candidate,
            "delta_pq": candidate["pq"] - baseline["pq"],
        }
    return result


def main() -> None:
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (6, 0):
            raise RuntimeError("Sliced recovery requires a Kaggle P100")
        progress("gpu_verified", gpu=torch.cuda.get_device_name(), torch=torch.__version__)
        base.seed_everything(CONFIG["seed"])
        paths = base.discover_competition_paths()
        coco = base.load_coco(paths["json"])
        indices = base.build_coco_indices(base.restrict_coco(coco, set(CONFIG["validation_names"])))
        cache_path = source_file(
            "validation-candidates.json", CONFIG["validation_candidates_sha256"]
        )
        raw_cache = json.loads(cache_path.read_text())
        if set(raw_cache) != set(CONFIG["validation_names"]):
            raise ValueError("Validation candidate membership mismatch")
        detector_path = source_file("detector.pt", CONFIG["detector_sha256"])
        refiner_path = source_file("refiner.pt", CONFIG["refiner_sha256"])
        from ultralytics import YOLO

        detector = YOLO(str(detector_path))
        device = torch.device("cuda")
        refiner = base.SeedRefiner(pretrained=False).to(device).eval()
        refiner.load_state_dict(torch.load(refiner_path, map_location=device, weights_only=True))
        cfg = base.CFG()
        cfg.use_tta = False
        baseline_stats, sliced_stats, candidate_cache = {}, {}, {}
        source_counts = {}
        all_crop_sources = [
            f"crop:{x0}:{y0}" for x0, y0 in core.crop_origins(2048, 2048, CONFIG["crop_size"])
        ]
        source_ablation_stats = {source: {} for source in all_crop_sources}
        baseline_size_totals, sliced_size_totals = {}, {}
        started = time.monotonic()
        for number, filename in enumerate(CONFIG["validation_names"], 1):
            image = cv2.imread(str(paths["train"] / filename), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (2048, 2048):
                raise ValueError(f"Invalid image {filename}")
            whole_candidates = whole_rows(raw_cache[filename])
            crop_candidates = crop_rows(image, detector, refiner, cfg, device)
            eligible_crops = [
                row
                for row in crop_candidates
                if row["score"] >= CONFIG["score_threshold"]
                and int(row["mask"].sum()) >= CONFIG["minimum_area"]
            ]
            combined = whole_candidates + eligible_crops
            masks, provenance = core.reconcile(
                combined,
                minimum_area=CONFIG["minimum_area"],
                duplicate_iou=CONFIG["cross_view_duplicate_iou"],
            )
            truth = truth_rles(filename, indices)
            baseline_masks = control_masks(raw_cache[filename])
            baseline_stats[filename] = metric_core.score(truth, baseline_masks).tolist()
            sliced_stats[filename] = metric_core.score(truth, masks).tolist()
            update_size_recall(baseline_size_totals, truth, baseline_masks)
            update_size_recall(sliced_size_totals, truth, masks)
            for source in all_crop_sources:
                source_masks, _ = core.reconcile(
                    whole_candidates + [row for row in eligible_crops if row["source"] == source],
                    minimum_area=CONFIG["minimum_area"],
                    duplicate_iou=CONFIG["cross_view_duplicate_iou"],
                )
                source_ablation_stats.setdefault(source, {})[filename] = metric_core.score(
                    truth, source_masks
                ).tolist()
            candidate_cache[filename] = serialized(crop_candidates)
            source_counts[filename] = {
                "whole_input": len(whole_candidates),
                "crop_input": len(crop_candidates),
                "crop_eligible": len(eligible_crops),
                "crop_border_touching": sum(row["border_touching"] for row in crop_candidates),
                "assembled": len(masks),
                "assembled_by_source": {
                    source: sum(row["source"] == source for row in provenance)
                    for source in sorted({row["source"] for row in provenance})
                },
            }
            if number % 10 == 0 or number == len(CONFIG["validation_names"]):
                progress(
                    "sliced_inference",
                    images_processed=number,
                    images_total=len(CONFIG["validation_names"]),
                    seconds=time.monotonic() - started,
                )
        baseline = metric_core.metrics(np.sum(list(baseline_stats.values()), axis=0))
        if abs(baseline["pq"] - CONFIG["expected_control_pq"]) > 1e-8:
            raise ValueError(f"Control parity failure: {baseline}")
        sliced = metric_core.metrics(np.sum(list(sliced_stats.values()), axis=0))
        assessment = sorted(set(CONFIG["validation_names"]) - set(CONFIG["calibration_names"]))
        baseline82 = metric_core.metrics(
            np.sum([baseline_stats[name] for name in assessment], axis=0)
        )
        sliced82 = metric_core.metrics(np.sum([sliced_stats[name] for name in assessment], axis=0))
        interval = metric_core.bootstrap_delta(sliced_stats, baseline_stats, assessment, 2000)
        source_ablation = {}
        for source, stats_by_name in sorted(source_ablation_stats.items()):
            names = sorted(stats_by_name)
            source_metrics = metric_core.metrics(
                np.sum([stats_by_name[name] for name in names], axis=0)
            )
            control_metrics = metric_core.metrics(
                np.sum([baseline_stats[name] for name in names], axis=0)
            )
            source_ablation[source] = {
                "images": len(names),
                "baseline": control_metrics,
                "whole_plus_source": source_metrics,
                "delta_pq": source_metrics["pq"] - control_metrics["pq"],
                "delta_tp": source_metrics["tp"] - control_metrics["tp"],
                "delta_fp": source_metrics["fp"] - control_metrics["fp"],
            }
        candidates_output = WORK / "sliced-validation-candidates.json"
        write_json(candidates_output, candidate_cache)
        report = {
            "status": "complete",
            "run_id": CONFIG["run_id"],
            "source_sha256": CONFIG["source_sha256"],
            "membership_sha256": CONFIG["membership_sha256"],
            "evidence_scope": "historical validation sliced-proposal diagnostic",
            "baseline": baseline,
            "sliced": sliced,
            "delta": sliced["pq"] - baseline["pq"],
            "baseline82": baseline82,
            "sliced82": sliced82,
            "delta82": sliced82["pq"] - baseline82["pq"],
            "retrospective_paired_block_bootstrap82": interval,
            "source_counts": source_counts,
            "source_ablation": source_ablation,
            "truth_size_recall": {
                "baseline": size_recall(baseline_size_totals),
                "sliced": size_recall(sliced_size_totals),
            },
            "subgroups": {
                "site": grouped_comparison(
                    baseline_stats, sliced_stats, lambda name: Path(name).stem[-2:]
                ),
                "year": grouped_comparison(
                    baseline_stats, sliced_stats, lambda name: int(name[:4])
                ),
            },
            "elapsed_seconds": time.monotonic() - started,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
            "candidate_cache_sha256": digest(candidates_output),
            "config": CONFIG,
        }
        report_path = WORK / "sliced-report.json"
        write_json(report_path, report)
        STATUS.update(status="complete", result=report)
        progress("complete")
    except Exception as error:
        STATUS.update(status="failed", error=str(error), traceback=traceback.format_exc())
        progress("failed")
        raise


if __name__ == "__main__":
    main()
