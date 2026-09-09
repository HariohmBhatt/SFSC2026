"""Fine-tune the proven detector on matched full-image and native-crop views."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import time
import traceback
from pathlib import Path

import cv2
import legacy_runtime as base
import matched_core
import metric_core
import numpy as np
import recovery_core
import torch

CONFIG = json.loads(Path("/kaggle/working/task-config.json").read_text())
WORK = Path("/kaggle/working/matched-scale")
WORK.mkdir(parents=True, exist_ok=True)
STARTED = time.monotonic()
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
    STATUS.update(event=event, elapsed_seconds=time.monotonic() - STARTED, **values)
    write_json(WORK / "progress.json", STATUS)
    print(json.dumps({"event": event, **values}), flush=True)


def source_file(filename: str, expected_hash: str) -> Path:
    matches = [
        path
        for path in Path("/kaggle/input").glob(f"**/{filename}")
        if digest(path) == expected_hash
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one verified {filename}, found {matches}")
    return matches[0]


def release() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def label_lines(masks) -> list[str]:
    result = []
    for mask in masks:
        line = base.mask_to_yolo_polygon(mask, mask.shape[1], mask.shape[0])
        if line is not None:
            result.append(line)
    return result


def materialize_training_data(paths, indices, selected_ids) -> tuple[Path, dict]:
    root = WORK / "matched_data"
    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    counts = {"whole": 0, "crop": 0, "instances": 0, "empty_views": 0}
    for number, filename in enumerate(CONFIG["train_names"], 1):
        image = cv2.imread(str(paths["train"] / filename), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (2048, 2048):
            raise ValueError(f"Invalid training image {filename}")
        image_id = selected_ids[filename]
        info = indices["images_by_id"][image_id]
        masks = [
            base.annotation_to_mask(row, int(info["height"]), int(info["width"]))
            for row in indices["annotations_by_image"].get(image_id, [])
            if not row.get("iscrowd", 0)
        ]
        for view, x0, y0, x1, y1 in matched_core.training_views(2048, 2048, 1280):
            safe_view = view.replace(":", "-")
            stem = f"{Path(filename).stem}__{safe_view}"
            destination = images / f"{stem}.jpg"
            if view == "whole":
                if not destination.exists():
                    os.symlink(paths["train"] / filename, destination)
                visible = masks
                counts["whole"] += 1
            else:
                if not destination.exists() and not cv2.imwrite(
                    str(destination),
                    image[y0:y1, x0:x1],
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                ):
                    raise OSError(f"Could not write {destination}")
                visible = [
                    cropped
                    for mask in masks
                    if (
                        cropped := matched_core.crop_instance(
                            mask, x0, y0, x1, y1, CONFIG["minimum_visible_pixels"]
                        )
                    )
                    is not None
                ]
                counts["crop"] += 1
            lines = label_lines(visible)
            counts["instances"] += len(lines)
            counts["empty_views"] += int(not lines)
            (labels / f"{stem}.txt").write_text("\n".join(lines))
        if number % 50 == 0 or number == len(CONFIG["train_names"]):
            progress("data_materialization", images=number, **counts)
    expected = len(CONFIG["train_names"]) * 5
    if len(list(images.iterdir())) != expected or len(list(labels.iterdir())) != expected:
        raise ValueError("Matched-view dataset completeness failure")
    yaml = root / "data.yaml"
    yaml.write_text(f"path: {root}\ntrain: images\nval: images\nnames:\n  0: filament\n")
    return yaml, counts


def train_detector(yaml: Path, initial: Path) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(initial))
    batches = [0]

    def on_batch(trainer):
        batches[0] += 1
        if batches[0] == 1 or batches[0] % 100 == 0:
            loss = float(trainer.loss.detach().cpu())
            if not np.isfinite(loss):
                raise RuntimeError("Non-finite detector loss")
            progress(
                "detector_training",
                epoch=int(trainer.epoch) + 1,
                batches=batches[0],
                loss=loss,
                peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30,
            )

    model.add_callback("on_train_batch_end", on_batch)
    model.train(
        data=str(yaml),
        epochs=CONFIG["epochs"],
        imgsz=1280,
        batch=2,
        device=0,
        patience=0,
        seed=CONFIG["seed"],
        single_cls=True,
        mask_ratio=2,
        overlap_mask=False,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.20,
        fliplr=0.5,
        flipud=0.5,
        degrees=5,
        scale=0.10,
        translate=0.02,
        mosaic=0.0,
        close_mosaic=0,
        optimizer="AdamW",
        lr0=2e-4,
        lrf=0.10,
        cos_lr=True,
        warmup_epochs=0.5,
        weight_decay=5e-4,
        workers=2,
        amp=True,
        val=False,
        plots=False,
        save_period=2,
        project=str(WORK / "detector_runs"),
        name="matched-scale-8",
        exist_ok=True,
        verbose=True,
    )
    output = WORK / "detector-matched-scale.pt"
    shutil.copy2(str(model.trainer.last), output)
    if digest(output) == CONFIG["initial_detector_sha256"]:
        raise ValueError("Detector checkpoint did not change")
    del model
    release()
    return output


def json_rle(mask: np.ndarray) -> dict:
    return metric_core.encode(mask)


def candidate_score(detector_score: float, refiner_score: float, long_object: bool) -> float:
    if long_object:
        return detector_score
    return detector_score * math.sqrt(max(refiner_score, 0.0))


@torch.inference_mode()
def predict_view(image, detector_input, source, detector, refiner, device, cfg, offset):
    prediction = detector.predict(
        detector_input,
        imgsz=1280,
        conf=CONFIG["detector_confidence"],
        iou=0.6,
        agnostic_nms=True,
        retina_masks=True,
        max_det=100,
        verbose=False,
    )[0]
    if prediction.masks is None:
        return []
    height, width = detector_input.shape[:2] if not isinstance(detector_input, str) else image.shape
    seeds = [mask.cpu().numpy().astype(bool) for mask in prediction.masks.data]
    if seeds and seeds[0].shape != (height, width):
        seeds = [
            cv2.resize(
                mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            for mask in seeds
        ]
    scores = prediction.boxes.conf.detach().cpu().numpy().astype(float).tolist()
    seeds, scores = base.remove_duplicate_seeds(seeds, scores)
    result = []
    x0, y0 = offset
    for seed, detector_score in zip(seeds, scores, strict=True):
        if int(seed.sum()) < 32:
            continue
        full_seed = np.zeros_like(image, bool)
        full_seed[y0 : y0 + height, x0 : x0 + width] = seed
        refined, refiner_score = base.refine_seed(image, full_seed, refiner, device, cfg)
        ys, xs = np.where(full_seed)
        object_width = int(xs.max() - xs.min() + 1)
        object_height = int(ys.max() - ys.min() + 1)
        long_object = (
            max(object_width, object_height) >= 720
            or max(object_width, object_height) / max(min(object_width, object_height), 1) >= 4.5
        )
        mask = full_seed if long_object or int(refined.sum()) < 32 else refined
        result.append(
            {
                "source": source,
                "detector_score": float(detector_score),
                "refiner_score": float(refiner_score),
                "score": candidate_score(detector_score, refiner_score, long_object),
                "long_object_path": long_object,
                "border_touching": bool(
                    seed[:4].any() or seed[-4:].any() or seed[:, :4].any() or seed[:, -4:].any()
                ),
                "mask": mask,
            }
        )
    return result


def serialize(rows) -> list[dict]:
    return [
        {
            **{key: value for key, value in row.items() if key != "mask"},
            "mask": json_rle(row["mask"]),
            "area": int(row["mask"].sum()),
        }
        for row in rows
    ]


def infer_validation(paths, detector_path, refiner_path, indices):
    from ultralytics import YOLO

    detector = YOLO(str(detector_path))
    device = torch.device("cuda")
    refiner = base.SeedRefiner(pretrained=False).to(device).eval()
    refiner.load_state_dict(torch.load(refiner_path, map_location=device, weights_only=True))
    cfg = base.CFG()
    cfg.use_tta = False
    cache = {}
    started = time.monotonic()
    for number, filename in enumerate(CONFIG["validation_names"], 1):
        path = paths["train"] / filename
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        rows = predict_view(image, str(path), "whole", detector, refiner, device, cfg, (0, 0))
        for x0, y0 in matched_core.crop_origins(2048, 2048, 1280):
            crop = image[y0 : y0 + 1280, x0 : x0 + 1280]
            rows.extend(
                predict_view(
                    image,
                    np.repeat(crop[..., None], 3, axis=2),
                    f"crop:{x0}:{y0}",
                    detector,
                    refiner,
                    device,
                    cfg,
                    (x0, y0),
                )
            )
        cache[filename] = serialize(rows)
        if number % 10 == 0 or number == len(CONFIG["validation_names"]):
            write_json(WORK / "validation-candidates.partial.json", cache)
            progress(
                "validation_inference",
                images=number,
                candidates=sum(map(len, cache.values())),
                seconds=time.monotonic() - started,
            )
    del detector, refiner
    release()
    output = WORK / "validation-candidates.json"
    write_json(output, cache)
    return cache, time.monotonic() - started


def truths(filename, indices):
    return [
        [
            base.annotation_to_rle(
                row,
                int(indices["images_by_id"][image_id]["height"]),
                int(indices["images_by_id"][image_id]["width"]),
            )
            for row in indices["annotations_by_image"].get(image_id, [])
            if not row.get("iscrowd", 0)
        ]
        for image_id in indices["image_ids_by_file"][filename]
    ]


def assemble(rows, threshold, mode):
    candidates = [
        {
            "mask": metric_core.decode(row["mask"]),
            "score": float(row["score"]),
            "source": row["source"],
        }
        for row in rows
        if float(row["score"]) >= threshold
        and int(row["area"]) >= 64
        and (mode == "multiscale" or row["source"] == "whole")
    ]
    return recovery_core.reconcile(candidates, minimum_area=64, duplicate_iou=0.6)[0]


def baseline_masks(rows):
    candidates = []
    for row in rows:
        detector_score = float(row["detector_score"])
        score = (
            detector_score
            if row.get("long_object_path", False)
            else detector_score * math.sqrt(max(float(row["refiner_score"]), 0.0))
        )
        if score >= 0.3 and int(row["refined_area"]) >= 64:
            candidates.append((score, metric_core.decode(row["refined"])))
    masks = [mask for _, mask in candidates]
    scores = [score for score, _ in candidates]
    return base.resolve_overlaps(masks, scores, 64)[0]


def aggregate(stats, names):
    return metric_core.metrics(np.sum([stats[name] for name in names], axis=0))


def evaluate(cache, baseline_cache, indices):
    calibration = CONFIG["calibration_names"]
    assessment = sorted(set(CONFIG["validation_names"]) - set(calibration))
    truth_cache = {name: truths(name, indices) for name in CONFIG["validation_names"]}
    baseline_stats = {
        name: metric_core.score(truth_cache[name], baseline_masks(baseline_cache[name]))
        for name in CONFIG["validation_names"]
    }
    grid = []
    configurations = []
    for mode in ("whole", "multiscale"):
        for threshold in CONFIG["thresholds"]:
            stats = {
                name: metric_core.score(truth_cache[name], assemble(cache[name], threshold, mode))
                for name in calibration
            }
            metrics = aggregate(stats, calibration)
            row = {"mode": mode, "threshold": threshold, **metrics}
            grid.append(row)
            configurations.append(row)
    selected = dict(matched_core.select_configuration(configurations))
    selected_stats = {
        name: metric_core.score(
            truth_cache[name],
            assemble(cache[name], selected["threshold"], selected["mode"]),
        )
        for name in CONFIG["validation_names"]
    }
    baseline = aggregate(baseline_stats, CONFIG["validation_names"])
    candidate = aggregate(selected_stats, CONFIG["validation_names"])
    baseline82 = aggregate(baseline_stats, assessment)
    candidate82 = aggregate(selected_stats, assessment)
    interval = metric_core.bootstrap_delta(selected_stats, baseline_stats, assessment, 2000)
    return {
        "calibration_grid": grid,
        "selected": selected,
        "baseline": baseline,
        "candidate": candidate,
        "delta": candidate["pq"] - baseline["pq"],
        "baseline82": baseline82,
        "candidate82": candidate82,
        "delta82": candidate82["pq"] - baseline82["pq"],
        "paired_block_bootstrap82": interval,
    }


def main() -> None:
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (6, 0):
            raise RuntimeError("Matched-scale training requires a Kaggle P100")
        progress("gpu_verified", gpu=torch.cuda.get_device_name(), torch=torch.__version__)
        base.seed_everything(CONFIG["seed"])
        paths = base.discover_competition_paths()
        coco = base.load_coco(paths["json"])
        restricted = base.restrict_coco(
            coco, set(CONFIG["train_names"]) | set(CONFIG["validation_names"])
        )
        indices = base.build_coco_indices(restricted)
        selected_ids = base.choose_medoid_annotators(indices)
        initial = source_file("detector.pt", CONFIG["initial_detector_sha256"])
        refiner = source_file("refiner.pt", CONFIG["refiner_sha256"])
        baseline_path = source_file(
            "validation-candidates.json", CONFIG["baseline_candidates_sha256"]
        )
        yaml, data_counts = materialize_training_data(paths, indices, selected_ids)
        training_started = time.monotonic()
        detector = train_detector(yaml, initial)
        training_seconds = time.monotonic() - training_started
        candidates, inference_seconds = infer_validation(paths, detector, refiner, indices)
        baseline_cache = json.loads(baseline_path.read_text())
        if set(baseline_cache) != set(CONFIG["validation_names"]):
            raise ValueError("Baseline membership mismatch")
        result = evaluate(candidates, baseline_cache, indices)
        candidate_path = WORK / "validation-candidates.json"
        report = {
            "status": "complete",
            "run_id": CONFIG["run_id"],
            "source_sha256": CONFIG["source_sha256"],
            "membership_sha256": CONFIG["membership_sha256"],
            "evidence_scope": "matched-scale historical validation pilot",
            "data_counts": data_counts,
            "training_seconds": training_seconds,
            "inference_seconds": inference_seconds,
            "elapsed_seconds": time.monotonic() - STARTED,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
            "detector_sha256": digest(detector),
            "candidate_cache_sha256": digest(candidate_path),
            "config": CONFIG,
            **result,
            "screening_gate": bool(
                result["delta82"] >= 0.015 and result["paired_block_bootstrap82"]["ci95"][0] > 0
            ),
            "decision_scope": (
                "A passing screening gate justifies blocked replication; historical "
                "validation cannot authorize final fitting or submission."
            ),
            "automatic_submission": False,
        }
        if abs(report["baseline"]["pq"] - CONFIG["expected_control_pq"]) > 1e-8:
            raise ValueError(f"Baseline parity failure: {report['baseline']}")
        write_json(WORK / "matched-scale-report.json", report)
        STATUS.update(status="complete", result=report)
        progress("complete")
    except Exception as error:
        STATUS.update(status="failed", error=str(error), traceback=traceback.format_exc())
        progress("failed")
        raise


if __name__ == "__main__":
    main()
