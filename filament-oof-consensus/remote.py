"""Kaggle-only staged training worker. Bootstrapped with immutable source/config."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import shutil
import time
import traceback
from pathlib import Path

import core
import cv2
import legacy_runtime as base
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

WORK = Path("/kaggle/working/consensus")
WORK.mkdir(exist_ok=True)
CONFIG = json.loads(Path("/kaggle/working/task-config.json").read_text())
STARTED = time.monotonic()
STATUS = {
    "status": "starting",
    "stage": CONFIG["stage"],
    "run_id": CONFIG["run_id"],
    "source_sha256": CONFIG["source_sha256"],
    "config": CONFIG,
}


def write_json(path, payload):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(path)


def progress(phase, **values):
    STATUS.update(phase=phase, elapsed_seconds=time.monotonic() - STARTED, **values)
    write_json(WORK / "progress.json", STATUS)
    print(json.dumps({"event": phase, **values}), flush=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release():
    gc.collect()
    torch.cuda.empty_cache()


def truth_for(filename, indices):
    records = []
    for iid in indices["image_ids_by_file"][filename]:
        records.append(
            [
                base.annotation_to_rle(ann, 2048, 2048)
                for ann in indices["annotations_by_image"].get(iid, [])
                if not ann.get("iscrowd", 0)
            ]
        )
    return records


def source_file(suffix, expected_sha=None):
    paths = [p for p in Path("/kaggle/input").rglob(Path(suffix).name) if str(p).endswith(suffix)]
    if expected_sha:
        paths = [p for p in paths if sha(p) == expected_sha]
    if len(paths) != 1:
        raise RuntimeError(f"Expected unique input {suffix}, found {paths}")
    return paths[0]


def train_fixed(train_names, paths, indices, cfg):
    """Fixed durations, train-only diagnostics; no excluded labels select epochs."""
    from ultralytics import YOLO

    medoids = base.choose_medoid_annotators(indices)
    data = WORK / "yolo_data"
    base.write_yolo_split(data / "train", train_names, paths["train"], medoids, indices)
    base.write_yolo_split(data / "diagnostic", train_names[:12], paths["train"], medoids, indices)
    yaml = data / "data.yaml"
    yaml.write_text(
        f"path: {data}\ntrain: train/images\nval: diagnostic/images\nnames:\n  0: filament\n"
    )
    models = WORK / "models"
    models.mkdir(exist_ok=True)
    detector = YOLO(cfg.detector_model)
    steps = [0]

    def on_batch(trainer):
        steps[0] += 1
        if steps[0] == 1 or steps[0] % 100 == 0:
            loss = float(trainer.loss.detach().cpu())
            if not np.isfinite(loss):
                raise RuntimeError("Nonfinite detector loss")
            progress(
                "detector_training",
                epoch=int(trainer.epoch) + 1,
                optimizer_batches=steps[0],
                loss=loss,
                peak_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
            )

    detector.add_callback("on_train_batch_end", on_batch)
    progress("detector_start", train_images=len(train_names))
    detector.train(
        data=str(yaml),
        epochs=36,
        imgsz=1280,
        batch=2,
        device=0,
        patience=0,
        seed=cfg.seed,
        single_cls=True,
        mask_ratio=2,
        overlap_mask=False,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.30,
        fliplr=0.5,
        flipud=0.5,
        degrees=10,
        scale=0.20,
        translate=0.05,
        mosaic=0.0,
        close_mosaic=0,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.05,
        cos_lr=True,
        warmup_epochs=2.0,
        weight_decay=0.0005,
        workers=2,
        amp=True,
        val=False,
        save_period=6,
        project=str(WORK / "detector_runs"),
        name="fixed36",
        exist_ok=True,
    )
    shutil.copy2(str(detector.trainer.last), models / "detector.pt")
    del detector
    release()

    dataset = base.RefinerDataset(train_names, paths["train"], medoids, indices, cfg, True)
    loader = DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True, drop_last=True
    )
    refiner = base.SeedRefiner(pretrained=True).cuda()
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=3e-4, weight_decay=1e-4)
    # The control selected epoch 9 of a 10-epoch cosine schedule.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = torch.amp.GradScaler("cuda")
    for epoch in range(9):
        refiner.train()
        losses = []
        for features, targets in loader:
            features, targets = features.cuda(), targets.cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = refiner(features)
                loss = 0.45 * F.binary_cross_entropy_with_logits(
                    logits, targets
                ) + 0.55 * base.soft_dice_loss(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite refiner loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        torch.save(refiner.state_dict(), models / "refiner.pt")
        progress("refiner_training", epoch=epoch + 1, loss=float(np.mean(losses)))
    del refiner, optimizer, scaler, loader
    release()
    return models


def alternatives(image, seed, refined, detector_score, refiner_score, proposal_id):
    rows = []
    for kind, mask in [("seed", seed), ("refined", refined)]:
        if mask.sum() < 32:
            continue
        rows.append(
            {
                "proposal_id": proposal_id,
                "kind": kind,
                "rle": core.encode(mask),
                "area": int(mask.sum()),
                "detector_score": float(detector_score),
                "features": core.mask_features(
                    image, seed, mask, detector_score, refiner_score, kind
                ),
            }
        )
    return rows


@torch.inference_mode()
def infer_candidates(names, image_dir, models, cfg, indices=None):
    from ultralytics import YOLO

    detector = YOLO(str(models / "detector.pt"))
    refiner = base.SeedRefiner(pretrained=False).cuda().eval()
    refiner.load_state_dict(
        torch.load(models / "refiner.pt", weights_only=True, map_location="cuda")
    )
    cache = {}
    for number, filename in enumerate(names, 1):
        image = cv2.imread(str(image_dir / filename), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (2048, 2048):
            raise ValueError(f"Invalid image {filename}")
        prediction = detector.predict(
            str(image_dir / filename),
            imgsz=1280,
            conf=0.03,
            iou=0.6,
            agnostic_nms=True,
            retina_masks=True,
            max_det=100,
            verbose=False,
        )[0]
        candidates = []
        if prediction.masks is not None:
            masks = [mask.cpu().numpy().astype(bool) for mask in prediction.masks.data]
            masks = [
                cv2.resize(
                    mask.astype(np.uint8), (2048, 2048), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                for mask in masks
            ]
            scores = prediction.boxes.conf.cpu().numpy().tolist()
            masks, scores = base.remove_duplicate_seeds(masks, scores)
            for pid, (seed, score) in enumerate(zip(masks, scores, strict=True)):
                if seed.sum() < 32:
                    continue
                refined, confidence = base.refine_seed(
                    image, seed, refiner, torch.device("cuda"), cfg
                )
                candidates.extend(alternatives(image, seed, refined, score, confidence, pid))
        if indices is not None:
            core.annotate_targets(candidates, truth_for(filename, indices))
        cache[filename] = candidates
        if number % 10 == 0 or number == len(names):
            write_json(WORK / "candidates.partial.json", cache)
            progress("candidate_inference", images_processed=number, images_total=len(names))
    del detector, refiner
    release()
    return cache


def verify_fold_report(report, fold):
    if report.get("status") != "complete" or report.get("run_id") != CONFIG["run_id"]:
        raise ValueError("Incomplete or unrelated fold report")
    if report["source_sha256"] != CONFIG["source_sha256"]:
        raise ValueError("Fold source mismatch")
    if report["train_names"] != fold["train"] or report["held_names"] != fold["held"]:
        raise ValueError("Fold lineage mismatch")
    if set(report["train_names"]) & set(report["held_names"]):
        raise ValueError("Fold contains leakage")


def baseline_masks(items):
    masks, scores = [], []
    for item in items:
        value = float(item["detector_score"])
        if not item.get("long_object_path", False):
            value *= np.sqrt(max(float(item["refiner_score"]), 0.0))
        if value >= 0.3 and item["refined_area"] >= 64:
            masks.append(base.decode_json_rle(item["refined"]))
            scores.append(value)
    return base.resolve_overlaps(masks, scores, 64)[0]


def assess(paths, indices):
    oof = {}
    for fold in CONFIG["folds"]:
        report_path = source_file(f"fold-{fold['fold']}-report.json")
        report = json.loads(report_path.read_text())
        verify_fold_report(report, fold)
        candidate_path = source_file(
            f"fold-{fold['fold']}-candidates.json", report["candidates_sha256"]
        )
        candidates = json.loads(candidate_path.read_text())
        if set(candidates) != set(fold["held"]) or set(oof) & set(candidates):
            raise ValueError("OOF coverage or duplicate membership mismatch")
        oof.update(candidates)
    if set(oof) != set(CONFIG["train_names"]):
        raise ValueError("Incomplete OOF training coverage")
    quality = core.fit_quality(oof)
    write_json(WORK / "quality.json", quality)
    progress("quality_trained", examples=quality["examples"])

    raw_path = source_file(
        "sfsc-yolo-refiner-validation/validation-candidates.json",
        CONFIG["baseline_candidates_sha256"],
    )
    raw = json.loads(raw_path.read_text())
    validation_names = CONFIG["validation_names"]
    if set(raw) != set(validation_names):
        raise ValueError("Control validation membership mismatch")
    truth = {name: truth_for(name, indices) for name in validation_names}
    caches, baseline_stats = {}, {}
    for number, filename in enumerate(validation_names, 1):
        image = cv2.imread(str(paths["train"] / filename), cv2.IMREAD_GRAYSCALE)
        caches[filename] = []
        for pid, item in enumerate(raw[filename]):
            seed = base.decode_json_rle(item["seed"])
            refined = base.decode_json_rle(item["refined"])
            caches[filename].extend(
                alternatives(
                    image, seed, refined, item["detector_score"], item["refiner_score"], pid
                )
            )
        baseline_stats[filename] = core.score(
            truth[filename], baseline_masks(raw[filename])
        ).tolist()
        if number % 10 == 0:
            progress("validation_features", images_processed=number)
    baseline = core.metrics(np.sum(list(baseline_stats.values()), axis=0))
    if abs(baseline["pq"] - 0.4351790286733054) > 1e-7:
        raise ValueError(f"Control PQ failed reproduction: {baseline}")
    # Calibration labels may select one threshold, never fit the quality learner.
    grid = []
    for threshold in [0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40]:
        total = np.zeros(8)
        for name in CONFIG["calibration_names"]:
            total += core.score(truth[name], core.assemble(caches[name], quality, threshold))
        grid.append({"threshold": threshold, **core.metrics(total)})
    selected = max(grid, key=lambda row: (row["pq"], row["threshold"]))
    write_json(WORK / "selection.json", selected)
    candidate_stats = {
        name: core.score(
            truth[name], core.assemble(caches[name], quality, selected["threshold"])
        ).tolist()
        for name in validation_names
    }
    other = sorted(set(validation_names) - set(CONFIG["calibration_names"]))
    full = core.metrics(np.sum(list(candidate_stats.values()), axis=0))
    assessment = core.metrics(np.sum([candidate_stats[n] for n in other], axis=0))
    control_other = core.metrics(np.sum([baseline_stats[n] for n in other], axis=0))
    interval = core.bootstrap_delta(candidate_stats, baseline_stats, other)
    gate = (
        full["pq"] >= 0.460
        and assessment["pq"] >= 0.455
        and assessment["pq"] - control_other["pq"] >= 0.02
        and interval["ci95"][0] > 0
        and abs(full["pq"] - full["one_to_one_pq"]) < 1e-8
    )
    report = {
        "status": "complete",
        "run_id": CONFIG["run_id"],
        "source_sha256": CONFIG["source_sha256"],
        "selected": selected,
        "quality_sha256": sha(WORK / "quality.json"),
        "baseline": baseline,
        "full_validation": full,
        "assessment82": assessment,
        "control82": control_other,
        "paired_block_bootstrap": interval,
        "full_refit_gate": gate,
        "calibration_grid": grid,
        "baseline_per_image": baseline_stats,
        "candidate_per_image": candidate_stats,
        "train_names": CONFIG["train_names"],
        "calibration_names": CONFIG["calibration_names"],
        "assessment_names": other,
    }
    write_json(WORK / "assessment-report.json", report)
    return report


def final_inference(paths, models, cfg, validation):
    quality_path = source_file("consensus/quality.json", validation["quality_sha256"])
    quality = json.loads(quality_path.read_text())
    names = sorted(p.name for p in paths["test"].glob("*.jpeg"))
    if len(names) != 180:
        raise ValueError("Wrong test-image count")
    cache = infer_candidates(names, paths["test"], models, cfg)
    threshold = validation["selected"]["threshold"]
    counts = {}
    with (WORK / "submission.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filament_id", "segmentation_rle"])
        for name in names:
            masks = core.assemble(cache[name], quality, threshold)
            counts[name] = len(masks)
            occupied = np.zeros((2048, 2048), bool)
            for index, mask in enumerate(masks, 1):
                rle = core.encode(mask)
                if np.any(mask & occupied) or not np.array_equal(core.decode(rle), mask):
                    raise ValueError("Submission mask integrity failure")
                occupied |= mask
                writer.writerow([f"{Path(name).stem}_{index}", rle["counts"]])
    if not sum(counts.values()):
        raise ValueError("All-empty artifact")
    return {
        "status": "complete",
        "run_id": CONFIG["run_id"],
        "source_sha256": CONFIG["source_sha256"],
        "counts": counts,
        "submission_sha256": sha(WORK / "submission.csv"),
        "train_names": CONFIG["train_names"],
        "threshold": threshold,
        "quality_sha256": sha(quality_path),
        "detector_sha256": sha(models / "detector.pt"),
        "refiner_sha256": sha(models / "refiner.pt"),
    }


def main():
    try:
        import torchvision
        import ultralytics

        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (6, 0):
            raise RuntimeError("This job requires a Pascal P100 GPU")
        smoke = torch.ones((16, 16), device="cuda", requires_grad=True)
        (smoke @ smoke).sum().backward()
        progress(
            "gpu_verified",
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
            torchvision=torchvision.__version__,
            ultralytics=ultralytics.__version__,
        )
        cfg = base.CFG(seed=20260907)
        base.seed_everything(cfg.seed)
        paths = base.discover_competition_paths()
        allowed = set(CONFIG["train_names"]) | set(CONFIG.get("held_names", []))
        if CONFIG["stage"] == "assess":
            allowed |= set(CONFIG["validation_names"])
        coco = base.restrict_coco(base.load_coco(paths["json"]), allowed)
        indices = base.build_coco_indices(coco)
        stage = CONFIG["stage"]
        if stage.startswith("fold-"):
            train = CONFIG["train_names"]
            held = CONFIG["held_names"]
            if set(train) & set(held):
                raise ValueError("Fold overlap")
            training_indices = base.build_coco_indices(base.restrict_coco(coco, set(train)))
            models = train_fixed(train, paths, training_indices, cfg)
            cache = infer_candidates(held, paths["train"], models, cfg, indices)
            cache_path = WORK / f"{stage}-candidates.json"
            write_json(cache_path, cache)
            result = {
                "status": "complete",
                "run_id": CONFIG["run_id"],
                "source_sha256": CONFIG["source_sha256"],
                "train_names": train,
                "held_names": held,
                "candidates_sha256": sha(cache_path),
                "detector_sha256": sha(models / "detector.pt"),
                "refiner_sha256": sha(models / "refiner.pt"),
            }
            write_json(WORK / f"{stage}-report.json", result)
        elif stage == "assess":
            result = assess(paths, indices)
        elif stage == "final":
            validation = json.loads(source_file("consensus/assessment-report.json").read_text())
            if (
                (
                    not validation["full_refit_gate"]
                    and not CONFIG.get("manual_gate_override", False)
                )
                or validation["run_id"] != CONFIG["run_id"]
                or validation["source_sha256"] != CONFIG["source_sha256"]
            ):
                raise ValueError("Final training gate/lineage failed")
            models = train_fixed(CONFIG["train_names"], paths, indices, cfg)
            result = final_inference(paths, models, cfg, validation)
            result.update(
                manual_gate_override=bool(CONFIG.get("manual_gate_override", False)),
                assessment_full_refit_gate=validation["full_refit_gate"],
                execution_source_sha256=CONFIG["execution_source_sha256"],
            )
            write_json(WORK / "final-report.json", result)
        else:
            raise ValueError(f"Unknown stage {stage}")
        STATUS.update(status="complete", result=result)
        progress("complete")
    except Exception as error:
        STATUS.update(status="failed", error=str(error), traceback=traceback.format_exc())
        progress("failed")
        raise


if __name__ == "__main__":
    main()
