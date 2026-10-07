from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from pycocotools import mask as coco_mask
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from .config import Paths, sha256
from .data import FullDiskDataset, PromptDataset, annotation_mask
from .losses import multitask_loss
from .models import FullDiskProposalNet, PromptFPN


def emit_progress(event: str, **fields: Any) -> None:
    """Write one machine-readable progress event to the mirrored training log."""
    payload = {"event": event, "time": datetime.now(UTC).isoformat(), **fields}
    print("SFSC_PROGRESS " + json.dumps(payload, sort_keys=True), flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cv2.setNumThreads(0)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Archive member escapes destination: {member.filename}")
        bundle.extractall(destination)


def prepare_data(config: dict[str, Any], paths: Paths) -> Path:
    archive = paths.drive / config["data_archive"]
    manifest_path = paths.drive / config["data_manifest"]
    if not archive.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Dataset archive and manifest must already exist on Google Drive")
    manifest = json.loads(manifest_path.read_text())
    if sha256(archive) != manifest["zip"]["sha256"]:
        raise ValueError("Drive dataset archive hash mismatch")
    data_root = paths.scratch / "data"
    marker = data_root / ".complete"
    if not marker.is_file() or marker.read_text().strip() != manifest["zip"]["sha256"]:
        shutil.rmtree(data_root, ignore_errors=True)
        safe_extract(archive, data_root)
        marker.write_text(manifest["zip"]["sha256"] + "\n")
    roots = list(data_root.glob("**/SFSCDataset"))
    if (data_root / "train").is_dir():
        return data_root
    if len(roots) != 1:
        raise FileNotFoundError("Could not identify the extracted SFSCDataset root")
    return roots[0]


def read_split(path: Path) -> dict[str, list[str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        name: sorted(row["filename"] for row in rows if row["split"] == name)
        for name in ("train", "validation")
    }


def fold_for(filename: str, folds: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{filename}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def coco_records(coco: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict]]]:
    images = {str(row["id"]): row for row in coco["images"]}
    annotations: dict[str, list[dict]] = {}
    for row in coco["annotations"]:
        annotations.setdefault(str(row["image_id"]), []).append(row)
    by_filename: dict[str, list[dict]] = {}
    for image_id, image in images.items():
        by_filename.setdefault(Path(image["file_name"]).name, []).append(
            {"image": image, "annotations": annotations.get(image_id, [])}
        )
    return images, by_filename


def medoid_record(records: Sequence[dict]) -> dict:
    if len(records) == 1:
        return records[0]
    masks = []
    for record in records:
        image = record["image"]
        shape = (int(image["height"]), int(image["width"]))
        union = np.zeros(shape, dtype=bool)
        for annotation in record["annotations"]:
            if not annotation.get("iscrowd", 0):
                union |= annotation_mask(annotation, shape)
        masks.append(union)
    costs = []
    for left in masks:
        cost = 0.0
        for right in masks:
            intersection = np.logical_and(left, right).sum()
            denominator = left.sum() + right.sum()
            cost += 1 - (2 * intersection / denominator if denominator else 1.0)
        costs.append(cost)
    return records[int(np.argmin(costs))]


def write_yolo_dataset(
    data_root: Path,
    coco: dict[str, Any],
    train_names: Sequence[str],
    validation_names: Sequence[str],
    output: Path,
) -> Path:
    _, records = coco_records(coco)
    image_root = data_root / "train/train_images"
    for split_name, filenames in (("train", train_names), ("val", validation_names)):
        images_dir = output / split_name / "images"
        labels_dir = output / split_name / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            source = image_root / filename
            destination = images_dir / filename
            if not destination.exists():
                os.symlink(source, destination)
            record = medoid_record(records[filename])
            image = record["image"]
            height, width = int(image["height"]), int(image["width"])
            lines = []
            for annotation in record["annotations"]:
                if annotation.get("iscrowd", 0):
                    continue
                mask = annotation_mask(annotation, (height, width)).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if len(contour) < 3 or cv2.contourArea(contour) < 4:
                        continue
                    points = contour[:, 0, :].astype(np.float32)
                    points[:, 0] /= width
                    points[:, 1] /= height
                    lines.append("0 " + " ".join(f"{value:.7f}" for value in points.ravel()))
            (labels_dir / f"{Path(filename).stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
    yaml = output / "data.yaml"
    yaml.write_text(
        f"path: {output.resolve()}\ntrain: train/images\nval: val/images\nnames:\n  0: filament\n"
    )
    return yaml


def train_detectors(
    config: dict[str, Any], paths: Paths, data_root: Path, split: dict[str, list[str]], coco: dict
) -> list[Path]:
    from ultralytics import YOLO

    outputs = []
    for index, detector in enumerate(config["detectors"]):
        detector_train, detector_validation = inner_detector_split(
            split["train"], index, int(config["seed"])
        )
        dataset = write_yolo_dataset(
            data_root,
            coco,
            detector_train,
            detector_validation,
            paths.scratch / "yolo" / f"detector{index}",
        )
        name = f"detector-{index}-{Path(detector['model']).stem}"
        directory = paths.checkpoints / "detectors" / name
        best = directory / "weights/best.pt"
        complete = directory / "complete.json"
        if best.is_file() and complete.is_file():
            outputs.append(best)
            continue
        last = directory / "weights/last.pt"
        model = YOLO(str(last) if last.is_file() else detector["model"])
        arguments = dict(
            data=str(dataset),
            epochs=int(detector["epochs"]),
            imgsz=int(detector["image_size"]),
            batch=int(detector["batch"]),
            device=0,
            seed=int(config["seed"]) + int(detector["seed_offset"]),
            single_cls=True,
            mask_ratio=2,
            overlap_mask=False,
            optimizer="AdamW",
            lr0=5e-4,
            lrf=0.02,
            cos_lr=True,
            warmup_epochs=3,
            weight_decay=5e-4,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.25,
            degrees=15,
            translate=0.08,
            scale=0.3,
            fliplr=0.5,
            flipud=0.5,
            mosaic=0.2,
            close_mosaic=10,
            patience=20,
            amp=True,
            workers=int(config["num_workers"]),
            project=str(directory.parent),
            name=directory.name,
            exist_ok=True,
            save_period=5,
            plots=False,
        )
        model.train(resume=True) if last.is_file() else model.train(**arguments)
        best = Path(model.trainer.best)
        write_json(complete, {"status": "complete", "sha256": sha256(best)})
        outputs.append(best)
    return outputs


def encode_mask(mask: np.ndarray) -> dict[str, Any]:
    encoded = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": list(encoded["size"]), "counts": encoded["counts"].decode("ascii")}


def native_rle(encoded: dict[str, Any]) -> dict[str, Any]:
    counts = encoded["counts"]
    return {
        "size": list(encoded["size"]),
        "counts": counts.encode("ascii") if isinstance(counts, str) else counts,
    }


def safely_negative(
    seed: dict[str, Any], truths: Sequence[dict[str, Any]], maximum_overlap: float
) -> bool:
    seed = native_rle(seed)
    seed_area = int(coco_mask.area(seed))
    if not seed_area:
        return False
    for truth in truths:
        truth = native_rle(truth)
        intersection = int(coco_mask.area(coco_mask.merge([seed, truth], intersect=True)))
        union = int(coco_mask.area(coco_mask.merge([seed, truth])))
        if intersection / max(union, 1) > maximum_overlap:
            return False
        if intersection / max(int(coco_mask.area(truth)), 1) > maximum_overlap:
            return False
    return True


def inner_detector_split(names: Sequence[str], fold: int, seed: int) -> tuple[list[str], list[str]]:
    validation = [name for name in names if fold_for(name, 10, seed + fold * 997) == 0]
    if not validation:
        validation = [sorted(names)[0]]
    training = sorted(set(names) - set(validation))
    if not training:
        raise ValueError("OOF detector inner split has no training images")
    return training, sorted(validation)


def train_oof_detector(
    config: dict[str, Any],
    paths: Paths,
    data_root: Path,
    coco: dict,
    fitting_names: Sequence[str],
    fold: int,
) -> Path:
    from ultralytics import YOLO

    detector = config["prompt_oof_detector"]
    directory = paths.checkpoints / "prompt-detectors" / f"fold{fold}"
    best = directory / "weights/best.pt"
    complete = directory / "complete.json"
    if best.is_file() and complete.is_file():
        return best
    train_names, validation_names = inner_detector_split(fitting_names, fold, int(config["seed"]))
    dataset = write_yolo_dataset(
        data_root,
        coco,
        train_names,
        validation_names,
        paths.scratch / "prompt-yolo" / f"fold{fold}",
    )
    last = directory / "weights/last.pt"
    model = YOLO(str(last) if last.is_file() else detector["model"])
    arguments = dict(
        data=str(dataset),
        epochs=int(detector["epochs"]),
        imgsz=int(detector["image_size"]),
        batch=int(detector["batch"]),
        device=0,
        seed=int(config["seed"]) + 10_000 + fold,
        single_cls=True,
        mask_ratio=2,
        overlap_mask=False,
        optimizer="AdamW",
        lr0=5e-4,
        lrf=0.02,
        cos_lr=True,
        warmup_epochs=3,
        weight_decay=5e-4,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.25,
        degrees=15,
        translate=0.08,
        scale=0.3,
        fliplr=0.5,
        flipud=0.5,
        mosaic=0.2,
        close_mosaic=10,
        patience=15,
        amp=True,
        workers=int(config["num_workers"]),
        project=str(directory.parent),
        name=directory.name,
        exist_ok=True,
        save_period=5,
        plots=False,
    )
    model.train(resume=True) if last.is_file() else model.train(**arguments)
    if not best.is_file():
        raise RuntimeError(f"OOF detector fold {fold} did not produce {best}")
    write_json(complete, {"status": "complete", "sha256": sha256(best)})
    return best


@torch.inference_mode()
def collect_oof_prompts(
    config: dict[str, Any],
    detector_path: Path,
    data_root: Path,
    coco: dict,
    holdout_names: Sequence[str],
    fold: int,
) -> list[dict[str, Any]]:
    from ultralytics import YOLO

    _, records_by_name = coco_records(coco)
    detector = YOLO(str(detector_path))
    image_root = data_root / "train/train_images"
    detector_config = config["prompt_oof_detector"]
    prompts = []
    total_names = len(holdout_names)
    for name_index, name in enumerate(holdout_names, 1):
        image = cv2.imread(str(image_root / name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_root / name)
        result = detector.predict(
            image,
            imgsz=int(detector_config["image_size"]),
            conf=float(config["detector_conf"]),
            iou=float(config["detector_iou"]),
            agnostic_nms=True,
            retina_masks=True,
            max_det=150,
            device=0,
            half=False,
            verbose=False,
        )[0]
        seeds = []
        if result.masks is not None:
            for value in result.masks.data.detach().cpu().numpy():
                if value.shape != image.shape:
                    value = cv2.resize(value, image.shape[::-1], interpolation=cv2.INTER_LINEAR)
                mask = value >= 0.5
                if mask.any():
                    seeds.append(mask)

        record = medoid_record(records_by_name[name])
        info = record["image"]
        shape = (int(info["height"]), int(info["width"]))
        truths = [
            annotation_mask(annotation, shape)
            for annotation in record["annotations"]
            if not annotation.get("iscrowd", 0)
        ]
        truth_rles = [encode_mask(mask) for mask in truths]
        seed_rles = [encode_mask(mask) for mask in seeds]
        ious = (
            np.asarray(
                coco_mask.iou(
                    [native_rle(row) for row in truth_rles],
                    [native_rle(row) for row in seed_rles],
                    [0] * len(seed_rles),
                )
            )
            if truth_rles and seed_rles
            else np.zeros((len(truth_rles), len(seed_rles)), dtype=np.float32)
        )
        all_truth_rles = []
        for annotation_record in records_by_name[name]:
            annotation_info = annotation_record["image"]
            annotation_shape = (
                int(annotation_info["height"]),
                int(annotation_info["width"]),
            )
            all_truth_rles.extend(
                encode_mask(annotation_mask(annotation, annotation_shape))
                for annotation in annotation_record["annotations"]
                if not annotation.get("iscrowd", 0)
            )
        rows, columns = linear_sum_assignment(ious, maximize=True)
        assigned = {
            int(seed_index): int(truth_index)
            for truth_index, seed_index in zip(rows, columns, strict=True)
            if ious[truth_index, seed_index] > float(config["prompt_positive_iou"])
        }
        for seed_index, seed_rle in enumerate(seed_rles):
            if seed_index in assigned:
                target = truth_rles[assigned[seed_index]]
            elif safely_negative(
                seed_rle,
                all_truth_rles,
                float(config["prompt_negative_maximum_overlap"]),
            ):
                target = None
            else:
                continue
            prompts.append(
                {
                    "filename": name,
                    "seed": seed_rle,
                    "target": target,
                    "annotation_record": str(info["id"]),
                    "oof_fold": fold,
                }
            )
        emit_progress(
            "progress",
            phase="prompt_collection",
            unit=f"fold{fold}",
            current=name_index,
            total=total_names,
            filename=name,
            prompts=len(prompts),
        )
    return prompts


def generate_oof_prompts(
    config: dict[str, Any],
    paths: Paths,
    data_root: Path,
    split: dict[str, list[str]],
    coco: dict,
) -> Path:
    output = paths.cache / "prompts/oof-train-prompts.json"
    if output.is_file():
        return output
    folds = int(config["prompt_oof_folds"])
    all_prompts = []
    checkpoints = {}
    assignments = {}
    for fold in range(folds):
        emit_progress(
            "unit_start",
            phase="prompts",
            unit=f"fold{fold}",
            unit_index=fold + 1,
            unit_total=folds,
        )
        holdout = [
            name for name in split["train"] if fold_for(name, folds, int(config["seed"])) == fold
        ]
        fitting = sorted(set(split["train"]) - set(holdout))
        checkpoint = train_oof_detector(config, paths, data_root, coco, fitting, fold)
        checkpoints[str(fold)] = {"path": str(checkpoint), "sha256": sha256(checkpoint)}
        assignments[str(fold)] = holdout
        all_prompts.extend(collect_oof_prompts(config, checkpoint, data_root, coco, holdout, fold))
        emit_progress(
            "unit_complete",
            phase="prompts",
            unit=f"fold{fold}",
            unit_index=fold + 1,
            unit_total=folds,
            prompts=len(all_prompts),
        )
    if not all_prompts or not any(row["target"] is not None for row in all_prompts):
        raise RuntimeError("OOF detectors produced no supported positive prompts")
    document = {
        "source": "out-of-fold-detectors",
        "folds": folds,
        "assignments": assignments,
        "checkpoints": checkpoints,
        "positives": sum(row["target"] is not None for row in all_prompts),
        "negatives": sum(row["target"] is None for row in all_prompts),
        "prompts": all_prompts,
    }
    write_json(output, document)
    return output


def load_prompts(config: dict[str, Any], paths: Paths) -> list[dict[str, Any]]:
    oof = paths.cache / "prompts/oof-train-prompts.json"
    if oof.is_file():
        return json.loads(oof.read_text())["prompts"]
    artifact = paths.drive / config["prompt_artifact"]
    if artifact.is_file():
        with zipfile.ZipFile(artifact) as archive:
            document = json.loads(archive.read("train-prompts.json"))
        return document["prompts"]
    prompt_files = sorted(paths.cache.glob("**/train-prompts.json"))
    if not prompt_files:
        if config.get("require_real_prompts", False):
            raise FileNotFoundError(f"Real-prompt artifact is required on Drive: {artifact}")
        return []
    documents = [json.loads(path.read_text()) for path in prompt_files]
    return [row for document in documents for row in document.get("prompts", [])]


def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float = 0.995) -> None:
    with torch.no_grad():
        for target, source in zip(ema.parameters(), model.parameters(), strict=True):
            target.lerp_(source, 1 - decay)
        for target, source in zip(ema.buffers(), model.buffers(), strict=True):
            target.copy_(source)


def train_one_refiner(
    config: dict[str, Any],
    paths: Paths,
    data_root: Path,
    coco: dict,
    train_names: list[str],
    validation_names: list[str],
    prompts: list[dict],
    backbone: str,
    fold: int,
    unit_index: int,
    unit_total: int,
) -> Path:
    identifier = backbone.replace("/", "_").replace(".", "-")
    output = paths.checkpoints / "refiners" / f"{identifier}-fold{fold}.pt"
    complete = output.with_suffix(".complete.json")
    if output.is_file() and complete.is_file():
        emit_progress(
            "unit_complete",
            phase="refiners",
            unit=f"{identifier}-fold{fold}",
            unit_index=unit_index,
            unit_total=unit_total,
            skipped=True,
        )
        return output
    dataset = PromptDataset(
        data_root / "train/train_images",
        coco,
        train_names,
        int(config["refiner_crop"]),
        config["refiner_contexts"],
        prompts,
        True,
        float(config["real_prompt_fraction"]),
    )
    validation = PromptDataset(
        data_root / "train/train_images",
        coco,
        validation_names,
        int(config["refiner_crop"]),
        config["refiner_contexts"],
        prompts,
        False,
        float(config["real_prompt_fraction"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["refiner_batch"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
        drop_last=True,
        persistent_workers=bool(config["num_workers"]),
    )
    validation_loader = DataLoader(validation, batch_size=2, shuffle=False, num_workers=2)
    model = PromptFPN(backbone).cuda()
    ema = PromptFPN(backbone, pretrained=False).cuda()
    ema.load_state_dict(model.state_dict())
    ema.eval()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["refiner_lr"]),
        weight_decay=float(config["refiner_weight_decay"]),
    )
    total_steps = int(config["refiner_epochs"]) * int(config["refiner_steps_per_epoch"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=float(config["refiner_lr"]), total_steps=total_steps, pct_start=0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["refiner_amp"]))
    best = float("inf")
    start_epoch = 0
    last = output.with_suffix(".last.pt")
    if last.is_file():
        state = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(state["training_model"])
        ema.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"])
        best = float(state["best_validation_loss"])

    def batches():
        while True:
            yield from loader

    stream = batches()
    output.parent.mkdir(parents=True, exist_ok=True)
    epochs = int(config["refiner_epochs"])
    steps_per_epoch = int(config["refiner_steps_per_epoch"])
    emit_progress(
        "unit_start",
        phase="refiners",
        unit=f"{identifier}-fold{fold}",
        unit_index=unit_index,
        unit_total=unit_total,
        epoch=start_epoch,
        epochs=epochs,
    )
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = torch.zeros((), device="cuda")
        for step in range(1, steps_per_epoch + 1):
            features, target = next(stream)
            features, target = features.cuda(non_blocking=True), target.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(config["refiner_amp"])):
                prediction = model(features)
                loss, _ = multitask_loss(
                    prediction["mask"], prediction["boundary"], prediction["quality"], target
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            update_ema(ema, model)
            running_loss += loss.detach()
            if step % 20 == 0 or step == steps_per_epoch:
                completed = (
                    (unit_index - 1) * epochs * steps_per_epoch
                    + epoch * steps_per_epoch
                    + step
                )
                emit_progress(
                    "progress",
                    phase="refiners",
                    unit=f"{identifier}-fold{fold}",
                    unit_index=unit_index,
                    unit_total=unit_total,
                    epoch=epoch + 1,
                    epochs=epochs,
                    step=step,
                    steps=steps_per_epoch,
                    current=completed,
                    total=unit_total * epochs * steps_per_epoch,
                    train_loss=float(running_loss) / step,
                )
        value, batches = 0.0, 0
        with torch.inference_mode():
            for features, target in validation_loader:
                features, target = features.cuda(non_blocking=True), target.cuda(non_blocking=True)
                prediction = ema(features)
                loss, _ = multitask_loss(
                    prediction["mask"], prediction["boundary"], prediction["quality"], target
                )
                value += float(loss)
                batches += 1
        value /= max(batches, 1)
        checkpoint = {
            "model": ema.state_dict(),
            "backbone": backbone,
            "fold": fold,
            "epoch": epoch + 1,
            "validation_loss": value,
            "config": config,
        }
        if value < best:
            best = value
            torch.save(checkpoint, output)
        torch.save(
            {
                **checkpoint,
                "training_model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_validation_loss": best,
            },
            last,
        )
        emit_progress(
            "epoch_complete",
            phase="refiners",
            unit=f"{identifier}-fold{fold}",
            unit_index=unit_index,
            unit_total=unit_total,
            epoch=epoch + 1,
            epochs=epochs,
            train_loss=float(running_loss) / steps_per_epoch,
            validation_loss=value,
            best_validation_loss=best,
        )
    write_json(complete, {"status": "complete", "sha256": sha256(output)})
    emit_progress(
        "unit_complete",
        phase="refiners",
        unit=f"{identifier}-fold{fold}",
        unit_index=unit_index,
        unit_total=unit_total,
    )
    return output


def train_refiners(
    config: dict[str, Any], paths: Paths, data_root: Path, split: dict[str, list[str]], coco: dict
) -> list[Path]:
    prompts = load_prompts(config, paths)
    outputs = []
    folds = int(config["refiner_folds"])
    unit_total = len(config["refiner_backbones"]) * folds
    for backbone_index, backbone in enumerate(config["refiner_backbones"]):
        for fold in range(folds):
            unit_index = backbone_index * folds + fold + 1
            validation_names = [
                name
                for name in split["train"]
                if fold_for(name, folds, int(config["seed"])) == fold
            ]
            training_names = sorted(set(split["train"]) - set(validation_names))
            outputs.append(
                train_one_refiner(
                    config,
                    paths,
                    data_root,
                    coco,
                    training_names,
                    validation_names,
                    prompts,
                    backbone,
                    fold,
                    unit_index,
                    unit_total,
                )
            )
    return outputs


def train_semantic(
    config: dict[str, Any], paths: Paths, data_root: Path, split: dict[str, list[str]], coco: dict
) -> Path:
    output = paths.checkpoints / "semantic" / "full-disk.pt"
    complete = output.with_suffix(".complete.json")
    if output.is_file() and complete.is_file():
        emit_progress("unit_complete", phase="semantic", unit="full-disk", skipped=True)
        return output
    dataset = FullDiskDataset(
        data_root / "train/train_images", coco, split["train"], int(config["semantic_size"])
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["semantic_batch"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
        persistent_workers=bool(config["num_workers"]),
    )
    model = FullDiskProposalNet(config["semantic_backbone"]).cuda()
    ema = FullDiskProposalNet(config["semantic_backbone"], pretrained=False).cuda()
    ema.load_state_dict(model.state_dict())
    ema.eval()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["semantic_lr"]), weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["semantic_epochs"])
    )
    start_epoch = 0
    last = output.with_suffix(".last.pt")
    if last.is_file():
        state = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(state["training_model"])
        ema.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"])
    output.parent.mkdir(parents=True, exist_ok=True)
    epochs = int(config["semantic_epochs"])
    steps_per_epoch = len(loader)
    emit_progress(
        "unit_start",
        phase="semantic",
        unit="full-disk",
        epoch=start_epoch,
        epochs=epochs,
    )
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = torch.zeros((), device="cuda")
        for step, (features, target, boundary_target) in enumerate(loader, 1):
            features = features.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            boundary_target = boundary_target.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                prediction = model(features)
                loss, _ = multitask_loss(
                    prediction["mask"],
                    prediction["boundary"],
                    prediction["quality"],
                    target,
                    boundary_target,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema, model)
            running_loss += loss.detach()
            if step % 10 == 0 or step == steps_per_epoch:
                completed = epoch * steps_per_epoch + step
                emit_progress(
                    "progress",
                    phase="semantic",
                    unit="full-disk",
                    epoch=epoch + 1,
                    epochs=epochs,
                    step=step,
                    steps=steps_per_epoch,
                    current=completed,
                    total=epochs * steps_per_epoch,
                    train_loss=float(running_loss) / step,
                )
        scheduler.step()
        torch.save(
            {
                "model": ema.state_dict(),
                "training_model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch + 1,
                "config": config,
            },
            last,
        )
        emit_progress(
            "epoch_complete",
            phase="semantic",
            unit="full-disk",
            epoch=epoch + 1,
            epochs=epochs,
            train_loss=float(running_loss) / steps_per_epoch,
        )
    torch.save(
        {"model": ema.state_dict(), "epoch": int(config["semantic_epochs"]), "config": config},
        output,
    )
    write_json(complete, {"status": "complete", "sha256": sha256(output)})
    emit_progress("unit_complete", phase="semantic", unit="full-disk")
    return output
