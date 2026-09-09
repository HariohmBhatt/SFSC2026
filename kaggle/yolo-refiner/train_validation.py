# ruff: noqa: B905, E402, E501, E701, E702, I001, SIM300, UP035
"""Full leakage-safe Solar Filament train/OOF/test pipeline for Kaggle.

Primary metric: Panoptic Quality (competition update, 7 Aug 2026).
The module uses only official train annotations and H-alpha images.
"""

from __future__ import annotations

import gc
import base64
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--force-reinstall",
        "torch==2.7.1",
        "torchvision==0.22.1",
        "--index-url",
        TORCH_INDEX,
    ]
)
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--no-deps",
        "ultralytics==8.4.103",
        "pycocotools==2.0.11",
    ]
)

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


@dataclass
class CFG:
    seed: int = 20260906
    n_folds: int = 1
    image_size: int = 1280
    detector_model: str = "yolo11l-seg.pt"
    detector_epochs: int = 36
    detector_batch: int = 2
    detector_patience: int = 10
    detector_conf: float = 0.03
    detector_fallback_conf: float = 0.01
    detector_iou: float = 0.60
    mask_ratio: int = 2
    refiner_crop: int = 384
    refiner_context: float = 1.6
    refiner_min_crop: int = 128
    refiner_epochs: int = 10
    refiner_batch: int = 8
    refiner_lr: float = 3e-4
    refiner_threshold: float = 0.55
    minimum_area: int = 64
    pq_match_iou: float = 0.50
    ensemble_iou: float = 0.50
    minimum_fold_support: int = 2
    singleton_score: float = 0.90
    num_workers: int = 2
    use_tta: bool = False
    save_oof_masks: bool = False
    debug: bool = False


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def discover_competition_paths() -> dict[str, Path]:
    roots = sorted(Path("/kaggle/input").glob("**/MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if len(roots) != 1:
        raise RuntimeError(f"Expected one official train JSON, found {roots}")
    annotation_path = roots[0]
    data_root = annotation_path.parent.parent
    train_dir = data_root / "train" / "train_images"
    test_dir = data_root / "test" / "test_images"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(f"Invalid competition tree under {data_root}")
    return {"root": data_root, "json": annotation_path, "train": train_dir, "test": test_dir}


def load_coco(annotation_path: Path) -> dict[str, Any]:
    with annotation_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def restrict_coco(coco: Mapping[str, Any], filenames: set[str]) -> dict[str, Any]:
    images = [image for image in coco["images"] if image["file_name"] in filenames]
    image_ids = {str(image["id"]) for image in images}
    annotations = [
        annotation
        for annotation in coco["annotations"]
        if str(annotation["image_id"]) in image_ids
    ]
    return {"images": images, "annotations": annotations}


def build_coco_indices(coco: Mapping[str, Any]) -> dict[str, Any]:
    images_by_id = {str(image["id"]): image for image in coco["images"]}
    annotations_by_image: dict[str, list[Mapping[str, Any]]] = {}
    for annotation in coco["annotations"]:
        annotations_by_image.setdefault(
            str(annotation["image_id"]), []
        ).append(annotation)
    image_ids_by_file: dict[str, list[str]] = {}
    for image_id, image in images_by_id.items():
        image_ids_by_file.setdefault(image["file_name"], []).append(image_id)
    for filename in image_ids_by_file:
        image_ids_by_file[filename].sort()
    return {
        "images_by_id": images_by_id,
        "annotations_by_image": annotations_by_image,
        "image_ids_by_file": image_ids_by_file,
    }


def annotation_to_rle(annotation: Mapping[str, Any], height: int, width: int) -> dict[str, Any]:
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, dict):
        rle = dict(segmentation)
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        return rle
    polygons = segmentation or []
    if not polygons:
        return mask_utils.encode(np.asfortranarray(np.zeros((height, width), dtype=np.uint8)))
    rles = mask_utils.frPyObjects(polygons, height, width)
    return mask_utils.merge(rles)


def annotation_to_mask(annotation: Mapping[str, Any], height: int, width: int) -> np.ndarray:
    return mask_utils.decode(annotation_to_rle(annotation, height, width)).astype(bool)


def semantic_rle_for_record(
    image_id: int,
    images_by_id: Mapping[int, Mapping[str, Any]],
    annotations_by_image: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    image = images_by_id[image_id]
    rles = [
        annotation_to_rle(annotation, int(image["height"]), int(image["width"]))
        for annotation in annotations_by_image.get(image_id, [])
        if not annotation.get("iscrowd", 0)
    ]
    if not rles:
        return mask_utils.encode(
            np.asfortranarray(np.zeros((int(image["height"]), int(image["width"])), dtype=np.uint8))
        )
    return mask_utils.merge(rles)


def choose_medoid_annotators(indices: Mapping[str, Any]) -> dict[str, int]:
    """Choose the annotator closest to the others, avoiding max-area bias."""
    selected: dict[str, int] = {}
    for filename, image_ids in indices["image_ids_by_file"].items():
        if len(image_ids) == 1:
            selected[filename] = image_ids[0]
            continue
        union_rles = [
            semantic_rle_for_record(
                image_id, indices["images_by_id"], indices["annotations_by_image"]
            )
            for image_id in image_ids
        ]
        pairwise_iou = np.asarray(mask_utils.iou(union_rles, union_rles, [0] * len(union_rles)))
        mean_agreement = (pairwise_iou.sum(axis=1) - 1.0) / max(len(image_ids) - 1, 1)
        selected[filename] = image_ids[int(np.argmax(mean_agreement))]
    return selected


def build_observation_table(indices: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for filename, image_ids in sorted(indices["image_ids_by_file"].items()):
        counts = [len(indices["annotations_by_image"].get(image_id, [])) for image_id in image_ids]
        rows.append(
            {
                "filename": filename,
                "site": Path(filename).stem[-2:],
                "n_annotators": len(image_ids),
                "mean_instances": float(np.mean(counts)),
                "max_instances": max(counts, default=0),
            }
        )
    return pd.DataFrame(rows)


def assign_grouped_folds(observations: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    frame = observations.copy().sort_values("filename").reset_index(drop=True)
    count_bin = pd.qcut(
        frame["mean_instances"].rank(method="first"),
        q=min(5, len(frame)), labels=False, duplicates="drop",
    ).astype(int)
    frame["stratum"] = frame["site"] + "_q" + count_bin.astype(str)
    frame["fold"] = -1
    fold_sizes = np.zeros(cfg.n_folds, dtype=int)
    rng = np.random.default_rng(cfg.seed)
    strata = sorted(frame.groupby("stratum").groups.items(), key=lambda item: (-len(item[1]), item[0]))
    for _, row_indices in strata:
        row_indices = np.asarray(list(row_indices), dtype=int)
        rng.shuffle(row_indices)
        for row_index in row_indices:
            choices = np.flatnonzero(fold_sizes == fold_sizes.min())
            fold = int(rng.choice(choices))
            frame.loc[row_index, "fold"] = fold
            fold_sizes[fold] += 1
    assert frame["filename"].is_unique
    assert (frame["fold"] >= 0).all()
    return frame

# %%
def mask_to_yolo_polygon(mask: np.ndarray, width: int, height: int) -> str | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if len(contour) < 3:
        return None
    contour = contour.astype(np.float32)
    contour[:, 0] = np.clip(contour[:, 0] / width, 0, 1)
    contour[:, 1] = np.clip(contour[:, 1] / height, 0, 1)
    return "0 " + " ".join(f"{value:.6f}" for value in contour.ravel())


def write_yolo_split(
    split_dir: Path,
    filenames: Sequence[str],
    train_image_dir: Path,
    selected_image_ids: Mapping[str, int],
    indices: Mapping[str, Any],
) -> None:
    image_output = split_dir / "images"
    label_output = split_dir / "labels"
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        source = train_image_dir / filename
        destination = image_output / filename
        if not destination.exists():
            os.symlink(source, destination)
        image_id = selected_image_ids[filename]
        info = indices["images_by_id"][image_id]
        lines = []
        for annotation in indices["annotations_by_image"].get(image_id, []):
            if annotation.get("iscrowd", 0):
                continue
            mask = annotation_to_mask(annotation, int(info["height"]), int(info["width"]))
            line = mask_to_yolo_polygon(mask, int(info["width"]), int(info["height"]))
            if line is not None:
                lines.append(line)
        (label_output / f"{Path(filename).stem}.txt").write_text("\n".join(lines), encoding="utf-8")


def prepare_yolo_fold(
    fold: int,
    fold_manifest: pd.DataFrame,
    train_image_dir: Path,
    selected_image_ids: Mapping[str, int],
    indices: Mapping[str, Any],
    work_dir: Path,
) -> Path:
    fold_dir = work_dir / "yolo_data" / f"fold_{fold}"
    marker = fold_dir / ".ready"
    if marker.exists():
        return fold_dir / "data.yaml"
    train_files = fold_manifest.loc[fold_manifest["fold"] != fold, "filename"].tolist()
    val_files = [
        filename
        for filename in fold_manifest.loc[
            fold_manifest["fold"] == fold, "filename"
        ].tolist()
        if filename in CALIBRATION_FILES
    ]
    write_yolo_split(fold_dir / "train", train_files, train_image_dir, selected_image_ids, indices)
    write_yolo_split(fold_dir / "val", val_files, train_image_dir, selected_image_ids, indices)
    yaml_text = (
        f"path: {fold_dir}\n"
        "train: train/images\n"
        "val: val/images\n"
        "names:\n  0: filament\n"
    )
    yaml_path = fold_dir / "data.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    marker.write_text("ready", encoding="utf-8")
    return yaml_path


def train_detector(fold: int, data_yaml: Path, cfg: CFG, work_dir: Path) -> Path:
    from ultralytics import YOLO

    output_dir = work_dir / "models" / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "detector.pt"
    if final_path.exists():
        return final_path
    epochs = 2 if cfg.debug else cfg.detector_epochs
    model = YOLO(cfg.detector_model)
    gpu_count = torch.cuda.device_count()
    device: int | str = ",".join(map(str, range(gpu_count))) if gpu_count > 1 else (0 if gpu_count == 1 else "cpu")
    model.train(
        data=str(data_yaml), epochs=epochs, imgsz=cfg.image_size,
        batch=cfg.detector_batch, device=device, patience=cfg.detector_patience,
        seed=cfg.seed + fold, single_cls=True, mask_ratio=cfg.mask_ratio,
        overlap_mask=False, hsv_h=0.0, hsv_s=0.0, hsv_v=0.30,
        fliplr=0.5, flipud=0.5, degrees=10, scale=0.20, translate=0.05,
        mosaic=0.0, close_mosaic=0, optimizer="AdamW", lr0=0.001,
        lrf=0.05, cos_lr=True, warmup_epochs=2.0, weight_decay=0.0005,
        save_period=6, workers=cfg.num_workers, amp=True,
        project=str(work_dir / "detector_runs"), name=f"fold_{fold}",
        exist_ok=True, verbose=True,
    )
    shutil.copy2(str(model.trainer.best), final_path)
    shutil.copy2(str(model.trainer.best), output_dir / "detector-ap-best.pt")
    shutil.copy2(str(model.trainer.last), output_dir / "detector-last.pt")
    return final_path

# %%
def square_bounds(mask: np.ndarray, context: float, minimum: int) -> tuple[int, int, int, int]:
    height, width = mask.shape
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, width, height
    center_x = (xs.min() + xs.max() + 1) / 2
    center_y = (ys.min() + ys.max() + 1) / 2
    side = int(math.ceil(max(np.ptp(xs) + 1, np.ptp(ys) + 1) * context))
    side = min(max(side, minimum), min(height, width))
    x0, y0 = int(round(center_x - side / 2)), int(round(center_y - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0: x1 -= x0; x0 = 0
    if y0 < 0: y1 -= y0; y0 = 0
    if x1 > width: x0 -= x1 - width; x1 = width
    if y1 > height: y0 -= y1 - height; y1 = height
    return x0, y0, x1, y1


def jitter_seed_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    seed = mask.astype(np.uint8)
    radius = int(rng.integers(-5, 6))
    if radius != 0:
        kernel_size = 2 * abs(radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        seed = cv2.dilate(seed, kernel) if radius > 0 else cv2.erode(seed, kernel)
    shift_x, shift_y = int(rng.integers(-8, 9)), int(rng.integers(-8, 9))
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    seed = cv2.warpAffine(seed, matrix, (seed.shape[1], seed.shape[0]), flags=cv2.INTER_NEAREST)
    return seed.astype(bool) if seed.any() else mask.astype(bool)


class RefinerDataset(Dataset):
    def __init__(
        self,
        filenames: Sequence[str],
        image_dir: Path,
        selected_image_ids: Mapping[str, int],
        indices: Mapping[str, Any],
        cfg: CFG,
        augment: bool,
    ) -> None:
        self.image_dir, self.indices, self.cfg, self.augment = image_dir, indices, cfg, augment
        self.samples = []
        for filename in filenames:
            image_id = selected_image_ids[filename]
            for annotation in indices["annotations_by_image"].get(image_id, []):
                if not annotation.get("iscrowd", 0):
                    self.samples.append((filename, image_id, annotation))
            if augment:
                self.samples.append((filename, image_id, None))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        filename, image_id, annotation = self.samples[index]
        info = self.indices["images_by_id"][image_id]
        image = cv2.imread(str(self.image_dir / filename), cv2.IMREAD_GRAYSCALE)
        rng = np.random.default_rng(self.cfg.seed + index + random.randint(0, 1_000_000))
        if annotation is None:
            target = np.zeros(image.shape, dtype=bool)
            occupied = np.zeros(image.shape, dtype=bool)
            for neighbor in self.indices["annotations_by_image"].get(image_id, []):
                occupied |= annotation_to_mask(neighbor, *image.shape)
            residual = cv2.GaussianBlur(image, (0, 0), 25).astype(np.float32) - image
            residual[occupied | (image < 12)] = -np.inf
            center_y, center_x = np.unravel_index(int(np.argmax(residual)), image.shape)
            seed = np.zeros(image.shape, dtype=np.uint8)
            radius_x = int(rng.integers(8, 40))
            radius_y = int(rng.integers(5, 24))
            cv2.ellipse(seed, (int(center_x), int(center_y)), (radius_x, radius_y),
                        float(rng.uniform(0, 180)), 0, 360, 1, -1)
            seed = seed.astype(bool)
        else:
            target = annotation_to_mask(annotation, int(info["height"]), int(info["width"]))
            seed = jitter_seed_mask(target, rng) if self.augment else target
        x0, y0, x1, y1 = square_bounds(seed, self.cfg.refiner_context, self.cfg.refiner_min_crop)
        image_crop = cv2.resize(image[y0:y1, x0:x1], (self.cfg.refiner_crop,) * 2, interpolation=cv2.INTER_LINEAR)
        target_crop = cv2.resize(target[y0:y1, x0:x1].astype(np.uint8), (self.cfg.refiner_crop,) * 2, interpolation=cv2.INTER_NEAREST)
        seed_crop = cv2.resize(seed[y0:y1, x0:x1].astype(np.uint8), (self.cfg.refiner_crop,) * 2, interpolation=cv2.INTER_NEAREST)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image_crop)
        features = np.stack([image_crop, clahe, seed_crop * 255], axis=0).astype(np.float32) / 255.0
        if self.augment:
            k = random.randrange(4)
            features = np.rot90(features, k, axes=(1, 2)).copy()
            target_crop = np.rot90(target_crop, k).copy()
            if random.random() < 0.5:
                features = features[:, :, ::-1].copy(); target_crop = target_crop[:, ::-1].copy()
            gain = random.uniform(0.9, 1.1)
            features[:2] = np.clip(features[:2] * gain, 0, 1)
        return torch.from_numpy(features), torch.from_numpy(target_crop[None].astype(np.float32))


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True),
        )


class SeedRefiner(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        if pretrained:
            with torch.no_grad():
                encoder.conv1.weight[:, 2].zero_()
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool, self.layer1, self.layer2, self.layer3, self.layer4 = (
            encoder.maxpool, encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4
        )
        self.up4, self.dec4 = nn.ConvTranspose2d(512, 256, 2, 2), ConvBlock(512, 256)
        self.up3, self.dec3 = nn.ConvTranspose2d(256, 128, 2, 2), ConvBlock(256, 128)
        self.up2, self.dec2 = nn.ConvTranspose2d(128, 64, 2, 2), ConvBlock(128, 64)
        self.up1, self.dec1 = nn.ConvTranspose2d(64, 32, 2, 2), ConvBlock(96, 32)
        self.up0, self.head = nn.ConvTranspose2d(32, 16, 2, 2), nn.Conv2d(16, 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.0]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 1.0]).view(1, 3, 1, 1))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = (tensor - self.mean) / self.std
        x0 = self.stem(tensor); x1 = self.layer1(self.pool(x0)); x2 = self.layer2(x1)
        x3 = self.layer3(x2); x4 = self.layer4(x3)
        x = self.dec4(torch.cat([self.up4(x4), x3], dim=1))
        x = self.dec3(torch.cat([self.up3(x), x2], dim=1))
        x = self.dec2(torch.cat([self.up2(x), x1], dim=1))
        x = self.dec1(torch.cat([self.up1(x), x0], dim=1))
        return self.head(self.up0(x))


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()


def train_refiner(
    fold: int,
    fold_manifest: pd.DataFrame,
    image_dir: Path,
    selected_image_ids: Mapping[str, int],
    indices: Mapping[str, Any],
    cfg: CFG,
    work_dir: Path,
) -> Path:
    output_dir = work_dir / "models" / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "refiner.pt"
    if checkpoint.exists():
        return checkpoint
    train_files = fold_manifest.loc[fold_manifest["fold"] != fold, "filename"].tolist()
    val_files = [
        filename
        for filename in fold_manifest.loc[
            fold_manifest["fold"] == fold, "filename"
        ].tolist()
        if filename in CALIBRATION_FILES
    ]
    if cfg.debug:
        train_files, val_files = train_files[:24], val_files[:8]
    train_dataset = RefinerDataset(train_files, image_dir, selected_image_ids, indices, cfg, True)
    val_dataset = RefinerDataset(val_files, image_dir, selected_image_ids, indices, cfg, False)
    train_loader = DataLoader(train_dataset, batch_size=cfg.refiner_batch, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.refiner_batch * 2, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SeedRefiner(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.refiner_lr, weight_decay=1e-4)
    epochs = 2 if cfg.debug else cfg.refiner_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    best_dice = -1.0
    best_epoch = 0
    for epoch in range(epochs):
        model.train()
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(features)
                loss = 0.45 * F.binary_cross_entropy_with_logits(logits, targets) + 0.55 * soft_dice_loss(logits, targets)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        model.eval(); dice_total = sample_count = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device).bool()
                predictions = model(features).sigmoid() >= cfg.refiner_threshold
                intersection = (predictions & targets).sum(dim=(1, 2, 3)).float()
                denominator = predictions.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
                dice_total += ((2 * intersection + 1e-7) / (denominator + 1e-7)).sum().item()
                sample_count += len(features)
        validation_dice = dice_total / max(sample_count, 1)
        print(f"fold={fold} refiner_epoch={epoch + 1} val_crop_dice={validation_dice:.5f}")
        if validation_dice > best_dice:
            best_dice = validation_dice
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint)
        if epoch + 1 in {5, epochs}:
            torch.save(model.state_dict(), output_dir / f"refiner-epoch-{epoch + 1}.pt")
        scheduler.step()
    shutil.copy2(checkpoint, output_dir / "refiner-crop-best.pt")
    (output_dir / "refiner-crop-selection.json").write_text(
        json.dumps({"best_epoch": best_epoch, "best_dice": best_dice}, indent=2),
        encoding="utf-8",
    )
    return checkpoint

# %%
def component_count(mask: np.ndarray) -> int:
    return max(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1, 0)


def binary_dice(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    return float((2 * intersection + 1e-7) / (first.sum() + second.sum() + 1e-7))


@torch.no_grad()
def refine_seed(
    image: np.ndarray,
    seed_mask: np.ndarray,
    model: SeedRefiner,
    device: torch.device,
    cfg: CFG,
) -> tuple[np.ndarray, float]:
    x0, y0, x1, y1 = square_bounds(seed_mask, cfg.refiner_context, cfg.refiner_min_crop)
    image_crop = cv2.resize(image[y0:y1, x0:x1], (cfg.refiner_crop,) * 2, interpolation=cv2.INTER_LINEAR)
    seed_crop = cv2.resize(seed_mask[y0:y1, x0:x1].astype(np.uint8), (cfg.refiner_crop,) * 2, interpolation=cv2.INTER_NEAREST)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image_crop)
    features = np.stack([image_crop, clahe, seed_crop * 255], axis=0).astype(np.float32) / 255.0
    tensor = torch.from_numpy(features[None]).to(device)
    transforms = [lambda value: value]
    inverses = [lambda value: value]
    if cfg.use_tta:
        transforms += [lambda value: value.flip(-1), lambda value: value.flip(-2), lambda value: value.flip((-2, -1))]
        inverses += [lambda value: value.flip(-1), lambda value: value.flip(-2), lambda value: value.flip((-2, -1))]
    probabilities = []
    for transform, inverse in zip(transforms, inverses):
        probability = inverse(model(transform(tensor)).sigmoid())[0, 0].cpu().numpy()
        probabilities.append(probability)
    base_probability = probabilities[0]
    candidate_probability = np.mean(probabilities, axis=0)
    base_mask = base_probability >= cfg.refiner_threshold
    candidate_mask = candidate_probability >= cfg.refiner_threshold
    use_candidate = True
    if base_mask.any():
        area_ratio = candidate_mask.sum() / max(base_mask.sum(), 1)
        use_candidate = (
            0.60 <= area_ratio <= 1.70
            and binary_dice(base_mask, candidate_mask) >= 0.62
            and component_count(candidate_mask) <= component_count(base_mask) + 1
        )
    probability = candidate_probability if use_candidate else base_probability
    resized_probability = cv2.resize(probability, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    crop_mask = resized_probability >= cfg.refiner_threshold
    full_mask = np.zeros_like(seed_mask, dtype=bool)
    full_mask[y0:y1, x0:x1] = crop_mask
    confidence = float(resized_probability[crop_mask].mean()) if crop_mask.any() else 0.0
    return full_mask, confidence


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / union) if union else 0.0


def remove_duplicate_seeds(masks: list[np.ndarray], scores: list[float], threshold: float = 0.60) -> tuple[list[np.ndarray], list[float]]:
    kept_masks, kept_scores = [], []
    for index in np.argsort(scores)[::-1]:
        mask = masks[int(index)]
        if all(mask_iou(mask, kept) < threshold for kept in kept_masks):
            kept_masks.append(mask); kept_scores.append(float(scores[int(index)]))
    return kept_masks, kept_scores


def resolve_overlaps(masks: list[np.ndarray], scores: list[float], minimum_area: int) -> tuple[list[np.ndarray], list[float]]:
    if not masks:
        return [], []
    assigned = np.zeros_like(masks[0], dtype=bool)
    output_masks, output_scores = [], []
    for index in np.argsort(scores)[::-1]:
        clean = masks[int(index)].astype(bool) & ~assigned
        if clean.sum() >= minimum_area:
            output_masks.append(clean); output_scores.append(float(scores[int(index)])); assigned |= clean
    return output_masks, output_scores


def load_fold_models(fold: int, cfg: CFG, work_dir: Path) -> tuple[Any, SeedRefiner, torch.device]:
    from ultralytics import YOLO

    model_dir = work_dir / "models" / f"fold_{fold}"
    detector = YOLO(str(model_dir / "detector.pt"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    refiner = SeedRefiner(pretrained=False)
    refiner.load_state_dict(torch.load(model_dir / "refiner.pt", map_location=device))
    refiner.to(device).eval()
    return detector, refiner, device


def classical_fallback_seed(image: np.ndarray, minimum_area: int) -> np.ndarray:
    """Image-only last resort so every test observation has a valid prediction."""
    blurred = cv2.GaussianBlur(image, (0, 0), 25)
    darkness = blurred.astype(np.float32) - image.astype(np.float32)
    disk = image > 12
    values = darkness[disk]
    if values.size == 0:
        disk = np.ones_like(image, dtype=bool)
        values = darkness.ravel()
    threshold = float(np.quantile(values, 0.995))
    candidate = ((darkness >= threshold) & disk).astype(np.uint8)
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    valid = [index for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] >= minimum_area]
    if not valid:
        valid = [1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))] if count > 1 else []
    if not valid:
        fallback = np.zeros_like(image, dtype=np.uint8)
        center_y, center_x = np.unravel_index(int(np.argmax(darkness)), darkness.shape)
        cv2.circle(fallback, (int(center_x), int(center_y)), 6, 1, -1)
        return fallback.astype(bool)
    best = max(valid, key=lambda index: stats[index, cv2.CC_STAT_AREA])
    return labels == best


@torch.no_grad()
def predict_instances(
    image_path: Path,
    detector: Any,
    refiner: SeedRefiner,
    device: torch.device,
    cfg: CFG,
) -> tuple[list[np.ndarray], list[float]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    result = detector.predict(
        str(image_path), imgsz=cfg.image_size, conf=cfg.detector_conf,
        iou=cfg.detector_iou, agnostic_nms=True, retina_masks=True, verbose=False,
    )[0]
    if result.masks is None:
        result = detector.predict(
            str(image_path), imgsz=cfg.image_size, conf=cfg.detector_fallback_conf,
            iou=cfg.detector_iou, agnostic_nms=True, retina_masks=True, verbose=False,
        )[0]
    if result.masks is None:
        fallback_seed = classical_fallback_seed(image, cfg.minimum_area)
        fallback_mask, fallback_confidence = refine_seed(image, fallback_seed, refiner, device, cfg)
        return [fallback_mask], [0.01 * fallback_confidence]
    seed_masks = [mask.cpu().numpy().astype(bool) for mask in result.masks.data]
    if seed_masks and seed_masks[0].shape != image.shape:
        seed_masks = [cv2.resize(mask.astype(np.uint8), image.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool) for mask in seed_masks]
    detector_scores = result.boxes.conf.detach().cpu().numpy().astype(float).tolist()
    seed_masks, detector_scores = remove_duplicate_seeds(seed_masks, detector_scores)
    masks, scores = [], []
    for seed_mask, detector_score in zip(seed_masks, detector_scores):
        if seed_mask.sum() < cfg.minimum_area:
            continue
        refined_mask, refiner_score = refine_seed(image, seed_mask, refiner, device, cfg)
        masks.append(refined_mask); scores.append(float(detector_score * refiner_score))
    return resolve_overlaps(masks, scores, cfg.minimum_area)

# %%
def pairwise_iou(gt_masks: Sequence[np.ndarray], pred_masks: Sequence[np.ndarray]) -> np.ndarray:
    if not gt_masks or not pred_masks:
        return np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float64)
    gt_rles = [mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in gt_masks]
    pred_rles = [mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in pred_masks]
    return np.asarray(mask_utils.iou(gt_rles, pred_rles, [0] * len(pred_rles)), dtype=np.float64)


def panoptic_quality(
    gt_masks: Sequence[np.ndarray],
    pred_masks: Sequence[np.ndarray],
    match_iou: float = 0.50,
) -> dict[str, float | int]:
    """August 7 metric. TP uses the original PQ criterion IoU > 0.5."""
    n_gt, n_pred = len(gt_masks), len(pred_masks)
    if n_gt == 0 and n_pred == 0:
        return {"pq": 1.0, "sq": 1.0, "rq": 1.0, "tp": 0, "fp": 0,
                "fn": 0, "sum_iou": 0.0, "one_to_one_tp": 0,
                "one_to_one_sum_iou": 0.0}
    iou = pairwise_iou(gt_masks, pred_masks)
    matched = iou > match_iou
    tp = int(matched.sum())
    fp = int(n_pred - matched.any(axis=0).sum())
    fn = int(n_gt - matched.any(axis=1).sum())
    sum_iou = float(iou[matched].sum())
    eligible = np.where(matched, iou, 0.0)
    rows, columns = linear_sum_assignment(eligible, maximize=True)
    one_to_one_values = eligible[rows, columns]
    one_to_one_values = one_to_one_values[one_to_one_values > 0]
    denominator = tp + 0.5 * fp + 0.5 * fn
    pq = sum_iou / denominator if denominator else 1.0
    sq = sum_iou / tp if tp else 0.0
    rq = tp / denominator if denominator else 1.0
    return {"pq": pq, "sq": sq, "rq": rq, "tp": tp, "fp": fp, "fn": fn,
            "sum_iou": sum_iou, "one_to_one_tp": int(len(one_to_one_values)),
            "one_to_one_sum_iou": float(one_to_one_values.sum())}


def ground_truth_sets_for_file(filename: str, indices: Mapping[str, Any]) -> list[list[np.ndarray]]:
    result = []
    for image_id in indices["image_ids_by_file"][filename]:
        info = indices["images_by_id"][image_id]
        masks = [
            annotation_to_mask(annotation, int(info["height"]), int(info["width"]))
            for annotation in indices["annotations_by_image"].get(image_id, [])
            if not annotation.get("iscrowd", 0)
        ]
        result.append(masks)
    return result


def evaluate_prediction_for_file(
    filename: str,
    pred_masks: Sequence[np.ndarray],
    indices: Mapping[str, Any],
    cfg: CFG,
) -> list[dict[str, Any]]:
    rows = []
    for annotator_index, gt_masks in enumerate(ground_truth_sets_for_file(filename, indices)):
        rows.append({"filename": filename, "annotator": annotator_index,
                     **panoptic_quality(gt_masks, pred_masks, cfg.pq_match_iou),
                     "n_gt": len(gt_masks), "n_pred": len(pred_masks)})
    return rows

# %%
def train_all_folds(
    fold_manifest: pd.DataFrame,
    paths: Mapping[str, Path],
    selected_image_ids: Mapping[str, int],
    indices: Mapping[str, Any],
    cfg: CFG,
    work_dir: Path,
) -> None:
    for fold in range(cfg.n_folds):
        print(f"\n===== TRAIN FOLD {fold}/{cfg.n_folds - 1} =====")
        data_yaml = prepare_yolo_fold(fold, fold_manifest, paths["train"], selected_image_ids, indices, work_dir)
        train_detector(fold, data_yaml, cfg, work_dir)
        train_refiner(fold, fold_manifest, paths["train"], selected_image_ids, indices, cfg, work_dir)
        gc.collect(); torch.cuda.empty_cache()


def run_oof(
    fold_manifest: pd.DataFrame,
    paths: Mapping[str, Path],
    indices: Mapping[str, Any],
    cfg: CFG,
    work_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotator_rows = []
    oof_dir = work_dir / "oof_masks"; oof_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(cfg.n_folds):
        detector, refiner, device = load_fold_models(fold, cfg, work_dir)
        validation_files = fold_manifest.loc[fold_manifest["fold"] == fold, "filename"].tolist()
        if cfg.debug: validation_files = validation_files[:8]
        for number, filename in enumerate(validation_files, 1):
            masks, scores = predict_instances(paths["train"] / filename, detector, refiner, device, cfg)
            rows = evaluate_prediction_for_file(filename, masks, indices, cfg)
            for row in rows: row["fold"] = fold
            annotator_rows.extend(rows)
            if cfg.save_oof_masks:
                shape = (2048, 2048)
                stack = np.stack(masks).astype(np.uint8) if masks else np.zeros((0, *shape), np.uint8)
                np.savez_compressed(oof_dir / f"{Path(filename).stem}.npz", masks=stack, scores=np.asarray(scores))
            if number % 20 == 0: print(f"fold={fold} OOF {number}/{len(validation_files)}")
        del detector, refiner; gc.collect(); torch.cuda.empty_cache()
    annotator_df = pd.DataFrame(annotator_rows)
    metrics = ["pq", "sq", "rq", "tp", "fp", "fn", "sum_iou", "n_gt", "n_pred"]
    observation_df = annotator_df.groupby(["filename", "fold"], as_index=False)[metrics].mean()
    annotator_df.to_csv(work_dir / "oof_by_annotator.csv", index=False)
    observation_df.to_csv(work_dir / "oof_by_observation.csv", index=False)
    return observation_df, annotator_df


def summarize_oof(observation_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["pq", "sq", "rq", "tp", "fp", "fn", "n_gt", "n_pred"]
    fold_summary = observation_df.groupby("fold")[metrics].mean().reset_index()
    overall = pd.DataFrame([{"fold": "overall", **observation_df[metrics].mean().to_dict()}])
    return pd.concat([fold_summary, overall], ignore_index=True)

# %%
def fuse_fold_predictions(
    predictions: list[tuple[int, np.ndarray, float]],
    cfg: CFG,
) -> tuple[list[np.ndarray], list[float]]:
    clusters: list[list[tuple[int, np.ndarray, float]]] = []
    for item in sorted(predictions, key=lambda value: value[2], reverse=True):
        best_cluster, best_iou = None, 0.0
        for cluster in clusters:
            representative = cluster[0][1]
            overlap = mask_iou(item[1], representative)
            if overlap > best_iou:
                best_cluster, best_iou = cluster, overlap
        if best_cluster is not None and best_iou >= cfg.ensemble_iou:
            best_cluster.append(item)
        else:
            clusters.append([item])
    fused_masks, fused_scores = [], []
    for cluster in clusters:
        support = len({item[0] for item in cluster})
        maximum_score = max(item[2] for item in cluster)
        if support < cfg.minimum_fold_support and maximum_score < cfg.singleton_score:
            continue
        votes = np.stack([item[1] for item in cluster]).sum(axis=0)
        required_votes = max(1, math.ceil(len(cluster) / 2))
        fused = votes >= required_votes
        if fused.sum() >= cfg.minimum_area:
            fused_masks.append(fused); fused_scores.append(float(np.mean([item[2] for item in cluster])))
    return resolve_overlaps(fused_masks, fused_scores, cfg.minimum_area)


def encode_submission_mask(mask: np.ndarray) -> str:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("ascii")


def run_test_ensemble(paths: Mapping[str, Path], cfg: CFG, work_dir: Path) -> pd.DataFrame:
    test_files = sorted(path for path in paths["test"].iterdir() if path.suffix.lower() == ".jpeg")
    models = [load_fold_models(fold, cfg, work_dir) for fold in range(cfg.n_folds)]
    submission_rows = []
    for image_number, image_path in enumerate(test_files, 1):
        all_predictions = []
        for fold, (detector, refiner, device) in enumerate(models):
            masks, scores = predict_instances(image_path, detector, refiner, device, cfg)
            all_predictions.extend((fold, mask, score) for mask, score in zip(masks, scores))
        masks, _ = fuse_fold_predictions(all_predictions, cfg)
        image_id = image_path.stem
        for instance_index, mask in enumerate(masks, 1):
            submission_rows.append({
                "filament_id": f"{image_id}_{instance_index}",
                "segmentation_rle": encode_submission_mask(mask),
            })
        if image_number % 20 == 0:
            print(f"test {image_number}/{len(test_files)} rows={len(submission_rows)}")
    submission = pd.DataFrame(submission_rows, columns=["filament_id", "segmentation_rle"])
    if submission.empty or not submission["filament_id"].is_unique:
        raise RuntimeError("Invalid empty or duplicate-ID submission")
    submission.to_csv(work_dir / "submission.csv", index=False)
    return submission


def config_fingerprint(cfg: CFG, fold_manifest: pd.DataFrame) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True) + "\n" + "\n".join(
        f"{row.filename},{row.fold}" for row in fold_manifest.sort_values("filename").itertuples()
    )
    return hashlib.sha256(payload.encode()).hexdigest()


__all__ = [name for name in globals() if not name.startswith("_")]


# zlib-compressed JSON containing the repository's immutable 495/106 split and
# its pre-existing 24-image calibration subset. Filled by prepare.py.
SPLIT_B64 = "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"
SPLIT = json.loads(zlib.decompress(base64.b64decode(SPLIT_B64)))
CALIBRATION_FILES = frozenset(SPLIT["calibration"])
WORK_DIR = Path("/kaggle/working/sfsc-yolo-refiner-validation")
REPORT_PATH = WORK_DIR / "validation-report.json"


def write_status(status: dict[str, Any]) -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


def fixed_manifest() -> pd.DataFrame:
    rows = [*(dict(filename=name, fold=1) for name in SPLIT["train"])]
    rows.extend(dict(filename=name, fold=0) for name in SPLIT["validation"])
    frame = pd.DataFrame(rows).sort_values("filename").reset_index(drop=True)
    assert len(frame) == 601 and frame["filename"].nunique() == 601
    assert (frame["fold"] == 1).sum() == 495
    assert (frame["fold"] == 0).sum() == 106
    assert CALIBRATION_FILES <= set(SPLIT["validation"])
    return frame


def prepare_fixed_yolo(
    manifest: pd.DataFrame,
    image_dir: Path,
    selected_image_ids: Mapping[str, int],
    indices: Mapping[str, Any],
) -> Path:
    data_dir = WORK_DIR / "yolo_data"
    marker = data_dir / ".ready"
    yaml_path = data_dir / "data.yaml"
    if marker.exists():
        return yaml_path
    train_files = manifest.loc[manifest["fold"] == 1, "filename"].tolist()
    calibration_files = sorted(CALIBRATION_FILES)
    write_yolo_split(data_dir / "train", train_files, image_dir, selected_image_ids, indices)
    write_yolo_split(
        data_dir / "val", calibration_files, image_dir, selected_image_ids, indices
    )
    yaml_path.write_text(
        f"path: {data_dir}\ntrain: train/images\nval: val/images\n"
        "names:\n  0: filament\n",
        encoding="utf-8",
    )
    marker.write_text("ready\n", encoding="utf-8")
    return yaml_path


def json_rle(mask: np.ndarray) -> dict[str, Any]:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": encoded["size"], "counts": encoded["counts"].decode("ascii")}


def decode_json_rle(encoded: Mapping[str, Any]) -> np.ndarray:
    payload = {"size": encoded["size"], "counts": encoded["counts"].encode("ascii")}
    return mask_utils.decode(payload).astype(bool)


def seed_geometry(mask: np.ndarray) -> tuple[int, int, float]:
    ys, xs = np.where(mask)
    if not len(xs):
        return 0, 0, 0.0
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    return width, height, max(width, height) / max(min(width, height), 1)


@torch.inference_mode()
def cache_validation_candidates(
    paths: Mapping[str, Path],
    manifest: pd.DataFrame,
    cfg: CFG,
    filenames: Sequence[str] | None = None,
    cache_name: str = "validation-candidates.json",
) -> dict[str, list[dict[str, Any]]]:
    cache_path = WORK_DIR / cache_name
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    detector, refiner, device = load_fold_models(0, cfg, WORK_DIR)
    cache: dict[str, list[dict[str, Any]]] = {}
    validation_files = (
        list(filenames)
        if filenames is not None
        else manifest.loc[manifest["fold"] == 0, "filename"].tolist()
    )
    for number, filename in enumerate(validation_files, 1):
        image_path = paths["train"] / filename
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        result = detector.predict(
            str(image_path),
            imgsz=cfg.image_size,
            conf=cfg.detector_conf,
            iou=cfg.detector_iou,
            agnostic_nms=True,
            retina_masks=True,
            max_det=100,
            verbose=False,
        )[0]
        candidates: list[dict[str, Any]] = []
        if result.masks is not None:
            seeds = [mask.cpu().numpy().astype(bool) for mask in result.masks.data]
            if seeds and seeds[0].shape != image.shape:
                seeds = [
                    cv2.resize(
                        mask.astype(np.uint8), image.shape[::-1], interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                    for mask in seeds
                ]
            detector_scores = result.boxes.conf.detach().cpu().numpy().astype(float).tolist()
            seeds, detector_scores = remove_duplicate_seeds(seeds, detector_scores)
            for seed, detector_score in zip(seeds, detector_scores):
                if seed.sum() < 32:
                    continue
                refined, refiner_score = refine_seed(image, seed, refiner, device, cfg)
                width, height, aspect = seed_geometry(seed)
                preserve_seed = max(width, height) >= 720 or aspect >= 4.5
                hybrid = seed if preserve_seed or refined.sum() < 32 else refined
                candidates.append(
                    {
                        "seed": json_rle(seed),
                        "refined": json_rle(refined),
                        "hybrid": json_rle(hybrid),
                        "detector_score": detector_score,
                        "refiner_score": refiner_score,
                        "seed_area": int(seed.sum()),
                        "refined_area": int(refined.sum()),
                        "hybrid_area": int(hybrid.sum()),
                        "long_object_path": preserve_seed,
                    }
                )
        cache[filename] = candidates
        if number % 10 == 0 or number == len(validation_files):
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
            print(f"validation inference {number}/{len(validation_files)}", flush=True)
    del detector, refiner
    gc.collect()
    torch.cuda.empty_cache()
    return cache


def checkpoint_epoch(checkpoint: Path, default: int) -> int:
    match = re.search(r"epoch[-]?(\d+)", checkpoint.stem)
    if match:
        return int(match.group(1))
    if "last" in checkpoint.stem:
        return default
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return int(payload.get("epoch", default - 1)) + 1


def choose_refiner_checkpoint(
    paths: Mapping[str, Path],
    manifest: pd.DataFrame,
    indices: Mapping[str, Any],
    cfg: CFG,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model_dir = WORK_DIR / "models/fold_0"
    selection = json.loads(
        (model_dir / "refiner-crop-selection.json").read_text(encoding="utf-8")
    )
    candidates = [model_dir / "refiner-crop-best.pt"]
    candidates.extend(sorted(model_dir.glob("refiner-epoch-*.pt")))
    rows = []
    calibration_files = sorted(CALIBRATION_FILES)
    for index, checkpoint in enumerate(candidates):
        shutil.copy2(checkpoint, model_dir / "refiner.pt")
        cache = cache_validation_candidates(
            paths,
            manifest,
            cfg,
            filenames=calibration_files,
            cache_name=f"refiner-{index}-calibration-candidates.json",
        )
        selected, _ = calibrate(cache, indices)
        epoch = (
            int(selection["best_epoch"])
            if "crop-best" in checkpoint.stem
            else checkpoint_epoch(checkpoint, cfg.refiner_epochs)
        )
        rows.append(
            {
                "checkpoint": str(checkpoint.relative_to(WORK_DIR)),
                "epoch": epoch,
                "selected": selected,
            }
        )
    best = max(rows, key=lambda row: row["selected"]["pq"])
    shutil.copy2(WORK_DIR / best["checkpoint"], model_dir / "refiner.pt")
    print(f"PQ-selected refiner checkpoint: {best}", flush=True)
    return rows, best


def choose_detector_checkpoint(
    paths: Mapping[str, Path],
    manifest: pd.DataFrame,
    indices: Mapping[str, Any],
    cfg: CFG,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model_dir = WORK_DIR / "models/fold_0"
    run_weights = WORK_DIR / "detector_runs/fold_0/weights"
    candidates = [model_dir / "detector-ap-best.pt", model_dir / "detector-last.pt"]
    periodic = sorted(run_weights.glob("epoch*.pt"))
    if periodic:
        candidates.extend(periodic[1::2])
    unique = []
    seen_hashes = set()
    for checkpoint in candidates:
        if not checkpoint.exists():
            continue
        checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if checksum not in seen_hashes:
            unique.append((checkpoint, checksum))
            seen_hashes.add(checksum)
    rows = []
    calibration_files = sorted(CALIBRATION_FILES)
    for index, (checkpoint, checksum) in enumerate(unique):
        shutil.copy2(checkpoint, model_dir / "detector.pt")
        cache = cache_validation_candidates(
            paths,
            manifest,
            cfg,
            filenames=calibration_files,
            cache_name=f"checkpoint-{index}-calibration-candidates.json",
        )
        selected, _ = calibrate(cache, indices)
        rows.append(
            {
                "checkpoint": str(checkpoint.relative_to(WORK_DIR)),
                "sha256": checksum,
                "epoch": checkpoint_epoch(checkpoint, cfg.detector_epochs),
                "selected": selected,
            }
        )
    best = max(rows, key=lambda row: row["selected"]["pq"])
    chosen = WORK_DIR / best["checkpoint"]
    shutil.copy2(chosen, model_dir / "detector.pt")
    print(f"PQ-selected detector checkpoint: {best}", flush=True)
    return rows, best


def candidate_score(candidate: Mapping[str, Any], kind: str) -> float:
    detector_score = float(candidate["detector_score"])
    if kind == "seed" or candidate.get("long_object_path", False):
        return detector_score
    return detector_score * math.sqrt(max(float(candidate["refiner_score"]), 0.0))


def evaluate_candidates(
    cache: Mapping[str, Sequence[Mapping[str, Any]]],
    filenames: Sequence[str],
    indices: Mapping[str, Any],
    kind: str,
    score_threshold: float,
    minimum_area: int,
) -> dict[str, float | int]:
    totals = {"sum_iou": 0.0, "tp": 0, "fp": 0, "fn": 0, "records": 0,
              "one_to_one_sum_iou": 0.0, "one_to_one_tp": 0}
    for filename in filenames:
        masks, scores = [], []
        for candidate in cache[filename]:
            score = candidate_score(candidate, kind)
            if score < score_threshold or int(candidate[f"{kind}_area"]) < minimum_area:
                continue
            mask = decode_json_rle(candidate[kind])
            if mask.sum() >= minimum_area:
                masks.append(mask)
                scores.append(score)
        masks, _ = resolve_overlaps(masks, scores, minimum_area)
        for truth in ground_truth_sets_for_file(filename, indices):
            row = panoptic_quality(truth, masks, 0.5)
            totals["sum_iou"] += float(row["sum_iou"])
            totals["tp"] += int(row["tp"])
            totals["fp"] += int(row["fp"])
            totals["fn"] += int(row["fn"])
            totals["one_to_one_sum_iou"] += float(row["one_to_one_sum_iou"])
            totals["one_to_one_tp"] += int(row["one_to_one_tp"])
            totals["records"] += 1
    denominator = totals["tp"] + 0.5 * (totals["fp"] + totals["fn"])
    totals["pq"] = totals["sum_iou"] / denominator if denominator else 0.0
    totals["sq"] = totals["sum_iou"] / totals["tp"] if totals["tp"] else 0.0
    totals["rq"] = totals["tp"] / denominator if denominator else 0.0
    one_to_one_fp = totals["fp"] + totals["tp"] - totals["one_to_one_tp"]
    one_to_one_fn = totals["fn"] + totals["tp"] - totals["one_to_one_tp"]
    one_to_one_denominator = totals["one_to_one_tp"] + 0.5 * (
        one_to_one_fp + one_to_one_fn
    )
    totals["one_to_one_pq"] = (
        totals["one_to_one_sum_iou"] / one_to_one_denominator
        if one_to_one_denominator
        else 0.0
    )
    return totals


def calibrate(
    cache: Mapping[str, Sequence[Mapping[str, Any]]],
    indices: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grids = {
        "seed": [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
        "refined": [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40],
        "hybrid": [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40],
    }
    rows = []
    calibration_files = sorted(CALIBRATION_FILES)
    for kind, thresholds in grids.items():
        for threshold in thresholds:
            for minimum_area in [64, 96, 128, 192, 256, 384, 512]:
                metrics = evaluate_candidates(
                    cache, calibration_files, indices, kind, threshold, minimum_area
                )
                rows.append(
                    {
                        "kind": kind,
                        "score_threshold": threshold,
                        "minimum_area": minimum_area,
                        **metrics,
                    }
                )
    best = max(rows, key=lambda row: (row["pq"], -row["fp"], row["tp"]))
    return best, rows


def main() -> None:
    started = time.perf_counter()
    status: dict[str, Any] = {
        "status": "started",
        "method": "YOLO11l-seg 1280 + negative-aware prompted ResNet18 crop refiner",
        "source_basis": "Adapted and corrected from ektarr/new-metric-solar-filament",
        "config": asdict(CFG()),
        "split": {"train": 495, "validation": 106, "calibration": 24},
    }
    write_status(status)
    try:
        import torchvision
        import ultralytics

        assert torch.cuda.is_available()
        assert torch.cuda.get_device_capability() == (6, 0)
        cfg = CFG()
        seed_everything(cfg.seed)
        paths = discover_competition_paths()
        manifest = fixed_manifest()
        coco = restrict_coco(load_coco(paths["json"]), set(manifest["filename"]))
        indices = build_coco_indices(coco)
        medoids = choose_medoid_annotators(indices)
        assert set(manifest["filename"]) <= set(indices["image_ids_by_file"])
        manifest.to_csv(WORK_DIR / "fixed-manifest.csv", index=False)
        data_yaml = prepare_fixed_yolo(manifest, paths["train"], medoids, indices)
        train_detector(0, data_yaml, cfg, WORK_DIR)
        train_refiner(0, manifest, paths["train"], medoids, indices, cfg, WORK_DIR)
        refiner_checkpoint_comparison, selected_refiner = choose_refiner_checkpoint(
            paths, manifest, indices, cfg
        )
        checkpoint_comparison, selected_detector = choose_detector_checkpoint(
            paths, manifest, indices, cfg
        )
        cache = cache_validation_candidates(paths, manifest, cfg)
        selected, grid = calibrate(cache, indices)
        all_validation = sorted(SPLIT["validation"])
        other_validation = sorted(set(all_validation) - CALIBRATION_FILES)
        kwargs = {
            "kind": selected["kind"],
            "score_threshold": selected["score_threshold"],
            "minimum_area": selected["minimum_area"],
        }
        full_metrics = evaluate_candidates(cache, all_validation, indices, **kwargs)
        other_metrics = evaluate_candidates(cache, other_validation, indices, **kwargs)
        status.update(
            {
                "status": "complete",
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "ultralytics": ultralytics.__version__,
                "gpu": torch.cuda.get_device_name(),
                "selected": selected,
                "detector_checkpoint_comparison": checkpoint_comparison,
                "refiner_checkpoint_comparison": refiner_checkpoint_comparison,
                "selected_detector_epoch": selected_detector["epoch"],
                "selected_refiner_epoch": selected_refiner["epoch"],
                "full_validation": full_metrics,
                "non_calibration_validation": other_metrics,
                "control_pq": 0.38117396647989144,
                "delta_vs_control": full_metrics["pq"] - 0.38117396647989144,
                "full_refit_gate": min(
                    full_metrics["pq"], full_metrics["one_to_one_pq"]
                ) >= 0.40117396647989144,
                "grid": grid,
                "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    except Exception as error:
        status.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        raise
    finally:
        write_status(status)
        print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
