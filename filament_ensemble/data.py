from __future__ import annotations

import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from pycocotools import mask as coco_mask
from torch.utils.data import Dataset


def native_rle(rle: dict[str, Any]) -> dict[str, Any]:
    counts = rle["counts"]
    return {
        "size": list(rle["size"]),
        "counts": counts.encode("ascii") if isinstance(counts, str) else counts,
    }


def decode_rle(rle: dict[str, Any] | None, shape: tuple[int, int]) -> np.ndarray:
    if rle is None:
        return np.zeros(shape, dtype=bool)
    return coco_mask.decode(native_rle(rle)).astype(bool)


def annotation_mask(annotation: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, dict):
        encoded = native_rle(segmentation)
    else:
        encoded = coco_mask.merge(coco_mask.frPyObjects(segmentation, *shape))
    return coco_mask.decode(encoded).astype(bool)


def corrupt_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    seed = mask.astype(np.uint8)
    radius = int(rng.integers(-13, 14))
    if radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(radius) + 1,) * 2)
        seed = cv2.dilate(seed, kernel) if radius > 0 else cv2.erode(seed, kernel)
    angle = float(rng.uniform(-8, 8))
    scale = float(rng.uniform(0.92, 1.08))
    shift = rng.integers(-18, 19, size=2)
    center = (seed.shape[1] / 2, seed.shape[0] / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[:, 2] += shift
    seed = cv2.warpAffine(seed, matrix, seed.shape[::-1], flags=cv2.INTER_NEAREST)
    if rng.random() < 0.35 and seed.any():
        ys, xs = np.where(seed)
        for _ in range(int(rng.integers(1, 4))):
            index = int(rng.integers(len(xs)))
            cv2.circle(seed, (int(xs[index]), int(ys[index])), int(rng.integers(6, 28)), 0, -1)
    return seed.astype(bool) if seed.any() else mask.copy()


def crop_bounds(mask: np.ndarray, context: float, minimum: int = 160) -> tuple[int, int, int, int]:
    height, width = mask.shape
    ys, xs = np.where(mask)
    if not len(xs):
        side = min(height, width)
        return (width - side) // 2, (height - side) // 2, (width + side) // 2, (height + side) // 2
    center_x = (xs.min() + xs.max() + 1) / 2
    center_y = (ys.min() + ys.max() + 1) / 2
    side = int(math.ceil(max(np.ptp(xs) + 1, np.ptp(ys) + 1) * context))
    side = min(max(side, minimum), min(height, width))
    x0, y0 = int(round(center_x - side / 2)), int(round(center_y - side / 2))
    x0, y0 = min(max(x0, 0), width - side), min(max(y0, 0), height - side)
    return x0, y0, x0 + side, y0 + side


def feature_stack(image: np.ndarray, seed: np.ndarray, size: int) -> np.ndarray:
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    seed = cv2.resize(seed.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    local = cv2.GaussianBlur(image, (0, 0), 17)
    local = np.clip(image.astype(np.float32) - local.astype(np.float32) + 128, 0, 255)
    inside = cv2.distanceTransform(seed, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - seed, cv2.DIST_L2, 5)
    signed = np.clip(inside - outside, -64, 64) / 128 + 0.5
    gradient_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.clip(np.hypot(gradient_x, gradient_y), 0, 255)
    return np.stack(
        [image / 255, clahe / 255, local / 255, seed, signed, gradient / 255], axis=0
    ).astype(np.float32)


def full_disk_features(image: np.ndarray, size: int) -> np.ndarray:
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    local_small = np.clip(
        image.astype(np.float32) - cv2.GaussianBlur(image, (0, 0), 9).astype(np.float32) + 128,
        0,
        255,
    )
    local_large = np.clip(
        image.astype(np.float32) - cv2.GaussianBlur(image, (0, 0), 31).astype(np.float32) + 128,
        0,
        255,
    )
    gradient_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.clip(np.hypot(gradient_x, gradient_y), 0, 255)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    radius = np.hypot(xx - (size - 1) / 2, yy - (size - 1) / 2)
    radius /= radius.max()
    return np.stack(
        [image / 255, clahe / 255, local_small / 255, local_large / 255, gradient / 255, radius],
        axis=0,
    ).astype(np.float32)


def augment(features: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation = random.randrange(4)
    features = np.rot90(features, rotation, axes=(1, 2)).copy()
    target = np.rot90(target, rotation, axes=(-2, -1)).copy()
    if random.random() < 0.5:
        features, target = features[:, :, ::-1].copy(), target[..., :, ::-1].copy()
    if random.random() < 0.5:
        features, target = features[:, ::-1, :].copy(), target[..., ::-1, :].copy()
    gain = random.uniform(0.82, 1.18)
    gamma = random.uniform(0.8, 1.25)
    features[:3] = np.clip((features[:3] * gain) ** gamma, 0, 1)
    if random.random() < 0.35:
        features[:3] = np.clip(features[:3] + np.random.normal(0, 0.025, features[:3].shape), 0, 1)
    return features, target


class PromptDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        coco: dict[str, Any],
        filenames: Sequence[str],
        crop_size: int,
        contexts: Sequence[float],
        prompts: Sequence[dict[str, Any]] | None = None,
        augment_data: bool = True,
        real_prompt_fraction: float = 0.7,
    ) -> None:
        self.image_dir = image_dir
        self.crop_size = crop_size
        self.contexts = tuple(contexts)
        self.augment_data = augment_data
        self.real_prompt_fraction = real_prompt_fraction
        allowed = set(filenames)
        images = {
            str(row["id"]): row for row in coco["images"] if Path(row["file_name"]).name in allowed
        }
        self.synthetic = []
        for annotation in coco["annotations"]:
            image = images.get(str(annotation["image_id"]))
            if image is not None and not annotation.get("iscrowd", 0):
                self.synthetic.append((Path(image["file_name"]).name, annotation, image))
        self.prompts = [row for row in (prompts or []) if row["filename"] in allowed]
        self.length = max(len(self.synthetic), len(self.prompts), 1)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = random.randrange(2**32) if self.augment_data else index + 20261006
        rng = np.random.default_rng(seed)
        use_real = bool(self.prompts) and (
            random.random() < self.real_prompt_fraction if self.augment_data else index % 2 == 0
        )
        if use_real:
            row = self.prompts[index % len(self.prompts)]
            filename = row["filename"]
            image = cv2.imread(str(self.image_dir / filename), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(self.image_dir / filename)
            seed = decode_rle(row["seed"], image.shape)
            target = decode_rle(row.get("target"), image.shape)
        else:
            filename, annotation, info = self.synthetic[index % len(self.synthetic)]
            image = cv2.imread(str(self.image_dir / filename), cv2.IMREAD_GRAYSCALE)
            target = annotation_mask(annotation, (int(info["height"]), int(info["width"])))
            seed = corrupt_mask(target, rng)
        if image is None:
            raise FileNotFoundError(self.image_dir / filename)
        x0, y0, x1, y1 = crop_bounds(seed, float(rng.choice(self.contexts)))
        target_crop = cv2.resize(
            target[y0:y1, x0:x1].astype(np.uint8),
            (self.crop_size,) * 2,
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        features = feature_stack(image[y0:y1, x0:x1], seed[y0:y1, x0:x1], self.crop_size)
        if self.augment_data:
            features, target_crop = augment(features, target_crop)
        return torch.from_numpy(features), torch.from_numpy(target_crop[None])


class FullDiskDataset(Dataset):
    def __init__(
        self, image_dir: Path, coco: dict[str, Any], filenames: Sequence[str], size: int
    ) -> None:
        self.image_dir, self.size = image_dir, size
        allowed = set(filenames)
        annotations: dict[str, list[dict[str, Any]]] = {}
        for row in coco["annotations"]:
            annotations.setdefault(str(row["image_id"]), []).append(row)
        self.samples = []
        for image in coco["images"]:
            name = Path(image["file_name"]).name
            if name in allowed:
                self.samples.append((name, image, annotations.get(str(image["id"]), [])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        name, info, annotations = self.samples[index]
        image = cv2.imread(str(self.image_dir / name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(self.image_dir / name)
        shape = (int(info["height"]), int(info["width"]))
        target = np.zeros(shape, dtype=bool)
        boundary = np.zeros(shape, dtype=bool)
        kernel = np.ones((5, 5), dtype=np.uint8)
        for annotation in annotations:
            if not annotation.get("iscrowd", 0):
                instance = annotation_mask(annotation, shape)
                target |= instance
                boundary |= cv2.morphologyEx(
                    instance.astype(np.uint8), cv2.MORPH_GRADIENT, kernel
                ).astype(bool)
        features = full_disk_features(image, self.size)
        target = cv2.resize(
            target.astype(np.uint8), (self.size,) * 2, interpolation=cv2.INTER_NEAREST
        )
        boundary = cv2.resize(
            boundary.astype(np.uint8), (self.size,) * 2, interpolation=cv2.INTER_NEAREST
        )
        features, targets = augment(features, np.stack([target, boundary]).astype(np.float32))
        return (
            torch.from_numpy(features),
            torch.from_numpy(targets[0:1]),
            torch.from_numpy(targets[1:2]),
        )
