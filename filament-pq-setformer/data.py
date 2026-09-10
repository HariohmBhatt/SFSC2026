"""Deterministic coherent-record data adapter for local audits and P100 training."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask


def discover_dataset_root(input_root: Path) -> Path:
    """Locate one complete SFSC training directory under a Kaggle input mount."""
    filename = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    candidates = Path(input_root).glob(f"**/{filename}")
    roots = {
        path.parent.resolve() for path in candidates if (path.parent / "train_images").is_dir()
    }
    if len(roots) != 1:
        raise ValueError(f"Expected one SFSC training root, found {len(roots)}")
    return roots.pop()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed(*values: object) -> int:
    text = ":".join(map(str, values))
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


@dataclass(frozen=True)
class AnnotationRecord:
    record_id: str
    annotations: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class AnnotationIndex:
    records_by_file: dict[str, tuple[AnnotationRecord, ...]]
    sha256: str
    record_count: int
    maximum_instances: int

    @property
    def filenames(self) -> tuple[str, ...]:
        return tuple(sorted(self.records_by_file))


def load_annotation_index(path: Path, allowed_filenames: set[str] | None = None) -> AnnotationIndex:
    path = Path(path)
    payload = json.loads(path.read_text())
    images = {str(image["id"]): str(image["file_name"]) for image in payload["images"]}
    annotations: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        image_id = str(annotation["image_id"])
        if image_id not in images:
            raise ValueError(f"Annotation references unknown image record {image_id}")
        annotations[image_id].append(annotation)

    records: dict[str, list[AnnotationRecord]] = defaultdict(list)
    for record_id, filename in images.items():
        if allowed_filenames is not None and filename not in allowed_filenames:
            continue
        record_annotations = tuple(annotations[record_id])
        if not record_annotations:
            raise ValueError(f"Annotation record {record_id} is empty")
        records[filename].append(AnnotationRecord(record_id, record_annotations))
    frozen = {filename: tuple(values) for filename, values in records.items()}
    if allowed_filenames is not None and set(frozen) != allowed_filenames:
        missing = sorted(allowed_filenames - set(frozen))
        raise ValueError(f"Missing requested annotation filenames: {missing[:5]}")
    return AnnotationIndex(
        records_by_file=frozen,
        sha256=_sha256(path),
        record_count=sum(map(len, frozen.values())),
        maximum_instances=max(
            (len(record.annotations) for values in frozen.values() for record in values),
            default=0,
        ),
    )


def select_annotation_record(
    records: Sequence[AnnotationRecord],
    filename: str,
    *,
    presentation: int,
    seed: int,
) -> AnnotationRecord:
    if not records:
        raise ValueError("Cannot sample an empty annotation-record collection")
    index = _seed("annotation", seed, presentation, filename) % len(records)
    return records[index]


def filename_for_presentation(filenames: Sequence[str], presentation: int, *, seed: int) -> str:
    """Return a deterministic shuffled virtual-epoch filename."""
    names = tuple(sorted(filenames))
    if not names or presentation < 0:
        raise ValueError("Virtual presentations require filenames and a nonnegative index")
    epoch, position = divmod(presentation, len(names))
    ordered = sorted(names, key=lambda name: (_seed("order", seed, epoch, name), name))
    return ordered[position]


def rasterize_record(record: AnnotationRecord, *, native_size: int, output_size: int) -> np.ndarray:
    if native_size <= 0 or output_size <= 0:
        raise ValueError("Raster dimensions must be positive")
    scale = output_size / native_size
    masks = []
    for annotation in record.annotations:
        polygons = annotation.get("segmentation", [])
        if not polygons:
            raise ValueError(f"Annotation {annotation.get('id')} has no segmentation")
        scaled = [[float(value) * scale for value in polygon] for polygon in polygons]
        rles = coco_mask.frPyObjects(scaled, output_size, output_size)
        mask = coco_mask.decode(coco_mask.merge(rles)).astype(bool)
        if mask.ndim != 2 or not mask.any():
            raise ValueError(f"Annotation {annotation.get('id')} rasterized empty")
        masks.append(mask)
    return np.stack(masks)


def _augment(image: np.ndarray, masks: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if rng.random() < 0.5:
        image, masks = image[:, ::-1], masks[:, :, ::-1]
    if rng.random() < 0.5:
        image, masks = image[::-1, :], masks[:, ::-1, :]
    gamma = rng.uniform(0.85, 1.15)
    contrast = rng.uniform(0.85, 1.15)
    brightness = rng.uniform(-0.05, 0.05) * 255
    normalized = np.power(np.clip(image / 255.0, 0, 1), gamma)
    normalized = (normalized - normalized.mean()) * contrast + normalized.mean()
    image = np.clip(normalized * 255 + brightness, 0, 255).astype(np.float32)
    return np.ascontiguousarray(image), np.ascontiguousarray(masks)


def build_training_item(
    *,
    index: AnnotationIndex,
    image_directory: Path,
    filenames: Sequence[str],
    presentation: int,
    input_size: int,
    seed: int,
    augment: bool,
) -> dict[str, Any]:
    """Build one Detectron2 input while keeping torch imports Kaggle-only."""
    import torch
    from detectron2.structures import BitMasks, Instances

    filename = filename_for_presentation(filenames, presentation, seed=seed)
    record = select_annotation_record(
        index.records_by_file[filename], filename, presentation=presentation, seed=seed
    )
    image_path = Path(image_directory) / filename
    with Image.open(image_path) as source:
        if source.mode != "L" or source.size != (2048, 2048):
            raise ValueError(
                f"Unexpected SFSC image format for {filename}: {source.mode}/{source.size}"
            )
        image = np.asarray(
            source.resize((input_size, input_size), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
    masks = rasterize_record(record, native_size=2048, output_size=input_size)
    if augment:
        image, masks = _augment(
            image,
            masks,
            seed=_seed("augment", seed, presentation, filename),
        )
    mask_tensor = torch.from_numpy(np.ascontiguousarray(masks))
    instances = Instances((input_size, input_size))
    instances.gt_masks = mask_tensor
    instances.gt_boxes = BitMasks(mask_tensor).get_bounding_boxes()
    instances.gt_classes = torch.zeros(len(mask_tensor), dtype=torch.int64)
    return {
        "image": torch.from_numpy(image[None]),
        "height": input_size,
        "width": input_size,
        "instances": instances,
        "filename": filename,
        "annotation_record": record.record_id,
        "presentation": presentation,
    }


class VirtualTrainingDataset:
    """Fixed-length deterministic presentation stream for accumulation-safe training."""

    def __init__(
        self,
        *,
        index: AnnotationIndex,
        image_directory: Path,
        filenames: Sequence[str],
        presentations: int,
        input_size: int,
        seed: int,
        augment: bool = True,
    ) -> None:
        if presentations <= 0:
            raise ValueError("presentations must be positive")
        if not set(filenames) <= set(index.records_by_file):
            raise ValueError("Training filenames are absent from the annotation index")
        self.index = index
        self.image_directory = Path(image_directory)
        self.filenames = tuple(filenames)
        self.presentations = presentations
        self.input_size = input_size
        self.seed = seed
        self.augment = augment

    def __len__(self) -> int:
        return self.presentations

    def __getitem__(self, presentation: int) -> dict[str, Any]:
        return build_training_item(
            index=self.index,
            image_directory=self.image_directory,
            filenames=self.filenames,
            presentation=presentation,
            input_size=self.input_size,
            seed=self.seed,
            augment=self.augment,
        )


def instance_collate(batch: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(batch)
