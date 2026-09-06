"""Shared source-image splits, annotation sets, and prediction contracts."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask
from scipy.optimize import linear_sum_assignment

from sfsc.contracts import require
from sfsc.evaluation import PQResult, pairwise_iou, panoptic_quality


class Dataset:
    def __init__(self, data_root: Path, manifest: Path):
        self.root = Path(data_root)
        with Path(manifest).open() as stream:
            self.rows = list(csv.DictReader(stream))
        self.by_filename = {row["filename"]: row for row in self.rows}
        annotation_path = self.root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
        annotation = json.loads(annotation_path.read_text())
        self.images = annotation["images"]
        self.annotations = defaultdict(list)
        for item in annotation["annotations"]:
            self.annotations[item["image_id"]].append(item)
        self.records = defaultdict(list)
        for image in self.images:
            self.records[Path(image["file_name"]).name].append(image)
        require(set(self.records) == set(self.by_filename), "Dataset and manifest filenames differ")
        for filename, records in self.records.items():
            expected = set(self.by_filename[filename]["annotation_ids"].split(";"))
            require(
                {str(record["id"]) for record in records} == expected,
                f"Annotation set drift: {filename}",
            )

    def filenames(self, *splits: str) -> list[str]:
        require(bool(splits) and set(splits) <= {"train", "validation", "test"}, "Invalid splits")
        return sorted(row["filename"] for row in self.rows if row["split"] in splits)

    def image_path(self, filename: str, competition: bool = False) -> Path:
        require(Path(filename).name == filename, "Expected a source filename")
        return self.root / ("test/test_images" if competition else "train/train_images") / filename

    def truth(self, image: dict) -> list[np.ndarray]:
        result = []
        for annotation in self.annotations[image["id"]]:
            rles = coco_mask.frPyObjects(
                annotation["segmentation"], image["height"], image["width"]
            )
            result.append(coco_mask.decode(coco_mask.merge(rles)).astype(bool))
        return result


def label_masks(labels: np.ndarray, shape: tuple[int, int]) -> list[np.ndarray]:
    require(labels.ndim == 2 and labels.shape == shape, f"Expected original-resolution map {shape}")
    require(
        np.issubdtype(labels.dtype, np.integer) and np.all(labels >= 0),
        "Label maps must contain nonnegative integers; 0 is background",
    )
    return [labels == value for value in np.unique(labels) if value != 0]


def prediction_files(folder: Path, filenames: list[str]) -> dict[str, Path]:
    require(folder.is_dir(), "Prediction directory is missing")
    expected = {f"{Path(name).stem}.npy": name for name in filenames}
    require(len(expected) == len(filenames), "Image stems must be unique")
    actual = {path.name: path for path in folder.glob("*.npy")}
    require(
        set(actual) == set(expected),
        "Prediction coverage mismatch: every requested image needs exactly one .npy label map",
    )
    return {expected[name]: path for name, path in actual.items()}


def summary(total: PQResult) -> dict:
    return {
        "pq": total.pq,
        "matched_iou": total.matched_iou,
        "true_positives": total.true_positives,
        "false_positives": total.false_positives,
        "false_negatives": total.false_negatives,
        "recognition_quality": total.recognition_quality,
        "segmentation_quality": total.segmentation_quality,
    }


def size_band(mask: np.ndarray) -> str:
    area = int(mask.sum())
    return "tiny" if area < 1024 else "medium" if area < 4096 else "large"


def evaluate(dataset: Dataset, folder: Path, split: str) -> dict:
    files = prediction_files(folder, dataset.filenames(split))
    total = PQResult()
    grouped = defaultdict(PQResult)
    per_record = []
    matched_ious = []
    size_totals = defaultdict(PQResult)
    for filename, path in files.items():
        image = dataset.records[filename][0]
        labels = np.load(path, allow_pickle=False)
        prediction = label_masks(labels, (image["height"], image["width"]))
        row = dataset.by_filename[filename]
        for record in dataset.records[filename]:
            truth = dataset.truth(record)
            result = panoptic_quality(truth, prediction)
            total += result
            for key in (
                f"site:{row['site']}",
                f"year:{row['year']}",
                f"multiplicity:{len(dataset.records[filename])}",
            ):
                grouped[key] += result
            overlaps = pairwise_iou(truth, prediction)
            eligible = np.where(overlaps > 0.5, overlaps, 0)
            truth_indices, prediction_indices = linear_sum_assignment(eligible, maximize=True)
            pairs = [
                (int(t), int(p), float(eligible[t, p]))
                for t, p in zip(truth_indices, prediction_indices, strict=True)
                if eligible[t, p] > 0
            ]
            matched_ious.extend(value for _, _, value in pairs)
            matched_truth = {t for t, _, _ in pairs}
            matched_prediction = {p for _, p, _ in pairs}
            for t, _, value in pairs:
                size_totals[size_band(truth[t])] += PQResult(matched_iou=value, true_positives=1)
            for t, mask in enumerate(truth):
                if t not in matched_truth:
                    size_totals[size_band(mask)] += PQResult(false_negatives=1)
            for p, mask in enumerate(prediction):
                if p not in matched_prediction:
                    size_totals[size_band(mask)] += PQResult(false_positives=1)
            per_record.append(
                {
                    "image_id": record["id"],
                    "filename": filename,
                    **summary(result),
                    "truth_count": len(truth),
                    "prediction_count": len(prediction),
                    "fragmented_truths": int(np.count_nonzero((overlaps > 0).sum(axis=1) > 1)),
                    "merged_predictions": int(np.count_nonzero((overlaps > 0).sum(axis=0) > 1)),
                }
            )
    return {
        **summary(total),
        "split": split,
        "images": len(files),
        "annotation_records": len(per_record),
        "groups": {key: summary(value) for key, value in grouped.items()},
        "records": per_record,
        "matched_iou_distribution": matched_ious,
        "matched_dice_distribution": [2 * value / (1 + value) for value in matched_ious],
        "size_bands": {key: summary(value) for key, value in size_totals.items()},
        "size_band_policy": "Global matching; TP/FN by truth area, FP by prediction area. "
        "tiny <1024, medium <4096, large >=4096 pixels.",
    }


def submission(data_root: Path, folder: Path, output: Path) -> dict:
    from PIL import Image

    images = sorted((data_root / "test/test_images").glob("*.jpeg"))
    require(bool(images), "Competition test images missing")
    files = prediction_files(folder, [path.name for path in images])
    counts = Counter()
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["filament_id", "segmentation_rle"])
        for image in images:
            with Image.open(image) as source:
                shape = (source.height, source.width)
            masks = label_masks(np.load(files[image.name], allow_pickle=False), shape)
            for index, mask in enumerate(masks, 1):
                encoded = coco_mask.encode(np.asfortranarray(mask, dtype=np.uint8))
                require(
                    np.array_equal(coco_mask.decode(encoded).astype(bool), mask),
                    "COCO RLE failed round-trip validation",
                )
                writer.writerow([f"{image.stem}_{index}", encoded["counts"].decode("ascii")])
            counts[image.name] = len(masks)
    return {
        "images": len(images),
        "instances": sum(counts.values()),
        "empty_images": sum(value == 0 for value in counts.values()),
        "counts": dict(counts),
    }
