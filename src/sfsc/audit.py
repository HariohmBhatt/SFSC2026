"""Audit the local competition data and generate the fixed development split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sfsc.splits import SPLITS, assign_grouped_splits

DEFAULT_ANNOTATION = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        f"p{round(quantile * 100):02d}": round(float(np.quantile(array, quantile)), 4)
        for quantile in QUANTILES
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _duplicate_hash_groups(paths: Iterable[Path]) -> tuple[dict[str, str], list[list[str]]]:
    hashes: dict[str, str] = {}
    by_hash: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        digest = _sha256(path)
        hashes[path.name] = digest
        by_hash[digest].append(path.name)
    duplicates = [sorted(names) for names in by_hash.values() if len(names) > 1]
    return hashes, sorted(duplicates)


def _inspect_images(paths: Sequence[Path], sample_size: int) -> dict[str, Any]:
    modes: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    bytes_total = 0
    image_means: list[float] = []
    image_stds: list[float] = []
    low_clipped: list[float] = []
    high_clipped: list[float] = []
    per_site_mean: dict[str, list[float]] = defaultdict(list)

    for path in paths:
        bytes_total += path.stat().st_size
        with Image.open(path) as image:
            image.load()
            modes[image.mode] += 1
            dimensions[f"{image.width}x{image.height}"] += 1
            grayscale = image.convert("L")
            grayscale.thumbnail((sample_size, sample_size), Image.Resampling.BOX)
            pixels = np.asarray(grayscale, dtype=np.uint8)
        mean = float(pixels.mean())
        image_means.append(mean)
        image_stds.append(float(pixels.std()))
        low_clipped.append(float(np.mean(pixels == 0)))
        high_clipped.append(float(np.mean(pixels == 255)))
        per_site_mean[path.stem[-2:]].append(mean)

    return {
        "count": len(paths),
        "bytes": bytes_total,
        "modes": dict(sorted(modes.items())),
        "dimensions": dict(sorted(dimensions.items())),
        "sample_size": sample_size,
        "per_image_mean_quantiles": _quantiles(image_means),
        "per_image_std_quantiles": _quantiles(image_stds),
        "zero_pixel_fraction_quantiles": _quantiles(low_clipped),
        "max_pixel_fraction_quantiles": _quantiles(high_clipped),
        "mean_intensity_by_site": {
            site: round(float(np.mean(values)), 4) for site, values in sorted(per_site_mean.items())
        },
    }


def _build_groups(
    images: Sequence[Mapping[str, Any]], annotations: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    annotations_by_image: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        annotations_by_image[str(annotation["image_id"])].append(annotation)

    image_records_by_file: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for image in images:
        image_records_by_file[str(image["file_name"])].append(image)

    groups: list[dict[str, Any]] = []
    for filename, records in sorted(image_records_by_file.items()):
        counts = [len(annotations_by_image[str(record["id"])]) for record in records]
        areas = [
            float(annotation["area"])
            for record in records
            for annotation in annotations_by_image[str(record["id"])]
        ]
        groups.append(
            {
                "filename": filename,
                "site": Path(filename).stem[-2:],
                "year": int(Path(filename).stem[:4]),
                "annotation_set_count": len(records),
                "annotation_ids": [str(record["id"]) for record in records],
                "instance_count_total": sum(counts),
                "instance_count_mean": float(np.mean(counts)),
                "instance_count_min": min(counts),
                "instance_count_max": max(counts),
                "instance_count_range": max(counts) - min(counts),
                "area_total": sum(areas),
                "area_mean": float(np.mean(areas)) if areas else 0.0,
            }
        )
    return groups


def _audit_annotations(data: Mapping[str, Any], train_files: set[str]) -> dict[str, Any]:
    images = data["images"]
    annotations = data["annotations"]
    categories = {int(category["id"]): str(category["name"]) for category in data["categories"]}
    image_ids = [str(image["id"]) for image in images]
    image_id_set = set(image_ids)
    json_files = {str(image["file_name"]) for image in images}

    errors: list[str] = []
    if len(image_ids) != len(image_id_set):
        errors.append("Image-record IDs are not unique")
    missing_references = sum(
        str(annotation["image_id"]) not in image_id_set for annotation in annotations
    )
    if missing_references:
        errors.append(f"{missing_references} annotations reference unknown image-record IDs")
    missing_files = sorted(json_files - train_files)
    unannotated_files = sorted(train_files - json_files)
    if missing_files:
        errors.append(f"{len(missing_files)} annotated filenames are absent from train_images")
    if unannotated_files:
        errors.append(f"{len(unannotated_files)} train images are absent from the annotation JSON")

    invalid_polygons = 0
    outside_coordinates = 0
    nonpositive_areas = 0
    nonpositive_boxes = 0
    for annotation in annotations:
        polygons = annotation.get("segmentation", [])
        if len(polygons) != 1 or len(polygons[0]) < 6 or len(polygons[0]) % 2:
            invalid_polygons += 1
        if any(
            coordinate < 0 or coordinate > 2048 for polygon in polygons for coordinate in polygon
        ):
            outside_coordinates += 1
        if float(annotation["area"]) <= 0:
            nonpositive_areas += 1
        if float(annotation["bbox"][2]) <= 0 or float(annotation["bbox"][3]) <= 0:
            nonpositive_boxes += 1

    integrity_counts = {
        "orphan_annotation_references": missing_references,
        "missing_annotated_files": len(missing_files),
        "unannotated_train_files": len(unannotated_files),
        "invalid_polygons": invalid_polygons,
        "outside_coordinates": outside_coordinates,
        "nonpositive_areas": nonpositive_areas,
        "nonpositive_boxes": nonpositive_boxes,
    }
    if any((invalid_polygons, outside_coordinates, nonpositive_areas, nonpositive_boxes)):
        errors.append(f"Invalid annotation geometry found: {integrity_counts}")

    areas = [float(annotation["area"]) for annotation in annotations]
    box_widths = [float(annotation["bbox"][2]) for annotation in annotations]
    box_heights = [float(annotation["bbox"][3]) for annotation in annotations]
    vertices = [len(annotation["segmentation"][0]) // 2 for annotation in annotations]
    instances_by_image = Counter(str(annotation["image_id"]) for annotation in annotations)
    records_by_filename: dict[str, list[str]] = defaultdict(list)
    for image in images:
        records_by_filename[str(image["file_name"])].append(str(image["id"]))
    disagreement_ranges = [
        max(instances_by_image[image_id] for image_id in record_ids)
        - min(instances_by_image[image_id] for image_id in record_ids)
        for record_ids in records_by_filename.values()
        if len(record_ids) > 1
    ]

    category_counts = Counter(int(annotation["category_id"]) for annotation in annotations)
    annotation_set_counts = Counter(len(record_ids) for record_ids in records_by_filename.values())
    return {
        "errors": errors,
        "integrity": integrity_counts,
        "image_record_count": len(images),
        "unique_filename_count": len(records_by_filename),
        "duplicate_image_record_count": len(images) - len(records_by_filename),
        "annotation_set_count_distribution": {
            str(key): value for key, value in sorted(annotation_set_counts.items())
        },
        "annotation_count": len(annotations),
        "annotations_per_record_quantiles": _quantiles(list(instances_by_image.values())),
        "category_counts": {
            categories[category_id]: category_counts.get(category_id, 0)
            for category_id in sorted(categories)
        },
        "iscrowd_counts": dict(sorted(Counter(str(a["iscrowd"]) for a in annotations).items())),
        "segment_components": dict(
            sorted(Counter(str(len(a["segmentation"])) for a in annotations).items())
        ),
        "annotations_with_spines": sum(bool(annotation.get("spine")) for annotation in annotations),
        "area_quantiles": _quantiles(areas),
        "bbox_width_quantiles": _quantiles(box_widths),
        "bbox_height_quantiles": _quantiles(box_heights),
        "polygon_vertex_quantiles": _quantiles(vertices),
        "area_threshold_counts": {
            f"below_{threshold}": sum(area < threshold for area in areas)
            for threshold in (16, 64, 256, 1024)
        },
        "bbox_touches_frame": sum(
            float(annotation["bbox"][0]) <= 0
            or float(annotation["bbox"][1]) <= 0
            or float(annotation["bbox"][0]) + float(annotation["bbox"][2]) >= 2048
            or float(annotation["bbox"][1]) + float(annotation["bbox"][3]) >= 2048
            for annotation in annotations
        ),
        "multiply_annotated_files": len(disagreement_ranges),
        "multiply_annotated_same_instance_count": sum(value == 0 for value in disagreement_ranges),
        "multiply_annotated_different_instance_count": sum(
            value > 0 for value in disagreement_ranges
        ),
        "annotation_instance_range_distribution": {
            str(key): value for key, value in sorted(Counter(disagreement_ranges).items())
        },
        "max_annotation_instance_count_range": max(disagreement_ranges, default=0),
    }


def _write_manifest(path: Path, groups: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "filename",
        "split",
        "site",
        "year",
        "annotation_set_count",
        "annotation_ids",
        "instance_count_total",
        "instance_count_mean",
        "instance_count_min",
        "instance_count_max",
        "instance_count_range",
        "area_total",
        "area_mean",
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for group in sorted(groups, key=lambda row: str(row["filename"])):
            row = dict(group)
            row["annotation_ids"] = ";".join(row["annotation_ids"])
            writer.writerow({field: row[field] for field in fields})


def _split_summary(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in SPLITS:
        rows = [group for group in groups if group["split"] == split]
        result[split] = {
            "unique_images": len(rows),
            "annotation_records": sum(int(row["annotation_set_count"]) for row in rows),
            "instances": sum(int(row["instance_count_total"]) for row in rows),
            "sites": dict(sorted(Counter(str(row["site"]) for row in rows).items())),
            "years": dict(sorted(Counter(str(row["year"]) for row in rows).items())),
            "annotation_set_counts": dict(
                sorted(Counter(str(row["annotation_set_count"]) for row in rows).items())
            ),
            "instance_count_mean_quantiles": _quantiles(
                [float(row["instance_count_mean"]) for row in rows]
            ),
        }
    return result


def _markdown_report(report: Mapping[str, Any]) -> str:
    annotation = report["annotations"]
    train = report["images"]["train"]
    competition_test = report["images"]["competition_test"]
    split = report["split"]
    category_rows = "\n".join(
        f"| {name} | {count:,} |" for name, count in annotation["category_counts"].items()
    )
    split_rows = "\n".join(
        f"| {name} | {values['unique_images']:,} | {values['annotation_records']:,} | "
        f"{values['instances']:,} |"
        for name, values in split["counts"].items()
    )
    area = annotation["area_quantiles"]
    width = annotation["bbox_width_quantiles"]
    height = annotation["bbox_height_quantiles"]
    vertices = annotation["polygon_vertex_quantiles"]
    geometry_rows = "\n".join(
        f"| {label} | {values['p01']} | {values['p05']} | {values['p50']} | "
        f"{values['p95']} | {values['p99']} | {values['p100']} |"
        for label, values in (
            ("Area", area),
            ("Box width", width),
            ("Box height", height),
            ("Vertices", vertices),
        )
    )

    return f"""# Generated Dataset Assessment

Generated by `sfsc-audit` from the local download. Pixel statistics use a
{train["sample_size"]} x {train["sample_size"]} area-resampled view of every image;
file modes, dimensions, counts, hashes, and annotation statistics are exact.

## Inventory

| Item | Value |
|---|---:|
| Training JPEGs | {train["count"]:,} |
| Competition-test JPEGs | {competition_test["count"]:,} |
| COCO image records | {annotation["image_record_count"]:,} |
| Unique annotated filenames | {annotation["unique_filename_count"]:,} |
| Duplicate annotation records | {annotation["duplicate_image_record_count"]:,} |
| Filament instances | {annotation["annotation_count"]:,} |

All training images have modes `{train["modes"]}` and dimensions
`{train["dimensions"]}`. Competition-test images have modes
`{competition_test["modes"]}` and dimensions `{competition_test["dimensions"]}`.

## Integrity

- Annotation integrity counters: `{annotation["integrity"]}`.
- Exact train duplicate hash groups: `{report["hashes"]["train_duplicate_groups"]}`.
- Exact test duplicate hash groups: `{report["hashes"]["test_duplicate_groups"]}`.
- Train/test filename overlap: `{report["hashes"]["cross_filename_overlap"]}`.
- Train/test content-hash overlap: `{report["hashes"]["cross_hash_overlap"]}`.
- Bounding boxes touching the frame: `{annotation["bbox_touches_frame"]}`.

## Categories

| Category | Instances |
|---|---:|
{category_rows}

## Instance Geometry

| Measure | p01 | p05 | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
{geometry_rows}

Tiny-object counts are `{annotation["area_threshold_counts"]}`. This confirms
that downsampling full frames is unsafe for a substantial part of the target
distribution.

## Multiple Annotation Sets

- Annotation sets per source: `{annotation["annotation_set_count_distribution"]}`.
- Multiply annotated sources: {annotation["multiply_annotated_files"]:,}.
- Sources whose annotators disagree on instance count:
  {annotation["multiply_annotated_different_instance_count"]:,}.
- Maximum instance-count range for one source:
  {annotation["max_annotation_instance_count_range"]}.
- Full disagreement range distribution:
  `{annotation["annotation_instance_range_distribution"]}`.

This is the principal leakage and label-noise risk. Splits must be grouped by
source filename, and one annotation set must remain an indivisible target during
training/evaluation.

## Approximate Pixel Statistics

Training per-image mean intensity quantiles:
`{train["per_image_mean_quantiles"]}`.

Training per-image standard-deviation quantiles:
`{train["per_image_std_quantiles"]}`.

Mean training intensity by site: `{train["mean_intensity_by_site"]}`.

Competition-test per-image mean intensity quantiles:
`{competition_test["per_image_mean_quantiles"]}`.

Competition-test per-image standard-deviation quantiles:
`{competition_test["per_image_std_quantiles"]}`.

These summaries are useful for detecting broad domain shift, but normalization
parameters used by a model must be fitted from training-manifest rows only.

## Fixed Split

Seed: `{split["seed"]}`. Manifest SHA-256: `{split["manifest_sha256"]}`.

| Split | Unique JPEGs | Annotation records | Instances |
|---|---:|---:|---:|
{split_rows}

The complete site/year distributions are in `splits/split_summary.json` and the
source-level assignments are in `splits/split_manifest.csv`.

## Conclusions

1. Use native-resolution tiled inference and foreground-aware training crops.
2. Keep all annotation records for one JPEG in one split.
3. Score each annotation image-record independently, then aggregate global PQ.
4. Track false merges/fragments and stratify diagnostics by site, year, size,
   and annotation multiplicity.
5. Treat category/spine metadata as out of scope unless organizers explicitly
   confirm it is allowed as a training target.
"""


def run_audit(
    data_root: Path,
    manifest_path: Path,
    summary_path: Path,
    report_json_path: Path,
    report_md_path: Path,
    *,
    seed: int,
    sample_size: int,
) -> dict[str, Any]:
    train_dir = data_root / "train" / "train_images"
    competition_test_dir = data_root / "test" / "test_images"
    annotation_path = data_root / "train" / DEFAULT_ANNOTATION
    for path in (train_dir, competition_test_dir, annotation_path):
        if not path.exists():
            raise FileNotFoundError(f"Required dataset path does not exist: {path}")

    train_paths = sorted(train_dir.glob("*.jpeg"))
    competition_test_paths = sorted(competition_test_dir.glob("*.jpeg"))
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    required_keys = {"info", "licenses", "categories", "images", "annotations"}
    if set(data) != required_keys:
        raise ValueError(f"Unexpected annotation keys: {set(data)}")

    train_files = {path.name for path in train_paths}
    competition_test_files = {path.name for path in competition_test_paths}
    annotation_audit = _audit_annotations(data, train_files)
    if annotation_audit["errors"]:
        raise ValueError("; ".join(annotation_audit["errors"]))

    train_hashes, train_duplicates = _duplicate_hash_groups(train_paths)
    test_hashes, test_duplicates = _duplicate_hash_groups(competition_test_paths)
    cross_hashes = sorted(set(train_hashes.values()) & set(test_hashes.values()))
    cross_filenames = sorted(train_files & competition_test_files)
    if train_duplicates or test_duplicates or cross_hashes or cross_filenames:
        raise ValueError("Exact image duplicates or train/test overlap detected")

    groups = _build_groups(data["images"], data["annotations"])
    assignments = assign_grouped_splits(groups, seed=seed)
    for group in groups:
        group["split"] = assignments[str(group["filename"])]
    _write_manifest(manifest_path, groups)
    manifest_sha256 = _sha256(manifest_path)

    counts = _split_summary(groups)
    split_summary = {
        "seed": seed,
        "fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "group_key": "source JPEG filename",
        "manifest_sha256": manifest_sha256,
        "counts": counts,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(split_summary, indent=2) + "\n", encoding="utf-8")

    report = {
        "dataset_root": str(data_root),
        "annotation_file": str(annotation_path),
        "images": {
            "train": _inspect_images(train_paths, sample_size),
            "competition_test": _inspect_images(competition_test_paths, sample_size),
            "train_sites": dict(sorted(Counter(path.stem[-2:] for path in train_paths).items())),
            "competition_test_sites": dict(
                sorted(Counter(path.stem[-2:] for path in competition_test_paths).items())
            ),
            "train_years": dict(sorted(Counter(path.stem[:4] for path in train_paths).items())),
            "competition_test_years": dict(
                sorted(Counter(path.stem[:4] for path in competition_test_paths).items())
            ),
        },
        "annotations": annotation_audit,
        "hashes": {
            "algorithm": "SHA-256",
            "train_duplicate_groups": train_duplicates,
            "test_duplicate_groups": test_duplicates,
            "cross_filename_overlap": cross_filenames,
            "cross_hash_overlap": cross_hashes,
        },
        "split": split_summary,
    }
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.write_text(_markdown_report(report), encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("SFSC_DATA_ROOT", "SFSCDataset")),
    )
    parser.add_argument("--manifest", type=Path, default=Path("splits/split_manifest.csv"))
    parser.add_argument("--summary", type=Path, default=Path("splits/split_summary.json"))
    parser.add_argument("--report-json", type=Path, default=Path("reports/data_assessment.json"))
    parser.add_argument("--report-md", type=Path, default=Path("reports/data_assessment.md"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample-size", type=int, default=256)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    report = run_audit(
        arguments.data_root,
        arguments.manifest,
        arguments.summary,
        arguments.report_json,
        arguments.report_md,
        seed=arguments.seed,
        sample_size=arguments.sample_size,
    )
    counts = report["split"]["counts"]
    print(
        "Audit passed. "
        + ", ".join(f"{split}={counts[split]['unique_images']}" for split in SPLITS)
        + f"; manifest={report['split']['manifest_sha256']}"
    )


if __name__ == "__main__":
    main()
