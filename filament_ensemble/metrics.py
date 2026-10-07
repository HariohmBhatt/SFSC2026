from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as coco_mask
from scipy.optimize import linear_sum_assignment


def native(rle: Mapping[str, Any]) -> dict[str, Any]:
    counts = rle["counts"]
    return {
        "size": list(rle["size"]),
        "counts": counts.encode("ascii") if isinstance(counts, str) else counts,
    }


def annotation_sets(coco: Mapping[str, Any]) -> dict[str, list[list[dict]]]:
    annotations: dict[str, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[str(annotation["image_id"])].append(annotation)
    result: dict[str, list[list[dict]]] = defaultdict(list)
    for image in sorted(coco["images"], key=lambda row: str(row["id"])):
        shape = (int(image["height"]), int(image["width"]))
        masks = []
        for annotation in annotations[str(image["id"])]:
            if annotation.get("iscrowd", 0):
                continue
            segmentation = annotation["segmentation"]
            mask = (
                native(segmentation)
                if isinstance(segmentation, dict)
                else coco_mask.merge(coco_mask.frPyObjects(segmentation, *shape))
            )
            masks.append(mask)
        result[Path(image["file_name"]).name].append(masks)
    return dict(result)


def match(truth: Sequence[dict], prediction: Sequence[dict]) -> dict[str, float | int]:
    truth, prediction = [native(row) for row in truth], [native(row) for row in prediction]
    matrix = (
        np.asarray(coco_mask.iou(truth, prediction, [0] * len(prediction)))
        if truth and prediction
        else np.zeros((len(truth), len(prediction)))
    )
    eligible = np.where(matrix > 0.5, matrix, 0)
    rows, columns = linear_sum_assignment(eligible, maximize=True)
    values = [
        float(eligible[row, column])
        for row, column in zip(rows, columns, strict=True)
        if eligible[row, column] > 0
    ]
    return {
        "sum_iou": sum(values),
        "tp": len(values),
        "fp": len(prediction) - len(values),
        "fn": len(truth) - len(values),
        "records": 1,
    }


def score(
    predictions: Mapping[str, Sequence[dict]], truth: Mapping[str, Sequence[Sequence[dict]]]
) -> dict[str, float | int]:
    total: dict[str, float | int] = {"sum_iou": 0.0, "tp": 0, "fp": 0, "fn": 0, "records": 0}
    for filename, records in truth.items():
        for record in records:
            row = match(record, predictions.get(filename, []))
            for key in total:
                total[key] += row[key]
    denominator = total["tp"] + 0.5 * (total["fp"] + total["fn"])
    return {
        **total,
        "denominator": denominator,
        "pq": total["sum_iou"] / denominator if denominator else 1.0,
        "sq": total["sum_iou"] / total["tp"] if total["tp"] else 0.0,
        "rq": total["tp"] / denominator if denominator else 1.0,
    }
