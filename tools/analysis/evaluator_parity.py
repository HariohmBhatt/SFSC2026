"""Compare cached official pair-counting and local matching on validation only."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = ROOT / "outputs/precision-v2-20260906-052615/parity/experiment/predictions"
SOURCE = (
    ROOT / "reports/competition_watch/cache/azimahmadzadeh__self-evaluation-notebook/source.txt"
)
REPORT = ROOT / "outputs/precision-v2-20260906-052615/results/experiment/report.json"


def summarize(stats):
    iou, tp, fp, fn = stats
    denominator = tp + 0.5 * (fp + fn)
    return {
        "matched_iou": float(iou),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "pq": float(iou / denominator) if denominator else 0.0,
    }


def stats(matrix, pair_counting=False, inclusive=False):
    hit = matrix >= 0.5 if inclusive else matrix > 0.5
    if pair_counting:
        return np.array([matrix[hit].sum(), hit.sum(), (~hit.any(0)).sum(), (~hit.any(1)).sum()])
    eligible = np.where(hit, matrix, 0)
    rows, cols = linear_sum_assignment(eligible, maximize=True)
    selected = eligible[rows, cols]
    tp = int(sum(selected > 0))
    return np.array([selected.sum(), tp, matrix.shape[1] - tp, matrix.shape[0] - tp])


def main():
    names = {
        r["filename"]
        for r in csv.DictReader((ROOT / "splits/split_manifest.csv").open())
        if r["split"] == "validation"
    }
    expected = {Path(n).stem for n in names}
    paths = {p.stem: p for p in PREDICTIONS.glob("*.npy")}
    assert len(names) == 106 and paths.keys() == expected, "Requires exact 106 validation maps"
    # Only validation images/annotation objects are retained or rasterized.
    raw = json.loads(
        (ROOT / "SFSCDataset/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json").read_text()
    )
    records = [r for r in raw["images"] if Path(r["file_name"]).name in names]
    allowed_ids = {r["id"] for r in records}
    annotations = defaultdict(list)
    for ann in raw["annotations"]:
        if ann["image_id"] in allowed_ids:
            annotations[ann["image_id"]].append(ann)
    del raw
    expected_records = {
        r["image_id"]: r for r in json.loads(REPORT.read_text())["metrics"]["records"]
    }
    totals = {k: np.zeros(4) for k in ["local", "official_pairs", "inclusive_local"]}
    differences, exact_half, duplicate_edges = [], 0, 0
    for record in records:
        name = Path(record["file_name"]).name
        labels = np.load(paths[Path(name).stem], allow_pickle=False)
        assert labels.shape == (record["height"], record["width"])
        pred = [
            coco_mask.encode(np.asfortranarray(labels == k, dtype=np.uint8))
            for k in np.unique(labels)
            if k
        ]
        truth = [
            coco_mask.merge(
                coco_mask.frPyObjects(a["segmentation"], record["height"], record["width"])
            )
            for a in annotations[record["id"]]
        ]
        matrix = (
            np.asarray(coco_mask.iou(truth, pred, [0] * len(pred)))
            if truth and pred
            else np.zeros((len(truth), len(pred)))
        )
        local = stats(matrix)
        cached = expected_records[record["id"]]
        expected_stats = [
            cached[k]
            for k in ["matched_iou", "true_positives", "false_positives", "false_negatives"]
        ]
        assert np.allclose(local, expected_stats, atol=1e-10), record["id"]
        official = stats(matrix, pair_counting=True)
        inclusive = stats(matrix, inclusive=True)
        for key, values in zip(totals, [local, official, inclusive], strict=True):
            totals[key] += values
        exact_half += int((matrix == 0.5).sum())
        duplicate_edges += int(((matrix > 0.5).sum(0) > 1).sum())
        if not np.allclose(local, official, atol=1e-10):
            differences.append(
                {"record": record["id"], "local": summarize(local), "official": summarize(official)}
            )
    result = {
        "images": len(names),
        "annotation_records": len(records),
        "official_source": str(SOURCE.relative_to(ROOT)),
        "official_source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "implementation": "NumPy translation of cached get_pq_score; exact-integer COCO IoU",
        "live_official_evaluator_verified": False,
        "totals": {k: summarize(v) for k, v in totals.items()},
        "exact_half_iou_pairs": exact_half,
        "predictions_hitting_multiple_truths_above_half": duplicate_edges,
        "different_records": differences,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
