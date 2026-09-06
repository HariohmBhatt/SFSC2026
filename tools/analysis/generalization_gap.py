"""Read-only audit of saved development metrics and unlabeled test images.

Never opens annotations or internal-test images. Outputs JSON summary to stdout.
Run from repository root with Python, NumPy, Pillow, and SciPy available.
"""

import csv
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "outputs/precision-v2-20260906-052615/results/experiment/report.json"
CONTROL = ROOT / "outputs/precision-v2-20260906-052615/results/experiment/baseline-metrics.json"
RNG = np.random.default_rng(20260906)
N_BOOT = 20000
FIELDS = ("matched_iou", "true_positives", "false_positives", "false_negatives")


def aggregate(records):
    out = defaultdict(lambda: np.zeros(4))
    for row in records:
        out[row["filename"]] += [row[k] for k in FIELDS]
    return dict(out)


def pq(stats):
    return stats[..., 0] / np.maximum(stats[..., 1] + 0.5 * (stats[..., 2] + stats[..., 3]), 1e-12)


def describe(stats):
    stats = np.asarray(stats)
    draws = RNG.integers(len(stats), size=(N_BOOT, len(stats)))
    return {
        "images": len(stats),
        "pq": float(pq(stats.sum(0))),
        "ci95": np.quantile(pq(stats[draws].sum(1)), [0.025, 0.975]).tolist(),
        "stats": dict(zip(FIELDS, stats.sum(0).tolist(), strict=True)),
    }


def timestamp(name):
    return datetime.strptime(name[:14], "%Y%m%d%H%M%S").replace(tzinfo=UTC).timestamp()


def main():
    result = {}
    final = json.loads(REPORT.read_text())["metrics"]
    control = json.loads(CONTROL.read_text())
    f, b = aggregate(final["records"]), aggregate(control["records"])
    assert f.keys() == b.keys()
    names = sorted(f)
    fa, ba = np.array([f[n] for n in names]), np.array([b[n] for n in names])
    draws = RNG.integers(len(names), size=(N_BOOT, len(names)))
    delta = pq(fa[draws].sum(1)) - pq(ba[draws].sum(1))
    result["paired_bootstrap"] = {
        "draws": N_BOOT,
        "unit": "source JPEG",
        "seed": 20260906,
        "final": describe(fa),
        "control": describe(ba),
        "delta": final["pq"] - control["pq"],
        "delta_ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
        "bootstrap_fraction_delta_le_zero": float(np.mean(delta <= 0)),
    }
    calibration = set(
        json.loads((REPORT.parent / "checkpoint/calibration.json").read_text())["filenames"]
    )
    result["calibration_membership"] = {}
    for label, selected in [
        ("calibration", calibration),
        ("other_validation", set(names) - calibration),
    ]:
        idx = [i for i, n in enumerate(names) if n in selected]
        result["calibration_membership"][label] = {
            "final": describe(fa[idx]),
            "control": describe(ba[idx]),
        }
    # Only training and validation membership is analyzed.
    manifest = {
        r["filename"]: r
        for r in csv.DictReader((ROOT / "splits/split_manifest.csv").open())
        if r["split"] in ("train", "validation")
    }
    train = [n for n, r in manifest.items() if r["split"] == "train"]
    test = sorted(p.name for p in (ROOT / "SFSCDataset/test/test_images").glob("*.jpeg"))
    assert len(names) == 106 and len(train) == 495 and len(test) == 180
    train_ts = np.array([timestamp(n) for n in train])
    distances = np.array([np.min(np.abs(timestamp(n) - train_ts)) / 86400 for n in names])
    test_distances = np.array([np.min(np.abs(timestamp(n) - train_ts)) / 86400 for n in test])
    result["temporal"] = {
        "validation_distance_days_quantiles": np.quantile(
            distances, [0, 0.25, 0.5, 0.75, 1]
        ).tolist(),
        "test_distance_days_quantiles": np.quantile(
            test_distances, [0, 0.25, 0.5, 0.75, 1]
        ).tolist(),
        "cumulative_counts": {
            str(t): {"validation": int(sum(distances <= t)), "test": int(sum(test_distances <= t))}
            for t in [1, 3, 7, 14]
        },
        "strata": {
            label: describe(fa[mask])
            for label, mask in [
                ("<=1 day", distances <= 1),
                ("1-3 days", (distances > 1) & (distances <= 3)),
                ("3-7 days", (distances > 3) & (distances <= 7)),
                (">7 days", distances > 7),
            ]
        },
        "spearman_per_image_pq_vs_distance": dict(
            zip(["rho", "p_value"], map(float, spearmanr(distances, pq(fa))), strict=True)
        ),
    }
    for attr, getter in [("site", lambda n: n[14:16]), ("year", lambda n: n[:4])]:
        vc, tc = Counter(map(getter, names)), Counter(map(getter, test))
        total = np.zeros(4)
        result[attr] = {}
        for group in sorted(vc.keys() | tc.keys()):
            members = [i for i, n in enumerate(names) if getter(n) == group]
            item = {"validation_images": vc[group], "test_images": tc[group]}
            if members:
                item.update(describe(fa[members]))
                total += fa[members].sum(0) * tc[group] / vc[group]
            result[attr][group] = item
        result[attr + "_reweighted_pq"] = float(pq(total))
    result["validation_multiplicity"] = {
        str(k): {
            "images": sum(int(manifest[n]["annotation_set_count"]) == k for n in names),
            "metrics": final["groups"]["multiplicity:" + str(k)],
        }
        for k in [1, 2, 3]
    }
    result["size_bands"] = final["size_bands"]
    result["diagnostics"] = {
        k: sum(r[k] for r in final["records"]) for k in ["fragmented_truths", "merged_predictions"]
    }
    # Fixed image-only summaries, sampled on a 256x256 grid with no label access.
    yy, xx = np.mgrid[:256, :256]
    radius = np.hypot(xx - 127.5, yy - 127.5) / 128
    center = radius < 0.6
    corners = radius > 1.2
    paths = {p.name: p for p in (ROOT / "SFSCDataset/train").rglob("*.jpeg") if p.name in f}

    def covariates(name, istest=False):
        path = ROOT / "SFSCDataset/test/test_images" / name if istest else paths[name]
        with Image.open(path) as im:
            a = (
                np.asarray(
                    im.convert("L").resize((256, 256), Image.Resampling.BILINEAR), dtype=float
                )
                / 255
            )
        c = a[center]
        return [
            float(np.median(c)),
            float(np.quantile(c, 0.9) - np.quantile(c, 0.1)),
            float(np.median(a[corners])),
            float(np.mean(np.abs(np.diff(a, axis=0))[center[:-1]])),
        ]

    va = np.array([covariates(n) for n in names])
    ta = np.array([covariates(n, True) for n in test])
    result["image_covariates"] = {}
    for i, label in enumerate(
        ["central_median", "central_p90_p10", "corner_median", "central_vertical_gradient"]
    ):
        pooled = np.sqrt((va[:, i].var(ddof=1) + ta[:, i].var(ddof=1)) / 2)
        correlation = (
            dict(zip(["rho", "p_value"], map(float, spearmanr(va[:, i], pq(fa))), strict=True))
            if np.ptp(va[:, i])
            else None
        )
        result["image_covariates"][label] = {
            "validation_median_iqr": np.quantile(va[:, i], [0.25, 0.5, 0.75]).tolist(),
            "test_median_iqr": np.quantile(ta[:, i], [0.25, 0.5, 0.75]).tolist(),
            "standardized_mean_difference_test_minus_validation": float(
                (ta[:, i].mean() - va[:, i].mean()) / pooled
            )
            if pooled
            else None,
            "spearman_validation_pq": correlation,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
