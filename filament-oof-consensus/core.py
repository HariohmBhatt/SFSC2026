"""CPU implementation shared by training, validation and final inference."""

from __future__ import annotations

import hashlib
from datetime import datetime

import numpy as np
from pycocotools import mask as coco_mask
from scipy import ndimage
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import expit

FEATURES = [
    "detector_logit",
    "foreground_mean",
    "log_area",
    "log_seed_area",
    "area_ratio",
    "seed_iou",
    "log_aspect",
    "fill",
    "components",
    "inside_intensity",
    "ring_contrast",
    "kind_refined",
]


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def encode(mask):
    rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": list(mask.shape), "counts": rle["counts"].decode("ascii")}


def decode(rle):
    return coco_mask.decode({"size": rle["size"], "counts": rle["counts"].encode("ascii")}).astype(
        bool
    )


def make_folds(filenames, folds=3, block_days=60, purge_days=7):
    """Date blocks are shared across sites; purge is mandatory and auditable."""
    days = {
        name: datetime.strptime(name[:14], "%Y%m%d%H%M%S").date().toordinal() for name in filenames
    }
    assignments = {
        name: int(hashlib.sha256(f"sfsc-v3:{day // block_days}".encode()).hexdigest()[:8], 16)
        % folds
        for name, day in days.items()
    }
    result = []
    for fold in range(folds):
        held = sorted(n for n in filenames if assignments[n] == fold)
        held_days = np.array([days[n] for n in held])
        train = sorted(
            n
            for n in filenames
            if assignments[n] != fold and np.min(np.abs(held_days - days[n])) > purge_days
        )
        purged = sorted(set(filenames) - set(train) - set(held))
        if len(held) < 50 or len(train) < 150:
            raise ValueError(f"Insufficient frozen fold {fold}: {len(train)}/{len(held)}")
        result.append({"fold": fold, "train": train, "held": held, "purged": purged})
    return result


def mask_features(image, seed, mask, detector_score, foreground_mean, kind):
    area = int(mask.sum())
    if not area:
        raise ValueError("Cannot score an empty mask")
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - 5), min(mask.shape[0], ys.max() + 6)
    x0, x1 = max(0, xs.min() - 5), min(mask.shape[1], xs.max() + 6)
    local = mask[y0:y1, x0:x1]
    pixels = image[y0:y1, x0:x1].astype(float) / 255
    ring = ndimage.binary_dilation(local, iterations=4) & ~local
    inside = float(pixels[local].mean())
    contrast = float(pixels[ring].mean()) - inside if ring.any() else 0.0
    width, height = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    seed_area = int(seed.sum())
    intersection = np.logical_and(seed, mask).sum()
    score = np.clip(detector_score, 1e-5, 1 - 1e-5)
    return [
        float(np.log(score / (1 - score))),
        float(foreground_mean),
        float(np.log1p(area)),
        float(np.log1p(seed_area)),
        float(np.clip(area / max(seed_area, 1), 0, 8)),
        float(intersection / max(seed_area + area - intersection, 1)),
        float(np.log(max(width, height) / min(width, height))),
        float(area / (width * height)),
        float(min(ndimage.label(local, structure=np.ones((3, 3)))[1], 20)),
        inside,
        contrast,
        float(kind == "refined"),
    ]


def annotate_targets(candidates, truth_records):
    """Direct expected IoU contribution, averaged over independent annotation sets.

    This is single-candidate quality, not a duplicate-sensitive assignment label.
    Final set matching and non-overlap are enforced by the evaluator/assembler.
    """
    if not candidates:
        return
    predictions = [
        dict(size=c["rle"]["size"], counts=c["rle"]["counts"].encode("ascii")) for c in candidates
    ]
    targets = np.zeros(len(candidates))
    for truth in truth_records:
        if truth:
            iou = np.asarray(coco_mask.iou(truth, predictions, [0] * len(predictions)))
            best = iou.max(axis=0)
            targets += best * (best > 0.5)
    targets /= max(len(truth_records), 1)
    for candidate, target in zip(candidates, targets, strict=True):
        candidate["target"] = float(target)
        candidate["record_weight"] = len(truth_records)


def fit_quality(caches, regularization=0.02):
    rows = [c for items in caches.values() for c in items]
    if not rows:
        raise ValueError("No OOF quality examples")
    x = np.array([c["features"] for c in rows], dtype=float)
    y = np.array([c["target"] for c in rows])
    w = np.array([c["record_weight"] for c in rows], dtype=float)
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 0.05)
    x = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    w /= w.sum()

    def objective(beta):
        logits = x @ beta
        penalty = regularization * np.dot(beta[1:], beta[1:]) / 2
        loss = np.dot(w, np.logaddexp(0, logits) - y * logits) + penalty
        grad = x.T @ (w * (expit(logits) - y))
        grad[1:] += regularization * beta[1:]
        return loss, grad

    fitted = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
    if not fitted.success or not np.isfinite(fitted.fun):
        raise ValueError(f"Quality fitting failed: {fitted.message}")
    return {
        "feature_names": FEATURES,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": fitted.x.tolist(),
        "regularization": regularization,
        "examples": len(rows),
        "images": len(caches),
        "loss": float(fitted.fun),
        "target": "E[IoU * 1(IoU > 0.5)]",
    }


def quality(candidate, model):
    if model["feature_names"] != FEATURES:
        raise ValueError("Quality feature schema mismatch")
    x = (np.array(candidate["features"]) - model["mean"]) / model["scale"]
    return float(expit(model["coefficients"][0] + np.dot(x, model["coefficients"][1:])))


def assemble(candidates, model, threshold, minimum_area=64):
    """Select one alternative per prompt, suppress duplicates, then paint overlaps."""
    best = {}
    for candidate in candidates:
        value = quality(candidate, model)
        if candidate["area"] < minimum_area or value < threshold:
            continue
        identity = candidate["proposal_id"]
        if identity not in best or value > best[identity][0]:
            best[identity] = (value, candidate)
    accepted = []
    occupied = None
    for value, candidate in sorted(best.values(), key=lambda item: item[0], reverse=True):
        mask = decode(candidate["rle"])
        if occupied is None:
            occupied = np.zeros_like(mask)
        duplicate = False
        for _, previous in accepted:
            inter = np.logical_and(mask, previous).sum()
            if inter / max(np.logical_or(mask, previous).sum(), 1) >= 0.6:
                duplicate = True
                break
        if duplicate:
            continue
        mask &= ~occupied
        if mask.sum() >= minimum_area:
            accepted.append((value, mask))
            occupied |= mask
    return [mask for _, mask in accepted]


def score(truth_records, masks):
    pred = [coco_mask.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in masks]
    totals = np.zeros(8, dtype=float)
    for truth in truth_records:
        iou = (
            np.asarray(coco_mask.iou(truth, pred, [0] * len(pred)))
            if truth and pred
            else (np.zeros((len(truth), len(pred))))
        )
        eligible = iou > 0.5
        tp = eligible.sum()
        fp = len(pred) - eligible.any(axis=0).sum()
        fn = len(truth) - eligible.any(axis=1).sum()
        rows, cols = linear_sum_assignment(np.where(eligible, iou, 0), maximize=True)
        matched = iou[rows, cols]
        matched = matched[matched > 0.5]
        totals += [
            iou[eligible].sum(),
            tp,
            fp,
            fn,
            matched.sum(),
            len(matched),
            len(pred) - len(matched),
            len(truth) - len(matched),
        ]
    return totals


def metrics(stats):
    s, tp, fp, fn, s1, tp1, fp1, fn1 = np.asarray(stats, dtype=float)
    den, den1 = tp + (fp + fn) / 2, tp1 + (fp1 + fn1) / 2
    return {
        "pq": float(s / den) if den else 0.0,
        "one_to_one_pq": float(s1 / den1) if den1 else 0.0,
        "sq": float(s / tp) if tp else 0.0,
        "rq": float(tp / den) if den else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "sum_iou": float(s),
    }


def bootstrap_delta(candidate, baseline, filenames, replicates=2000):
    groups = {}
    for name in filenames:
        day = datetime.strptime(name[:14], "%Y%m%d%H%M%S").date().toordinal()
        groups.setdefault(day // 60, []).append(name)
    first = np.array(
        [sum((np.array(candidate[n]) for n in names), np.zeros(8)) for names in groups.values()]
    )
    second = np.array(
        [sum((np.array(baseline[n]) for n in names), np.zeros(8)) for names in groups.values()]
    )
    rng = np.random.default_rng(20260907)
    deltas = []
    for _ in range(replicates):
        sample = rng.integers(0, len(first), len(first))
        deltas.append(
            metrics(first[sample].sum(axis=0))["pq"] - metrics(second[sample].sum(axis=0))["pq"]
        )
    return {
        "blocks": len(groups),
        "replicates": replicates,
        "ci95": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "median": float(np.median(deltas)),
    }
