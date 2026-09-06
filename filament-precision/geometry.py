"""CPU image/instance operations, independent of TensorFlow."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


def input_channels(gray: np.ndarray) -> np.ndarray:
    integer_input = np.issubdtype(np.asarray(gray).dtype, np.integer)
    gray = np.asarray(gray, dtype=np.float32)
    if integer_input:
        gray = gray / 255.0
    local = ndimage.uniform_filter(gray, size=65, mode="reflect")
    return np.stack((gray * 2 - 1, np.clip((gray - local) * 5, -1, 1)), axis=-1)


@lru_cache(maxsize=16)
def read_image(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")).copy()


def polygon_bounds(annotation: dict) -> tuple[float, float, float, float]:
    points = np.concatenate([np.asarray(p).reshape(-1, 2) for p in annotation["segmentation"]])
    low, high = points.min(axis=0), points.max(axis=0)
    return (
        float(low[0]),
        float(low[1]),
        float(high[0] - low[0]),
        float(high[1] - low[1]),
    )


def crop_targets(annotations: list[dict], x: int, y: int, size: int) -> np.ndarray:
    """Rasterize one complete annotator set, with halo to avoid false crop-edge contours."""
    margin = 3
    side = size + 2 * margin
    foreground = np.zeros((side, side), dtype=bool)
    interior = np.zeros_like(foreground)
    boundary = np.zeros_like(foreground)
    for annotation in annotations:
        bx, by, bw, bh = polygon_bounds(annotation)
        if bx + bw < x - margin or by + bh < y - margin or bx > x + side or by > y + side:
            continue
        polygons = []
        for polygon in annotation["segmentation"]:
            points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
            points -= np.array([x - margin, y - margin])
            polygons.append(points.ravel().tolist())
        rles = coco_mask.frPyObjects(polygons, side, side)
        mask = coco_mask.decode(coco_mask.merge(rles)).astype(bool)
        core = ndimage.binary_erosion(mask, iterations=1)
        if mask.any() and not core.any():
            core = mask.copy()  # Preserve genuinely thin/tiny components.
        foreground |= mask
        interior |= core
        boundary |= mask & ~core
    target = np.stack((foreground, interior, boundary), axis=-1).astype(np.float32)
    return target[margin : margin + size, margin : margin + size]


def sample_crop(dataset, filename: str, config: dict, rng: np.random.Generator):
    image = read_image(str(dataset.image_path(filename)))
    records = dataset.records[filename]
    record = records[int(rng.integers(len(records)))]
    annotations = dataset.annotations[record["id"]]
    size = config["crop_size"]
    scale = rng.uniform(*config["scale_range"])
    side = min(round(size / scale), min(image.shape))
    if annotations and rng.random() < config["foreground_probability"]:
        annotation = annotations[int(rng.integers(len(annotations)))]
        bx, by, bw, bh = polygon_bounds(annotation)
        cx, cy = bx + bw / 2, by + bh / 2
        x = int(
            np.clip(
                cx - side / 2 + rng.uniform(-side / 4, side / 4),
                0,
                image.shape[1] - side,
            )
        )
        y = int(
            np.clip(
                cy - side / 2 + rng.uniform(-side / 4, side / 4),
                0,
                image.shape[0] - side,
            )
        )
    else:
        x = int(rng.integers(image.shape[1] - side + 1))
        y = int(rng.integers(image.shape[0] - side + 1))
    gray = image[y : y + side, x : x + side]
    target = crop_targets(annotations, x, y, side)
    gray = (
        np.asarray(
            Image.fromarray(gray).resize((size, size), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        / 255
    )
    target = np.stack(
        [
            np.asarray(
                Image.fromarray(target[..., i]).resize((size, size), Image.Resampling.NEAREST)
            )
            for i in range(3)
        ],
        axis=-1,
    )
    turns = int(rng.integers(4))
    gray, target = np.rot90(gray, turns), np.rot90(target, turns)
    if rng.random() < 0.5:
        gray, target = gray[:, ::-1], target[:, ::-1]
    gray = np.clip((gray - 0.5) * rng.uniform(0.8, 1.2) + 0.5 + rng.uniform(-0.05, 0.05), 0, 1)
    gray = gray ** rng.uniform(0.8, 1.2)
    if rng.random() < 0.15:
        gray = ndimage.gaussian_filter(gray, sigma=rng.uniform(0.2, 0.7))
    return np.ascontiguousarray(input_channels(gray)), np.ascontiguousarray(target)


def tile_starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    return sorted(set([*range(0, length - tile + 1, stride), length - tile]))


def blend_window(side: int) -> np.ndarray:
    window = np.maximum(np.hanning(side), 0.05)
    return np.outer(window, window).astype(np.float32)[..., None]


def compact_labels(labels: np.ndarray, minimum_area: int) -> np.ndarray:
    labels = np.maximum(labels, 0).astype(np.int32)
    counts = np.bincount(labels.ravel())
    keep = counts >= minimum_area
    keep[0] = False
    mapping = np.zeros(len(counts), dtype=np.int32)
    mapping[keep] = np.arange(1, int(keep.sum()) + 1)
    return mapping[labels]


def instances(probabilities: np.ndarray, parameters: dict) -> np.ndarray:
    """Seed from connected predicted interiors; watershed only within each foreground component."""
    foreground = probabilities[..., 0] >= parameters["foreground"]
    regions, _ = ndimage.label(foreground, structure=np.ones((3, 3)))
    regions = compact_labels(regions, parameters["minimum_area"])
    output = np.zeros(foreground.shape, dtype=np.int32)
    next_id = 0
    for component, bounds in enumerate(ndimage.find_objects(regions), 1):
        if bounds is None:
            continue
        mask = regions[bounds] == component
        probs = probabilities[bounds]
        mode = parameters.get("mode", "legacy")
        confidence = parameters.get("confidence", parameters["foreground"])
        # A confident patch licenses the whole weakly connected object. Do not
        # turn multiple bright patches along one filament into separate instances.
        strong = mask & (probs[..., 0] >= confidence)
        if not strong.any():
            continue
        core = mask & (probs[..., 1] >= parameters["interior"]) & (probs[..., 2] < 0.5)
        seeds, _ = ndimage.label(core, structure=np.ones((3, 3)))
        seeds = compact_labels(seeds, parameters.get("minimum_seed_area", 4))
        if mode == "hysteresis" or seeds.max(initial=0) <= 1:
            # Includes seedless filaments: keep the whole connected candidate.
            labels = mask.astype(np.int32)
        else:
            # Boundary likelihood plus low foreground creates the watershed relief.
            cost = np.clip(0.65 * probs[..., 2] + 0.35 * (1 - probs[..., 0]), 0, 1)
            cost[~mask] = 1
            markers = seeds.astype(np.int32)
            markers[~mask] = -1
            labels = ndimage.watershed_ift(
                (cost * 254).astype(np.uint8), markers, structure=np.ones((3, 3))
            )
            labels[~mask] = 0
            labels = compact_labels(labels, parameters["minimum_area"])
        if mode != "legacy":
            # Reject low-confidence watershed fragments independently too.
            accepted = np.unique(labels[strong])
            labels[~np.isin(labels, accepted)] = 0
            labels = compact_labels(labels, parameters["minimum_area"])
        positive = labels > 0
        output[bounds][positive] = labels[positive] + next_id
        next_id += int(labels.max(initial=0))
    return output


def pq_totals(truth: list[np.ndarray], labels: np.ndarray) -> np.ndarray:
    """Same strict matching as sfsc.evaluation, using label histograms for speed."""
    ids, counts = np.unique(labels, return_counts=True)
    areas = dict(zip(ids.tolist(), counts.tolist(), strict=True))
    ids = ids[ids != 0]
    iou = np.zeros((len(truth), len(ids)), dtype=np.float64)
    for row, mask in enumerate(truth):
        overlapping, intersections = np.unique(labels[mask], return_counts=True)
        intersection = dict(zip(overlapping.tolist(), intersections.tolist(), strict=True))
        area = int(mask.sum())
        for col, label_id in enumerate(ids):
            common = intersection.get(int(label_id), 0)
            iou[row, col] = common / (area + areas[int(label_id)] - common)
    eligible = np.where(iou > 0.5, iou, 0)
    rows, cols = linear_sum_assignment(eligible, maximize=True)
    values = eligible[rows, cols]
    values = values[values > 0]
    return np.array(
        [values.sum(), len(values), len(ids) - len(values), len(truth) - len(values)],
        dtype=np.float64,
    )


def pq_score(totals: np.ndarray) -> float:
    denominator = totals[1] + 0.5 * (totals[2] + totals[3])
    return float(totals[0] / denominator) if denominator else 1.0


def stratified_validation(dataset, per_site: int) -> list[str]:
    by_site = {}
    for name in dataset.filenames("validation"):
        by_site.setdefault(dataset.by_filename[name]["site"], []).append(name)
    # Spread across dates within each site; deterministic, validation only.
    return sorted(
        {
            names[int(i)]
            for names in by_site.values()
            for i in np.linspace(0, len(names) - 1, min(per_site, len(names)))
        }
    )
