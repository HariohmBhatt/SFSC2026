"""Train and assess the first native-reconstruction pilot on a Kaggle P100."""

from __future__ import annotations

import hashlib
import json
import math
import time
import traceback
from collections import deque
from functools import lru_cache
from pathlib import Path

import core
import cv2
import legacy_runtime as base
import metric_core
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

CONFIG = json.loads(Path("/kaggle/working/task-config.json").read_text())
WORK = Path("/kaggle/working/native-pilot")
WORK.mkdir(parents=True, exist_ok=True)
STATUS = {"status": "starting", "run_id": CONFIG["run_id"]}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def progress(event: str, **values) -> None:
    STATUS.update(event=event, **values)
    write_json(WORK / "progress.json", STATUS)
    print(json.dumps({"event": event, **values}), flush=True)


def source_file(filename: str, expected_hash: str | None = None) -> Path:
    matches = list(Path("/kaggle/input").glob(f"**/{filename}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {filename}, found {matches}")
    if expected_hash is not None and digest(matches[0]) != expected_hash:
        raise ValueError(f"Hash mismatch for {filename}")
    return matches[0]


def decode(value) -> np.ndarray:
    return metric_core.decode(value)


def encode(mask: np.ndarray) -> dict:
    return metric_core.encode(mask)


def truth_sets(filename: str, indices) -> list[list[np.ndarray]]:
    return base.ground_truth_sets_for_file(filename, indices)


def selected_truths(filename: str, indices, medoids) -> list[np.ndarray]:
    image_id = medoids[filename]
    info = indices["images_by_id"][image_id]
    return [
        base.annotation_to_mask(annotation, int(info["height"]), int(info["width"]))
        for annotation in indices["annotations_by_image"].get(image_id, [])
        if not annotation.get("iscrowd", 0)
    ]


def load_oof_candidates() -> dict[str, list[dict]]:
    combined = {}
    for fold, expected in enumerate(CONFIG["oof_candidate_sha256"]):
        path = source_file(f"fold-{fold}-candidates.json", expected)
        payload = json.loads(path.read_text())
        if set(payload) & set(combined):
            raise ValueError("OOF candidate images overlap")
        combined.update(payload)
    if set(combined) != set(CONFIG["train_names"]):
        raise ValueError("OOF candidates do not cover the frozen training membership")
    return combined


def build_samples(candidates, indices) -> tuple[list[dict], dict]:
    medoids = base.choose_medoid_annotators(indices)
    positives, negatives, ambiguous = [], [], 0
    for filename in sorted(candidates):
        selected = selected_truths(filename, indices, medoids)
        all_records = truth_sets(filename, indices)
        groups = {}
        for candidate in candidates[filename]:
            groups.setdefault(int(candidate["proposal_id"]), {})[candidate["kind"]] = candidate
        for proposal_id, alternatives in sorted(groups.items()):
            if set(alternatives) != {"seed", "refined"}:
                ambiguous += 1
                continue
            seed_row, global_row = alternatives["seed"], alternatives["refined"]
            seed = decode(seed_row["rle"])
            association = core.associate_prompt(seed, selected, all_records)
            sample = {
                "filename": filename,
                "proposal_id": proposal_id,
                "seed": seed_row["rle"],
                "coarse_mask": global_row["rle"],
                "detector_score": float(seed_row["detector_score"]),
            }
            if association["status"] == "positive":
                sample["target"] = encode(selected[int(association["target_index"])])
                sample["status"] = "positive"
                positives.append(sample)
            elif association["status"] == "negative":
                sample["target"] = encode(np.zeros(seed.shape, bool))
                sample["status"] = "negative"
                negatives.append(sample)
            else:
                ambiguous += 1
    negatives.sort(
        key=lambda sample: hashlib.sha256(
            f"{sample['filename']}:{sample['proposal_id']}".encode()
        ).hexdigest()
    )
    negatives = negatives[: max(1, len(positives) // 3)]
    samples = sorted(positives + negatives, key=lambda row: (row["filename"], row["proposal_id"]))
    report = {
        "positive": len(positives),
        "hard_negative": len(negatives),
        "ambiguous_or_incomplete": ambiguous,
        "total": len(samples),
        "images": len({sample["filename"] for sample in samples}),
    }
    return samples, report


def resized_global_inputs(image, seed, bounds, size=384) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    image_crop = cv2.resize(image[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_LINEAR)
    seed_crop = cv2.resize(
        seed[y0:y1, x0:x1].astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image_crop)
    return np.stack([image_crop, clahe, seed_crop * 255]).astype(np.float32) / 255.0


def local_input(image_roi, seed_roi, coarse_roi, x0, y0, window=512):
    height, width = image_roi.shape
    actual_height, actual_width = min(window, height - y0), min(window, width - x0)
    image_tile = image_roi[y0 : y0 + actual_height, x0 : x0 + actual_width]
    seed_tile = seed_roi[y0 : y0 + actual_height, x0 : x0 + actual_width]
    coarse_tile = coarse_roi[y0 : y0 + actual_height, x0 : x0 + actual_width]
    image_tile, valid = core.pad_to_window(image_tile, window, "reflect")
    seed_tile, _ = core.pad_to_window(seed_tile.astype(np.float32), window)
    coarse_tile, _ = core.pad_to_window(coarse_tile.astype(np.float32), window)
    xs = np.broadcast_to(
        np.linspace(-1, 1, max(width, 2), dtype=np.float32)[None, :], (height, width)
    )[y0 : y0 + actual_height, x0 : x0 + actual_width]
    ys = np.broadcast_to(
        np.linspace(-1, 1, max(height, 2), dtype=np.float32)[:, None], (height, width)
    )[y0 : y0 + actual_height, x0 : x0 + actual_width]
    xs, _ = core.pad_to_window(xs, window)
    ys, _ = core.pad_to_window(ys, window)
    features = np.stack(
        [image_tile / 255.0, seed_tile, coarse_tile, xs, ys, valid.astype(np.float32)]
    ).astype(np.float32)
    return features, valid, actual_width, actual_height


@lru_cache(maxsize=48)
def cached_decode(counts: str) -> np.ndarray:
    return decode({"size": [2048, 2048], "counts": counts})


class NativeDataset(Dataset):
    def __init__(self, samples, image_dir: Path, seed: int):
        self.samples, self.image_dir, self.seed = samples, image_dir, seed
        self.epoch = 0
        self.cached_filename = None
        self.cached_image = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self):
        return len(self.samples)

    def image(self, filename):
        if self.cached_filename != filename:
            self.cached_filename = filename
            self.cached_image = cv2.imread(str(self.image_dir / filename), cv2.IMREAD_GRAYSCALE)
            if self.cached_image is None:
                raise FileNotFoundError(filename)
        return self.cached_image

    def __getitem__(self, index):
        sample = self.samples[index]
        image = self.image(sample["filename"])
        seed = cached_decode(sample["seed"]["counts"])
        coarse_mask = cached_decode(sample["coarse_mask"]["counts"])
        target = cached_decode(sample["target"]["counts"])
        bounds = core.conditioning_bounds(seed, coarse_mask)
        bx0, by0, bx1, by1 = bounds
        image_roi = image[by0:by1, bx0:bx1]
        seed_roi = seed[by0:by1, bx0:bx1]
        coarse_roi = coarse_mask[by0:by1, bx0:bx1]
        target_roi = target[by0:by1, bx0:bx1]
        starts = [
            (x, y)
            for y in core.axis_starts(image_roi.shape[0])
            for x in core.axis_starts(image_roi.shape[1])
        ]
        foreground = [(x, y) for x, y in starts if target_roi[y : y + 512, x : x + 512].any()]
        rng = np.random.default_rng(self.seed + self.epoch * len(self.samples) + index)
        choices = foreground if foreground and rng.random() < 0.7 else starts
        x0, y0 = choices[int(rng.integers(0, len(choices)))]
        local, valid, actual_width, actual_height = local_input(
            image_roi, seed_roi, coarse_roi, x0, y0
        )
        target_tile = target_roi[y0 : y0 + actual_height, x0 : x0 + actual_width]
        target_tile, _ = core.pad_to_window(target_tile.astype(np.float32), 512)
        global_tensor = resized_global_inputs(image, seed, bounds)
        if rng.random() < 0.5:
            local = local[:, :, ::-1].copy()
            local[3] *= -1
            target_tile = target_tile[:, ::-1].copy()
            valid = valid[:, ::-1].copy()
            global_tensor = global_tensor[:, :, ::-1].copy()
        if rng.random() < 0.5:
            local = local[:, ::-1, :].copy()
            local[4] *= -1
            target_tile = target_tile[::-1, :].copy()
            valid = valid[::-1, :].copy()
            global_tensor = global_tensor[:, ::-1, :].copy()
        gain = float(rng.uniform(0.9, 1.1))
        local[0] = np.clip(local[0] * gain, 0, 1)
        global_tensor[:2] = np.clip(global_tensor[:2] * gain, 0, 1)
        return (
            torch.from_numpy(local),
            torch.from_numpy(global_tensor),
            torch.from_numpy(target_tile[None]),
            torch.from_numpy(valid[None]),
        )


class GroupNormBlock(nn.Sequential):
    def __init__(self, input_channels, output_channels):
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, output_channels),
            nn.ReLU(inplace=True),
        )


class NativeRefiner(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.local_encoder = resnet18(weights=weights)
        pretrained_conv = self.local_encoder.conv1
        self.local_encoder.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.local_encoder.conv1.weight.zero_()
                self.local_encoder.conv1.weight[:, 0] = pretrained_conv.weight.mean(dim=1)
        self.global_encoder = resnet18(weights=weights)
        if pretrained:
            with torch.no_grad():
                self.global_encoder.conv1.weight[:, 2].zero_()
        for parameter in self.global_encoder.parameters():
            parameter.requires_grad = False
        self.context = nn.Linear(512, 64)
        self.fuse = nn.Sequential(
            nn.Conv2d(576, 512, 1, bias=False), nn.GroupNorm(8, 512), nn.ReLU(inplace=True)
        )
        self.up4, self.dec4 = nn.ConvTranspose2d(512, 256, 2, 2), GroupNormBlock(512, 256)
        self.up3, self.dec3 = nn.ConvTranspose2d(256, 128, 2, 2), GroupNormBlock(256, 128)
        self.up2, self.dec2 = nn.ConvTranspose2d(128, 64, 2, 2), GroupNormBlock(128, 64)
        self.up1, self.dec1 = nn.ConvTranspose2d(64, 32, 2, 2), GroupNormBlock(96, 32)
        self.up0, self.head = nn.ConvTranspose2d(32, 16, 2, 2), nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("local_mean", torch.tensor([0.485, 0, 0, 0, 0, 0]).view(1, 6, 1, 1))
        self.register_buffer("local_std", torch.tensor([0.229, 1, 1, 1, 1, 1]).view(1, 6, 1, 1))
        self.register_buffer("global_mean", torch.tensor([0.485, 0.456, 0.0]).view(1, 3, 1, 1))
        self.register_buffer("global_std", torch.tensor([0.229, 0.224, 1.0]).view(1, 3, 1, 1))

    @staticmethod
    def features(encoder, tensor):
        x0 = encoder.relu(encoder.bn1(encoder.conv1(tensor)))
        x1 = encoder.layer1(encoder.maxpool(x0))
        x2 = encoder.layer2(x1)
        x3 = encoder.layer3(x2)
        x4 = encoder.layer4(x3)
        return x0, x1, x2, x3, x4

    def train(self, mode=True):
        super().train(mode)
        self.global_encoder.eval()
        for module in self.local_encoder.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, local, global_tensor):
        prompt_prior = torch.maximum(local[:, 1:2], local[:, 2:3]).clamp(0.05, 0.95)
        prompt_logit = torch.logit(prompt_prior)
        local = (local - self.local_mean) / self.local_std
        global_tensor = (global_tensor - self.global_mean) / self.global_std
        x0, x1, x2, x3, x4 = self.features(self.local_encoder, local)
        with torch.no_grad():
            global_feature = self.features(self.global_encoder, global_tensor)[-1]
            global_feature = F.adaptive_avg_pool2d(global_feature, 1).flatten(1)
        context = self.context(global_feature.detach())[:, :, None, None]
        context = context.expand(-1, -1, x4.shape[-2], x4.shape[-1])
        x = self.fuse(torch.cat([x4, context], dim=1))
        x = self.dec4(torch.cat([self.up4(x), x3], dim=1))
        x = self.dec3(torch.cat([self.up3(x), x2], dim=1))
        x = self.dec2(torch.cat([self.up2(x), x1], dim=1))
        x = self.dec1(torch.cat([self.up1(x), x0], dim=1))
        return self.head(self.up0(x)) + prompt_logit


def masked_loss(logits, targets, valid):
    logits = logits.float()
    targets = targets.float()
    valid = valid.float()
    bce = (F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * valid).sum()
    bce = bce / valid.sum().clamp_min(1)
    probabilities = logits.sigmoid() * valid
    targets = targets * valid
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (1 - (2 * intersection + 1) / (denominator + 1)).mean()
    return 0.45 * bce + 0.55 * dice


def train_model(samples, image_dir: Path, device):
    dataset = NativeDataset(samples, image_dir, CONFIG["seed"])
    loader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(CONFIG["seed"]),
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    model = NativeRefiner(pretrained=True).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"]
    )
    updates, total_updates = 0, CONFIG["optimizer_updates"]
    accumulation = CONFIG["accumulation"]

    def multiplier(step):
        if step < CONFIG["warmup_updates"]:
            return 0.05 + 0.95 * step / max(CONFIG["warmup_updates"], 1)
        ratio = (step - CONFIG["warmup_updates"]) / max(total_updates - CONFIG["warmup_updates"], 1)
        return 0.05 + 0.475 * (1 + math.cos(math.pi * min(ratio, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    amp_enabled = bool(CONFIG["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    recent_losses = deque(maxlen=100)
    all_losses = []
    micro_steps, epoch = 0, 0
    while updates < total_updates:
        dataset.set_epoch(epoch)
        model.train()
        for local, global_tensor, targets, valid in loader:
            local = local.to(device, non_blocking=True)
            global_tensor = global_tensor.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss = masked_loss(model(local, global_tensor), targets, valid)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss before backward at update {updates}, micro-step {micro_steps}"
                )
            scaler.scale(loss / accumulation).backward()
            loss_value = float(loss.detach())
            recent_losses.append(loss_value)
            all_losses.append(loss_value)
            micro_steps += 1
            if micro_steps % accumulation:
                continue
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters, CONFIG["gradient_clip"], error_if_nonfinite=True
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            updates += 1
            if updates % 500 == 0:
                nonfinite = [
                    name
                    for name, parameter in model.named_parameters()
                    if not torch.isfinite(parameter).all()
                ]
                if nonfinite:
                    raise FloatingPointError(
                        f"Non-finite parameters after update {updates}: {nonfinite[:5]}"
                    )
                torch.save(model.state_dict(), WORK / "last-finite-native-refiner.pt")
            if updates % 100 == 0 or updates == total_updates:
                elapsed = time.monotonic() - started
                progress(
                    "native_training",
                    optimizer_update=updates,
                    optimizer_updates=total_updates,
                    rolling_loss=float(np.mean(recent_losses)),
                    last_loss=loss_value,
                    gradient_norm=float(gradient_norm),
                    amp_scale=float(scaler.get_scale()),
                    seconds_per_update=elapsed / updates,
                    peak_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
                    peak_reserved_gb=torch.cuda.max_memory_reserved() / 2**30,
                )
            if updates >= total_updates:
                break
        epoch += 1
    checkpoint = WORK / "native-refiner.pt"
    nonfinite_parameters = [
        name for name, parameter in model.named_parameters() if not torch.isfinite(parameter).all()
    ]
    if nonfinite_parameters:
        raise FloatingPointError(f"Non-finite model parameters: {nonfinite_parameters[:5]}")
    torch.save(model.state_dict(), checkpoint)
    return (
        model.eval(),
        checkpoint,
        {
            "optimizer_updates": updates,
            "micro_steps": micro_steps,
            "epochs_started": epoch,
            "amp": amp_enabled,
            "nonfinite_batches": 0,
            "last_loss": all_losses[-1],
            "minimum_loss": min(all_losses),
            "maximum_loss": max(all_losses),
            "seconds": time.monotonic() - started,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
        },
    )


@torch.inference_mode()
def native_refine(image, seed, coarse_mask, model, device):
    bounds = core.conditioning_bounds(seed, coarse_mask)
    bx0, by0, bx1, by1 = bounds
    image_roi = image[by0:by1, bx0:bx1]
    seed_roi = seed[by0:by1, bx0:bx1]
    coarse_roi = coarse_mask[by0:by1, bx0:bx1]
    global_tensor = torch.from_numpy(resized_global_inputs(image, seed, bounds))
    prepared, positions = [], []
    for y0 in core.axis_starts(image_roi.shape[0]):
        for x0 in core.axis_starts(image_roi.shape[1]):
            local, _, _, _ = local_input(image_roi, seed_roi, coarse_roi, x0, y0)
            prepared.append(torch.from_numpy(local))
            positions.append((x0, y0))
    probabilities = []
    for start in range(0, len(prepared), CONFIG["inference_batch"]):
        local_batch = torch.stack(prepared[start : start + CONFIG["inference_batch"]]).to(device)
        global_batch = global_tensor.expand(len(local_batch), -1, -1, -1).to(device)
        with torch.amp.autocast("cuda", enabled=bool(CONFIG["amp"])):
            output = model(local_batch, global_batch).sigmoid().float().cpu().numpy()[:, 0]
        if not np.isfinite(output).all():
            raise FloatingPointError("Non-finite probabilities during native inference")
        probabilities.extend(output)
    blended = core.blend_tiles(
        image_roi.shape,
        [(probability, x, y) for probability, (x, y) in zip(probabilities, positions, strict=True)],
    )
    return blended, bounds, len(prepared)


def full_mask(probability, bounds, shape, threshold):
    x0, y0, x1, y1 = bounds
    result = np.zeros(shape, bool)
    result[y0:y1, x0:x1] = probability >= threshold
    return result


def records_as_rle(filename, indices):
    result = []
    for image_id in indices["image_ids_by_file"][filename]:
        info = indices["images_by_id"][image_id]
        result.append(
            [
                base.annotation_to_rle(annotation, int(info["height"]), int(info["width"]))
                for annotation in indices["annotations_by_image"].get(image_id, [])
                if not annotation.get("iscrowd", 0)
            ]
        )
    return result


def candidate_score(candidate):
    detector_score = float(candidate["detector_score"])
    if candidate.get("long_object_path", False):
        return detector_score
    return detector_score * math.sqrt(max(float(candidate["refiner_score"]), 0.0))


def assemble(candidates, kind):
    masks, scores = [], []
    for candidate in candidates:
        score = candidate_score(candidate)
        encoded = candidate[kind]
        area = int(candidate[f"{kind}_area"])
        if score < CONFIG["score_threshold"] or area < CONFIG["minimum_area"]:
            continue
        mask = decode(encoded)
        if mask.sum() >= CONFIG["minimum_area"]:
            masks.append(mask)
            scores.append(score)
    return base.resolve_overlaps(masks, scores, CONFIG["minimum_area"])[0]


def evaluate(cache, names, indices, kind):
    per_image = {}
    for name in names:
        masks = assemble(cache[name], kind)
        per_image[name] = metric_core.score(records_as_rle(name, indices), masks).tolist()
    return metric_core.metrics(np.sum(list(per_image.values()), axis=0)), per_image


def comparison_strata(native_per_image, baseline_per_image, names):
    groups = {}
    for name in names:
        year = int(name[:4])
        period = "2011-2014" if year <= 2014 else "2015-2018" if year <= 2018 else "2019-2022"
        groups.setdefault(f"site:{Path(name).stem[-2:]}", []).append(name)
        groups.setdefault(f"period:{period}", []).append(name)
    result = {}
    for group, members in sorted(groups.items()):
        baseline = metric_core.metrics(
            np.sum([baseline_per_image[name] for name in members], axis=0)
        )
        native = metric_core.metrics(np.sum([native_per_image[name] for name in members], axis=0))
        result[group] = {
            "images": len(members),
            "baseline": baseline,
            "native": native,
            "delta": native["pq"] - baseline["pq"],
        }
    return result


def assess(model, paths, indices, device):
    validation_path = source_file(
        "validation-candidates.json", CONFIG["validation_candidates_sha256"]
    )
    cache = json.loads(validation_path.read_text())
    names = CONFIG["validation_names"]
    if set(cache) != set(names):
        raise ValueError("Validation cache membership mismatch")
    baseline, baseline_per_image = evaluate(cache, names, indices, "refined")
    if abs(baseline["pq"] - CONFIG["expected_control_pq"]) > 1e-10:
        raise ValueError("Failed to reproduce the frozen control PQ")
    thresholds = [float(value) for value in CONFIG["mask_threshold_grid"]]
    total_windows = 0
    probability_summaries = []
    started = time.monotonic()
    for number, name in enumerate(names, 1):
        image = cv2.imread(str(paths["train"] / name), cv2.IMREAD_GRAYSCALE)
        for candidate in cache[name]:
            seed = decode(candidate["seed"])
            coarse_mask = decode(candidate["refined"])
            probability, bounds, windows = native_refine(image, seed, coarse_mask, model, device)
            probability_summaries.append(
                np.quantile(probability, [0.0, 0.5, 0.9, 0.99, 1.0]).tolist()
            )
            for threshold in thresholds:
                key = f"native_{threshold:.2f}".replace(".", "_")
                native_mask = full_mask(probability, bounds, seed.shape, threshold)
                candidate[key] = encode(native_mask)
                candidate[f"{key}_area"] = int(native_mask.sum())
            total_windows += windows
        if number % 10 == 0:
            progress(
                "native_inference",
                images_processed=number,
                images_total=len(names),
                windows=total_windows,
            )
    calibration = set(CONFIG["calibration_names"])
    other = [name for name in names if name not in calibration]
    calibration_grid = []
    for threshold in thresholds:
        key = f"native_{threshold:.2f}".replace(".", "_")
        metrics, _ = evaluate(cache, sorted(calibration), indices, key)
        calibration_grid.append({"threshold": threshold, **metrics})
    selected_threshold = core.choose_threshold(
        [(row["threshold"], row["pq"]) for row in calibration_grid],
        CONFIG["mask_threshold_default"],
    )
    selected_key = f"native_{selected_threshold:.2f}".replace(".", "_")
    for candidates in cache.values():
        for candidate in candidates:
            candidate["native"] = candidate[selected_key]
            candidate["native_area"] = candidate[f"{selected_key}_area"]
            for threshold in thresholds:
                key = f"native_{threshold:.2f}".replace(".", "_")
                candidate.pop(key)
                candidate.pop(f"{key}_area")
    native, native_per_image = evaluate(cache, names, indices, "native")
    baseline_other = metric_core.metrics(
        np.sum([baseline_per_image[name] for name in other], axis=0)
    )
    native_other = metric_core.metrics(np.sum([native_per_image[name] for name in other], axis=0))
    bootstrap = metric_core.bootstrap_delta(native_per_image, baseline_per_image, other, 2000)
    nonempty_candidates = sum(
        int(candidate["native_area"] > 0)
        for candidates in cache.values()
        for candidate in candidates
    )
    total_candidates = sum(len(candidates) for candidates in cache.values())
    if nonempty_candidates == 0:
        raise RuntimeError("Native reconstruction collapsed to all-empty masks")
    gate = (
        native["pq"] - baseline["pq"] >= 0.015
        and native_other["pq"] - baseline_other["pq"] >= 0.01
        and bootstrap["ci95"][0] > 0
        and abs(native["pq"] - native["one_to_one_pq"]) <= 1e-8
    )
    cache_path = WORK / "native-validation-candidates.json"
    write_json(cache_path, cache)
    return {
        "baseline": baseline,
        "native": native,
        "delta": native["pq"] - baseline["pq"],
        "baseline82": baseline_other,
        "native82": native_other,
        "delta82": native_other["pq"] - baseline_other["pq"],
        "paired_block_bootstrap82": bootstrap,
        "diagnostic_gate": gate,
        "evidence_scope": "historical-validation geometry diagnostic",
        "selected_mask_threshold": selected_threshold,
        "calibration_grid": calibration_grid,
        "nonempty_candidates": nonempty_candidates,
        "total_candidates": total_candidates,
        "probability_quantiles": np.quantile(
            np.asarray(probability_summaries), [0.0, 0.5, 1.0], axis=0
        ).tolist(),
        "strata": comparison_strata(native_per_image, baseline_per_image, other),
        "inference_seconds": time.monotonic() - started,
        "windows": total_windows,
        "candidate_cache_sha256": digest(cache_path),
    }


def main():
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (6, 0):
            raise RuntimeError("This pilot requires a Pascal P100 GPU")
        smoke = torch.ones((16, 16), device="cuda", requires_grad=True)
        (smoke @ smoke).sum().backward()
        progress("gpu_verified", gpu=torch.cuda.get_device_name(), torch=torch.__version__)
        base.seed_everything(CONFIG["seed"])
        paths = base.discover_competition_paths()
        coco = base.load_coco(paths["json"])
        allowed = set(CONFIG["train_names"]) | set(CONFIG["validation_names"])
        indices = base.build_coco_indices(base.restrict_coco(coco, allowed))
        candidates = load_oof_candidates()
        samples, sample_report = build_samples(candidates, indices)
        if len(samples) < 1000 or sample_report["images"] < 450:
            raise ValueError(f"Insufficient honest prompt supervision: {sample_report}")
        manifest_path = WORK / "training-samples.json"
        write_json(manifest_path, samples)
        progress("samples_ready", **sample_report)
        device = torch.device("cuda")
        model, checkpoint, training = train_model(samples, paths["train"], device)
        assessment = assess(model, paths, indices, device)
        report = {
            "status": "complete",
            "run_id": CONFIG["run_id"],
            "source_sha256": CONFIG["source_sha256"],
            "membership_sha256": CONFIG["membership_sha256"],
            "sample_manifest_sha256": digest(manifest_path),
            "model_sha256": digest(checkpoint),
            "sample_report": sample_report,
            "training": training,
            "assessment": assessment,
            "config": CONFIG,
        }
        write_json(WORK / "pilot-report.json", report)
        STATUS.update(status="complete", result=report)
        progress("complete")
    except Exception as error:
        STATUS.update(status="failed", error=str(error), traceback=traceback.format_exc())
        progress("failed")
        raise


if __name__ == "__main__":
    main()
