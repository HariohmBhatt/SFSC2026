from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Candidate:
    mask: np.ndarray
    score: float
    source: str
    agreement: float = 1.0


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    return float(intersection / union) if union else 0.0


def weighted_mask_fusion(
    candidates: Sequence[Candidate], merge_iou: float = 0.72, threshold: float = 0.5
) -> list[Candidate]:
    """Fuse only strongly agreeing masks; preserve distinct touching filaments."""
    groups: list[list[Candidate]] = []
    for candidate in sorted(candidates, key=lambda row: row.score, reverse=True):
        best_index, best_iou = -1, 0.0
        for index, group in enumerate(groups):
            value = max(mask_iou(candidate.mask, member.mask) for member in group)
            if value > best_iou:
                best_index, best_iou = index, value
        if best_iou >= merge_iou:
            groups[best_index].append(candidate)
        else:
            groups.append([candidate])
    fused = []
    for group in groups:
        weights = np.asarray([max(row.score, 1e-4) for row in group], dtype=np.float32)
        probability = sum(
            row.mask.astype(np.float32) * weight for row, weight in zip(group, weights, strict=True)
        )
        probability /= weights.sum()
        mask = probability >= threshold
        if not mask.any():
            mask = max(group, key=lambda row: row.score).mask.copy()
        sources = sorted({row.source for row in group})
        fused.append(
            Candidate(
                mask=mask,
                score=float(np.average([row.score for row in group], weights=weights)),
                source="+".join(sources),
                agreement=len(sources),
            )
        )
    return fused


def soft_mask_nms(candidates: Sequence[Candidate], iou_threshold: float = 0.55) -> list[Candidate]:
    remaining = [
        Candidate(row.mask.copy(), row.score, row.source, row.agreement) for row in candidates
    ]
    selected = []
    while remaining:
        remaining.sort(key=lambda row: row.score, reverse=True)
        best = remaining.pop(0)
        selected.append(best)
        updated = []
        for candidate in remaining:
            overlap = mask_iou(best.mask, candidate.mask)
            if overlap >= 0.9:
                continue
            candidate.score *= float(np.exp(-(overlap * overlap) / 0.5))
            if overlap < iou_threshold or candidate.score >= 0.05:
                updated.append(candidate)
        remaining = updated
    return selected


def semantic_components(
    probability: np.ndarray,
    boundary: np.ndarray,
    native_shape: tuple[int, int],
    threshold: float = 0.45,
    minimum_area: int = 32,
) -> list[Candidate]:
    foreground = probability >= threshold
    foreground[boundary >= 0.55] = False
    foreground = cv2.morphologyEx(
        foreground.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    count, labels = cv2.connectedComponents(foreground, connectivity=8)
    candidates = []
    for label in range(1, count):
        mask = labels == label
        if mask.sum() < minimum_area:
            continue
        score = float(probability[mask].mean())
        native = cv2.resize(
            mask.astype(np.uint8), native_shape[::-1], interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        candidates.append(Candidate(native, score, "semantic"))
    return candidates


def dihedral_transforms(tensor: torch.Tensor) -> Iterable[tuple[torch.Tensor, callable]]:
    for rotation in range(4):
        rotated = torch.rot90(tensor, rotation, dims=(-2, -1))
        yield rotated, lambda value, k=rotation: torch.rot90(value, -k, dims=(-2, -1))
        flipped = torch.flip(rotated, dims=(-1,))
        yield (
            flipped,
            lambda value, k=rotation: torch.rot90(torch.flip(value, dims=(-1,)), -k, dims=(-2, -1)),
        )


@torch.inference_mode()
def tta_predict(
    models: Sequence[torch.nn.Module], tensor: torch.Tensor, dihedral: bool = False
) -> dict[str, torch.Tensor]:
    mask_sum = torch.zeros((tensor.shape[0], 1, *tensor.shape[-2:]), device=tensor.device)
    boundary_sum = torch.zeros_like(mask_sum)
    quality_sum = torch.zeros((tensor.shape[0], 1), device=tensor.device)
    count = 0
    transforms = dihedral_transforms(tensor) if dihedral else ((tensor, lambda value: value),)
    for transformed, invert in transforms:
        for model in models:
            output = model(transformed)
            mask_sum += invert(output["mask"].sigmoid())
            boundary_sum += invert(output["boundary"].sigmoid())
            quality_sum += output["quality"]
            count += 1
    return {
        "mask": mask_sum / count,
        "boundary": boundary_sum / count,
        "quality": quality_sum / count,
    }


def paint_instances(
    candidates: Sequence[Candidate], score_threshold: float, minimum_area: int
) -> list[Candidate]:
    accepted: list[Candidate] = []
    occupied = None
    for candidate in sorted(candidates, key=lambda row: (row.score, row.agreement), reverse=True):
        if candidate.score < score_threshold or candidate.mask.sum() < minimum_area:
            continue
        if occupied is None:
            occupied = np.zeros_like(candidate.mask, dtype=bool)
        mask = candidate.mask & ~occupied
        if mask.sum() >= minimum_area:
            accepted.append(Candidate(mask, candidate.score, candidate.source, candidate.agreement))
            occupied |= mask
    return accepted


def resize_probability(probability: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    return (
        F.interpolate(probability, size=size, mode="bilinear", align_corners=False)[0, 0]
        .float()
        .cpu()
        .numpy()
    )
