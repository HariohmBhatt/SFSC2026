from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).sum((1, 2, 3))
    denominator = probabilities.sum((1, 2, 3)) + targets.sum((1, 2, 3))
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = logits.sigmoid()
    pt = probability * targets + (1 - probability) * (1 - targets)
    return ((1 - pt).pow(gamma) * bce).mean()


def soft_erode(image: torch.Tensor) -> torch.Tensor:
    vertical = -F.max_pool2d(-image, (3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-image, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def soft_dilate(image: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(image, 3, stride=1, padding=1)


def soft_skeleton(image: torch.Tensor, iterations: int = 12) -> torch.Tensor:
    opened = soft_dilate(soft_erode(image))
    skeleton = F.relu(image - opened)
    for _ in range(iterations):
        image = soft_erode(image)
        opened = soft_dilate(soft_erode(image))
        skeleton = skeleton + F.relu(image - opened) * (1 - skeleton)
    return skeleton


def cldice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if max(logits.shape[-2:]) > 512:
        logits = F.interpolate(logits, size=(512, 512), mode="bilinear", align_corners=False)
        targets = F.interpolate(targets, size=(512, 512), mode="nearest")
    probabilities = logits.sigmoid()
    prediction_skeleton = soft_skeleton(probabilities)
    target_skeleton = soft_skeleton(targets)
    precision = (prediction_skeleton * targets).sum((1, 2, 3)) / (
        prediction_skeleton.sum((1, 2, 3)) + 1
    )
    sensitivity = (target_skeleton * probabilities).sum((1, 2, 3)) / (
        target_skeleton.sum((1, 2, 3)) + 1
    )
    return (1 - (2 * precision * sensitivity + 1e-6) / (precision + sensitivity + 1e-6)).mean()


def boundary_target(targets: torch.Tensor) -> torch.Tensor:
    return (soft_dilate(targets) - soft_erode(targets)).clamp(0, 1)


def multitask_loss(
    mask_logits: torch.Tensor,
    boundary_logits: torch.Tensor,
    quality: torch.Tensor,
    targets: torch.Tensor,
    boundary_targets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    dice = soft_dice_loss(mask_logits, targets)
    focal = focal_loss(mask_logits, targets)
    topology = cldice_loss(mask_logits, targets)
    boundary_targets = boundary_target(targets) if boundary_targets is None else boundary_targets
    boundary = 0.5 * soft_dice_loss(boundary_logits, boundary_targets) + 0.5 * focal_loss(
        boundary_logits, boundary_targets
    )
    with torch.no_grad():
        predicted = mask_logits.sigmoid() > 0.5
        intersection = (predicted & targets.bool()).sum((1, 2, 3)).float()
        union = (predicted | targets.bool()).sum((1, 2, 3)).float()
        quality_target = torch.where(union > 0, intersection / union, torch.ones_like(union))
    quality_loss = F.smooth_l1_loss(quality.flatten(), quality_target)
    parts = {
        "dice": dice,
        "focal": focal,
        "topology": topology,
        "boundary": boundary,
        "quality": quality_loss,
    }
    total = 0.35 * dice + 0.30 * focal + 0.15 * topology + 0.12 * boundary + 0.08 * quality_loss
    return total, parts
