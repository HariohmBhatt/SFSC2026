from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from pycocotools import mask as coco_mask

from .data import crop_bounds, feature_stack, full_disk_features
from .ensemble import (
    Candidate,
    paint_instances,
    semantic_components,
    soft_mask_nms,
    tta_predict,
    weighted_mask_fusion,
)
from .models import FullDiskProposalNet, PromptFPN


def encode(mask: np.ndarray) -> dict[str, Any]:
    encoded = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": list(encoded["size"]), "counts": encoded["counts"].decode("ascii")}


def load_refiners(checkpoints: Sequence[Path]) -> list[PromptFPN]:
    models = []
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = PromptFPN(checkpoint["backbone"], pretrained=False).cuda()
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models.append(model)
    return models


def load_semantic(path: Path, backbone: str) -> FullDiskProposalNet:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = FullDiskProposalNet(backbone, pretrained=False).cuda()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.inference_mode()
def detector_proposals(
    detectors: Sequence,
    image: np.ndarray,
    scales: Sequence[int],
    confidence: float,
    nms_iou: float,
    candidates_per_view: int,
    maximum_candidates: int,
) -> list[Candidate]:
    transforms = (
        (image, lambda mask: mask),
        (np.fliplr(image).copy(), lambda mask: np.fliplr(mask).copy()),
        (np.flipud(image).copy(), lambda mask: np.flipud(mask).copy()),
    )
    candidates = []
    for detector_index, detector in enumerate(detectors):
        for size in scales:
            for transformed, invert in transforms:
                result = detector.predict(
                    transformed,
                    imgsz=int(size),
                    conf=confidence,
                    iou=nms_iou,
                    agnostic_nms=True,
                    retina_masks=True,
                    max_det=150,
                    device=0,
                    verbose=False,
                )[0]
                if result.masks is None:
                    continue
                scores = result.boxes.conf.detach().float().cpu().numpy()
                masks = result.masks.data.detach().cpu().numpy()
                order = np.argsort(scores)[::-1][:candidates_per_view]
                for candidate_index in order:
                    mask = masks[candidate_index]
                    score = scores[candidate_index]
                    if mask.shape != transformed.shape:
                        mask = cv2.resize(
                            mask, transformed.shape[::-1], interpolation=cv2.INTER_LINEAR
                        )
                    candidates.append(
                        Candidate(
                            invert(mask >= 0.5),
                            float(score),
                            f"yolo{detector_index}@{size}",
                        )
                    )
    candidates = sorted(candidates, key=lambda row: row.score, reverse=True)[:maximum_candidates]
    return soft_mask_nms(weighted_mask_fusion(candidates), iou_threshold=0.55)


@torch.inference_mode()
def semantic_proposals(
    model: FullDiskProposalNet,
    image: np.ndarray,
    size: int,
    threshold: float,
    minimum_area: int,
    amp: bool,
) -> list[Candidate]:
    features = torch.from_numpy(full_disk_features(image, size))[None].cuda()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        prediction = model(features)
    probability = prediction["mask"].sigmoid()[0, 0].float().cpu().numpy()
    boundary = prediction["boundary"].sigmoid()[0, 0].float().cpu().numpy()
    return semantic_components(
        probability, boundary, image.shape, threshold=threshold, minimum_area=minimum_area
    )


@torch.inference_mode()
def refine_proposals(
    models: Sequence[PromptFPN],
    image: np.ndarray,
    proposals: Sequence[Candidate],
    crop_size: int,
    contexts: Sequence[float],
    proposal_batch: int,
    amp: bool,
    dihedral_tta: bool = False,
) -> list[list[Candidate]]:
    results: list[list[Candidate]] = [[] for _ in proposals]
    proposal_batch = max(1, int(proposal_batch))
    for start in range(0, len(proposals), proposal_batch):
        rows = proposals[start : start + proposal_batch]
        crops = []
        for local_index, proposal in enumerate(rows):
            for context in contexts:
                x0, y0, x1, y1 = crop_bounds(proposal.mask, float(context))
                features = feature_stack(
                    image[y0:y1, x0:x1], proposal.mask[y0:y1, x0:x1], crop_size
                )
                crops.append((local_index, context, (x0, y0, x1, y1), features))
        tensor = torch.from_numpy(np.stack([row[3] for row in crops])).cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            prediction = tta_predict(models, tensor, dihedral=dihedral_tta)
        probabilities = prediction["mask"][:, 0].float().cpu().numpy()
        qualities = prediction["quality"].flatten().float().cpu().numpy()
        for (local_index, context, bounds, _), probability, quality in zip(
            crops, probabilities, qualities, strict=True
        ):
            proposal = rows[local_index]
            x0, y0, x1, y1 = bounds
            restored_probability = cv2.resize(
                probability, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR
            )
            restored = np.zeros(image.shape, dtype=np.float32)
            restored[y0:y1, x0:x1] = restored_probability
            mask = restored >= 0.5
            if not mask.any():
                continue
            confidence = float(restored[mask].mean())
            score = proposal.score * math.sqrt(max(float(quality), 1e-4)) * math.sqrt(
                max(confidence, 1e-4)
            )
            results[start + local_index].append(
                Candidate(mask, score, f"refined:{proposal.source}@{context:g}")
            )
    return results


@torch.inference_mode()
def predict_candidates(
    image_path: Path,
    detectors: Sequence,
    refiners: Sequence[PromptFPN],
    semantic: FullDiskProposalNet,
    config: dict[str, Any],
) -> tuple[list[Candidate], dict[str, Any]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    detector_rows = detector_proposals(
        detectors,
        image,
        config["detector_tta_scales"],
        float(config["detector_conf"]),
        float(config["detector_iou"]),
        int(config["detector_candidates_per_view"]),
        int(config["maximum_detector_candidates"]),
    )
    semantic_rows = semantic_proposals(
        semantic,
        image,
        int(config["semantic_size"]),
        float(config["semantic_mask_threshold"]),
        int(config["semantic_minimum_area"]),
        bool(config["inference_amp"]),
    )
    proposals = soft_mask_nms(
        weighted_mask_fusion(
            [*detector_rows, *semantic_rows], merge_iou=float(config["ensemble_merge_iou"])
        ),
        iou_threshold=float(config["mask_nms_iou"]),
    )
    proposals = sorted(proposals, key=lambda row: (row.score, row.agreement), reverse=True)[
        : int(config["maximum_refiner_proposals"])
    ]
    refined = []
    refined_rows = refine_proposals(
        refiners,
        image,
        proposals,
        int(config["refiner_crop"]),
        config["refiner_contexts"],
        int(config["refiner_proposal_batch"]),
        bool(config["inference_amp"]),
        bool(config["refiner_dihedral_tta"]),
    )
    for proposal, rows in zip(proposals, refined_rows, strict=True):
        refined.extend(
            rows
            or [
                Candidate(
                    proposal.mask,
                    proposal.score * 0.5,
                    f"unrefined:{proposal.source}",
                    proposal.agreement,
                )
            ]
        )
    fused = weighted_mask_fusion(refined, merge_iou=float(config["ensemble_merge_iou"]))
    return fused, {
        "detector_proposals": len(detector_rows),
        "semantic_proposals": len(semantic_rows),
        "fused_proposals": len(proposals),
        "refined_candidates": len(refined),
    }


@torch.inference_mode()
def predict_image(
    image_path: Path,
    detectors: Sequence,
    refiners: Sequence[PromptFPN],
    semantic: FullDiskProposalNet,
    config: dict[str, Any],
    score_threshold: float,
    minimum_area: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fused, diagnostics = predict_candidates(image_path, detectors, refiners, semantic, config)
    instances = paint_instances(fused, score_threshold, minimum_area)
    return [encode(row.mask) for row in instances], {**diagnostics, "instances": len(instances)}


def save_predictions(path: Path, predictions: dict[str, list[dict]], diagnostics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"predictions": predictions, "diagnostics": diagnostics}, indent=2) + "\n"
    )
