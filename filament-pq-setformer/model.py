"""Kaggle-only MaskDINO E1 model adapter with a minimal train/predict interface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.structures import ImageList
from maskdino import add_maskdino_config
from torch import nn
from torch.nn import functional as F

DATASET_NAME = "sfsc_pq_setformer"


def _register_metadata() -> None:
    if DATASET_NAME not in DatasetCatalog.list():
        DatasetCatalog.register(DATASET_NAME, lambda: [])
    MetadataCatalog.get(DATASET_NAME).set(
        thing_classes=["filament"],
        thing_dataset_id_to_contiguous_id={1: 0},
    )


def build_maskdino_config(experiment: Any):
    """Translate the typed experiment config into one explicit MaskDINO config."""
    _register_metadata()
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)

    cfg.DATASETS.TRAIN = (DATASET_NAME,)
    cfg.DATASETS.TEST = ()
    cfg.MODEL.META_ARCHITECTURE = "MaskDINO"
    cfg.INPUT.FORMAT = "L"
    cfg.INPUT.DATASET_MAPPER_NAME = "detr_sfsc"
    cfg.INPUT.IMAGE_SIZE = experiment.model.input_size
    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.PIXEL_MEAN = [127.5]
    cfg.MODEL.PIXEL_STD = [127.5]
    cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.RESNETS.DEPTH = 50
    cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 64
    cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]

    cfg.MODEL.SEM_SEG_HEAD.NAME = "MaskDINOHead"
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = experiment.model.hidden_dim
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = experiment.model.mask_dim
    cfg.MODEL.SEM_SEG_HEAD.NORM = "GN"
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "MaskDINOEncoder"
    cfg.MODEL.SEM_SEG_HEAD.DIM_FEEDFORWARD = experiment.model.ffn_dim
    cfg.MODEL.SEM_SEG_HEAD.NUM_FEATURE_LEVELS = 3
    cfg.MODEL.SEM_SEG_HEAD.TOTAL_NUM_FEATURE_LEVELS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = [
        "res3",
        "res4",
        "res5",
    ]
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = experiment.model.encoder_layers
    cfg.MODEL.SEM_SEG_HEAD.FEATURE_ORDER = "low2high"

    cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES = experiment.model.queries
    cfg.MODEL.MaskDINO.HIDDEN_DIM = experiment.model.hidden_dim
    cfg.MODEL.MaskDINO.NHEADS = experiment.model.attention_heads
    cfg.MODEL.MaskDINO.DROPOUT = experiment.model.dropout
    cfg.MODEL.MaskDINO.DIM_FEEDFORWARD = experiment.model.ffn_dim
    cfg.MODEL.MaskDINO.DEC_LAYERS = experiment.model.decoder_layers
    cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS = experiment.model.train_mask_points
    cfg.MODEL.MaskDINO.OVERSAMPLE_RATIO = experiment.model.oversample_ratio
    cfg.MODEL.MaskDINO.IMPORTANCE_SAMPLE_RATIO = experiment.model.importance_sample_ratio
    cfg.MODEL.MaskDINO.CLASS_WEIGHT = experiment.loss.classification
    cfg.MODEL.MaskDINO.COST_CLASS_WEIGHT = experiment.loss.classification
    cfg.MODEL.MaskDINO.MASK_WEIGHT = experiment.loss.mask_bce
    cfg.MODEL.MaskDINO.COST_MASK_WEIGHT = experiment.loss.mask_bce
    cfg.MODEL.MaskDINO.DICE_WEIGHT = experiment.loss.mask_dice
    cfg.MODEL.MaskDINO.COST_DICE_WEIGHT = experiment.loss.mask_dice
    cfg.MODEL.MaskDINO.BOX_WEIGHT = experiment.loss.box_l1
    cfg.MODEL.MaskDINO.COST_BOX_WEIGHT = experiment.loss.box_l1
    cfg.MODEL.MaskDINO.GIOU_WEIGHT = experiment.loss.box_giou
    cfg.MODEL.MaskDINO.COST_GIOU_WEIGHT = experiment.loss.box_giou
    cfg.MODEL.MaskDINO.NO_OBJECT_WEIGHT = experiment.loss.no_object
    cfg.MODEL.MaskDINO.DEEP_SUPERVISION = True
    cfg.MODEL.MaskDINO.TWO_STAGE = True
    cfg.MODEL.MaskDINO.DN = "seg"
    cfg.MODEL.MaskDINO.DN_NUM = experiment.model.denoising_queries
    cfg.MODEL.MaskDINO.INITIALIZE_BOX_TYPE = "mask2box"
    cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON = False
    cfg.MODEL.MaskDINO.TEST.INSTANCE_ON = False
    cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON = False
    cfg.MODEL.MaskDINO.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = True
    cfg.TEST.DETECTIONS_PER_IMAGE = experiment.model.queries
    cfg.freeze()
    return cfg


class PQSetFormer(nn.Module):
    """Keep upstream MaskDINO complexity behind two competition-facing methods."""

    def __init__(self, experiment: Any) -> None:
        super().__init__()
        self.experiment = experiment
        self.base = build_model(build_maskdino_config(experiment))

    @property
    def device(self) -> torch.device:
        return self.base.pixel_mean.device

    def train_batch(self, images: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        self.train()
        return self.base(list(images))

    @torch.no_grad()
    def predict(
        self,
        images: Sequence[Mapping[str, Any]],
        native_shapes: Sequence[tuple[int, int]],
        *,
        presence_floor: float,
        mask_threshold: float,
        upsample_chunk: int = 8,
    ) -> list[list[dict[str, Any]]]:
        """Return raw query masks without conventional NMS or component splitting."""
        if len(images) != len(native_shapes):
            raise ValueError("Image and native-shape counts differ")
        was_training = self.training
        self.eval()
        tensors = [item["image"].to(self.device, non_blocking=True) for item in images]
        normalized = [(tensor - self.base.pixel_mean) / self.base.pixel_std for tensor in tensors]
        batch = ImageList.from_tensors(normalized, self.base.size_divisibility)
        features = self.base.backbone(batch.tensor)
        output, _ = self.base.sem_seg_head(features)
        presence = output["pred_logits"].sigmoid()[..., 0]
        mask_logits = output["pred_masks"]
        results: list[list[dict[str, Any]]] = []
        for batch_index, (height, width) in enumerate(native_shapes):
            keep = torch.where(presence[batch_index] >= presence_floor)[0]
            candidates = []
            for start in range(0, len(keep), upsample_chunk):
                indices = keep[start : start + upsample_chunk]
                probabilities = F.interpolate(
                    mask_logits[batch_index, indices, None],
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0].sigmoid()
                for local_index, query_index in enumerate(indices):
                    probability = probabilities[local_index]
                    mask = probability >= mask_threshold
                    area = int(mask.sum())
                    if not area:
                        continue
                    mask_confidence = float(probability[mask].mean())
                    candidates.append(
                        {
                            "query_id": f"q{int(query_index):03d}",
                            "presence": float(presence[batch_index, query_index]),
                            "predicted_iou": mask_confidence,
                            "quality_kind": "mask_confidence_proxy",
                            "mask": mask.cpu().numpy(),
                        }
                    )
                del probabilities
            results.append(candidates)
        if was_training:
            self.train()
        return results


def optimizer_groups(model: PQSetFormer, experiment: Any) -> list[dict[str, Any]]:
    backbone = []
    remaining = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (backbone if name.startswith("base.backbone.") else remaining).append(parameter)
    if not backbone or not remaining:
        raise ValueError("Could not construct distinct backbone and decoder parameter groups")
    return [
        {"params": remaining, "lr": experiment.training.decoder_lr, "name": "decoder"},
        {"params": backbone, "lr": experiment.training.backbone_lr, "name": "backbone"},
    ]


def finite_parameters(model: nn.Module) -> bool:
    return all(
        bool(torch.isfinite(parameter).all())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
