"""Kaggle-only MaskDINO E1 model adapter with a minimal train/predict interface."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
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
    cfg.MODEL.RESNETS.NORM = experiment.model.normalization
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


class QualityContractError(RuntimeError):
    """Raised when the pinned upstream output cannot safely support E2."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _detectron_key_from_torchvision(key: str) -> str | None:
    """Map the public ImageNet R50 state-dict naming onto Detectron2 ResNet."""
    key = key.removeprefix("module.")
    if key == "conv1.weight":
        return "stem.conv1.weight"
    if key.startswith("bn1."):
        return "stem.conv1.norm." + key.removeprefix("bn1.")
    layers = (("layer1", "res2"), ("layer2", "res3"), ("layer3", "res4"), ("layer4", "res5"))
    for layer, res in layers:
        if key.startswith(layer + "."):
            result = res + key.removeprefix(layer)
            result = result.replace(".downsample.0.", ".shortcut.")
            result = result.replace(".downsample.1.", ".shortcut.norm.")
            for name in ("bn1", "bn2", "bn3"):
                result = result.replace(
                    "." + name + ".", "." + name.replace("bn", "conv") + ".norm."
                )
            return result
    return None


def initialize_backbone_from_declared_weights(base: nn.Module, experiment: Any) -> dict[str, str]:
    """Load only a declared local generic ImageNet R50, never downloading it.

    A generic torchvision-style state dict is converted into Detectron2's
    naming.  The RGB stem is averaged into one channel.  Missing non-classifier
    tensors are a hard error, preventing a partially/randomly initialised model
    from being reported as pretrained.
    """
    model_config = experiment.model
    if not model_config.pretrained_backbone:
        return {"kind": "random", "normalization": model_config.normalization}
    path = Path(model_config.backbone_weights_path)
    if not path.is_file() or _sha256(path) != model_config.backbone_weights_sha256:
        raise QualityContractError(
            "Declared ImageNet R50 weight file is unavailable or hash-mismatched"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source = payload.get("state_dict", payload) if isinstance(payload, Mapping) else None
    if not isinstance(source, Mapping):
        raise QualityContractError("Declared ImageNet R50 file has no state dictionary")
    target = base.backbone.bottom_up.state_dict()
    converted: dict[str, torch.Tensor] = {}
    for source_key, value in source.items():
        if not isinstance(value, torch.Tensor):
            continue
        target_key = _detectron_key_from_torchvision(str(source_key))
        if target_key not in target:
            continue
        if target_key == "stem.conv1.weight":
            if value.ndim != 4 or value.shape[1] != 3:
                raise QualityContractError(
                    "ImageNet R50 stem must have exactly three input channels"
                )
            value = value.mean(dim=1, keepdim=True)
        if value.shape == target[target_key].shape:
            converted[target_key] = value
    required = {key for key in target if not key.endswith("num_batches_tracked")}
    missing = sorted(required - set(converted))
    if missing:
        raise QualityContractError(
            "Declared ImageNet R50 is not a complete compatible generic backbone; missing "
            + ", ".join(missing[:5])
        )
    base.backbone.bottom_up.load_state_dict(converted, strict=True)
    return {
        "kind": "generic_imagenet_r50_rgb_mean_stem",
        "path": str(path),
        "sha256": model_config.backbone_weights_sha256,
        "normalization": model_config.normalization,
    }


class PQSetFormer(nn.Module):
    """Keep upstream MaskDINO complexity behind two competition-facing methods."""

    def __init__(self, experiment: Any) -> None:
        super().__init__()
        self.experiment = experiment
        self.base = build_model(build_maskdino_config(experiment))
        self.initialization_provenance = initialize_backbone_from_declared_weights(
            self.base, experiment
        )
        hidden = experiment.model.hidden_dim
        self.conditional_iou_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    @property
    def device(self) -> torch.device:
        return self.base.pixel_mean.device

    def train_batch(self, images: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        self.train()
        tensors = [item["image"].to(self.device, non_blocking=True) for item in images]
        normalized = [(tensor - self.base.pixel_mean) / self.base.pixel_std for tensor in tensors]
        batch = ImageList.from_tensors(normalized, self.base.size_divisibility)
        instances = [item["instances"].to(self.device) for item in images]
        if "detr" not in self.base.data_loader:
            raise QualityContractError("PQSetFormer requires the declared Detectron2 DETR mapper")
        targets = self.base.prepare_targets_detr(instances, batch)
        outputs, mask_dict = self.base.sem_seg_head(
            self.base.backbone(batch.tensor), targets=targets
        )
        losses = self.base.criterion(outputs, targets, mask_dict)
        for name in list(losses):
            if name in self.base.criterion.weight_dict:
                losses[name] *= self.base.criterion.weight_dict[name]
            else:
                losses.pop(name)
        if self.experiment.loss.quality > 0:
            losses["loss_quality"] = (
                self._conditional_iou_loss(outputs, targets) * self.experiment.loss.quality
            )
        return losses

    def conditional_iou_logits(
        self, query_features: torch.Tensor, mask_logits: torch.Tensor, pixel_features: torch.Tensor
    ) -> torch.Tensor:
        """E2's detached mask-weighted pixel pooling, exposed for patched upstream use."""
        if query_features.ndim != 3 or mask_logits.ndim != 4 or pixel_features.ndim != 4:
            raise QualityContractError(
                "Conditional-IoU adapter received malformed MaskDINO features"
            )
        weights = mask_logits.sigmoid().detach()
        if weights.shape[-2:] != pixel_features.shape[-2:]:
            weights = F.interpolate(
                weights, size=pixel_features.shape[-2:], mode="bilinear", align_corners=False
            )
        denominator = weights.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
        pooled = torch.einsum("bqhw,bchw->bqc", weights / denominator, pixel_features)
        return self.conditional_iou_head(torch.cat((query_features, pooled), dim=-1)).squeeze(-1)

    def _conditional_iou_loss(
        self, outputs: Mapping[str, torch.Tensor], targets: Sequence[Mapping[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Supervise only Hungarian-matched queries with detached realized IoU."""
        try:
            query_features = outputs["pred_query_features"]
            pixel_features = outputs["pixel_features"]
        except KeyError as error:
            raise QualityContractError(
                "Missing PQ MaskDINO quality adapter outputs; apply the pinned vendor patch"
            ) from error
        quality_logits = self.conditional_iou_logits(
            query_features, outputs["pred_masks"], pixel_features
        )
        matched = self.base.criterion.matcher(outputs, targets)
        values = []
        for batch_index, (query_indices, target_indices) in enumerate(matched):
            if not len(query_indices):
                continue
            prediction = outputs["pred_masks"][batch_index, query_indices]
            truth = targets[batch_index]["masks"][target_indices].float()
            truth = F.interpolate(truth[:, None], size=prediction.shape[-2:], mode="nearest")[:, 0]
            predicted_binary = prediction.sigmoid().detach() >= 0.5
            target_binary = truth.detach() >= 0.5
            intersection = (predicted_binary & target_binary).flatten(1).sum(1)
            union = (predicted_binary | target_binary).flatten(1).sum(1).clamp_min(1)
            realized_iou = (intersection / union).detach().to(quality_logits.dtype)
            values.append(
                F.binary_cross_entropy_with_logits(
                    quality_logits[batch_index, query_indices], realized_iou
                )
            )
        if not values:
            # Preserve a finite connected scalar for empty-target microbatches.
            return quality_logits.sum() * 0
        return torch.stack(values).mean()

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
        try:
            conditional_iou = self.conditional_iou_logits(
                output["pred_query_features"], mask_logits, output["pixel_features"]
            ).sigmoid()
        except KeyError as error:
            raise QualityContractError(
                "Missing PQ MaskDINO quality adapter outputs; apply the pinned vendor patch"
            ) from error
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
                    candidates.append(
                        {
                            "query_id": f"q{int(query_index):03d}",
                            "presence": float(presence[batch_index, query_index]),
                            "presence_logit": float(
                                output["pred_logits"][batch_index, query_index, 0]
                            ),
                            "predicted_iou": float(conditional_iou[batch_index, query_index]),
                            "quality_kind": "conditional_mask_iou",
                            "mask": mask.cpu().numpy(),
                            "mask_probability": probability.cpu().numpy(),
                            "source": "identity",
                        }
                    )
                del probabilities
            results.append(candidates)
        if was_training:
            self.train()
        return results


def optimizer_groups(model: PQSetFormer, experiment: Any) -> list[dict[str, Any]]:
    groups: dict[tuple[str, bool], list[torch.Tensor]] = {}
    special_modules = (nn.modules.batchnorm._NormBase, nn.GroupNorm, nn.LayerNorm, nn.Embedding)
    special_names = {
        parameter_name
        for module_name, module in model.named_modules()
        if isinstance(module, special_modules)
        for parameter_name, _ in module.named_parameters(prefix=module_name, recurse=False)
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        bucket = "backbone" if name.startswith("base.backbone.") else "decoder"
        no_decay = name in special_names or name.endswith(".bias")
        groups.setdefault((bucket, no_decay), []).append(parameter)
    if not groups.get(("backbone", False)) or not groups.get(("decoder", False)):
        raise ValueError("Could not construct distinct backbone and decoder parameter groups")
    return [
        {
            "params": parameters,
            "lr": (
                experiment.training.backbone_lr
                if bucket == "backbone"
                else experiment.training.decoder_lr
            ),
            "weight_decay": 0.0 if no_decay else experiment.training.weight_decay,
            "name": f"{bucket}-{'no-decay' if no_decay else 'decay'}",
        }
        for (bucket, no_decay), parameters in sorted(groups.items())
        if parameters
    ]


def finite_parameters(model: nn.Module) -> bool:
    return all(
        bool(torch.isfinite(parameter).all())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
