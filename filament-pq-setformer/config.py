"""Typed, canonical configuration for PQ-SetFormer experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from typing import Any, TypeVar

T = TypeVar("T")


def _require_number(name: str, value: float, low: float, high: float) -> None:
    if isinstance(value, bool) or not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")


def _construct(cls: type[T], value: dict[str, Any]) -> T:
    expected = {field.name for field in fields(cls)}
    unknown = set(value) - expected
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**value)


@dataclass(frozen=True)
class ModelConfig:
    family: str = "MaskDINO-R50"
    input_size: int = 1280
    input_channels: int = 1
    queries: int = 64
    audited_max_instances: int = 26
    hidden_dim: int = 256
    mask_dim: int = 256
    ffn_dim: int = 1024
    attention_heads: int = 8
    dropout: float = 0.0
    decoder_layers: int = 6
    encoder_layers: int = 6
    train_mask_points: int = 12_544
    oversample_ratio: float = 3.0
    importance_sample_ratio: float = 0.75
    denoising_queries: int = 32
    pretrained_backbone: bool = False
    backbone_weights_path: str | None = None
    backbone_weights_sha256: str | None = None
    normalization: str = "GN"
    quality_pooling: str = "detached_mask_weighted_pixel_features"
    candidate_probability_cache_bytes: int = 262_144
    peak_vram_gb: float = 14.5

    def __post_init__(self) -> None:
        if self.family != "MaskDINO-R50":
            raise ValueError("Only the frozen MaskDINO-R50 family is supported")
        if self.input_size < 1024 or self.input_size % 32:
            raise ValueError("input_size must be at least 1024 and divisible by 32")
        if self.input_channels != 1:
            raise ValueError("SFSC input must remain one-channel grayscale")
        if self.pretrained_backbone:
            if not self.backbone_weights_path:
                raise ValueError("pretrained_backbone requires a declared local weight path")
            if (
                not self.backbone_weights_sha256
                or len(self.backbone_weights_sha256) != 64
                or any(char not in "0123456789abcdef" for char in self.backbone_weights_sha256)
            ):
                raise ValueError("pretrained_backbone requires a declared sha256")
            if self.normalization != "FrozenBN":
                raise ValueError("pretrained ImageNet ResNet-50 requires FrozenBN provenance")
        elif self.backbone_weights_path or self.backbone_weights_sha256:
            raise ValueError("random initialization must not declare pretrained backbone weights")
        elif self.normalization != "GN":
            raise ValueError("random initialization requires trainable GroupNorm")
        if self.quality_pooling != "detached_mask_weighted_pixel_features":
            raise ValueError("Only detached mask-weighted conditional-IoU pooling is supported")
        if not 1_024 <= self.candidate_probability_cache_bytes <= 1_048_576:
            raise ValueError("candidate probability cache bound is unsafe")
        if self.queries <= self.audited_max_instances:
            raise ValueError("Model needs more queries than the audited maximum instance count")
        if self.queries > 96:
            raise ValueError("queries must not exceed the frozen P100 search range")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        _require_number("dropout", self.dropout, 0.0, 0.5)
        _require_number("importance_sample_ratio", self.importance_sample_ratio, 0.0, 1.0)
        if self.oversample_ratio < 1 or self.denoising_queries < 0:
            raise ValueError("point oversampling and denoising settings are invalid")
        if not 1.0 <= self.peak_vram_gb < 15.0:
            raise ValueError("peak_vram_gb must stay below 15 GB")


@dataclass(frozen=True)
class LossConfig:
    classification: float = 2.0
    mask_bce: float = 5.0
    mask_dice: float = 5.0
    box_l1: float = 1.0
    box_giou: float = 0.5
    quality: float = 1.0
    foreground: float = 0.0
    boundary: float = 0.0
    cldice: float = 0.0
    no_object: float = 0.1

    def __post_init__(self) -> None:
        for field in fields(self):
            _require_number(field.name, getattr(self, field.name), 0.0, 10.0)


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 20260910
    physical_batch: int = 1
    accumulation: int = 4
    target_microbatches: int = 6_000
    smoke_microbatches: int = 20
    decoder_lr: float = 1e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 0.05
    warmup_steps: int = 100
    gradient_clip: float = 0.5
    amp: bool = True
    workers: int = 4
    reserve_seconds: int = 3_600

    def __post_init__(self) -> None:
        if self.physical_batch != 1 or self.accumulation < 1:
            raise ValueError("P100 training requires physical batch 1 and positive accumulation")
        if self.target_microbatches < self.accumulation:
            raise ValueError("target_microbatches must reach an optimizer step")
        for name in ("decoder_lr", "backbone_lr", "weight_decay", "gradient_clip"):
            _require_number(name, getattr(self, name), 0.0, 1.0)
        if self.workers not in {2, 4, 6}:
            raise ValueError("workers must be one of the benchmarked values")


@dataclass(frozen=True)
class AssemblyConfig:
    presence_floor: float = 0.01
    mask_threshold: float = 0.5
    utility_threshold: float = 0.2
    duplicate_iou: float = 0.7
    containment: float = 0.9
    minimum_area: int = 32
    ownership_gamma: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "presence_floor",
            "mask_threshold",
            "utility_threshold",
            "duplicate_iou",
            "containment",
        ):
            _require_number(name, getattr(self, name), 0.0, 1.0)
        if self.minimum_area not in {0, 32, 64, 96}:
            raise ValueError("minimum_area must be in the frozen calibration grid")


@dataclass(frozen=True)
class LoggingConfig:
    project: str = "sfsc2026-pq-setformer"
    mode: str = "offline"
    wandb_version: str = "0.30.0"

    def __post_init__(self) -> None:
        if self.mode != "offline":
            raise ValueError("Kaggle stages must use offline W&B logging")
        if not self.project or not self.wandb_version:
            raise ValueError("W&B project and pinned version are required")


@dataclass(frozen=True)
class ExperimentConfig:
    version: str = "pq-setformer-v1"
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    training: TrainingConfig = TrainingConfig()
    assembly: AssemblyConfig = AssemblyConfig()
    logging: LoggingConfig = LoggingConfig()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExperimentConfig:
        expected = {field.name for field in fields(cls)}
        unknown = set(value) - expected
        if unknown:
            raise ValueError(f"Unknown ExperimentConfig fields: {sorted(unknown)}")
        return cls(
            version=value.get("version", cls.version),
            model=_construct(ModelConfig, value.get("model", {})),
            loss=_construct(LossConfig, value.get("loss", {})),
            training=_construct(TrainingConfig, value.get("training", {})),
            assembly=_construct(AssemblyConfig, value.get("assembly", {})),
            logging=_construct(LoggingConfig, value.get("logging", {})),
        )
