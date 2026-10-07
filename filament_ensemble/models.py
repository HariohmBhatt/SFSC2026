from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        groups = min(16, output_channels)
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
        )


class PromptFPN(nn.Module):
    """Six-channel prompt refiner with mask, boundary, and IoU-quality heads."""

    def __init__(self, backbone: str, pretrained: bool = True, channels: int = 128) -> None:
        super().__init__()
        import timm

        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            in_chans=6,
            out_indices=(0, 1, 2, 3),
        )
        encoder_channels = self.encoder.feature_info.channels()
        self.lateral = nn.ModuleList(nn.Conv2d(value, channels, 1) for value in encoder_channels)
        self.smooth = nn.ModuleList(ConvNormAct(channels, channels) for _ in encoder_channels)
        self.fusion = ConvNormAct(channels * len(encoder_channels), channels)
        self.mask_head = nn.Conv2d(channels, 1, 1)
        self.boundary_head = nn.Conv2d(channels, 1, 1)
        self.quality_head = nn.Sequential(
            nn.Linear(encoder_channels[-1], channels),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        output_size = tensor.shape[-2:]
        features = self.encoder(tensor)
        pyramid: list[torch.Tensor] = [torch.empty(0)] * len(features)
        top = self.lateral[-1](features[-1])
        pyramid[-1] = self.smooth[-1](top)
        for index in range(len(features) - 2, -1, -1):
            lateral = self.lateral[index](features[index])
            top = lateral + F.interpolate(
                top, size=lateral.shape[-2:], mode="bilinear", align_corners=False
            )
            pyramid[index] = self.smooth[index](top)
        fusion_size = pyramid[0].shape[-2:]
        fused = self.fusion(
            torch.cat(
                [
                    F.interpolate(level, fusion_size, mode="bilinear", align_corners=False)
                    for level in pyramid
                ],
                dim=1,
            )
        )
        mask = F.interpolate(
            self.mask_head(fused), size=output_size, mode="bilinear", align_corners=False
        )
        boundary = F.interpolate(
            self.boundary_head(fused), size=output_size, mode="bilinear", align_corners=False
        )
        quality = self.quality_head(F.adaptive_avg_pool2d(features[-1], 1).flatten(1))
        return {
            "mask": mask,
            "boundary": boundary,
            "quality": quality,
        }


class FullDiskProposalNet(nn.Module):
    """Full-frame semantic and boundary model used to add detector-missed proposals."""

    def __init__(self, backbone: str, pretrained: bool = True) -> None:
        super().__init__()
        self.network = PromptFPN(backbone, pretrained=pretrained)

    def forward(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        if tensor.shape[1] != 6:
            raise ValueError("FullDiskProposalNet expects six derived image channels")
        return self.network(tensor)
