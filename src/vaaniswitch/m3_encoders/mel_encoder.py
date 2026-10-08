"""ResNet18 feature trunk producing sixteen Mel tokens."""

from __future__ import annotations

from typing import Self, cast

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from vaaniswitch.m3_encoders.config import MelEncoderConfig
from vaaniswitch.m3_encoders.preprocessing import preprocess_mel


class MelTokenEncoder(nn.Module):
    def __init__(
        self,
        config: MelEncoderConfig,
        *,
        token_dimension: int,
        load_pretrained: bool,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if load_pretrained else None
        network = resnet18(weights=weights)
        self.preprocessor_config = config
        self.backbone = nn.Sequential(*list(network.children())[:-2])
        self.spatial_pool = nn.AdaptiveAvgPool2d(config.adaptive_pool)
        self.projection = nn.Linear(512, token_dimension)
        self.layer_norm = nn.LayerNorm(token_dimension)
        self.pretrained_weights_identifier = (
            "ResNet18_Weights.IMAGENET1K_V1" if weights is not None else "NONE"
        )
        self.backbone_frozen = False
        self.set_backbone_trainable(not config.freeze_backbone_initially)

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.backbone_frozen = not trainable
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable
        if self.backbone_frozen:
            self.backbone.eval()

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        values = preprocess_mel(
            mel,
            db_min=self.preprocessor_config.fixed_db_min,
            db_max=self.preprocessor_config.fixed_db_max,
            imagenet_normalization=self.preprocessor_config.imagenet_normalization,
        )
        features = self.backbone(values)
        pooled = self.spatial_pool(features)
        tokens = pooled.flatten(2).transpose(1, 2)
        return cast(torch.Tensor, self.layer_norm(self.projection(tokens)))
