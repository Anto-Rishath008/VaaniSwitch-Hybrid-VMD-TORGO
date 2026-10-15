"""Compact recurrence-plot CNN producing sixteen spatial tokens."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from .native_b2_contract import RecurrenceEncoderConfig
from .native_b2_preprocessing import preprocess_recurrence


class RecurrenceTokenEncoder(nn.Module):
    def __init__(
        self,
        config: RecurrenceEncoderConfig,
        *,
        token_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        input_channels = config.input_channels
        for index, output_channels in enumerate(config.channels):
            layers: list[nn.Module] = [
                nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                _group_norm(output_channels),
                nn.GELU(),
                nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
                _group_norm(output_channels),
                nn.GELU(),
            ]
            if index < len(config.channels) - 1:
                layers.append(nn.MaxPool2d(2))
            if dropout:
                layers.append(nn.Dropout2d(dropout))
            blocks.append(nn.Sequential(*layers))
            input_channels = output_channels
        self.blocks = nn.Sequential(*blocks)
        self.spatial_pool = nn.AdaptiveAvgPool2d(config.adaptive_pool)
        self.projection = nn.Linear(config.channels[-1], token_dimension)
        self.layer_norm = nn.LayerNorm(token_dimension)

    def forward(self, recurrence: torch.Tensor) -> torch.Tensor:
        values = preprocess_recurrence(recurrence)
        features = self.spatial_pool(self.blocks(values))
        tokens = features.flatten(2).transpose(1, 2)
        return cast(torch.Tensor, self.layer_norm(self.projection(tokens)))


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)
