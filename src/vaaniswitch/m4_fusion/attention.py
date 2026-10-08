"""One Mel-query cross-attention block with explicit residual and per-head weights."""

from typing import cast

import torch
from torch import nn


class MelCrossAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(128, 4, dropout=0.10, batch_first=True)
        self.layer_norm = nn.LayerNorm(128)

    def forward(
        self,
        mel: torch.Tensor,
        nonlinear: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if mel.ndim != 3 or mel.shape[0] < 1 or mel.shape[1:] != (16, 128):
            raise ValueError("Mel tokens must have shape [B,16,128] with B positive")
        if nonlinear.shape != (mel.shape[0], 17, 128):
            raise ValueError("nonlinear tokens must have shape [B,17,128]")
        if mel.dtype != nonlinear.dtype or mel.device != nonlinear.device:
            raise ValueError("token dtype and device must agree")
        if not mel.is_floating_point() or not (
            torch.isfinite(mel).all() and torch.isfinite(nonlinear).all()
        ):
            raise ValueError("tokens must be finite floating point values")
        output, weights = self.attention(
            query=mel,
            key=nonlinear,
            value=nonlinear,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        attended = cast(torch.Tensor, self.layer_norm(mel + output))
        return attended, output, weights
