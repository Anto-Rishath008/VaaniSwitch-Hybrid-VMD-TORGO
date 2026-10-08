"""On-the-fly signed Module 3 encoders plus the locked lightweight classifier."""

from typing import cast

import torch
from torch import nn

from vaaniswitch.m3_encoders.config import Module3Config
from vaaniswitch.m3_encoders.models import ParallelTokenEncoders
from vaaniswitch.m3_encoders.schemas import EncoderBatch
from vaaniswitch.m4_fusion.attention import MelCrossAttention
from vaaniswitch.m4_fusion.schemas import FusionOutput


class TokenFusionClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cross_attention = MelCrossAttention()
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        mel: torch.Tensor,
        nonlinear: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> FusionOutput:
        attended, attention_output, weights = self.cross_attention(
            mel,
            nonlinear,
            return_attention=return_attention,
        )
        mel_pool = attended.mean(dim=1)
        nonlinear_pool = nonlinear.mean(dim=1)
        vector = torch.cat((mel_pool, nonlinear_pool), dim=-1)
        logits = cast(torch.Tensor, self.classifier(vector)).squeeze(-1)
        if not torch.isfinite(logits).all():
            raise ValueError("classifier produced non-finite logits")
        return FusionOutput(
            logits,
            attended,
            mel_pool,
            nonlinear_pool,
            vector,
            attention_output,
            weights,
        )


class DysarthriaClassifier(nn.Module):
    def __init__(self, encoder_config: Module3Config, *, load_pretrained: bool) -> None:
        super().__init__()
        self.encoders = ParallelTokenEncoders(encoder_config, load_pretrained=load_pretrained)
        self.fusion = TokenFusionClassifier()

    def verify_frozen_backbone(self) -> None:
        mel = self.encoders.mel_encoder
        if (
            not mel.backbone_frozen
            or mel.backbone.training
            or any(p.requires_grad or p.grad is not None for p in mel.backbone.parameters())
        ):
            raise ValueError("Module 4 requires the ResNet backbone frozen and in eval mode")

    def forward(self, batch: EncoderBatch, *, return_attention: bool = False) -> FusionOutput:
        self.verify_frozen_backbone()
        tokens = self.encoders(batch)
        return cast(
            FusionOutput,
            self.fusion(
                tokens.mel_tokens,
                tokens.nonlinear_tokens,
                return_attention=return_attention,
            ),
        )
