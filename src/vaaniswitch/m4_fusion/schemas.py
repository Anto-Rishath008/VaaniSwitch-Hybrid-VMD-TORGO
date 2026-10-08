"""Public typed output: logits for loss, sigmoid only as an explicit convenience."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class FusionOutput:
    logits: torch.Tensor
    attended_mel_tokens: torch.Tensor
    mel_pool: torch.Tensor
    nonlinear_pool: torch.Tensor
    fusion_vector: torch.Tensor
    attention_output: torch.Tensor
    attention_weights: torch.Tensor | None = None

    @property
    def probabilities(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)
