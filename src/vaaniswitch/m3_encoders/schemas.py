"""Typed Module 3 sample, batch, and token contracts."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class SampleProvenance:
    logical_id: str
    speaker_id: str
    condition: str
    gender: str
    prompt_type: str
    primary_microphone: str
    evaluation_fold: int


@dataclass(frozen=True, slots=True)
class EncoderSample:
    mel: torch.Tensor
    recurrence: torch.Tensor
    graph_nodes: torch.Tensor
    graph_edges: torch.Tensor
    provenance: SampleProvenance


@dataclass(frozen=True, slots=True)
class EncoderBatch:
    mel: torch.Tensor
    recurrence: torch.Tensor
    graph_nodes: torch.Tensor
    graph_node_mask: torch.Tensor
    graph_edges: tuple[torch.Tensor, ...]
    provenance: tuple[SampleProvenance, ...]

    def to(self, device: torch.device | str) -> EncoderBatch:
        return EncoderBatch(
            mel=self.mel.to(device),
            recurrence=self.recurrence.to(device),
            graph_nodes=self.graph_nodes.to(device),
            graph_node_mask=self.graph_node_mask.to(device),
            graph_edges=tuple(edges.to(device) for edges in self.graph_edges),
            provenance=self.provenance,
        )


@dataclass(frozen=True, slots=True)
class EncoderTokenOutput:
    mel_tokens: torch.Tensor
    graph_token: torch.Tensor
    recurrence_tokens: torch.Tensor

    @property
    def nonlinear_tokens(self) -> torch.Tensor:
        return torch.cat((self.graph_token, self.recurrence_tokens), dim=1)

    def validate(self, token_dimension: int = 128) -> None:
        batch_size = self.mel_tokens.shape[0]
        expected = (
            (self.mel_tokens, (batch_size, 16, token_dimension), "Mel"),
            (self.graph_token, (batch_size, 1, token_dimension), "graph"),
            (self.recurrence_tokens, (batch_size, 16, token_dimension), "recurrence"),
            (self.nonlinear_tokens, (batch_size, 17, token_dimension), "nonlinear"),
        )
        for tensor, shape, name in expected:
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} token shape is invalid: {tuple(tensor.shape)}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} tokens contain non-finite values")
