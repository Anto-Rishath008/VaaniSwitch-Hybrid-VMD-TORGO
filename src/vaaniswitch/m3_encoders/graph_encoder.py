"""Two-layer dense pure-PyTorch GCN for natural visibility graphs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import nn

from vaaniswitch.m3_encoders.config import GraphEncoderConfig
from vaaniswitch.m3_encoders.graph_ops import batched_normalized_adjacency


class GraphConvolution(nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dimension, output_dimension)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.linear(torch.bmm(adjacency, node_features)))


class VisibilityGraphTokenEncoder(nn.Module):
    def __init__(
        self,
        config: GraphEncoderConfig,
        *,
        token_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gcn1 = GraphConvolution(config.input_dim, config.hidden_dim)
        self.gcn2 = GraphConvolution(config.hidden_dim, config.graph_feature_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(2 * config.graph_feature_dim, token_dimension)
        self.layer_norm = nn.LayerNorm(token_dimension)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_indices: Sequence[torch.Tensor],
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            node_features.ndim != 3
            or node_features.shape[:2] != node_mask.shape
            or node_features.shape[2] != 2
            or not torch.isfinite(node_features).all()
        ):
            raise ValueError("graph node features/mask violate [B,N,2] contract")
        adjacency = batched_normalized_adjacency(
            edge_indices, node_mask, dtype=node_features.dtype
        )
        mask = node_mask.unsqueeze(-1)
        hidden = self.gcn1(node_features * mask, adjacency)
        hidden = self.dropout(self.activation(hidden)) * mask
        hidden = self.gcn2(hidden, adjacency)
        hidden = self.activation(hidden) * mask
        counts = node_mask.sum(dim=1, keepdim=True).to(hidden.dtype)
        mean_pool = hidden.sum(dim=1) / counts
        max_pool = hidden.masked_fill(~mask, -torch.inf).max(dim=1).values
        pooled = torch.cat((mean_pool, max_pool), dim=1)
        result = self.layer_norm(self.projection(pooled)).unsqueeze(1)
        return cast(torch.Tensor, result)
