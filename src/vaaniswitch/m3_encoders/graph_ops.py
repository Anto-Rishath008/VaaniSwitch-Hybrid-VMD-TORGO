"""Dense pure-PyTorch graph construction for at-most-256-node NVGs."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def normalized_adjacency(
    edge_index: torch.Tensor,
    node_count: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return D^-1/2 (A + I) D^-1/2 after strict stored-edge checks."""

    if not 1 <= node_count <= 256:
        raise ValueError("node count must be in [1,256]")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E]")
    if edge_index.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError("edge_index must contain integers")
    edges = edge_index.to(torch.int64)
    if edges.numel():
        if int(torch.min(edges)) < 0 or int(torch.max(edges)) >= node_count:
            raise ValueError("edge index is outside the valid-node range")
        if torch.any(edges[0] == edges[1]):
            raise ValueError("stored visibility graph must not contain self-loops")
        if torch.unique(edges, dim=1).shape[1] != edges.shape[1]:
            raise ValueError("stored visibility graph contains duplicate directed edges")
    adjacency = torch.zeros(
        (node_count, node_count), dtype=dtype, device=edge_index.device
    )
    if edges.numel():
        adjacency[edges[0], edges[1]] = 1
        if not torch.equal(adjacency, adjacency.T):
            raise ValueError("stored visibility graph edges must be bidirectional")
    adjacency.fill_diagonal_(1)
    degree = adjacency.sum(dim=1)
    inverse_sqrt = degree.rsqrt()
    return inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]


def batched_normalized_adjacency(
    edge_indices: Sequence[torch.Tensor],
    node_mask: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Pad normalized adjacencies with zeros so invalid nodes never contribute."""

    if node_mask.ndim != 2 or node_mask.dtype != torch.bool:
        raise ValueError("node mask must be boolean [B,N]")
    if len(edge_indices) != node_mask.shape[0]:
        raise ValueError("edge list count must equal batch size")
    batch_size, maximum_nodes = node_mask.shape
    result = torch.zeros(
        (batch_size, maximum_nodes, maximum_nodes),
        dtype=dtype,
        device=node_mask.device,
    )
    positions = torch.arange(maximum_nodes, device=node_mask.device)
    for index, edges in enumerate(edge_indices):
        count = int(node_mask[index].sum())
        expected_mask = positions < count
        if count == 0 or not torch.equal(node_mask[index], expected_mask):
            raise ValueError("valid graph nodes must form a non-empty prefix")
        result[index, :count, :count] = normalized_adjacency(
            edges.to(node_mask.device), count, dtype=dtype
        )
    return result
