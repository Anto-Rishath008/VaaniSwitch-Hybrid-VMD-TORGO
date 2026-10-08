"""Hand-verifiable graph normalization, masking, pooling, and GCN tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.m3_encoders.helpers import make_batch, make_model_config
from vaaniswitch.m3_encoders.graph_encoder import VisibilityGraphTokenEncoder
from vaaniswitch.m3_encoders.graph_ops import (
    batched_normalized_adjacency,
    normalized_adjacency,
)


def test_two_node_graph_adds_self_loops_once_and_normalizes_symmetrically() -> None:
    edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    actual = normalized_adjacency(edges, 2)
    expected = torch.full((2, 2), 0.5)
    torch.testing.assert_close(actual, expected)


def test_three_node_path_has_hand_verifiable_degree_normalization() -> None:
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.int64)
    actual = normalized_adjacency(edges, 3)
    root6 = 1 / (6**0.5)
    expected = torch.tensor(
        [[0.5, root6, 0.0], [root6, 1 / 3, root6], [0.0, root6, 0.5]]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("edges", "message"),
    [
        (torch.tensor([[0], [0]]), "self-loops"),
        (torch.tensor([[0, 0], [1, 1]]), "duplicate"),
        (torch.tensor([[0], [1]]), "bidirectional"),
        (torch.tensor([[0, 2], [1, 0]]), "outside"),
        (torch.zeros((3, 1), dtype=torch.int64), "shape"),
    ],
)
def test_invalid_stored_graphs_are_rejected(edges: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalized_adjacency(edges, 2)


def test_batched_adjacency_zeros_padding_and_rejects_nonprefix_mask() -> None:
    edges = (torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),)
    mask = torch.tensor([[True, True, False]])
    adjacency = batched_normalized_adjacency(edges, mask, dtype=torch.float32)
    assert torch.count_nonzero(adjacency[0, 2]) == 0
    assert torch.count_nonzero(adjacency[0, :, 2]) == 0
    with pytest.raises(ValueError, match="prefix"):
        batched_normalized_adjacency(
            edges, torch.tensor([[True, False, True]]), dtype=torch.float32
        )


def test_gcn_output_shape_determinism_and_one_node_safety(tmp_path: Path) -> None:
    config = make_model_config(tmp_path)
    encoder = VisibilityGraphTokenEncoder(
        config.graph, token_dimension=128, dropout=config.dropout
    ).eval()
    batch = make_batch(2, (3, 1))
    first = encoder(batch.graph_nodes, batch.graph_edges, batch.graph_node_mask)
    second = encoder(batch.graph_nodes, batch.graph_edges, batch.graph_node_mask)
    assert first.shape == (2, 1, 128)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


def test_padded_nodes_cannot_change_gcn_mean_or_max_pooling(tmp_path: Path) -> None:
    config = make_model_config(tmp_path)
    encoder = VisibilityGraphTokenEncoder(
        config.graph, token_dimension=128, dropout=0.0
    ).eval()
    batch = make_batch(1, (3,))
    enlarged_nodes = torch.cat((batch.graph_nodes, torch.full((1, 2, 2), 9999.0)), dim=1)
    enlarged_mask = torch.tensor([[True, True, True, False, False]])
    expected = encoder(batch.graph_nodes, batch.graph_edges, batch.graph_node_mask)
    actual = encoder(enlarged_nodes, batch.graph_edges, enlarged_mask)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
