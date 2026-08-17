"""Recurrence and strict natural-visibility numerical regression tests."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from tests.m2_representations.helpers import make_config
from vaaniswitch.m2_representations.recurrence import generate_recurrence
from vaaniswitch.m2_representations.reduction import block_rms, zscore
from vaaniswitch.m2_representations.visibility import (
    generate_visibility_graph,
    natural_visibility_edges,
)


def _signal(size: int = 2_000) -> npt.NDArray[np.float64]:
    samples = np.arange(size, dtype=np.float64)
    return cast(
        npt.NDArray[np.float64],
        (np.sin(0.017 * samples) + 0.25 * np.cos(0.043 * samples)).astype(np.float64),
    )


def test_recurrence_is_deterministic_exact_shape_uint8_symmetric_and_finite(tmp_path: Path) -> None:
    config = make_config(tmp_path).recurrence
    first = generate_recurrence(_signal(), config)
    second = generate_recurrence(_signal(), config)

    np.testing.assert_array_equal(first.image, second.image)
    assert first.image.shape == (224, 224)
    assert first.image.dtype == np.uint8
    assert np.array_equal(first.image, first.image.T)
    assert np.all(np.isfinite(first.image))


def test_recurrence_diagonal_values_density_and_state_contract(tmp_path: Path) -> None:
    result = generate_recurrence(_signal(), make_config(tmp_path).recurrence)
    assert np.all(np.diag(result.image) == 1)
    assert set(np.unique(result.image)) <= {0, 1}
    assert result.scalar_state_count == 228
    assert result.embedded_state_count == 224
    assert 0.15 <= result.recurrence_density <= 0.16


def test_recurrence_handles_constant_short_signal_safely(tmp_path: Path) -> None:
    result = generate_recurrence(np.ones(3, dtype=np.float64), make_config(tmp_path).recurrence)
    assert result.image.shape == (224, 224)
    assert np.all(result.image == 1)


def test_block_rms_uses_contiguous_anti_alias_aggregation() -> None:
    signal = np.asarray([3.0, 4.0, 0.0, 0.0], dtype=np.float64)
    np.testing.assert_allclose(block_rms(signal, 2), [np.sqrt(12.5), 0.0])
    assert np.array_equal(block_rms(np.asarray([2.0]), 3), [2.0, 2.0, 2.0])


def test_zscore_constant_series_is_finite_zero() -> None:
    assert np.array_equal(zscore(np.ones(5), 1.0e-12), np.zeros(5))


def _undirected(edges: np.ndarray[tuple[int, int], np.dtype[np.int16]]) -> set[tuple[int, int]]:
    return {(int(left), int(right)) for left, right in edges.T if left < right}


def test_natural_visibility_tiny_peak_series_is_hand_verifiable() -> None:
    edges = natural_visibility_edges(np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
    assert _undirected(edges) == {(0, 1), (1, 2)}


def test_natural_visibility_tiny_valley_series_is_hand_verifiable() -> None:
    edges = natural_visibility_edges(np.asarray([1.0, 0.0, 1.0], dtype=np.float64))
    assert _undirected(edges) == {(0, 1), (0, 2), (1, 2)}


def test_collinear_intermediate_point_blocks_strict_natural_visibility() -> None:
    edges = natural_visibility_edges(np.asarray([0.0, 1.0, 2.0], dtype=np.float64))
    assert _undirected(edges) == {(0, 1), (1, 2)}


def test_visibility_graph_is_deterministic_bounded_and_compact(tmp_path: Path) -> None:
    config = make_config(tmp_path).visibility_graph
    first = generate_visibility_graph(_signal(8_000), config)
    second = generate_visibility_graph(_signal(8_000), config)

    np.testing.assert_array_equal(first.node_features, second.node_features)
    np.testing.assert_array_equal(first.edge_index, second.edge_index)
    assert first.node_features.shape == (256, 2)
    assert first.node_features.dtype == np.float32
    assert first.edge_index.dtype == np.int16
    assert first.node_features.shape[0] <= 256


def test_visibility_edges_have_valid_unique_bidirectional_indices_no_self_loops(
    tmp_path: Path,
) -> None:
    graph = generate_visibility_graph(_signal(), make_config(tmp_path).visibility_graph)
    pairs = {(int(left), int(right)) for left, right in graph.edge_index.T}
    assert len(pairs) == graph.edge_index.shape[1]
    assert all(0 <= left < graph.node_features.shape[0] for left, _ in pairs)
    assert all(0 <= right < graph.node_features.shape[0] for _, right in pairs)
    assert all(left != right for left, right in pairs)
    assert all((right, left) in pairs for left, right in pairs)


def test_visibility_graph_is_connected_without_isolates_for_multinode_signal(
    tmp_path: Path,
) -> None:
    graph = generate_visibility_graph(_signal(), make_config(tmp_path).visibility_graph)
    assert graph.isolated_node_count == 0
    assert graph.connected_component_count == 1
    assert 0 < graph.graph_density <= 1


def test_visibility_graph_handles_one_sample_signal(tmp_path: Path) -> None:
    graph = generate_visibility_graph(np.asarray([0.5]), make_config(tmp_path).visibility_graph)
    assert graph.node_features.shape == (1, 2)
    assert graph.edge_index.shape == (2, 0)
    assert graph.isolated_node_count == 1
    assert graph.connected_component_count == 1
