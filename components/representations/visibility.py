"""Natural visibility graphs from bounded anti-aliased waveform envelopes."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from components.representations.config import VisibilityConfig
from components.representations.reduction import block_rms, zscore
from components.representations.schemas import GraphResult


def natural_visibility_edges(series: npt.NDArray[np.float64]) -> npt.NDArray[np.int16]:
    """Return bidirectional strict natural-visibility edges in O(N^2) time."""

    if series.ndim != 1 or series.size == 0 or not np.all(np.isfinite(series)):
        raise ValueError("series must be non-empty, one-dimensional, and finite")
    if series.size > np.iinfo(np.int16).max:
        raise ValueError("series is too large for int16 edge storage")
    undirected: list[tuple[int, int]] = []
    for left in range(series.size - 1):
        maximum_slope = -np.inf
        for right in range(left + 1, series.size):
            slope = float((series[right] - series[left]) / (right - left))
            if slope > maximum_slope:
                undirected.append((left, right))
            maximum_slope = max(maximum_slope, slope)
    directed = [edge for left, right in undirected for edge in ((left, right), (right, left))]
    if not directed:
        return np.empty((2, 0), dtype=np.int16)
    return np.asarray(directed, dtype=np.int16).T


def generate_visibility_graph(
    waveform: npt.NDArray[np.float64], config: VisibilityConfig
) -> GraphResult:
    """Build a <=256-node NVG with time and standardized-envelope node features."""

    if waveform.ndim != 1 or waveform.size == 0 or not np.all(np.isfinite(waveform)):
        raise ValueError("waveform must be non-empty, one-dimensional, and finite")
    node_count = min(config.max_nodes, waveform.size)
    amplitude = zscore(block_rms(waveform, node_count), config.epsilon)
    time = np.linspace(0.0, 1.0, node_count, dtype=np.float64)
    nodes = np.column_stack((time, amplitude)).astype(np.float32)
    edges = natural_visibility_edges(amplitude)
    undirected_edges = edges.shape[1] // 2
    possible = node_count * (node_count - 1) / 2
    density = float(undirected_edges / possible) if possible else 0.0
    degrees = np.bincount(edges[0].astype(np.int64), minlength=node_count)
    isolated = int(np.count_nonzero(degrees == 0))
    components = _connected_components(node_count, edges)
    return GraphResult(
        node_features=nodes,
        edge_index=edges,
        graph_density=density,
        isolated_node_count=isolated,
        connected_component_count=components,
    )


def _connected_components(node_count: int, edges: npt.NDArray[np.int16]) -> int:
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    for source, target in edges.T:
        adjacency[int(source)].append(int(target))
    visited: set[int] = set()
    components = 0
    for start in range(node_count):
        if start in visited:
            continue
        components += 1
        stack = [start]
        visited.add(start)
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
    return components
