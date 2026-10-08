"""Variable-size visibility-graph collation with explicit valid-node masks."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from vaaniswitch.m3_encoders.schemas import EncoderBatch, EncoderSample


def collate_encoder_samples(samples: Sequence[EncoderSample]) -> EncoderBatch:
    if not samples:
        raise ValueError("cannot collate an empty sample sequence")
    maximum_nodes = max(sample.graph_nodes.shape[0] for sample in samples)
    batch_size = len(samples)
    graph_nodes = torch.zeros((batch_size, maximum_nodes, 2), dtype=torch.float32)
    graph_mask = torch.zeros((batch_size, maximum_nodes), dtype=torch.bool)
    for index, sample in enumerate(samples):
        node_count = sample.graph_nodes.shape[0]
        if (
            sample.mel.shape != (1, 224, 224)
            or sample.recurrence.shape != (1, 224, 224)
            or sample.graph_nodes.ndim != 2
            or sample.graph_nodes.shape[1] != 2
            or not 1 <= node_count <= 256
            or sample.graph_edges.ndim != 2
            or sample.graph_edges.shape[0] != 2
        ):
            raise ValueError("sample tensor shapes violate the Module 3 contract")
        graph_nodes[index, :node_count] = sample.graph_nodes
        graph_mask[index, :node_count] = True
    return EncoderBatch(
        mel=torch.stack([sample.mel for sample in samples]),
        recurrence=torch.stack([sample.recurrence for sample in samples]),
        graph_nodes=graph_nodes,
        graph_node_mask=graph_mask,
        graph_edges=tuple(sample.graph_edges for sample in samples),
        provenance=tuple(sample.provenance for sample in samples),
    )
