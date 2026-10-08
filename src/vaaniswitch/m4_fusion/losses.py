"""Canonical Module 0 labels and training-partition-only weighted BCE."""

from collections.abc import Sequence

import torch
from torch import nn

from vaaniswitch.m0_data.schemas import Condition, ManifestRecord
from vaaniswitch.m3_encoders.schemas import EncoderBatch

LABEL_MAPPING = {Condition.CONTROL.value: 0, Condition.DYSARTHRIC.value: 1}


def label_for_condition(condition: str) -> int:
    if condition not in LABEL_MAPPING:
        raise ValueError(f"unknown condition: {condition}")
    return LABEL_MAPPING[condition]


def batch_labels(batch: EncoderBatch) -> torch.Tensor:
    return torch.tensor(
        [label_for_condition(item.condition) for item in batch.provenance],
        dtype=torch.float32,
        device=batch.mel.device,
    )


def training_pos_weight(
    records: Sequence[ManifestRecord],
    train_speakers: tuple[str, ...],
) -> float:
    """Reject mixed partitions instead of silently weighting validation or test records."""
    if not records or any(r.speaker_id not in train_speakers for r in records):
        raise ValueError("pos_weight accepts only nonempty training records")
    positive = sum(label_for_condition(r.condition.value) for r in records)
    negative = len(records) - positive
    if min(positive, negative) == 0:
        raise ValueError("training records must contain both classes")
    return negative / positive


def make_loss(pos_weight: float, device: torch.device | str = "cpu") -> nn.BCEWithLogitsLoss:
    weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    if not torch.isfinite(weight) or weight <= 0:
        raise ValueError("pos_weight must be finite and positive")
    return nn.BCEWithLogitsLoss(pos_weight=weight)
