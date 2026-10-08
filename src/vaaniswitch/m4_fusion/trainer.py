"""Bounded CPU training and validation; test data never enter a loader."""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from typing import Any, cast

import torch
from torch.utils.data import DataLoader

from vaaniswitch.m0_data.schemas import ManifestRecord
from vaaniswitch.m3_encoders.collate import collate_encoder_samples
from vaaniswitch.m3_encoders.dataset import EncoderRepresentationDataset
from vaaniswitch.m3_encoders.schemas import EncoderBatch
from vaaniswitch.m4_fusion.losses import batch_labels, make_loss
from vaaniswitch.m4_fusion.metrics import binary_metrics
from vaaniswitch.m4_fusion.model import DysarthriaClassifier
from vaaniswitch.m4_fusion.source import DevelopmentContext
from vaaniswitch.m4_fusion.splits import Partition, guard_batch, partition_records


def make_optimizer(model: DysarthriaClassifier, context: DevelopmentContext) -> torch.optim.AdamW:
    model.verify_frozen_backbone()
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=context.config.training.learning_rate,
        weight_decay=context.config.training.weight_decay,
    )


def tiny_records(
    records: Sequence[ManifestRecord], *, count: int = 4
) -> tuple[ManifestRecord, ...]:
    """Fixed train/validation-only smoke selection, first covering condition and gender."""
    selected: list[ManifestRecord] = []
    strata: set[tuple[str, str]] = set()
    for record in records:
        stratum = (record.condition.value, record.gender.value)
        if stratum not in strata:
            strata.add(stratum)
            selected.append(record)
    selected = selected[:count]
    for record in records:
        if len(selected) >= count:
            break
        if record not in selected:
            selected.append(record)
    if {r.condition.value for r in selected} != {"control", "dysarthric"}:
        raise ValueError("tiny panel must contain both classes")
    return tuple(selected)


def selected_records(
    context: DevelopmentContext,
    partition: Partition,
    *,
    smoke: bool,
) -> tuple[ManifestRecord, ...]:
    records = partition_records(context.source.contract, context.split, partition)
    return tiny_records(records) if smoke else records


def batches(
    context: DevelopmentContext,
    partition: Partition,
    *,
    smoke: bool,
    epoch: int = 0,
) -> Iterator[EncoderBatch]:
    records = selected_records(context, partition, smoke=smoke)
    dataset = EncoderRepresentationDataset(
        context.encoder_config,
        logical_ids=[r.logical_id for r in records],
        source=context.source,
    )
    generator = torch.Generator().manual_seed(context.config.training.seed + epoch)
    loader = DataLoader(
        dataset,
        batch_size=min(context.config.training.batch_size, 4)
        if smoke
        else (context.config.training.batch_size),
        shuffle=partition == "train",
        num_workers=0,
        drop_last=False,
        collate_fn=collate_encoder_samples,
        generator=generator,
    )
    for batch in loader:
        yield cast(EncoderBatch, batch)


def finite_gradients(model: DysarthriaClassifier) -> None:
    model.verify_frozen_backbone()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and (
            parameter.grad is None or not torch.isfinite(parameter.grad).all()
        ):
            raise ValueError(f"missing or non-finite gradient: {name}")


def run_epoch(
    model: DysarthriaClassifier,
    context: DevelopmentContext,
    partition: Partition,
    *,
    pos_weight: float,
    smoke: bool,
    epoch: int = 0,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    if (partition == "train") != (optimizer is not None):
        raise ValueError("optimizer is required only for the training partition")
    context.split.speakers(partition)
    model.train(optimizer is not None)
    criterion = make_loss(pos_weight)
    total_loss, count = 0.0, 0
    labels: list[int] = []
    probabilities: list[float] = []
    start = time.perf_counter()
    for batch in batches(context, partition, smoke=smoke, epoch=epoch):
        guard_batch(batch, context.source.contract, context.split, partition)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            output = model(batch)
            targets = batch_labels(batch)
            loss = criterion(output.logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("non-finite BCE loss")
            if optimizer is not None:
                loss.backward()
                finite_gradients(model)
                optimizer.step()
        size = len(batch.provenance)
        total_loss += float(loss.detach()) * size
        count += size
        if partition == "validation":
            labels.extend(int(value) for value in targets.tolist())
            probabilities.extend(output.probabilities.detach().tolist())
    if count == 0:
        raise ValueError("empty development partition")
    result: dict[str, Any] = {
        "loss": total_loss / count,
        "record_count": count,
        "elapsed_seconds": time.perf_counter() - start,
    }
    if partition == "validation":
        result.update(binary_metrics(labels, probabilities))
    return result
