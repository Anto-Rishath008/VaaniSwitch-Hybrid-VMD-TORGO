"""Guarded loaders: split metadata are checked before source load and before forward."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

import torch
from torch.utils.data import DataLoader, Dataset

from vaaniswitch.m0_data.schemas import ManifestRecord
from vaaniswitch.m3_encoders.collate import collate_encoder_samples
from vaaniswitch.m3_encoders.dataset import EncoderRepresentationDataset
from vaaniswitch.m3_encoders.schemas import EncoderBatch, EncoderSample, SampleProvenance
from vaaniswitch.m4_fusion.losses import label_for_condition, training_pos_weight
from vaaniswitch.m4_fusion.trainer import tiny_records
from vaaniswitch.m5_experiments.ledger import Ledger
from vaaniswitch.m5_experiments.no_vmd import NoVMDCache
from vaaniswitch.m5_experiments.source import ExperimentContext
from vaaniswitch.m5_experiments.splits import NestedSplit, Role


@dataclass(frozen=True)
class OuterPermit:
    ledger: Ledger
    run_id: str
    token: str

    def verify(self, split: NestedSplit, ablation: str) -> None:
        self.ledger.verify_claim(self.run_id, self.token)
        identity = self.ledger.get(self.run_id)["identity"]
        if identity["split_hash"] != split.split_hash or identity["ablation"] != ablation:
            raise ValueError("outer permit belongs to a different experiment")


def selected_records(
    context: ExperimentContext, split: NestedSplit, role: Role, *, smoke: bool
) -> tuple[ManifestRecord, ...]:
    if smoke and (role == "outer_test" or split.outer_fold != 1):
        raise ValueError("real smoke is restricted to Fold 1 development speakers")
    records = split.records(context.predecessor.source.contract, role)
    return tiny_records(records) if smoke else records


def pos_weight(context: ExperimentContext, split: NestedSplit, role: Role, *, smoke: bool) -> float:
    if role not in {"inner_train", "development"}:
        raise ValueError("class weights accept fitting roles only")
    return training_pos_weight(
        selected_records(context, split, role, smoke=smoke), split.speakers(role)
    )


class ExperimentDataset(Dataset[EncoderSample]):
    def __init__(
        self,
        context: ExperimentContext,
        split: NestedSplit,
        role: Role,
        ablation: str,
        *,
        smoke: bool = False,
        permit: OuterPermit | None = None,
    ) -> None:
        self.context, self.split, self.role, self.ablation = context, split, role, ablation
        self.permit = permit
        self.check_permission()
        self.records = selected_records(context, split, role, smoke=smoke)
        self.loaded: list[str] = []
        self.a0 = NoVMDCache(context) if ablation == "A0" else None
        self.signed = (
            None
            if self.a0
            else EncoderRepresentationDataset(
                context.predecessor.encoder_config,
                logical_ids=[r.logical_id for r in self.records],
                source=context.predecessor.source,
            )
        )

    def check_permission(self) -> None:
        if self.role == "outer_test":
            if self.permit is None:
                raise ValueError("outer dataset blocked before selected config and final retrain")
            self.permit.verify(self.split, self.ablation)
        elif self.permit is not None:
            raise ValueError("outer permit cannot be used for fitting")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> EncoderSample:
        self.check_permission()
        record = self.records[index]
        if record.speaker_id not in self.split.speakers(self.role):
            raise ValueError("source load forbidden outside assigned speakers")
        if self.a0 is not None:
            assert record.primary_microphone is not None
            provenance = SampleProvenance(
                record.logical_id,
                record.speaker_id,
                record.condition.value,
                record.gender.value,
                record.prompt_type.value,
                record.primary_microphone.value,
                self.context.predecessor.source.contract.speaker_to_evaluation_fold[
                    record.speaker_id
                ],
            )
            sample = EncoderSample(
                torch.from_numpy(self.a0.load(record).copy()).unsqueeze(0),
                torch.zeros((1, 224, 224), dtype=torch.uint8),
                torch.zeros((1, 2)),
                torch.empty((2, 0), dtype=torch.int64),
                provenance,
            )
        else:
            assert self.signed is not None
            sample = self.signed[index]
        self.loaded.append(record.logical_id)
        return sample

    def guard(self, batch: EncoderBatch) -> None:
        self.check_permission()
        expected = {r.logical_id: r for r in self.records}
        if len(batch.provenance) != batch.mel.shape[0] or not batch.provenance:
            raise ValueError("batch provenance size mismatch")
        for item in batch.provenance:
            record = expected.get(item.logical_id)
            if (
                record is None
                or item.speaker_id != record.speaker_id
                or item.condition != record.condition.value
                or item.evaluation_fold
                != self.context.predecessor.source.contract.speaker_to_evaluation_fold[
                    record.speaker_id
                ]
            ):
                raise ValueError("batch violates assigned speaker/record partition")
            label_for_condition(item.condition)


def batches(
    dataset: ExperimentDataset, *, epoch: int = 0, smoke: bool = False
) -> Iterator[EncoderBatch]:
    loader = DataLoader(
        dataset,
        batch_size=4 if smoke else 16,
        shuffle=dataset.role in {"inner_train", "development"},
        num_workers=0,
        drop_last=False,
        collate_fn=collate_encoder_samples,
        generator=torch.Generator().manual_seed(42 + epoch),
    )
    for batch in loader:
        yield cast(EncoderBatch, batch)
