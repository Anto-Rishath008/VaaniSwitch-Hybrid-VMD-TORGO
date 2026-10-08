"""PyTorch dataset over integrity-checked Module 2 representations."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset

from vaaniswitch.m0_data.schemas import ManifestRecord
from vaaniswitch.m3_encoders.config import Module3Config
from vaaniswitch.m3_encoders.schemas import EncoderSample, SampleProvenance
from vaaniswitch.m3_encoders.source import SignedRepresentationSource


class EncoderRepresentationDataset(Dataset[EncoderSample]):
    """Load deterministic arrays on demand without caching learned tokens."""

    def __init__(
        self,
        config: Module3Config,
        *,
        logical_ids: Sequence[str] | None = None,
        source: SignedRepresentationSource | None = None,
    ) -> None:
        self.config = config
        self.source = source or SignedRepresentationSource(config)
        if logical_ids is None:
            self.records = self.source.contract.records
        else:
            chosen: list[ManifestRecord] = []
            seen: set[str] = set()
            for logical_id in logical_ids:
                if logical_id in seen:
                    raise ValueError(f"duplicate requested logical ID: {logical_id}")
                record = self.source.contract.record_by_id.get(logical_id)
                if record is None:
                    raise ValueError(f"requested record is outside signed CORE: {logical_id}")
                seen.add(logical_id)
                chosen.append(record)
            if not chosen:
                raise ValueError("dataset selection must not be empty")
            self.records = tuple(chosen)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> EncoderSample:
        record = self.records[index]
        entry = self.source.load(record.logical_id)
        primary = record.primary_microphone
        if primary is None:
            raise ValueError(f"signed CORE record has no primary microphone: {record.logical_id}")
        provenance = SampleProvenance(
            logical_id=record.logical_id,
            speaker_id=record.speaker_id,
            condition=record.condition.value,
            gender=record.gender.value,
            prompt_type=record.prompt_type.value,
            primary_microphone=primary.value,
            evaluation_fold=self.source.contract.speaker_to_evaluation_fold[record.speaker_id],
        )
        return EncoderSample(
            mel=torch.from_numpy(entry.mel.copy()).unsqueeze(0),
            recurrence=torch.from_numpy(entry.recurrence.copy()).unsqueeze(0),
            graph_nodes=torch.from_numpy(entry.graph_nodes.copy()),
            graph_edges=torch.from_numpy(entry.graph_edges.astype("int64", copy=True)),
            provenance=provenance,
        )
