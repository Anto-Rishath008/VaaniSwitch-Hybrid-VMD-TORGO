"""Metadata-only nested roles; outer samples require a separate evaluation permit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from vaaniswitch.m0_data.schemas import ManifestRecord
from vaaniswitch.m3_encoders.source import SignedSourceContract
from vaaniswitch.m4_fusion.splits import SpeakerSplit, create_inner_split
from vaaniswitch.m5_experiments.config import digest

Role = Literal["inner_train", "inner_validation", "development", "outer_test"]


@dataclass(frozen=True)
class NestedSplit:
    inner: SpeakerSplit

    @property
    def outer_fold(self) -> int:
        return self.inner.outer_test_fold

    @property
    def split_hash(self) -> str:
        return self.inner.split_hash

    @property
    def identity(self) -> dict[str, Any]:
        return self.inner.identity

    def speakers(self, role: Role) -> tuple[str, ...]:
        if role == "inner_train":
            return self.inner.train_speakers
        if role == "inner_validation":
            return self.inner.validation_speakers
        if role == "development":
            return tuple(sorted((*self.inner.train_speakers, *self.inner.validation_speakers)))
        if role == "outer_test":
            return self.inner.test_speakers
        raise ValueError("unknown nested role")

    def records(self, contract: SignedSourceContract, role: Role) -> tuple[ManifestRecord, ...]:
        allowed = self.speakers(role)
        if role != "outer_test" and set(allowed) & set(self.inner.test_speakers):
            raise ValueError("outer speakers entered development")
        return tuple(
            sorted(
                (r for r in contract.records if r.speaker_id in allowed), key=lambda r: r.logical_id
            )
        )


def nested_splits(contract: SignedSourceContract, fingerprint: str) -> tuple[NestedSplit, ...]:
    if set(contract.speaker_to_evaluation_fold.values()) != {1, 2, 3}:
        raise ValueError("expected three signed outer folds")
    result = tuple(
        NestedSplit(
            create_inner_split(
                contract, outer_test_fold=fold, seed=42, folds_fingerprint=fingerprint
            )
        )
        for fold in (1, 2, 3)
    )
    for split in result:
        if [
            len(split.speakers(role)) for role in ("inner_train", "inner_validation", "outer_test")
        ] != [8, 2, 5]:
            raise ValueError("expected 8/2/5 speaker split")
        # All associated microphones and duplicate waveform content remain grouped.
        roles = [
            split.records(contract, role)
            for role in ("inner_train", "inner_validation", "outer_test")
        ]
        for attribute in ("logical_id", "primary_audio_path", "primary_audio_sha256"):
            sets = [{getattr(r, attribute) for r in records} - {None} for records in roles]
            if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
                raise ValueError(f"cross-partition leakage: {attribute}")
    return result


def record_hash(records: tuple[ManifestRecord, ...]) -> str:
    return digest([r.logical_id for r in records])
