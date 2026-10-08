"""Deterministic inner speaker splits; no outer-test dataset or evaluation API."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Literal

from vaaniswitch.m0_data.schemas import ManifestRecord
from vaaniswitch.m3_encoders.schemas import EncoderBatch
from vaaniswitch.m3_encoders.source import SignedSourceContract
from vaaniswitch.m4_fusion.config import semantic_hash

Partition = Literal["train", "validation"]


@dataclass(frozen=True, slots=True)
class SpeakerSplit:
    outer_test_fold: int
    seed: int
    train_speakers: tuple[str, ...]
    validation_speakers: tuple[str, ...]
    test_speakers: tuple[str, ...]
    folds_fingerprint: str
    manifest_fingerprint: str
    method: str = "class_stratified_speaker_sha256_v1"

    @property
    def identity(self) -> dict[str, Any]:
        # JSON-compatible identity is identical before and after checkpoint serialization.
        return {
            **asdict(self),
            "train_speakers": list(self.train_speakers),
            "validation_speakers": list(self.validation_speakers),
            "test_speakers": list(self.test_speakers),
        }

    @property
    def split_hash(self) -> str:
        return semantic_hash(self.identity)

    def speakers(self, partition: Partition) -> tuple[str, ...]:
        if partition == "train":
            return self.train_speakers
        if partition == "validation":
            return self.validation_speakers
        raise ValueError("Module 4 exposes only train and validation partitions")


def create_inner_split(
    contract: SignedSourceContract,
    *,
    outer_test_fold: int,
    seed: int,
    folds_fingerprint: str,
) -> SpeakerSplit:
    if type(outer_test_fold) is not int or outer_test_fold not in set(
        contract.speaker_to_evaluation_fold.values()
    ):
        raise ValueError("outer test fold is absent from signed folds")
    if seed != 42:
        raise ValueError("Module 4 uses one baseline seed: 42")
    mapping = contract.speaker_to_evaluation_fold
    test = tuple(sorted(s for s, fold in mapping.items() if fold == outer_test_fold))
    development = set(mapping) - set(test)
    conditions: dict[str, str] = {}
    for record in contract.records:
        previous = conditions.setdefault(record.speaker_id, record.condition.value)
        if previous != record.condition.value:
            raise ValueError("speaker has inconsistent condition labels")
    validation: list[str] = []
    for condition in ("control", "dysarthric"):
        candidates = sorted(s for s in development if conditions[s] == condition)
        if len(candidates) < 2:
            raise ValueError("each development class needs at least two speakers for inner split")

        def rank(speaker: str, condition: str = condition) -> tuple[str, str]:
            payload = f"{seed}|{outer_test_fold}|{condition}|{speaker}".encode("ascii")
            return hashlib.sha256(payload).hexdigest(), speaker

        validation.append(min(candidates, key=rank))
    train = tuple(sorted(development - set(validation)))
    result = SpeakerSplit(
        outer_test_fold,
        seed,
        train,
        tuple(sorted(validation)),
        test,
        folds_fingerprint,
        contract.manifest_fingerprint,
    )
    if set(train) & set(validation) or (set(train) | set(validation)) & set(test):
        raise ValueError("speaker overlap in inner split")
    return result


def partition_records(
    contract: SignedSourceContract,
    split: SpeakerSplit,
    partition: Partition,
) -> tuple[ManifestRecord, ...]:
    speakers = set(split.speakers(partition))
    if speakers & set(split.test_speakers):
        raise ValueError("reserved outer-test speakers cannot enter a development dataset")
    return tuple(
        sorted(
            (r for r in contract.records if r.speaker_id in speakers),
            key=lambda r: r.logical_id,
        )
    )


def guard_batch(
    batch: EncoderBatch,
    contract: SignedSourceContract,
    split: SpeakerSplit,
    partition: Partition,
) -> None:
    allowed = set(split.speakers(partition))
    if len(batch.provenance) != batch.mel.shape[0] or not batch.provenance:
        raise ValueError("batch provenance length is invalid")
    for item in batch.provenance:
        record = contract.record_by_id.get(item.logical_id)
        if (
            item.speaker_id not in allowed
            or item.speaker_id in split.test_speakers
            or item.evaluation_fold == split.outer_test_fold
            or record is None
            or record.speaker_id != item.speaker_id
            or record.condition.value != item.condition
            or contract.speaker_to_evaluation_fold[item.speaker_id] != item.evaluation_fold
        ):
            raise ValueError("batch violates signed development partition; forward forbidden")


def split_audit(contract: SignedSourceContract, split: SpeakerSplit) -> dict[str, Any]:
    result: dict[str, Any] = {
        **split.identity,
        "split_hash": split.split_hash,
        "status": "PASS",
        "speaker_overlap_count": 0,
        "outer_test_forward_count": 0,
        "outer_test_policy": "metadata identity only; no dataset/forward/predictions/metrics",
    }
    for partition in ("train", "validation"):
        records = partition_records(contract, split, partition)
        result[partition] = {
            "record_count": len(records),
            "condition_counts": dict(Counter(r.condition.value for r in records)),
            "record_identity_hash": semantic_hash([r.logical_id for r in records]),
        }
    result["reserved_test_record_count"] = sum(
        r.speaker_id in split.test_speakers for r in contract.records
    )
    return result
