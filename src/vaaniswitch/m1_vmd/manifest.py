"""Strict Module 0 manifest consumption for Module 1."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from vaaniswitch.m0_data.manifest import load_manifest_jsonl
from vaaniswitch.m0_data.schemas import Condition, ManifestRecord, PromptType, UsabilityStatus

SubsetName = Literal["core", "extended"]


class ManifestInterfaceError(ValueError):
    """Raised when Module 0 outputs cannot be consumed safely."""


@dataclass(frozen=True, slots=True)
class ManifestSource:
    """Validated records and the fingerprint binding every cache entry."""

    path: Path
    fingerprint: str
    records: tuple[ManifestRecord, ...]


def load_manifest_source(path: Path) -> ManifestSource:
    """Load the strict Module 0 schema and fingerprint the exact JSONL bytes."""

    if not path.is_file():
        raise ManifestInterfaceError(f"manifest does not exist: {path}")
    fingerprint = _sha256_file(path)
    records = load_manifest_jsonl(path)
    if not records:
        raise ManifestInterfaceError("manifest contains no logical records")
    return ManifestSource(path=path, fingerprint=fingerprint, records=records)


def select_subset(
    records: tuple[ManifestRecord, ...],
    subset: str,
) -> tuple[ManifestRecord, ...]:
    """Apply the signed-off Module 0 CORE or EXTENDED policy."""

    if subset == "core":
        allowed_statuses = {UsabilityStatus.USABLE}
    elif subset == "extended":
        allowed_statuses = {UsabilityStatus.USABLE, UsabilityStatus.REVIEW}
    else:
        raise ManifestInterfaceError(f"unsupported subset: {subset!r}")

    allowed_prompts = {PromptType.WORD, PromptType.SENTENCE}
    selected = tuple(
        record
        for record in records
        if record.normal_speech_eligible
        and record.usability_status in allowed_statuses
        and record.prompt_type in allowed_prompts
    )
    for record in selected:
        if (
            record.usability_status is UsabilityStatus.EXCLUDED
            or record.primary_audio_path is None
            or record.primary_audio_sha256 is None
        ):
            raise ManifestInterfaceError(
                f"subset admitted an unprocessable logical record: {record.logical_id}"
            )
    return selected


def select_limited_records(
    records: tuple[ManifestRecord, ...],
    limit: int | None,
) -> tuple[ManifestRecord, ...]:
    """Choose a deterministic, condition-balanced smoke subset when limited."""

    if limit is None or limit >= len(records):
        return records
    if limit <= 0:
        raise ManifestInterfaceError("limit must be positive")

    by_speaker: dict[str, list[ManifestRecord]] = defaultdict(list)
    speaker_condition: dict[str, Condition] = {}
    for record in records:
        by_speaker[record.speaker_id].append(record)
        speaker_condition[record.speaker_id] = record.condition
    for speaker_records in by_speaker.values():
        speaker_records.sort(key=lambda item: item.logical_id)

    dysarthric = sorted(
        speaker
        for speaker, condition in speaker_condition.items()
        if condition is Condition.DYSARTHRIC
    )
    control = sorted(
        speaker
        for speaker, condition in speaker_condition.items()
        if condition is Condition.CONTROL
    )
    speaker_order: list[str] = []
    for index in range(max(len(dysarthric), len(control))):
        if index < len(dysarthric):
            speaker_order.append(dysarthric[index])
        if index < len(control):
            speaker_order.append(control[index])

    chosen: list[ManifestRecord] = []
    chosen_ids: set[str] = set()
    for speaker in speaker_order:
        record = by_speaker[speaker][0]
        chosen.append(record)
        chosen_ids.add(record.logical_id)
        if len(chosen) == limit:
            return tuple(chosen)
    for record in records:
        if record.logical_id in chosen_ids:
            continue
        chosen.append(record)
        if len(chosen) == limit:
            break
    return tuple(chosen)


def resolve_primary_audio(record: ManifestRecord, repo_root: Path) -> Path:
    """Resolve only Module 0's primary path and reject path escape."""

    if record.primary_audio_path is None:
        raise ManifestInterfaceError(f"record has no primary audio: {record.logical_id}")
    relative = Path(record.primary_audio_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ManifestInterfaceError(
            f"primary path is not repository-relative: {record.primary_audio_path}"
        )
    resolved = (repo_root / relative).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ManifestInterfaceError(
            f"primary path escapes repository: {record.primary_audio_path}"
        ) from exc
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
