"""Canonical TORGO manifest construction and deterministic serialization."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from components.data.config import TorgoConfig
from components.data.prompt_types import classify_prompt
from components.data.schemas import (
    AudioAuditRecord,
    AudioStructuralStatus,
    DiscoveryResult,
    ManifestRecord,
    Microphone,
    PrimarySelectionReason,
    PromptType,
    UsabilityStatus,
)


class ManifestError(RuntimeError):
    """Raised when canonical manifest construction is ambiguous or unsafe."""


@dataclass(frozen=True, slots=True)
class MicrophoneSelection:
    """Validity-aware primary/alternate decision with structural evidence."""

    primary_path: Path | None
    alternate_path: Path | None
    primary_microphone: Microphone | None
    primary_audit: AudioAuditRecord | None
    head_audio_status: AudioStructuralStatus
    head_audio_issues: tuple[str, ...]
    array_audio_status: AudioStructuralStatus
    array_audio_issues: tuple[str, ...]
    reason: PrimarySelectionReason


def select_microphones(
    head_path: Path | None,
    head_audit: AudioAuditRecord | None,
    array_path: Path | None,
    array_audit: AudioAuditRecord | None,
) -> MicrophoneSelection:
    """Choose HEAD when structurally usable, otherwise a structurally usable ARRAY."""

    head_status, head_issues = _microphone_evidence("head", head_path, head_audit)
    array_status, array_issues = _microphone_evidence("array", array_path, array_audit)
    if head_status is AudioStructuralStatus.USABLE:
        return MicrophoneSelection(
            primary_path=head_path,
            alternate_path=array_path,
            primary_microphone=Microphone.HEAD,
            primary_audit=head_audit,
            head_audio_status=head_status,
            head_audio_issues=head_issues,
            array_audio_status=array_status,
            array_audio_issues=array_issues,
            reason=PrimarySelectionReason.HEAD_STRUCTURALLY_USABLE,
        )
    if array_status is AudioStructuralStatus.USABLE:
        return MicrophoneSelection(
            primary_path=array_path,
            alternate_path=head_path,
            primary_microphone=Microphone.ARRAY,
            primary_audit=array_audit,
            head_audio_status=head_status,
            head_audio_issues=head_issues,
            array_audio_status=array_status,
            array_audio_issues=array_issues,
            reason=(
                PrimarySelectionReason.HEAD_UNUSABLE_ARRAY_FALLBACK
                if head_status is AudioStructuralStatus.UNUSABLE
                else PrimarySelectionReason.ARRAY_STRUCTURALLY_USABLE_HEAD_MISSING
            ),
        )
    return MicrophoneSelection(
        primary_path=None,
        alternate_path=None,
        primary_microphone=None,
        primary_audit=None,
        head_audio_status=head_status,
        head_audio_issues=head_issues,
        array_audio_status=array_status,
        array_audio_issues=array_issues,
        reason=PrimarySelectionReason.NO_STRUCTURALLY_USABLE_AUDIO,
    )


def build_manifest(
    discovery: DiscoveryResult,
    audio_audits: Sequence[AudioAuditRecord],
    config: TorgoConfig,
) -> tuple[ManifestRecord, ...]:
    """Join prompt metadata to validity-aware, head-preferred WAV audits."""

    audits_by_path = {item.audio_path: item for item in audio_audits}
    if len(audits_by_path) != len(audio_audits):
        raise ManifestError("audio audit contains duplicate paths")

    records: list[ManifestRecord] = []
    for utterance in discovery.utterances:
        head_relative = (
            config.relative_path(utterance.head_mic_path) if utterance.head_mic_path else None
        )
        array_relative = (
            config.relative_path(utterance.array_mic_path) if utterance.array_mic_path else None
        )
        head_audit = _required_audit(head_relative, audits_by_path)
        array_audit = _required_audit(array_relative, audits_by_path)
        selection = select_microphones(
            utterance.head_mic_path,
            head_audit,
            utterance.array_mic_path,
            array_audit,
        )
        primary_relative = (
            config.relative_path(selection.primary_path) if selection.primary_path else None
        )
        audit = selection.primary_audit

        prompt_type = classify_prompt(
            utterance.transcript,
            utterance.transcript_quality_flags,
        )
        quality_flags = list(utterance.transcript_quality_flags)
        exclusion_reasons: list[str] = []
        for microphone, microphone_status, issues in (
            ("head", selection.head_audio_status, selection.head_audio_issues),
            ("array", selection.array_audio_status, selection.array_audio_issues),
        ):
            if microphone_status is AudioStructuralStatus.UNUSABLE:
                quality_flags.extend(f"{microphone}_unusable_{issue}" for issue in issues)
        if selection.reason is PrimarySelectionReason.HEAD_UNUSABLE_ARRAY_FALLBACK:
            quality_flags.append(selection.reason.value)
        if utterance.transcript is None:
            quality_flags.append("missing_or_unreadable_transcript")
            exclusion_reasons.append("missing_or_unreadable_transcript")
        if selection.primary_path is None:
            structural_reason = (
                "missing_primary_audio"
                if selection.head_audio_status is AudioStructuralStatus.MISSING
                and selection.array_audio_status is AudioStructuralStatus.MISSING
                else "no_structurally_usable_audio"
            )
            quality_flags.append(structural_reason)
            exclusion_reasons.append(structural_reason)
        elif audit is not None:
            quality_flags.extend(audit.quality_flags)

        quality_flags = sorted(set(quality_flags))
        exclusion_reasons = list(dict.fromkeys(exclusion_reasons))
        if exclusion_reasons:
            record_status = UsabilityStatus.EXCLUDED
            exclusion_reason = "|".join(exclusion_reasons)
        elif config.quality.review_flags.intersection(quality_flags):
            record_status = UsabilityStatus.REVIEW
            exclusion_reason = None
        else:
            record_status = UsabilityStatus.USABLE
            exclusion_reason = None

        normal_speech_eligible = record_status is not UsabilityStatus.EXCLUDED and prompt_type in {
            PromptType.WORD,
            PromptType.SENTENCE,
        }
        record = ManifestRecord(
            logical_id=utterance.logical_id,
            speaker_id=utterance.speaker_id,
            speaker_group=utterance.speaker_group,
            condition=utterance.condition,
            gender=utterance.gender,
            session_id=utterance.session_id,
            prompt_id=utterance.prompt_id,
            transcript=utterance.transcript,
            prompt_path=config.relative_path(utterance.prompt_path),
            head_mic_path=head_relative,
            head_audio_status=selection.head_audio_status,
            head_audio_issues=selection.head_audio_issues,
            array_mic_path=array_relative,
            array_audio_status=selection.array_audio_status,
            array_audio_issues=selection.array_audio_issues,
            primary_audio_path=primary_relative,
            alternate_audio_path=(
                config.relative_path(selection.alternate_path) if selection.alternate_path else None
            ),
            primary_microphone=selection.primary_microphone,
            primary_selection_reason=selection.reason,
            sample_rate=audit.sample_rate if audit else None,
            channels=audit.channels if audit else None,
            sample_width_bytes=audit.sample_width_bytes if audit else None,
            bit_depth=audit.bit_depth if audit else None,
            frame_count=audit.frame_count if audit else None,
            duration_seconds=audit.duration_seconds if audit else None,
            file_size=audit.file_size if audit else None,
            primary_audio_sha256=audit.sha256 if audit else None,
            prompt_type=prompt_type,
            normal_speech_eligible=normal_speech_eligible,
            usability_status=record_status,
            exclusion_reason=exclusion_reason,
            quality_flags=tuple(quality_flags),
        )
        record.validate()
        records.append(record)

    _validate_unique_records(records)
    return tuple(records)


def _required_audit(
    relative_path: str | None,
    audits_by_path: Mapping[str, AudioAuditRecord],
) -> AudioAuditRecord | None:
    if relative_path is None:
        return None
    audit = audits_by_path.get(relative_path)
    if audit is None:
        raise ManifestError(f"missing audio audit for microphone path: {relative_path}")
    return audit


def _microphone_evidence(
    microphone: str,
    path: Path | None,
    audit: AudioAuditRecord | None,
) -> tuple[AudioStructuralStatus, tuple[str, ...]]:
    if path is None:
        if audit is not None:
            raise ManifestError(f"{microphone} audit exists without a microphone path")
        return AudioStructuralStatus.MISSING, ("missing_path",)
    if audit is None:
        raise ManifestError(f"{microphone} path exists without an audio audit: {path}")
    issues = audit.structural_issues
    if issues:
        return AudioStructuralStatus.UNUSABLE, issues
    return AudioStructuralStatus.USABLE, ()


def manifest_jsonl_text(records: Sequence[ManifestRecord]) -> str:
    """Return byte-stable JSONL content for records in canonical order."""

    return "".join(
        json.dumps(
            record.to_dict(),
            ensure_ascii=False,
            sort_keys=False,
            separators=(",", ":"),
        )
        + "\n"
        for record in records
    )


def manifest_fingerprint(records: Sequence[ManifestRecord]) -> str:
    """Hash the exact deterministic JSONL representation used on disk."""

    return hashlib.sha256(manifest_jsonl_text(records).encode("utf-8")).hexdigest()


def write_manifest_jsonl(path: Path, records: Sequence[ManifestRecord]) -> None:
    """Atomically write the canonical JSONL manifest."""

    _atomic_write_text(path, manifest_jsonl_text(records))


def write_manifest_csv(path: Path, records: Sequence[ManifestRecord]) -> None:
    """Atomically write a spreadsheet-compatible view of the manifest."""

    rows = [record.to_dict(csv_safe=True) for record in records]
    write_csv(path, ManifestRecord.CSV_FIELDS, rows)


def load_manifest_jsonl(path: Path) -> tuple[ManifestRecord, ...]:
    """Load and validate a previously generated JSONL manifest."""

    records: list[ManifestRecord] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("JSON value is not an object")
                records.append(ManifestRecord.from_dict(value))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ManifestError(
                    f"invalid manifest record at {path}:{line_number}: {exc}"
                ) from exc
    _validate_unique_records(records)
    return tuple(records)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write deterministic, human-readable JSON with a trailing newline."""

    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(path, text)


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Atomically write ordered CSV rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(fieldnames),
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _validate_unique_records(records: Sequence[ManifestRecord]) -> None:
    logical_ids: set[str] = set()
    primary_paths: set[str] = set()
    duplicate_logical_ids: set[str] = set()
    duplicate_primary_paths: set[str] = set()
    for record in records:
        if record.logical_id in logical_ids:
            duplicate_logical_ids.add(record.logical_id)
        logical_ids.add(record.logical_id)
        if record.primary_audio_path is None:
            continue
        if record.primary_audio_path in primary_paths:
            duplicate_primary_paths.add(record.primary_audio_path)
        primary_paths.add(record.primary_audio_path)
    if duplicate_logical_ids:
        raise ManifestError(f"duplicate logical IDs: {sorted(duplicate_logical_ids)}")
    if duplicate_primary_paths:
        raise ManifestError(f"duplicate primary audio paths: {sorted(duplicate_primary_paths)}")
