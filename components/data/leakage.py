"""Duplicate and train/evaluation leakage audits for Module 0."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from components.data.manifest import manifest_fingerprint
from components.data.schemas import AudioAuditRecord, ManifestRecord, OrphanAudio
from components.data.splits import validate_speaker_folds


class LeakageError(RuntimeError):
    """Raised when a hard scientific leakage invariant fails."""


def find_duplicate_logical_ids(records: Iterable[ManifestRecord]) -> list[str]:
    """Return canonical IDs that occur more than once."""

    counts = Counter(record.logical_id for record in records)
    return sorted(value for value, count in counts.items() if count > 1)


def find_duplicate_primary_paths(records: Iterable[ManifestRecord]) -> list[str]:
    """Return non-null primary paths assigned to multiple records."""

    counts = Counter(record.primary_audio_path for record in records if record.primary_audio_path)
    return sorted(value for value, count in counts.items() if count > 1)


def find_alternate_paths_used_as_other_primaries(
    records: Iterable[ManifestRecord],
) -> list[str]:
    """Find alternate copies incorrectly promoted by a different logical record."""

    materialized = list(records)
    primary_owners = {
        record.primary_audio_path: record.logical_id
        for record in materialized
        if record.primary_audio_path
    }
    return sorted(
        {
            record.alternate_audio_path
            for record in materialized
            if record.alternate_audio_path
            and (owner := primary_owners.get(record.alternate_audio_path)) is not None
            and owner != record.logical_id
        }
    )


def build_leakage_report(
    records: Sequence[ManifestRecord],
    audio_audits: Sequence[AudioAuditRecord],
    folds: dict[str, Any],
    orphan_audio: Sequence[OrphanAudio],
) -> dict[str, Any]:
    """Audit duplicates, microphone binding, and every outer train/eval split."""

    duplicate_logical_ids = find_duplicate_logical_ids(records)
    duplicate_primary_paths = find_duplicate_primary_paths(records)
    path_associations = _path_associations(records)
    duplicate_associated_paths = {
        path: sorted({item["logical_id"] for item in associations})
        for path, associations in path_associations.items()
        if len({item["logical_id"] for item in associations}) > 1
    }
    alternate_as_primary = find_alternate_paths_used_as_other_primaries(records)

    audits_by_path = {audit.audio_path: audit for audit in audio_audits}
    hash_groups: dict[str, list[str]] = defaultdict(list)
    for audit in audio_audits:
        if audit.sha256 is not None:
            hash_groups[audit.sha256].append(audit.audio_path)
    duplicate_hash_groups: list[dict[str, Any]] = []
    cross_speaker_hashes: list[dict[str, Any]] = []
    for digest, paths in sorted(hash_groups.items()):
        if len(paths) < 2:
            continue
        speakers = sorted(
            {
                str(association["speaker_id"])
                for path in paths
                for association in path_associations.get(path, [])
            }
        )
        logical_ids = sorted(
            {
                str(association["logical_id"])
                for path in paths
                for association in path_associations.get(path, [])
            }
        )
        group = {
            "sha256": digest,
            "paths": sorted(paths),
            "speaker_ids": speakers,
            "logical_ids": logical_ids,
            "cross_speaker": len(speakers) > 1,
        }
        duplicate_hash_groups.append(group)
        if len(speakers) > 1:
            cross_speaker_hashes.append(group)

    split_validation = validate_speaker_folds(records, folds)
    per_fold: list[dict[str, Any]] = []
    cross_fold_hash_violations: list[dict[str, Any]] = []
    for fold in folds["folds"]:
        fold_id = int(fold["fold_id"])
        train_speakers = set(fold["train_speakers"])
        evaluation_speakers = set(fold["evaluation_speakers"])
        train_records = [record for record in records if record.speaker_id in train_speakers]
        evaluation_records = [
            record for record in records if record.speaker_id in evaluation_speakers
        ]
        train_paths = _record_audio_paths(train_records)
        evaluation_paths = _record_audio_paths(evaluation_records)
        train_hashes = _content_hashes(train_paths, audits_by_path)
        evaluation_hashes = _content_hashes(evaluation_paths, audits_by_path)
        shared_hashes = sorted(train_hashes & evaluation_hashes)
        for digest in shared_hashes:
            cross_fold_hash_violations.append(
                {
                    "fold_id": fold_id,
                    "sha256": digest,
                    "paths": sorted(hash_groups[str(digest)]),
                }
            )

        train_prompt_ids = {record.prompt_id for record in train_records}
        evaluation_prompt_ids = {record.prompt_id for record in evaluation_records}
        train_transcripts = {
            record.transcript for record in train_records if record.transcript is not None
        }
        evaluation_transcripts = {
            record.transcript for record in evaluation_records if record.transcript is not None
        }
        per_fold.append(
            {
                "fold_id": fold_id,
                "train_evaluation_speaker_overlap_count": len(train_speakers & evaluation_speakers),
                "train_evaluation_logical_id_overlap_count": len(
                    {record.logical_id for record in train_records}
                    & {record.logical_id for record in evaluation_records}
                ),
                "train_evaluation_audio_path_overlap_count": len(train_paths & evaluation_paths),
                "train_evaluation_content_hash_overlap_count": len(shared_hashes),
                "shared_prompt_id_count": len(train_prompt_ids & evaluation_prompt_ids),
                "shared_transcript_count": len(train_transcripts & evaluation_transcripts),
                "shared_prompt_policy": ("expected_and_allowed_across_independent_speakers"),
            }
        )

    violations = {
        "duplicate_logical_ids": duplicate_logical_ids,
        "duplicate_primary_audio_paths": duplicate_primary_paths,
        "audio_paths_assigned_to_multiple_logical_ids": duplicate_associated_paths,
        "alternate_paths_used_as_other_primaries": alternate_as_primary,
        "cross_speaker_duplicate_content_hashes": cross_speaker_hashes,
        "cross_fold_duplicate_content_hashes": cross_fold_hash_violations,
    }
    violation_count = sum(len(value) for value in violations.values())
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "PASS" if violation_count == 0 else "FAIL",
        "source": {
            "manifest_sha256": manifest_fingerprint(records),
            "speaker_folds_sha256": speaker_folds_fingerprint(folds),
        },
        "policy": {
            "split_unit": "speaker_id",
            "logical_unit": "speaker_id+session_id+prompt_id",
            "microphone_copies": "bound_to_one_logical_record",
            "repeated_prompts_across_speakers": "allowed_and_reported",
            "hard_failure_on_speaker_or_microphone_leakage": True,
        },
        "manifest_audit": {
            "logical_record_count": len(records),
            "missing_transcript_count": sum(record.transcript is None for record in records),
            "missing_transcript_logical_ids": sorted(
                record.logical_id for record in records if record.transcript is None
            ),
            "missing_primary_audio_count": sum(
                record.primary_audio_path is None for record in records
            ),
            "missing_primary_audio_logical_ids": sorted(
                record.logical_id for record in records if record.primary_audio_path is None
            ),
            "duplicate_logical_id_count": len(duplicate_logical_ids),
            "duplicate_primary_path_count": len(duplicate_primary_paths),
        },
        "raw_audio_duplicate_audit": {
            "audited_wav_count": len(audio_audits),
            "unique_sha256_count": len(hash_groups),
            "duplicate_hash_group_count": len(duplicate_hash_groups),
            "duplicate_hash_groups": duplicate_hash_groups,
            "cross_speaker_duplicate_hash_group_count": len(cross_speaker_hashes),
        },
        "orphan_audio_audit": {
            "orphan_audio_path_count": len(orphan_audio),
            "orphan_logical_stem_count": len(
                {(item.speaker_id, item.session_id, item.prompt_id) for item in orphan_audio}
            ),
            "by_reason": dict(sorted(Counter(item.reason for item in orphan_audio).items())),
        },
        "split_validation": split_validation,
        "per_fold_audit": per_fold,
        "violations": violations,
        "violation_count": violation_count,
    }
    assert_no_leakage(report)
    return report


def assert_no_leakage(report: dict[str, Any]) -> None:
    """Abort the pipeline whenever a hard leakage finding is present."""

    violations = report.get("violations")
    if not isinstance(violations, dict):
        raise LeakageError("Module 0 leakage report has no violations mapping")
    calculated_count = 0
    for finding in violations.values():
        if not isinstance(finding, (dict, list)):
            raise LeakageError("Module 0 leakage finding must be a list or mapping")
        calculated_count += len(finding)
    reported_count = report.get("violation_count")
    expected_status = "PASS" if calculated_count == 0 else "FAIL"
    if reported_count != calculated_count or report.get("status") != expected_status:
        raise LeakageError(
            "Module 0 leakage report is internally inconsistent: "
            f"status={report.get('status')!r}, reported={reported_count!r}, "
            f"calculated={calculated_count}"
        )
    if calculated_count:
        raise LeakageError(f"Module 0 leakage audit failed: {violations}")


def speaker_folds_fingerprint(folds: dict[str, Any]) -> str:
    """Hash the deterministic speaker-fold document used by leakage checks."""

    payload = json.dumps(
        folds,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _path_associations(
    records: Iterable[ManifestRecord],
) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        for role, path in (
            ("primary", record.primary_audio_path),
            ("alternate", record.alternate_audio_path),
        ):
            if path:
                result[path].append(
                    {
                        "logical_id": record.logical_id,
                        "speaker_id": record.speaker_id,
                        "role": role,
                    }
                )
    return result


def _record_audio_paths(records: Iterable[ManifestRecord]) -> set[str]:
    result: set[str] = set()
    for record in records:
        if record.primary_audio_path:
            result.add(record.primary_audio_path)
        if record.alternate_audio_path:
            result.add(record.alternate_audio_path)
    return result


def _content_hashes(
    paths: Iterable[str],
    audits_by_path: dict[str, AudioAuditRecord],
) -> set[str]:
    hashes: set[str] = set()
    for path in paths:
        audit = audits_by_path.get(path)
        if audit is not None and audit.sha256 is not None:
            hashes.add(audit.sha256)
    return hashes
