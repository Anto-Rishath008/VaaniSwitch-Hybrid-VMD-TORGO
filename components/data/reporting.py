"""Deterministic Module 0 dataset, quality, coverage, and exclusion reports."""

from __future__ import annotations

import hashlib
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from components.data.config import DownstreamSubsetConfig, TorgoConfig
from components.data.schemas import (
    AudioAuditRecord,
    DiscoveryResult,
    ManifestRecord,
    Microphone,
    UsabilityStatus,
)

MICROPHONE_COVERAGE_FIELDS = (
    "speaker_group",
    "speaker_id",
    "session_id",
    "prompt_records",
    "both_microphones",
    "head_only",
    "array_only",
    "neither_microphone",
    "selected_head_primary",
    "selected_array_primary",
    "direct_head_wavs",
    "direct_array_wavs",
    "orphan_head_wavs",
    "orphan_array_wavs",
)

EXCLUSION_FIELDS = (
    "entity_type",
    "logical_id",
    "speaker_id",
    "session_id",
    "prompt_id",
    "prompt_path",
    "audio_path",
    "microphone",
    "reason",
)

AUDIO_QUALITY_CONTEXT_FIELDS = (
    "logical_id",
    "speaker_id",
    "session_id",
    "prompt_id",
    "microphone",
    "selection_role",
    "prompt_present",
)


def snapshot_raw_tree(raw_root: Path) -> dict[str, Any]:
    """Fingerprint raw file metadata before/after the read-only pipeline."""

    files = sorted(
        (path for path in raw_root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(raw_root).as_posix(),
    )
    extension_counts: Counter[str] = Counter()
    zero_byte_count = 0
    total_bytes = 0
    digest = hashlib.sha256()
    for path in files:
        stat = path.stat()
        relative = path.relative_to(raw_root).as_posix()
        total_bytes += stat.st_size
        zero_byte_count += stat.st_size == 0
        extension = path.suffix or (path.name if path.name.startswith(".") else "(none)")
        extension_counts[extension] += 1
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\n")
    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "zero_byte_file_count": zero_byte_count,
        "extension_counts": dict(
            sorted(extension_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "metadata_sha256": digest.hexdigest(),
    }


def build_microphone_coverage_rows(
    discovery: DiscoveryResult,
    records: Sequence[ManifestRecord],
    config: TorgoConfig,
) -> list[dict[str, Any]]:
    """Build per-session coverage with a final explicit total row."""

    grouped: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for utterance in discovery.utterances:
        grouped[(utterance.speaker_group, utterance.speaker_id, utterance.session_id)].append(
            utterance
        )

    direct_counts: Counter[tuple[str, str, str]] = Counter()
    for path in discovery.audio_paths:
        session = path.parent.parent.name
        speaker = path.parent.parent.parent.name
        speaker_group = path.parent.parent.parent.parent.name
        grouped.setdefault((speaker_group, speaker, session), [])
        microphone = _microphone_from_directory(path.parent.name, config)
        direct_counts[(speaker, session, microphone.value)] += 1
    for orphan in discovery.orphan_audio:
        grouped.setdefault(
            (orphan.speaker_group, orphan.speaker_id, orphan.session_id),
            [],
        )
    orphan_counts: Counter[tuple[str, str, str]] = Counter(
        (item.speaker_id, item.session_id, item.microphone.value) for item in discovery.orphan_audio
    )
    selected_counts: Counter[tuple[str, str, str]] = Counter(
        (record.speaker_id, record.session_id, record.primary_microphone.value)
        for record in records
        if record.primary_microphone is not None
    )

    rows: list[dict[str, Any]] = []
    for (speaker_group, speaker_id, session_id), utterances in sorted(
        grouped.items(),
        key=lambda item: (item[0][1], item[0][2]),
    ):
        both = sum(
            item.head_mic_path is not None and item.array_mic_path is not None
            for item in utterances
        )
        head_only = sum(
            item.head_mic_path is not None and item.array_mic_path is None for item in utterances
        )
        array_only = sum(
            item.head_mic_path is None and item.array_mic_path is not None for item in utterances
        )
        neither = len(utterances) - both - head_only - array_only
        rows.append(
            {
                "speaker_group": speaker_group,
                "speaker_id": speaker_id,
                "session_id": session_id,
                "prompt_records": len(utterances),
                "both_microphones": both,
                "head_only": head_only,
                "array_only": array_only,
                "neither_microphone": neither,
                "selected_head_primary": selected_counts[(speaker_id, session_id, "head")],
                "selected_array_primary": selected_counts[(speaker_id, session_id, "array")],
                "direct_head_wavs": direct_counts[(speaker_id, session_id, "head")],
                "direct_array_wavs": direct_counts[(speaker_id, session_id, "array")],
                "orphan_head_wavs": orphan_counts[(speaker_id, session_id, "head")],
                "orphan_array_wavs": orphan_counts[(speaker_id, session_id, "array")],
            }
        )

    total = {field: 0 for field in MICROPHONE_COVERAGE_FIELDS[3:]}
    for row in rows:
        for field in total:
            total[field] += int(row[field])
    rows.append(
        {
            "speaker_group": "__TOTAL__",
            "speaker_id": "__TOTAL__",
            "session_id": "__TOTAL__",
            **total,
        }
    )
    return rows


def build_exclusion_rows(
    records: Sequence[ManifestRecord],
    discovery: DiscoveryResult,
    config: TorgoConfig,
) -> list[dict[str, Any]]:
    """Log excluded logical records separately from orphan raw audio."""

    rows: list[dict[str, Any]] = []
    for record in records:
        if record.usability_status is not UsabilityStatus.EXCLUDED:
            continue
        rows.append(
            {
                "entity_type": "logical_utterance",
                "logical_id": record.logical_id,
                "speaker_id": record.speaker_id,
                "session_id": record.session_id,
                "prompt_id": record.prompt_id,
                "prompt_path": record.prompt_path,
                "audio_path": record.primary_audio_path or "",
                "microphone": (
                    record.primary_microphone.value if record.primary_microphone else ""
                ),
                "reason": record.exclusion_reason or "",
            }
        )
    for orphan in discovery.orphan_audio:
        rows.append(
            {
                "entity_type": "orphan_audio",
                "logical_id": "",
                "speaker_id": orphan.speaker_id,
                "session_id": orphan.session_id,
                "prompt_id": orphan.prompt_id,
                "prompt_path": "",
                "audio_path": config.relative_path(orphan.audio_path),
                "microphone": orphan.microphone.value,
                "reason": orphan.reason,
            }
        )
    return rows


def build_audio_quality_rows(
    records: Sequence[ManifestRecord],
    discovery: DiscoveryResult,
    audio_audits: Sequence[AudioAuditRecord],
    config: TorgoConfig,
) -> list[dict[str, Any]]:
    """Attach logical/microphone context to every direct WAV audit."""

    associations: dict[str, dict[str, Any]] = {}
    for record in records:
        for microphone, path in (
            (Microphone.HEAD, record.head_mic_path),
            (Microphone.ARRAY, record.array_mic_path),
        ):
            if path is None:
                continue
            if path == record.primary_audio_path:
                role = "primary"
            elif path == record.alternate_audio_path:
                role = "alternate"
            else:
                role = "associated"
            associations[path] = {
                "logical_id": record.logical_id,
                "speaker_id": record.speaker_id,
                "session_id": record.session_id,
                "prompt_id": record.prompt_id,
                "microphone": microphone.value,
                "selection_role": role,
                "prompt_present": True,
            }
    for orphan in discovery.orphan_audio:
        path = config.relative_path(orphan.audio_path)
        associations[path] = {
            "logical_id": "",
            "speaker_id": orphan.speaker_id,
            "session_id": orphan.session_id,
            "prompt_id": orphan.prompt_id,
            "microphone": orphan.microphone.value,
            "selection_role": "orphan",
            "prompt_present": False,
        }

    rows: list[dict[str, Any]] = []
    for audit in audio_audits:
        context = associations.get(
            audit.audio_path,
            {
                "logical_id": "",
                "speaker_id": "",
                "session_id": "",
                "prompt_id": "",
                "microphone": "",
                "selection_role": "unassociated",
                "prompt_present": False,
            },
        )
        rows.append({**context, **audit.to_dict(csv_safe=True)})
    return rows


def build_dataset_summary(
    *,
    records: Sequence[ManifestRecord],
    discovery: DiscoveryResult,
    audio_audits: Sequence[AudioAuditRecord],
    coverage_rows: Sequence[dict[str, Any]],
    folds: dict[str, Any],
    leakage_report: dict[str, Any],
    raw_snapshot_before: dict[str, Any],
    raw_snapshot_after: dict[str, Any],
    config: TorgoConfig,
) -> dict[str, Any]:
    """Build the authoritative machine-readable Module 0 metrics report."""

    if raw_snapshot_before != raw_snapshot_after:
        raise RuntimeError("immutable raw TORGO metadata changed during Module 0")
    total_coverage = coverage_rows[-1]
    audits_by_path = {audit.audio_path: audit for audit in audio_audits}
    primary_paths = {record.primary_audio_path for record in records if record.primary_audio_path}
    alternate_paths = {
        record.alternate_audio_path for record in records if record.alternate_audio_path
    }
    orphan_paths = {config.relative_path(orphan.audio_path) for orphan in discovery.orphan_audio}
    primary_audits = [audits_by_path[path] for path in sorted(primary_paths)]
    alternate_audits = [audits_by_path[path] for path in sorted(alternate_paths)]
    orphan_audits = [audits_by_path[path] for path in sorted(orphan_paths)]

    exclusion_reasons: Counter[str] = Counter()
    for record in records:
        if record.exclusion_reason:
            exclusion_reasons.update(record.exclusion_reason.split("|"))
    speaker_rows: list[dict[str, Any]] = []
    for speaker_id in sorted({record.speaker_id for record in records}):
        selected = [record for record in records if record.speaker_id == speaker_id]
        speaker_rows.append(
            {
                "speaker_id": speaker_id,
                "speaker_group": selected[0].speaker_group,
                "condition": selected[0].condition.value,
                "gender": selected[0].gender.value,
                "logical_records": len(selected),
                "usable": sum(
                    record.usability_status is UsabilityStatus.USABLE for record in selected
                ),
                "review": sum(
                    record.usability_status is UsabilityStatus.REVIEW for record in selected
                ),
                "excluded": sum(
                    record.usability_status is UsabilityStatus.EXCLUDED for record in selected
                ),
            }
        )

    actual_manual_values = {
        "extracted_files": raw_snapshot_before["file_count"],
        "wav_files": sum(
            count
            for extension, count in raw_snapshot_before["extension_counts"].items()
            if extension.casefold() == ".wav"
        ),
        "prompt_records": len(records),
        "microphone_both": total_coverage["both_microphones"],
        "microphone_head_only": total_coverage["head_only"],
        "microphone_array_only": total_coverage["array_only"],
        "microphone_neither": total_coverage["neither_microphone"],
        "direct_head_wavs": total_coverage["direct_head_wavs"],
        "direct_array_wavs": total_coverage["direct_array_wavs"],
    }
    manual_comparison = {
        name: {
            "documented": expected,
            "recalculated": actual_manual_values.get(name),
            "matches": actual_manual_values.get(name) == expected,
        }
        for name, expected in sorted(config.manual_audit_reference.items())
    }

    return {
        "schema_version": "1.0",
        "module": "0_torgo_manifest_evaluation",
        "dataset_id": config.dataset_id,
        "language": config.language,
        "source": {
            "manifest_sha256": leakage_report["source"]["manifest_sha256"],
            "speaker_folds_sha256": leakage_report["source"]["speaker_folds_sha256"],
        },
        "discovery_warnings": {
            "count": len(discovery.warnings),
            "items": list(discovery.warnings),
        },
        "raw_data_immutability": {
            "verified_unchanged_during_run": True,
            "metadata_sha256": raw_snapshot_before["metadata_sha256"],
            "raw_root": config.relative_path(config.raw_root),
        },
        "raw_inventory": raw_snapshot_before,
        "manual_audit_verification": {
            "all_documented_values_match": all(
                item["matches"] for item in manual_comparison.values()
            ),
            "values": manual_comparison,
            "note": "All values were recalculated; documented values were comparison targets only.",
        },
        "speakers": {
            "total": len(speaker_rows),
            "by_group": _speaker_count_by(records, "group"),
            "by_condition": _speaker_count_by(records, "condition"),
            "by_gender": _speaker_count_by(records, "gender"),
            "by_condition_gender": _speaker_count_by(records, "condition_gender"),
            "details": speaker_rows,
        },
        "sessions": {
            "total": len(coverage_rows) - 1,
            "by_speaker": {
                speaker: sorted(
                    {
                        str(row["session_id"])
                        for row in coverage_rows[:-1]
                        if row["speaker_id"] == speaker
                    }
                )
                for speaker in sorted({str(row["speaker_id"]) for row in coverage_rows[:-1]})
            },
        },
        "logical_utterances": {
            "total": len(records),
            "by_usability_status": _count_dict(record.usability_status.value for record in records),
            "excluded_count": sum(
                record.usability_status is UsabilityStatus.EXCLUDED for record in records
            ),
            "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
            "normal_speech_eligible_count": sum(
                record.normal_speech_eligible for record in records
            ),
        },
        "prompt_types": _count_dict(record.prompt_type.value for record in records),
        "microphone_coverage": {
            **{key: total_coverage[key] for key in MICROPHONE_COVERAGE_FIELDS[3:]},
            "primary_selection_reasons": _count_dict(
                record.primary_selection_reason.value for record in records
            ),
        },
        "downstream_subset_policy": {
            "default_subset": "core",
            "excluded_prompt_types": [
                "articulation_task",
                "instruction_or_nonstandard",
                "unknown",
            ],
            **{
                name: {
                    "record_count": sum(_record_in_subset(record, subset) for record in records),
                    "normal_speech_eligible": subset.normal_speech_eligible,
                    "usability_statuses": sorted(
                        status.value for status in subset.usability_statuses
                    ),
                    "prompt_types": sorted(
                        prompt_type.value for prompt_type in subset.prompt_types
                    ),
                    "experiment_role": subset.experiment_role,
                }
                for name, subset in sorted(config.downstream_subsets.items())
            },
        },
        "orphan_audio": {
            "path_count": len(discovery.orphan_audio),
            "logical_stem_count": len(
                {
                    (item.speaker_id, item.session_id, item.prompt_id)
                    for item in discovery.orphan_audio
                }
            ),
            "by_reason": _count_dict(item.reason for item in discovery.orphan_audio),
        },
        "audio_quality": {
            "all_direct_microphone_wavs": _summarize_audio(audio_audits),
            "selected_primary_wavs": _summarize_audio(primary_audits),
            "alternate_wavs": _summarize_audio(alternate_audits),
            "orphan_wavs": _summarize_audio(orphan_audits),
            "policy_note": (
                "Extremely short, near-silent, and possible clipping are review flags, not "
                "automatic deletions. HEAD is preferred only when structurally usable; a valid "
                "ARRAY is the primary when HEAD fails structural audit."
            ),
        },
        "speaker_evaluation": {
            "strategy": folds["strategy"],
            "folds": [
                {
                    "fold_id": fold["fold_id"],
                    "evaluation_speakers": fold["evaluation_speakers"],
                    "evaluation_counts": fold["evaluation_counts"],
                }
                for fold in folds["folds"]
            ],
            "validation": folds["validation"],
        },
        "leakage_and_duplicates": {
            "status": leakage_report["status"],
            "violation_count": leakage_report["violation_count"],
            "duplicate_logical_id_count": leakage_report["manifest_audit"][
                "duplicate_logical_id_count"
            ],
            "duplicate_primary_path_count": leakage_report["manifest_audit"][
                "duplicate_primary_path_count"
            ],
            "duplicate_hash_group_count": leakage_report["raw_audio_duplicate_audit"][
                "duplicate_hash_group_count"
            ],
            "cross_speaker_duplicate_hash_group_count": leakage_report["raw_audio_duplicate_audit"][
                "cross_speaker_duplicate_hash_group_count"
            ],
        },
        "outputs": {
            name: config.relative_path(path) for name, path in sorted(config.outputs.items())
        },
        "limitations": [
            (
                "Condition and gender come from TORGO directory groups; no medical "
                "severity is inferred."
            ),
            (
                "Repeated prompts across independent speakers are expected and allowed "
                "by the speaker split policy."
            ),
            "Quality flags are signal indicators and are not medical or perceptual diagnoses.",
            "Module 0 performs no resampling, VMD, feature extraction, ASR, or model training.",
        ],
    }


def _summarize_audio(audits: Sequence[AudioAuditRecord]) -> dict[str, Any]:
    readable = [audit for audit in audits if audit.readable]
    rms_values = [
        audit.rms_dbfs
        for audit in readable
        if audit.rms_dbfs is not None and math.isfinite(audit.rms_dbfs)
    ]
    durations = [audit.duration_seconds for audit in readable if audit.duration_seconds is not None]
    clipping_fractions = [
        audit.clipped_sample_fraction
        for audit in readable
        if audit.clipped_sample_fraction is not None
    ]
    return {
        "file_count": len(audits),
        "readable_count": len(readable),
        "unreadable_count": len(audits) - len(readable),
        "zero_byte_count": sum(audit.zero_byte for audit in audits),
        "empty_frame_count": sum(audit.frame_count == 0 for audit in readable),
        "extremely_short_count": sum(audit.extremely_short for audit in readable),
        "near_silent_count": sum(audit.near_silent for audit in readable),
        "possible_clipping_count": sum(audit.possible_clipping for audit in readable),
        "full_scale_sample_indicator_count": sum(
            (audit.full_scale_sample_count or 0) > 0 for audit in readable
        ),
        "sample_rate_distribution": _count_dict(
            str(audit.sample_rate) for audit in readable if audit.sample_rate is not None
        ),
        "channel_distribution": _count_dict(
            str(audit.channels) for audit in readable if audit.channels is not None
        ),
        "sample_width_bytes_distribution": _count_dict(
            str(audit.sample_width_bytes)
            for audit in readable
            if audit.sample_width_bytes is not None
        ),
        "bit_depth_distribution": _count_dict(
            str(audit.bit_depth) for audit in readable if audit.bit_depth is not None
        ),
        "compression_type_distribution": _count_dict(
            str(audit.compression_type) for audit in readable if audit.compression_type is not None
        ),
        "duration_seconds": _numeric_distribution(durations),
        "duration_bands": {
            "under_0.1": sum(value < 0.1 for value in durations),
            "under_0.25": sum(value < 0.25 for value in durations),
            "under_0.5": sum(value < 0.5 for value in durations),
            "under_1.0": sum(value < 1.0 for value in durations),
            "over_30.0": sum(value > 30.0 for value in durations),
        },
        "total_duration_seconds": _rounded(sum(durations)),
        "total_duration_hours": _rounded(sum(durations) / 3600.0),
        "file_size_bytes": _numeric_distribution(
            [float(audit.file_size) for audit in audits if audit.file_size is not None]
        ),
        "rms_dbfs": _numeric_distribution(rms_values),
        "rms_threshold_counts": {
            "at_or_below_minus_60": sum(value <= -60.0 for value in rms_values),
            "at_or_below_minus_50": sum(value <= -50.0 for value in rms_values),
            "at_or_below_minus_40": sum(value <= -40.0 for value in rms_values),
            "digital_silence": sum(audit.rms_dbfs == float("-inf") for audit in readable),
        },
        "clipped_sample_fraction": _numeric_distribution(clipping_fractions),
        "clipping_fraction_counts": {
            "at_or_above_0.01": sum(value >= 0.01 for value in clipping_fractions),
            "at_or_above_0.05": sum(value >= 0.05 for value in clipping_fractions),
            "at_or_above_0.10": sum(value >= 0.10 for value in clipping_fractions),
        },
        "quality_flag_counts": _count_dict(
            flag for audit in audits for flag in audit.quality_flags
        ),
        "structurally_usable_count": sum(audit.structurally_usable for audit in audits),
        "structurally_unusable_count": sum(not audit.structurally_usable for audit in audits),
        "structural_issue_counts": _count_dict(
            issue for audit in audits for issue in audit.structural_issues
        ),
    }


def _numeric_distribution(values: Sequence[float]) -> dict[str, Any]:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return {"count": 0}
    return {
        "count": len(finite),
        "min": _rounded(finite[0]),
        "p01": _rounded(_percentile(finite, 0.01)),
        "p05": _rounded(_percentile(finite, 0.05)),
        "p25": _rounded(_percentile(finite, 0.25)),
        "median": _rounded(statistics.median(finite)),
        "mean": _rounded(statistics.fmean(finite)),
        "p75": _rounded(_percentile(finite, 0.75)),
        "p95": _rounded(_percentile(finite, 0.95)),
        "p99": _rounded(_percentile(finite, 0.99)),
        "max": _rounded(finite[-1]),
    }


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _rounded(value: float) -> float:
    return round(value, 9)


def _speaker_count_by(
    records: Sequence[ManifestRecord],
    field: str,
) -> dict[str, int]:
    values: dict[str, str] = {}
    for record in records:
        if field == "condition":
            value = record.condition.value
        elif field == "gender":
            value = record.gender.value
        elif field == "group":
            value = record.speaker_group
        elif field == "condition_gender":
            value = f"{record.condition.value}|{record.gender.value}"
        else:
            raise ValueError(f"unsupported speaker count field: {field}")
        existing = values.setdefault(record.speaker_id, value)
        if existing != value:
            raise ValueError(f"inconsistent {field} for speaker {record.speaker_id}")
    return _count_dict(values.values())


def _record_in_subset(
    record: ManifestRecord,
    subset: DownstreamSubsetConfig,
) -> bool:
    return (
        record.normal_speech_eligible is subset.normal_speech_eligible
        and record.usability_status in subset.usability_statuses
        and record.prompt_type in subset.prompt_types
    )


def _count_dict(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _microphone_from_directory(directory: str, config: TorgoConfig) -> Microphone:
    for name, configured_directory in config.microphone_directories.items():
        if directory == configured_directory:
            return Microphone(name)
    raise ValueError(f"unknown microphone directory: {directory}")
