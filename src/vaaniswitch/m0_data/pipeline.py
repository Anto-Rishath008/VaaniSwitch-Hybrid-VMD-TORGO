"""End-to-end orchestration for the non-mutating TORGO Module 0 pipeline."""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vaaniswitch.m0_data.audio_audit import audit_audio_paths
from vaaniswitch.m0_data.config import TorgoConfig
from vaaniswitch.m0_data.discovery import discover_torgo
from vaaniswitch.m0_data.leakage import (
    assert_no_leakage,
    build_leakage_report,
    speaker_folds_fingerprint,
)
from vaaniswitch.m0_data.manifest import (
    build_manifest,
    load_manifest_jsonl,
    manifest_fingerprint,
    write_csv,
    write_json,
    write_manifest_csv,
    write_manifest_jsonl,
)
from vaaniswitch.m0_data.reporting import (
    AUDIO_QUALITY_CONTEXT_FIELDS,
    EXCLUSION_FIELDS,
    MICROPHONE_COVERAGE_FIELDS,
    build_audio_quality_rows,
    build_dataset_summary,
    build_exclusion_rows,
    build_microphone_coverage_rows,
    snapshot_raw_tree,
)
from vaaniswitch.m0_data.schemas import (
    AudioAuditRecord,
    AudioStructuralStatus,
    ManifestRecord,
    UsabilityStatus,
)
from vaaniswitch.m0_data.splits import create_speaker_folds, validate_speaker_folds

LOGGER = logging.getLogger(__name__)
_MODULE0_HANDLERS: list[logging.Handler] = []


def run_module0(config: TorgoConfig) -> dict[str, Any]:
    """Run discovery, full WAV audit, manifests, folds, and leakage checks."""

    _configure_logging(config.outputs["pipeline_log"])
    LOGGER.info("Starting Module 0 for %s", config.dataset_id)
    LOGGER.info(
        "Raw TORGO is read-only at %s",
        config.relative_path(config.raw_root),
    )

    raw_before = snapshot_raw_tree(config.raw_root)
    discovery = discover_torgo(config)
    for warning in discovery.warnings:
        LOGGER.warning("TORGO discovery warning: %s", warning)
    audio_audits = audit_audio_paths(discovery.audio_paths, config)
    records = build_manifest(discovery, audio_audits, config)
    manifest_sha256 = manifest_fingerprint(records)
    folds = create_speaker_folds(
        records,
        config,
        manifest_sha256=manifest_sha256,
    )
    leakage_report = build_leakage_report(
        records,
        audio_audits,
        folds,
        discovery.orphan_audio,
    )
    coverage_rows = build_microphone_coverage_rows(discovery, records, config)
    exclusion_rows = build_exclusion_rows(records, discovery, config)
    audio_quality_rows = build_audio_quality_rows(
        records,
        discovery,
        audio_audits,
        config,
    )
    raw_after = snapshot_raw_tree(config.raw_root)
    summary = build_dataset_summary(
        records=records,
        discovery=discovery,
        audio_audits=audio_audits,
        coverage_rows=coverage_rows,
        folds=folds,
        leakage_report=leakage_report,
        raw_snapshot_before=raw_before,
        raw_snapshot_after=raw_after,
        config=config,
    )

    write_manifest_jsonl(config.outputs["manifest_jsonl"], records)
    write_manifest_csv(config.outputs["manifest_csv"], records)
    write_json(config.outputs["speaker_folds"], folds)
    write_json(config.outputs["dataset_summary"], summary)
    write_csv(
        config.outputs["exclusions"],
        EXCLUSION_FIELDS,
        exclusion_rows,
    )
    write_csv(
        config.outputs["audio_quality"],
        AUDIO_QUALITY_CONTEXT_FIELDS + AudioAuditRecord.CSV_FIELDS,
        audio_quality_rows,
    )
    write_csv(
        config.outputs["microphone_coverage"],
        MICROPHONE_COVERAGE_FIELDS,
        coverage_rows,
    )
    write_json(config.outputs["leakage_report"], leakage_report)

    validate_generated_outputs(config)
    LOGGER.info(
        "Module 0 complete: %d logical records, %d exclusions, leakage=%s",
        len(records),
        summary["logical_utterances"]["excluded_count"],
        leakage_report["status"],
    )
    return summary


def validate_generated_outputs(config: TorgoConfig) -> dict[str, Any]:
    """Reload every generated artifact and re-run cross-artifact gates."""

    for name, path in sorted(config.outputs.items()):
        if not path.is_file():
            raise ValueError(f"required Module 0 output is missing: {name} ({path})")
        if path.stat().st_size == 0:
            raise ValueError(f"required Module 0 output is empty: {name} ({path})")

    records = load_manifest_jsonl(config.outputs["manifest_jsonl"])
    if not records:
        raise ValueError("generated manifest contains no logical utterances")
    manifest_rows = _read_csv_rows(
        config.outputs["manifest_csv"],
        ManifestRecord.CSV_FIELDS,
    )
    _validate_manifest_csv_parity(records, manifest_rows)

    folds = _read_json_object(config.outputs["speaker_folds"], "speaker_folds")
    split_validation = validate_speaker_folds(records, folds)
    expected_hash = folds.get("source", {}).get("manifest_sha256")
    actual_hash = manifest_fingerprint(records)
    if expected_hash != actual_hash:
        raise ValueError("speaker fold source fingerprint does not match generated manifest")
    folds_hash = speaker_folds_fingerprint(folds)

    leakage = _read_json_object(config.outputs["leakage_report"], "leakage")
    assert_no_leakage(leakage)
    leakage_source = leakage.get("source")
    if not isinstance(leakage_source, dict):
        raise ValueError("leakage output has no source fingerprints")
    if leakage_source.get("manifest_sha256") != actual_hash:
        raise ValueError("leakage report manifest fingerprint is stale or invalid")
    if leakage_source.get("speaker_folds_sha256") != folds_hash:
        raise ValueError("leakage report speaker-fold fingerprint is stale or invalid")

    summary = _read_json_object(config.outputs["dataset_summary"], "dataset_summary")
    _validate_summary(
        summary,
        records,
        actual_hash,
        folds_hash,
        leakage,
        snapshot_raw_tree(config.raw_root),
        config,
    )

    exclusion_rows = _read_csv_rows(
        config.outputs["exclusions"],
        EXCLUSION_FIELDS,
    )
    audio_quality_rows = _read_csv_rows(
        config.outputs["audio_quality"],
        AUDIO_QUALITY_CONTEXT_FIELDS + AudioAuditRecord.CSV_FIELDS,
    )
    coverage_rows = _read_csv_rows(
        config.outputs["microphone_coverage"],
        MICROPHONE_COVERAGE_FIELDS,
    )
    _validate_tabular_reports(
        records,
        summary,
        exclusion_rows,
        audio_quality_rows,
        coverage_rows,
    )

    pipeline_log = config.outputs["pipeline_log"].read_text(encoding="utf-8")
    if str(config.repo_root) in pipeline_log:
        raise ValueError("pipeline log exposes an absolute repository path")
    return {
        "manifest_records": len(records),
        "manifest_sha256": actual_hash,
        "split_validation": split_validation,
        "leakage_status": leakage["status"],
    }


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{label} output must contain a JSON object")
    return value


def _read_csv_rows(path: Path, expected_fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(expected_fields):
            raise ValueError(
                f"CSV schema mismatch for {path}: expected {list(expected_fields)}, "
                f"found {reader.fieldnames}"
            )
        rows: list[dict[str, str]] = []
        for line_number, raw_row in enumerate(reader, start=2):
            if None in raw_row or any(value is None for value in raw_row.values()):
                raise ValueError(f"malformed CSV row at {path}:{line_number}")
            rows.append({str(key): str(value) for key, value in raw_row.items()})
    return rows


def _validate_manifest_csv_parity(
    records: Sequence[ManifestRecord],
    rows: Sequence[dict[str, str]],
) -> None:
    if len(rows) != len(records):
        raise ValueError(f"manifest CSV/JSONL row-count mismatch: {len(rows)} != {len(records)}")
    for line_number, (record, row) in enumerate(zip(records, rows, strict=True), start=2):
        expected = {
            key: "" if value is None else str(value)
            for key, value in record.to_dict(csv_safe=True).items()
        }
        if row != expected:
            mismatched = sorted(
                key for key in ManifestRecord.CSV_FIELDS if row.get(key) != expected[key]
            )
            raise ValueError(
                f"manifest CSV/JSONL content mismatch at CSV line {line_number}: {mismatched}"
            )


def _validate_summary(
    summary: dict[str, Any],
    records: Sequence[ManifestRecord],
    manifest_sha256: str,
    folds_sha256: str,
    leakage: dict[str, Any],
    current_raw_snapshot: dict[str, Any],
    config: TorgoConfig,
) -> None:
    source = _require_mapping(summary, "source")
    if source.get("manifest_sha256") != manifest_sha256:
        raise ValueError("dataset summary manifest fingerprint is stale or invalid")
    if source.get("speaker_folds_sha256") != folds_sha256:
        raise ValueError("dataset summary speaker-fold fingerprint is stale or invalid")
    if summary.get("raw_inventory") != current_raw_snapshot:
        raise ValueError("dataset summary raw inventory is stale or invalid")
    raw_immutability = _require_mapping(summary, "raw_data_immutability")
    if raw_immutability.get("metadata_sha256") != current_raw_snapshot.get("metadata_sha256"):
        raise ValueError("dataset summary raw metadata fingerprint is inconsistent")

    logical = _require_mapping(summary, "logical_utterances")
    expected_statuses = dict(
        sorted(Counter(record.usability_status.value for record in records).items())
    )
    expected_excluded = expected_statuses.get(UsabilityStatus.EXCLUDED.value, 0)
    if logical.get("total") != len(records):
        raise ValueError("dataset summary logical utterance count is inconsistent")
    if logical.get("excluded_count") != expected_excluded:
        raise ValueError("dataset summary exclusion count is inconsistent")
    if logical.get("by_usability_status") != expected_statuses:
        raise ValueError("dataset summary usability distribution is inconsistent")
    if logical.get("normal_speech_eligible_count") != sum(
        record.normal_speech_eligible for record in records
    ):
        raise ValueError("dataset summary normal-speech count is inconsistent")

    speakers = _require_mapping(summary, "speakers")
    if speakers.get("total") != len({record.speaker_id for record in records}):
        raise ValueError("dataset summary speaker count is inconsistent")
    speaker_labels = {
        record.speaker_id: (
            record.speaker_group,
            record.condition.value,
            record.gender.value,
        )
        for record in records
    }
    expected_speaker_distributions = {
        "by_group": dict(sorted(Counter(value[0] for value in speaker_labels.values()).items())),
        "by_condition": dict(
            sorted(Counter(value[1] for value in speaker_labels.values()).items())
        ),
        "by_gender": dict(sorted(Counter(value[2] for value in speaker_labels.values()).items())),
        "by_condition_gender": dict(
            sorted(Counter(f"{value[1]}|{value[2]}" for value in speaker_labels.values()).items())
        ),
    }
    for field, expected in expected_speaker_distributions.items():
        if speakers.get(field) != expected:
            raise ValueError(f"dataset summary speaker distribution is inconsistent for {field}")
    prompt_types = dict(sorted(Counter(record.prompt_type.value for record in records).items()))
    if summary.get("prompt_types") != prompt_types:
        raise ValueError("dataset summary prompt-type distribution is inconsistent")

    subset_policy = _require_mapping(summary, "downstream_subset_policy")
    if subset_policy.get("default_subset") != "core":
        raise ValueError("dataset summary default downstream subset must be core")
    if subset_policy.get("excluded_prompt_types") != [
        "articulation_task",
        "instruction_or_nonstandard",
        "unknown",
    ]:
        raise ValueError("dataset summary excluded prompt types are inconsistent")
    for name, subset in config.downstream_subsets.items():
        reported = _require_mapping(subset_policy, name)
        expected_count = sum(
            record.normal_speech_eligible is subset.normal_speech_eligible
            and record.usability_status in subset.usability_statuses
            and record.prompt_type in subset.prompt_types
            for record in records
        )
        subset_expected: dict[str, Any] = {
            "record_count": expected_count,
            "normal_speech_eligible": subset.normal_speech_eligible,
            "usability_statuses": sorted(status.value for status in subset.usability_statuses),
            "prompt_types": sorted(prompt_type.value for prompt_type in subset.prompt_types),
            "experiment_role": subset.experiment_role,
        }
        if reported != subset_expected:
            raise ValueError(f"dataset summary downstream subset {name} is inconsistent")

    leakage_summary = _require_mapping(summary, "leakage_and_duplicates")
    if leakage_summary.get("status") != leakage.get("status") or leakage_summary.get(
        "violation_count"
    ) != leakage.get("violation_count"):
        raise ValueError("dataset summary leakage result is inconsistent")


def _validate_tabular_reports(
    records: Sequence[ManifestRecord],
    summary: dict[str, Any],
    exclusion_rows: Sequence[dict[str, str]],
    audio_rows: Sequence[dict[str, str]],
    coverage_rows: Sequence[dict[str, str]],
) -> None:
    excluded_records = {
        record.logical_id: record.exclusion_reason
        for record in records
        if record.usability_status is UsabilityStatus.EXCLUDED
    }
    logged_exclusions = {
        row["logical_id"]: row["reason"]
        for row in exclusion_rows
        if row["entity_type"] == "logical_utterance"
    }
    logical_exclusion_rows = [
        row for row in exclusion_rows if row["entity_type"] == "logical_utterance"
    ]
    if len(logged_exclusions) != len(logical_exclusion_rows):
        raise ValueError("logical exclusions CSV contains duplicate logical IDs")
    if logged_exclusions != excluded_records:
        raise ValueError("logical exclusions CSV is inconsistent with the manifest")

    allowed_entities = {"logical_utterance", "orphan_audio"}
    unexpected_entities = {
        row["entity_type"] for row in exclusion_rows if row["entity_type"] not in allowed_entities
    }
    if unexpected_entities:
        raise ValueError(f"exclusions CSV has unsupported entity types: {unexpected_entities}")

    orphan_rows = [row for row in exclusion_rows if row["entity_type"] == "orphan_audio"]
    orphan_summary = _require_mapping(summary, "orphan_audio")
    if len(orphan_rows) != orphan_summary.get("path_count"):
        raise ValueError("orphan exclusions CSV count is inconsistent with the summary")
    orphan_audio_paths = {row["audio_path"] for row in orphan_rows}
    if len(orphan_audio_paths) != len(orphan_rows):
        raise ValueError("orphan exclusions CSV contains duplicate audio paths")

    audio_summary = _require_mapping(summary, "audio_quality")
    all_direct = _require_mapping(audio_summary, "all_direct_microphone_wavs")
    if len(audio_rows) != all_direct.get("file_count"):
        raise ValueError("audio-quality CSV count is inconsistent with the summary")
    audio_paths = [row["audio_path"] for row in audio_rows]
    if len(audio_paths) != len(set(audio_paths)):
        raise ValueError("audio-quality CSV contains duplicate audio paths")
    if any(row["structurally_usable"] not in {"True", "False"} for row in audio_rows):
        raise ValueError("audio-quality CSV has an invalid structurally_usable value")
    expected_structurally_usable = sum(row["structurally_usable"] == "True" for row in audio_rows)
    if all_direct.get("structurally_usable_count") != expected_structurally_usable:
        raise ValueError("audio structural usability count is inconsistent with the summary")
    if all_direct.get("structurally_unusable_count") != (
        len(audio_rows) - expected_structurally_usable
    ):
        raise ValueError("audio structural failure count is inconsistent with the summary")
    manifest_audio_paths = {
        path
        for record in records
        for path in (record.head_mic_path, record.array_mic_path)
        if path is not None
    }
    if set(audio_paths) != manifest_audio_paths | orphan_audio_paths:
        raise ValueError("audio-quality CSV paths do not match manifest and orphan paths")
    audio_by_path = {row["audio_path"]: row for row in audio_rows}
    for record in records:
        for microphone, path in (
            ("head", record.head_mic_path),
            ("array", record.array_mic_path),
        ):
            if path is None:
                continue
            row = audio_by_path[path]
            if path == record.primary_audio_path:
                expected_role = "primary"
            elif path == record.alternate_audio_path:
                expected_role = "alternate"
            else:
                expected_role = "associated"
            expected_context = {
                "logical_id": record.logical_id,
                "speaker_id": record.speaker_id,
                "session_id": record.session_id,
                "prompt_id": record.prompt_id,
                "microphone": microphone,
                "selection_role": expected_role,
                "prompt_present": "True",
            }
            if any(row[field] != expected for field, expected in expected_context.items()):
                raise ValueError(f"audio-quality context is inconsistent for {path}")
            recorded_status = (
                record.head_audio_status if microphone == "head" else record.array_audio_status
            )
            recorded_issues = (
                record.head_audio_issues if microphone == "head" else record.array_audio_issues
            )
            audited_status = (
                AudioStructuralStatus.USABLE
                if row["structurally_usable"] == "True"
                else AudioStructuralStatus.UNUSABLE
            )
            audited_issues = tuple(issue for issue in row["structural_issues"].split("|") if issue)
            if recorded_status is not audited_status or recorded_issues != audited_issues:
                raise ValueError(
                    f"manifest structural microphone evidence is inconsistent for {path}"
                )
        if record.primary_audio_path is not None:
            row = audio_by_path[record.primary_audio_path]
            expected_metadata = {
                "sample_rate": record.sample_rate,
                "channels": record.channels,
                "sample_width_bytes": record.sample_width_bytes,
                "bit_depth": record.bit_depth,
                "frame_count": record.frame_count,
                "duration_seconds": record.duration_seconds,
                "file_size": record.file_size,
                "sha256": record.primary_audio_sha256,
            }
            for field, expected in expected_metadata.items():
                csv_expected = "" if expected is None else str(expected)
                if row[field] != csv_expected:
                    raise ValueError(
                        f"audio-quality metadata is inconsistent for {record.primary_audio_path}: "
                        f"{field}"
                    )
    for orphan in orphan_rows:
        row = audio_by_path[orphan["audio_path"]]
        if (
            row["selection_role"] != "orphan"
            or row["prompt_present"] != "False"
            or row["speaker_id"] != orphan["speaker_id"]
            or row["session_id"] != orphan["session_id"]
            or row["prompt_id"] != orphan["prompt_id"]
            or row["microphone"] != orphan["microphone"]
        ):
            raise ValueError(
                f"orphan audio-quality context is inconsistent for {orphan['audio_path']}"
            )

    if not coverage_rows:
        raise ValueError("microphone-coverage CSV contains no rows")
    total_rows = [row for row in coverage_rows if row["speaker_id"] == "__TOTAL__"]
    if len(total_rows) != 1 or coverage_rows[-1] is not total_rows[0]:
        raise ValueError("microphone-coverage CSV must end with one total row")
    total_row = total_rows[0]
    session_rows = coverage_rows[:-1]
    sessions_summary = _require_mapping(summary, "sessions")
    if sessions_summary.get("total") != len(session_rows):
        raise ValueError("session count is inconsistent between summary and coverage CSV")
    for field in MICROPHONE_COVERAGE_FIELDS[3:]:
        expected_total = sum(_csv_integer(row, field) for row in session_rows)
        if _csv_integer(total_row, field) != expected_total:
            raise ValueError(f"microphone-coverage total is inconsistent for {field}")

    expected_coverage = {
        "prompt_records": len(records),
        "both_microphones": sum(
            record.head_mic_path is not None and record.array_mic_path is not None
            for record in records
        ),
        "head_only": sum(
            record.head_mic_path is not None and record.array_mic_path is None for record in records
        ),
        "array_only": sum(
            record.head_mic_path is None and record.array_mic_path is not None for record in records
        ),
        "neither_microphone": sum(
            record.head_mic_path is None and record.array_mic_path is None for record in records
        ),
        "selected_head_primary": sum(
            record.primary_microphone is not None and record.primary_microphone.value == "head"
            for record in records
        ),
        "selected_array_primary": sum(
            record.primary_microphone is not None and record.primary_microphone.value == "array"
            for record in records
        ),
        "direct_head_wavs": sum(row["microphone"] == "head" for row in audio_rows),
        "direct_array_wavs": sum(row["microphone"] == "array" for row in audio_rows),
        "orphan_head_wavs": sum(row["microphone"] == "head" for row in orphan_rows),
        "orphan_array_wavs": sum(row["microphone"] == "array" for row in orphan_rows),
    }
    for field, expected in expected_coverage.items():
        if _csv_integer(total_row, field) != expected:
            raise ValueError(f"microphone coverage is inconsistent for {field}")
    summary_coverage = _require_mapping(summary, "microphone_coverage")
    expected_summary_coverage: dict[str, Any] = {
        field: _csv_integer(total_row, field) for field in MICROPHONE_COVERAGE_FIELDS[3:]
    }
    expected_summary_coverage["primary_selection_reasons"] = dict(
        sorted(Counter(record.primary_selection_reason.value for record in records).items())
    )
    if summary_coverage != expected_summary_coverage:
        raise ValueError("dataset summary microphone coverage is inconsistent")


def _require_mapping(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise ValueError(f"{key} must be a JSON object")
    return item


def _csv_integer(row: Mapping[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"CSV field {field!r} must be an integer") from exc


def _configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    for handler in _MODULE0_HANDLERS:
        if handler in root_logger.handlers:
            root_logger.removeHandler(handler)
        handler.close()
    _MODULE0_HANDLERS.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    _MODULE0_HANDLERS.extend((file_handler, console_handler))
    root_logger.setLevel(logging.INFO)
