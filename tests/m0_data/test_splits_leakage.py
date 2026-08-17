"""Speaker-fold determinism and leakage hard-gate tests."""

import random
from dataclasses import replace
from pathlib import Path

import pytest

from tests.m0_data.helpers import (
    balanced_speaker_records,
    make_config,
    make_manifest_record,
)
from vaaniswitch.m0_data.leakage import (
    LeakageError,
    assert_no_leakage,
    build_leakage_report,
    find_alternate_paths_used_as_other_primaries,
    find_duplicate_logical_ids,
    find_duplicate_primary_paths,
)
from vaaniswitch.m0_data.schemas import (
    AudioAuditRecord,
    AudioStructuralStatus,
    Condition,
    Gender,
)
from vaaniswitch.m0_data.splits import SplitError, create_speaker_folds, validate_speaker_folds


def test_duplicate_logical_id_detection() -> None:
    record = balanced_speaker_records()[0]

    assert find_duplicate_logical_ids([record, record]) == [record.logical_id]


def test_duplicate_primary_audio_path_detection() -> None:
    records = balanced_speaker_records()
    shared = records[0].primary_audio_path
    duplicate = replace(
        records[1],
        head_mic_path=shared,
        primary_audio_path=shared,
    )

    assert find_duplicate_primary_paths([records[0], duplicate]) == [shared]


def test_alternate_path_reused_as_another_primary_is_detected() -> None:
    records = balanced_speaker_records()
    shared_path = records[1].primary_audio_path
    records[0] = replace(
        records[0],
        array_mic_path=shared_path,
        array_audio_status=AudioStructuralStatus.USABLE,
        array_audio_issues=(),
        alternate_audio_path=shared_path,
    )

    assert find_alternate_paths_used_as_other_primaries(records) == [shared_path]


def test_fold_creation_is_deterministic_and_input_order_independent(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    records = balanced_speaker_records()
    shuffled = list(records)
    random.Random(123).shuffle(shuffled)

    first = create_speaker_folds(records, config, manifest_sha256="abc")
    second = create_speaker_folds(shuffled, config, manifest_sha256="abc")

    assert first == second
    assert first["strategy"]["selected_folds"] == 3


def test_five_fold_preference_falls_back_to_largest_stratum_safe_count(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path, requested_max_folds=5)
    records = balanced_speaker_records()
    records.extend(
        [
            make_manifest_record("M04", Condition.DYSARTHRIC, Gender.MALE),
            make_manifest_record("M05", Condition.DYSARTHRIC, Gender.MALE),
            make_manifest_record("MC04", Condition.CONTROL, Gender.MALE),
        ]
    )

    folds = create_speaker_folds(records, config, manifest_sha256="abc")

    assert folds["strategy"]["requested_max_folds"] == 5
    assert folds["strategy"]["selected_folds"] == 3
    assert "largest count" in folds["strategy"]["selection_reason"]


def test_every_fold_is_speaker_disjoint_and_contains_all_strata(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    records = balanced_speaker_records()
    folds = create_speaker_folds(records, config, manifest_sha256="abc")

    validation = validate_speaker_folds(records, folds)

    assert validation["speaker_disjoint"]
    assert validation["condition_gender_supported_in_every_fold"]
    for fold in folds["folds"]:
        assert set(fold["train_speakers"]).isdisjoint(fold["evaluation_speakers"])


def test_speaker_overlap_fails_loudly(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    records = balanced_speaker_records()
    folds = create_speaker_folds(records, config, manifest_sha256="abc")
    leaked_speaker = folds["folds"][0]["evaluation_speakers"][0]
    folds["folds"][0]["train_speakers"].append(leaked_speaker)

    with pytest.raises(SplitError, match="speaker leakage"):
        validate_speaker_folds(records, folds)


def test_tampered_speaker_to_fold_map_fails_loudly(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    records = balanced_speaker_records()
    folds = create_speaker_folds(records, config, manifest_sha256="abc")
    speaker = next(iter(folds["speaker_to_evaluation_fold"]))
    original = folds["speaker_to_evaluation_fold"][speaker]
    folds["speaker_to_evaluation_fold"][speaker] = original % 3 + 1

    with pytest.raises(SplitError, match="does not match"):
        validate_speaker_folds(records, folds)


def test_internally_inconsistent_pass_report_fails_loudly() -> None:
    report = {
        "status": "PASS",
        "violation_count": 0,
        "violations": {"speaker_overlap": ["F01"]},
    }

    with pytest.raises(LeakageError, match="internally inconsistent"):
        assert_no_leakage(report)


def test_distinct_cross_speaker_paths_with_same_hash_fail(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    records = balanced_speaker_records()
    folds = create_speaker_folds(records, config, manifest_sha256="abc")
    first = records[0]
    second = next(record for record in records if record.speaker_id != first.speaker_id)
    duplicate_digest = "a" * 64
    audits = (
        _audio_audit(first.primary_audio_path, duplicate_digest),
        _audio_audit(second.primary_audio_path, duplicate_digest),
    )

    with pytest.raises(LeakageError, match="cross_speaker_duplicate_content_hashes"):
        build_leakage_report(records, audits, folds, ())


def test_same_speaker_duplicate_hash_is_reported_but_group_safe(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    records = balanced_speaker_records()
    first = records[0]
    second = make_manifest_record(
        first.speaker_id,
        first.condition,
        first.gender,
        prompt_index=2,
    )
    records.append(second)
    folds = create_speaker_folds(records, config, manifest_sha256="abc")
    duplicate_digest = "b" * 64

    report = build_leakage_report(
        records,
        (
            _audio_audit(first.primary_audio_path, duplicate_digest),
            _audio_audit(second.primary_audio_path, duplicate_digest),
        ),
        folds,
        (),
    )

    assert report["status"] == "PASS"
    assert report["raw_audio_duplicate_audit"]["duplicate_hash_group_count"] == 1
    assert report["raw_audio_duplicate_audit"]["cross_speaker_duplicate_hash_group_count"] == 0


def test_microphone_copy_cross_fold_fails_loudly(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    records = balanced_speaker_records()
    folds = create_speaker_folds(records, config, manifest_sha256="abc")
    first_eval = folds["folds"][0]["evaluation_speakers"][0]
    first_train = folds["folds"][0]["train_speakers"][0]
    evaluation_record = next(record for record in records if record.speaker_id == first_eval)
    train_index = next(
        index for index, record in enumerate(records) if record.speaker_id == first_train
    )
    records[train_index] = replace(
        records[train_index],
        array_mic_path=evaluation_record.primary_audio_path,
        array_audio_status=AudioStructuralStatus.USABLE,
        array_audio_issues=(),
        alternate_audio_path=evaluation_record.primary_audio_path,
    )

    with pytest.raises(SplitError, match="microphone path leakage"):
        validate_speaker_folds(records, folds)


def _audio_audit(path: str | None, digest: str) -> AudioAuditRecord:
    assert path is not None
    return AudioAuditRecord(
        audio_path=path,
        file_size=6_444,
        readable=True,
        error=None,
        zero_byte=False,
        sample_rate=16_000,
        channels=1,
        sample_width_bytes=2,
        bit_depth=16,
        frame_count=3_200,
        duration_seconds=0.2,
        compression_type="NONE",
        sample_count=3_200,
        rms_dbfs=-30.0,
        peak_amplitude_ratio=0.1,
        full_scale_sample_count=0,
        clipped_sample_count=0,
        clipped_sample_fraction=0.0,
        extremely_short=False,
        near_silent=False,
        possible_clipping=False,
        sha256=digest,
        quality_flags=(),
    )
