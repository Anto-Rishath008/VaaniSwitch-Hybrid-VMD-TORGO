"""Discovery, microphone policy, and manifest-schema tests."""

from dataclasses import replace
from pathlib import Path

import pytest

from tests.m0_data.helpers import add_prompt, make_config, make_manifest_record, write_wav
from vaaniswitch.m0_data.audio_audit import audit_audio_paths
from vaaniswitch.m0_data.discovery import discover_torgo
from vaaniswitch.m0_data.manifest import build_manifest
from vaaniswitch.m0_data.reporting import (
    build_audio_quality_rows,
    build_microphone_coverage_rows,
)
from vaaniswitch.m0_data.schemas import (
    AudioStructuralStatus,
    Condition,
    Gender,
    ManifestValidationError,
    Microphone,
    PrimarySelectionReason,
    PromptType,
    UsabilityStatus,
)


def test_valid_head_and_valid_array_select_head(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    head, array = add_prompt(
        config,
        group="F",
        speaker="F01",
        session="Session1",
        prompt_id="0001",
        transcript="hello",
        head=True,
        array=True,
    )
    assert head is not None and array is not None

    discovery = discover_torgo(config)
    audits = audit_audio_paths(discovery.audio_paths, config)
    record = build_manifest(discovery, audits, config)[0]

    assert record.primary_audio_path == config.relative_path(head)
    assert record.alternate_audio_path == config.relative_path(array)
    assert record.primary_microphone is Microphone.HEAD
    assert record.primary_selection_reason is PrimarySelectionReason.HEAD_STRUCTURALLY_USABLE
    assert record.head_audio_status is AudioStructuralStatus.USABLE
    assert record.array_audio_status is AudioStructuralStatus.USABLE


def test_valid_head_only_selects_head(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    head, _ = add_prompt(
        config,
        group="F",
        speaker="F01",
        session="Session1",
        prompt_id="0001",
        transcript="hello",
        head=True,
    )
    assert head is not None

    discovery = discover_torgo(config)
    audits = audit_audio_paths(discovery.audio_paths, config)
    record = build_manifest(discovery, audits, config)[0]

    assert record.primary_audio_path == config.relative_path(head)
    assert record.alternate_audio_path is None
    assert record.primary_microphone is Microphone.HEAD
    assert record.array_audio_status is AudioStructuralStatus.MISSING


def test_array_only_prompt_builds_array_primary_without_alternate(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    _, array = add_prompt(
        config,
        group="F",
        speaker="F01",
        session="Session1",
        prompt_id="0001",
        transcript="hello",
        array=True,
    )
    assert array is not None

    discovery = discover_torgo(config)
    audits = audit_audio_paths(discovery.audio_paths, config)
    record = build_manifest(discovery, audits, config)[0]

    assert record.primary_audio_path == config.relative_path(array)
    assert record.alternate_audio_path is None
    assert record.primary_microphone is Microphone.ARRAY
    assert (
        record.primary_selection_reason
        is PrimarySelectionReason.ARRAY_STRUCTURALLY_USABLE_HEAD_MISSING
    )


def test_neither_microphone_excludes_logical_utterance(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    add_prompt(
        config,
        group="F",
        speaker="F01",
        session="Session1",
        prompt_id="0001",
        transcript="hello",
    )

    discovery = discover_torgo(config)
    records = build_manifest(discovery, (), config)

    assert len(records) == 1
    assert records[0].usability_status is UsabilityStatus.EXCLUDED
    assert records[0].exclusion_reason == "missing_primary_audio"


def test_discovery_is_prompt_driven_and_logs_orphan_audio(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    add_prompt(
        config,
        group="M",
        speaker="M01",
        session="Session2_3",
        prompt_id="0001",
        transcript="word",
        head=True,
    )
    orphan = config.raw_root / "M/M01/Session2_3/wav_arrayMic/0099.wav"
    write_wav(orphan)

    discovery = discover_torgo(config)

    assert len(discovery.utterances) == 1
    assert len(discovery.orphan_audio) == 1
    assert discovery.orphan_audio[0].prompt_id == "0099"
    assert discovery.orphan_audio[0].reason == "missing_prompt_metadata"


def test_literal_dot_wav_filename_is_audited_as_malformed_orphan(tmp_path: Path) -> None:
    """Regression: ``Path('.wav').suffix`` is empty despite the WAV filename ending."""

    config = make_config(tmp_path)
    add_prompt(
        config,
        group="M",
        speaker="M01",
        session="Session1",
        prompt_id="0001",
        transcript="word",
        head=True,
    )
    malformed = config.raw_root / "M/M01/Session1/wav_arrayMic/.wav"
    write_wav(malformed)

    discovery = discover_torgo(config)

    assert malformed in discovery.audio_paths
    finding = next(item for item in discovery.orphan_audio if item.audio_path == malformed)
    assert finding.reason == "malformed_audio_filename_or_prompt_id"


def test_orphan_only_session_is_included_in_microphone_coverage(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    orphan = config.raw_root / "F/F01/Session1/wav_headMic/0099.wav"
    write_wav(orphan)

    discovery = discover_torgo(config)
    coverage = build_microphone_coverage_rows(discovery, (), config)

    session_row = coverage[0]
    total_row = coverage[-1]
    assert session_row["prompt_records"] == 0
    assert session_row["direct_head_wavs"] == 1
    assert session_row["orphan_head_wavs"] == 1
    assert total_row["direct_head_wavs"] == 1
    assert total_row["orphan_head_wavs"] == 1


def test_malformed_session_directory_is_reported_as_a_warning(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    malformed = config.raw_root / "F/F01/Session_bad"
    malformed.mkdir(parents=True)

    discovery = discover_torgo(config)

    assert "ignored_malformed_session_directory:F/F01/Session_bad" in discovery.warnings


def test_zero_byte_head_uses_valid_array_fallback_and_is_retained(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    head, array = add_prompt(
        config,
        group="F",
        speaker="F01",
        session="Session1",
        prompt_id="0001",
        transcript="hello",
        head=True,
        array=True,
    )
    assert head is not None and array is not None
    head.write_bytes(b"")
    discovery = discover_torgo(config)
    audits = audit_audio_paths(discovery.audio_paths, config)

    record = build_manifest(discovery, audits, config)[0]

    assert record.primary_audio_path == config.relative_path(array)
    assert record.alternate_audio_path == config.relative_path(head)
    assert record.primary_microphone is Microphone.ARRAY
    assert record.usability_status is UsabilityStatus.USABLE
    assert record.exclusion_reason is None
    assert record.head_audio_status is AudioStructuralStatus.UNUSABLE
    assert record.head_audio_issues == ("unreadable", "zero_byte")
    assert record.primary_selection_reason is PrimarySelectionReason.HEAD_UNUSABLE_ARRAY_FALLBACK
    assert "head_unusable_array_fallback" in record.quality_flags


def test_corrupt_head_uses_valid_array_fallback(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    head, array = add_prompt(
        config,
        group="F",
        speaker="F01",
        session="Session1",
        prompt_id="0001",
        transcript="hello",
        head=True,
        array=True,
    )
    assert head is not None and array is not None
    head.write_bytes(b"not a WAV")

    discovery = discover_torgo(config)
    audits = audit_audio_paths(discovery.audio_paths, config)
    record = build_manifest(discovery, audits, config)[0]

    assert record.primary_audio_path == config.relative_path(array)
    assert record.alternate_audio_path == config.relative_path(head)
    assert record.usability_status is UsabilityStatus.USABLE
    assert record.head_audio_issues == ("unreadable",)


def test_empty_frame_head_uses_valid_array_fallback(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    head, array = add_prompt(
        config,
        group="F",
        speaker="F01",
        session="Session1",
        prompt_id="0001",
        transcript="hello",
        head=True,
        array=True,
    )
    assert head is not None and array is not None
    write_wav(head, samples=[])

    discovery = discover_torgo(config)
    audits = audit_audio_paths(discovery.audio_paths, config)
    record = build_manifest(discovery, audits, config)[0]

    assert record.primary_audio_path == config.relative_path(array)
    assert record.alternate_audio_path == config.relative_path(head)
    assert record.usability_status is UsabilityStatus.USABLE
    assert record.head_audio_issues == ("empty_audio",)


def test_valid_head_remains_primary_when_array_is_invalid(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    head, array = add_prompt(
        config,
        group="F",
        speaker="F01",
        session="Session1",
        prompt_id="0001",
        transcript="hello",
        head=True,
        array=True,
    )
    assert head is not None and array is not None
    array.write_bytes(b"not a WAV")

    discovery = discover_torgo(config)
    audits = audit_audio_paths(discovery.audio_paths, config)
    record = build_manifest(discovery, audits, config)[0]

    assert record.primary_audio_path == config.relative_path(head)
    assert record.alternate_audio_path == config.relative_path(array)
    assert record.primary_microphone is Microphone.HEAD
    assert record.array_audio_status is AudioStructuralStatus.UNUSABLE
    assert record.usability_status is UsabilityStatus.USABLE


def test_invalid_head_and_invalid_array_are_excluded(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    head, array = add_prompt(
        config,
        group="F",
        speaker="F01",
        session="Session1",
        prompt_id="0001",
        transcript="hello",
        head=True,
        array=True,
    )
    assert head is not None and array is not None
    head.write_bytes(b"")
    array.write_bytes(b"not a WAV")

    discovery = discover_torgo(config)
    audits = audit_audio_paths(discovery.audio_paths, config)
    record = build_manifest(discovery, audits, config)[0]

    assert record.primary_audio_path is None
    assert record.alternate_audio_path is None
    assert record.primary_microphone is None
    assert record.usability_status is UsabilityStatus.EXCLUDED
    assert record.exclusion_reason == "no_structurally_usable_audio"
    assert record.head_mic_path == config.relative_path(head)
    assert record.array_mic_path == config.relative_path(array)


def test_invalid_head_remains_in_full_audio_quality_audit(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    head, _ = add_prompt(
        config,
        group="F",
        speaker="F01",
        session="Session1",
        prompt_id="0001",
        transcript="hello",
        head=True,
        array=True,
    )
    assert head is not None
    head.write_bytes(b"")

    discovery = discover_torgo(config)
    audits = audit_audio_paths(discovery.audio_paths, config)
    records = build_manifest(discovery, audits, config)
    rows = build_audio_quality_rows(records, discovery, audits, config)
    head_row = next(row for row in rows if row["audio_path"] == config.relative_path(head))

    assert head_row["selection_role"] == "alternate"
    assert head_row["zero_byte"] is True
    assert head_row["readable"] is False
    assert head_row["structurally_usable"] is False
    assert "zero_byte" in head_row["structural_issues"]


def test_manifest_schema_rejects_absolute_production_path() -> None:
    record = make_manifest_record("F01", Condition.DYSARTHRIC, Gender.FEMALE)
    invalid = replace(record, prompt_path="C:/Users/example/prompt.txt")

    with pytest.raises(ManifestValidationError, match="repository-relative"):
        invalid.validate()


def test_manifest_schema_rejects_head_priority_violation() -> None:
    record = make_manifest_record(
        "F01",
        Condition.DYSARTHRIC,
        Gender.FEMALE,
        alternate_path="data/raw/torgo/extracted/F/F01/Session1/wav_arrayMic/0001.wav",
    )
    invalid = replace(
        record,
        primary_audio_path=record.alternate_audio_path,
        alternate_audio_path=record.head_mic_path,
        primary_microphone=Microphone.ARRAY,
    )

    with pytest.raises(ManifestValidationError, match="microphone policy"):
        invalid.validate()


def test_manifest_schema_accepts_array_fallback_only_with_head_failure_evidence() -> None:
    record = make_manifest_record(
        "F01",
        Condition.DYSARTHRIC,
        Gender.FEMALE,
        alternate_path="data/raw/torgo/extracted/F/F01/Session1/wav_arrayMic/0001.wav",
    )
    fallback = replace(
        record,
        head_audio_status=AudioStructuralStatus.UNUSABLE,
        head_audio_issues=("unreadable",),
        primary_audio_path=record.array_mic_path,
        alternate_audio_path=record.head_mic_path,
        primary_microphone=Microphone.ARRAY,
        primary_selection_reason=PrimarySelectionReason.HEAD_UNUSABLE_ARRAY_FALLBACK,
        quality_flags=("head_unusable_array_fallback", "head_unusable_unreadable"),
    )

    fallback.validate()
    without_evidence = replace(fallback, head_audio_issues=())
    with pytest.raises(ManifestValidationError, match="rejection evidence"):
        without_evidence.validate()


def test_manifest_schema_rejects_tampered_canonical_labels() -> None:
    record = make_manifest_record("F01", Condition.DYSARTHRIC, Gender.FEMALE)
    invalid_records = (
        replace(record, condition=Condition.CONTROL),
        replace(record, gender=Gender.MALE),
        replace(record, speaker_group="FC"),
    )

    for invalid in invalid_records:
        with pytest.raises(ManifestValidationError, match=r"group|label mismatch"):
            invalid.validate()


def test_manifest_schema_blocks_articulation_from_normal_speech_subset() -> None:
    record = make_manifest_record("F01", Condition.DYSARTHRIC, Gender.FEMALE)
    invalid = replace(record, prompt_type=PromptType.ARTICULATION_TASK)

    with pytest.raises(ManifestValidationError, match="normal_speech_eligible"):
        invalid.validate()


def test_manifest_schema_requires_no_microphone_record_to_be_excluded() -> None:
    record = make_manifest_record("F01", Condition.DYSARTHRIC, Gender.FEMALE)
    invalid = replace(
        record,
        head_mic_path=None,
        head_audio_status=AudioStructuralStatus.MISSING,
        head_audio_issues=("missing_path",),
        primary_audio_path=None,
        primary_microphone=None,
        primary_selection_reason=PrimarySelectionReason.NO_STRUCTURALLY_USABLE_AUDIO,
        sample_rate=None,
        channels=None,
        sample_width_bytes=None,
        bit_depth=None,
        frame_count=None,
        duration_seconds=None,
        file_size=None,
        primary_audio_sha256=None,
    )

    with pytest.raises(ManifestValidationError, match="must be excluded"):
        invalid.validate()


def test_manifest_schema_rejects_usable_record_without_transcript() -> None:
    record = make_manifest_record("F01", Condition.DYSARTHRIC, Gender.FEMALE)
    invalid = replace(
        record,
        transcript=None,
        prompt_type=PromptType.UNKNOWN,
        normal_speech_eligible=False,
    )

    with pytest.raises(ManifestValidationError, match="without a transcript"):
        invalid.validate()


def test_manifest_schema_rejects_usable_record_without_readable_metadata() -> None:
    record = make_manifest_record("F01", Condition.DYSARTHRIC, Gender.FEMALE)
    invalid = replace(record, sample_rate=None)

    with pytest.raises(ManifestValidationError, match="positive readable metadata"):
        invalid.validate()


def test_manifest_json_schema_does_not_coerce_boolean_strings() -> None:
    record = make_manifest_record("F01", Condition.DYSARTHRIC, Gender.FEMALE)
    payload = record.to_dict()
    payload["normal_speech_eligible"] = "false"

    with pytest.raises(ManifestValidationError, match="JSON boolean"):
        type(record).from_dict(payload)
