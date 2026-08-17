"""Tiny synthetic fixtures shared by Module 0 tests."""

from __future__ import annotations

import hashlib
import math
import struct
import wave
from pathlib import Path

from components.data.config import (
    DownstreamSubsetConfig,
    EvaluationConfig,
    QualityConfig,
    TorgoConfig,
)
from components.data.schemas import (
    AudioStructuralStatus,
    Condition,
    Gender,
    GroupMetadata,
    ManifestRecord,
    Microphone,
    PrimarySelectionReason,
    PromptType,
    UsabilityStatus,
)


def make_config(
    repo_root: Path,
    *,
    requested_max_folds: int = 3,
    minimum_folds: int = 2,
) -> TorgoConfig:
    """Construct a test config without coupling fixtures to production paths."""

    raw_root = repo_root / "data/raw/torgo/extracted"
    outputs = {
        "manifest_jsonl": repo_root / "data/manifests/torgo_manifest.jsonl",
        "manifest_csv": repo_root / "data/manifests/torgo_manifest.csv",
        "speaker_folds": repo_root / "data/splits/speaker_folds.json",
        "dataset_summary": repo_root / "artifacts/metrics/module0_dataset_summary.json",
        "exclusions": repo_root / "artifacts/logs/module0_exclusions.csv",
        "audio_quality": repo_root / "artifacts/logs/module0_audio_quality.csv",
        "microphone_coverage": repo_root / "artifacts/logs/module0_microphone_coverage.csv",
        "leakage_report": repo_root / "artifacts/logs/module0_leakage_report.json",
        "pipeline_log": repo_root / "artifacts/logs/module0_pipeline.log",
    }
    return TorgoConfig(
        schema_version="1.0",
        dataset_id="TORGO_TEST",
        language="en",
        repo_root=repo_root.resolve(),
        raw_root=raw_root.resolve(),
        logical_key=("speaker_id", "session_id", "prompt_id"),
        groups={
            "F": GroupMetadata(Condition.DYSARTHRIC, Gender.FEMALE, r"^F\d+$"),
            "M": GroupMetadata(Condition.DYSARTHRIC, Gender.MALE, r"^M\d+$"),
            "FC": GroupMetadata(Condition.CONTROL, Gender.FEMALE, r"^FC\d+$"),
            "MC": GroupMetadata(Condition.CONTROL, Gender.MALE, r"^MC\d+$"),
        },
        session_pattern=r"^Session\d+(?:_\d+)*$",
        prompt_id_pattern=r"^\d{4}$",
        prompt_directory="prompts",
        prompt_extension=".txt",
        transcript_encodings=("utf-8-sig",),
        microphone_directories={"head": "wav_headMic", "array": "wav_arrayMic"},
        microphone_priority=("head", "array"),
        quality=QualityConfig(
            chunk_frames=128,
            max_workers=2,
            compute_sha256=True,
            extremely_short_seconds=0.1,
            near_silent_dbfs=-60.0,
            clipping_amplitude_ratio=0.9999,
            clipping_fraction_threshold=0.01,
            review_flags=frozenset(
                {
                    "extremely_short",
                    "near_silent",
                    "possible_clipping",
                    "placeholder_transcript",
                    "malformed_transcript_control_characters",
                }
            ),
        ),
        evaluation=EvaluationConfig(
            group_by="speaker_id",
            requested_max_folds=requested_max_folds,
            minimum_folds=minimum_folds,
            hard_strata=("condition", "gender"),
            balance_statuses=frozenset({UsabilityStatus.USABLE, UsabilityStatus.REVIEW}),
            seed=42,
        ),
        downstream_subsets={
            "core": DownstreamSubsetConfig(
                normal_speech_eligible=True,
                usability_statuses=frozenset({UsabilityStatus.USABLE}),
                prompt_types=frozenset({PromptType.WORD, PromptType.SENTENCE}),
                experiment_role="default_clean_classification",
            ),
            "extended": DownstreamSubsetConfig(
                normal_speech_eligible=True,
                usability_statuses=frozenset({UsabilityStatus.USABLE, UsabilityStatus.REVIEW}),
                prompt_types=frozenset({PromptType.WORD, PromptType.SENTENCE}),
                experiment_role="quality_flag_robustness_ablation",
            ),
        },
        outputs={name: path.resolve() for name, path in outputs.items()},
        manual_audit_reference={},
    )


def add_prompt(
    config: TorgoConfig,
    *,
    group: str,
    speaker: str,
    session: str,
    prompt_id: str,
    transcript: str,
    head: bool = False,
    array: bool = False,
) -> tuple[Path | None, Path | None]:
    """Add one tiny prompt and optional microphone WAVs."""

    session_path = config.raw_root / group / speaker / session
    prompt_path = session_path / "prompts" / f"{prompt_id}.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(transcript, encoding="utf-8")
    head_path: Path | None = None
    array_path: Path | None = None
    if head:
        head_path = session_path / "wav_headMic" / f"{prompt_id}.wav"
        signature = sum(ord(character) for character in f"{group}/{speaker}/{session}")
        write_wav(head_path, amplitude=700 + signature, frequency=180 + signature)
    if array:
        array_path = session_path / "wav_arrayMic" / f"{prompt_id}.wav"
        signature = sum(ord(character) for character in f"{group}/{speaker}/{session}")
        write_wav(array_path, amplitude=300 + signature, frequency=260 + signature)
    return head_path, array_path


def write_wav(
    path: Path,
    *,
    sample_rate: int = 16_000,
    duration_seconds: float = 0.2,
    amplitude: int = 1_000,
    frequency: float = 220.0,
    samples: list[int] | None = None,
) -> None:
    """Write a tiny mono 16-bit PCM WAV for tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if samples is None:
        count = round(sample_rate * duration_seconds)
        samples = [
            round(amplitude * math.sin(2 * math.pi * frequency * index / sample_rate))
            for index in range(count)
        ]
    payload = struct.pack(f"<{len(samples)}h", *samples)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(payload)


def make_manifest_record(
    speaker_id: str,
    condition: Condition,
    gender: Gender,
    *,
    prompt_index: int = 1,
    primary_path: str | None = None,
    alternate_path: str | None = None,
    status: UsabilityStatus = UsabilityStatus.USABLE,
) -> ManifestRecord:
    """Build one schema-valid in-memory record for split tests."""

    group = (
        "F"
        if condition is Condition.DYSARTHRIC and gender is Gender.FEMALE
        else "M"
        if condition is Condition.DYSARTHRIC
        else "FC"
        if gender is Gender.FEMALE
        else "MC"
    )
    prompt_id = f"{prompt_index:04d}"
    head_path = (
        primary_path
        or f"data/raw/torgo/extracted/{group}/{speaker_id}/Session1/wav_headMic/{prompt_id}.wav"
    )
    record = ManifestRecord(
        logical_id=f"{speaker_id}/Session1/{prompt_id}",
        speaker_id=speaker_id,
        speaker_group=group,
        condition=condition,
        gender=gender,
        session_id="Session1",
        prompt_id=prompt_id,
        transcript="test word",
        prompt_path=f"data/raw/torgo/extracted/{group}/{speaker_id}/Session1/prompts/{prompt_id}.txt",
        head_mic_path=head_path,
        head_audio_status=AudioStructuralStatus.USABLE,
        head_audio_issues=(),
        array_mic_path=alternate_path,
        array_audio_status=(
            AudioStructuralStatus.USABLE
            if alternate_path is not None
            else AudioStructuralStatus.MISSING
        ),
        array_audio_issues=() if alternate_path is not None else ("missing_path",),
        primary_audio_path=head_path,
        alternate_audio_path=alternate_path,
        primary_microphone=Microphone.HEAD,
        primary_selection_reason=PrimarySelectionReason.HEAD_STRUCTURALLY_USABLE,
        sample_rate=16_000,
        channels=1,
        sample_width_bytes=2,
        bit_depth=16,
        frame_count=3_200,
        duration_seconds=0.2,
        file_size=6_444,
        primary_audio_sha256=hashlib.sha256(f"{speaker_id}/{prompt_id}".encode()).hexdigest(),
        prompt_type=PromptType.SENTENCE,
        normal_speech_eligible=status is not UsabilityStatus.EXCLUDED,
        usability_status=status,
        exclusion_reason="test_exclusion" if status is UsabilityStatus.EXCLUDED else None,
        quality_flags=(),
    )
    record.validate()
    return record


def balanced_speaker_records() -> list[ManifestRecord]:
    """Return three speakers for each condition-by-gender stratum."""

    records: list[ManifestRecord] = []
    for prefix, condition, gender in (
        ("F", Condition.DYSARTHRIC, Gender.FEMALE),
        ("M", Condition.DYSARTHRIC, Gender.MALE),
        ("FC", Condition.CONTROL, Gender.FEMALE),
        ("MC", Condition.CONTROL, Gender.MALE),
    ):
        for index in range(1, 4):
            records.append(
                make_manifest_record(
                    f"{prefix}{index:02d}",
                    condition,
                    gender,
                )
            )
    return records
