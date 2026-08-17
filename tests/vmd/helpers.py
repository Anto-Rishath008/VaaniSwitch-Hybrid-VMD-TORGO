"""Tiny repository fixtures for Module 1 integration tests."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import wave
from pathlib import Path

import yaml

from components.data.manifest import write_manifest_jsonl
from components.data.reporting import snapshot_raw_tree
from components.data.schemas import (
    AudioStructuralStatus,
    Condition,
    Gender,
    ManifestRecord,
    Microphone,
    PrimarySelectionReason,
    PromptType,
    UsabilityStatus,
)
from components.vmd.config import (
    AudioConfig,
    CacheConfig,
    DatasetConfig,
    ExecutionConfig,
    Module1Config,
    OutputConfig,
    ReconstructionConfig,
    SelectedModesAblationConfig,
    VMDAlgorithmConfig,
)
from components.vmd.manifest import load_manifest_source, select_subset


def write_wav(
    path: Path,
    *,
    sample_rate: int = 16_000,
    duration_seconds: float = 0.025,
    frequency: float = 220.0,
) -> None:
    """Write a tiny mono 16-bit sine WAV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = max(1, round(sample_rate * duration_seconds))
    samples = [
        round(2_000 * math.sin(2 * math.pi * frequency * index / sample_rate))
        for index in range(count)
    ]
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def make_record(
    repo_root: Path,
    *,
    speaker_id: str,
    condition: Condition,
    gender: Gender,
    prompt_id: str = "0001",
    status: UsabilityStatus = UsabilityStatus.USABLE,
    prompt_type: PromptType = PromptType.WORD,
    array_primary: bool = False,
    unusable_head: bool = False,
) -> ManifestRecord:
    """Create a schema-valid record and its primary WAV under a tiny raw tree."""

    if condition is Condition.DYSARTHRIC:
        group = "F" if gender is Gender.FEMALE else "M"
    else:
        group = "FC" if gender is Gender.FEMALE else "MC"
    base = Path("data/raw/torgo/extracted") / group / speaker_id / "Session1"
    prompt_relative = base / "prompts" / f"{prompt_id}.txt"
    prompt_path = repo_root / prompt_relative
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("hello", encoding="utf-8")

    head_relative: Path | None
    array_relative: Path | None
    head_status: AudioStructuralStatus
    head_issues: tuple[str, ...]
    array_status: AudioStructuralStatus
    array_issues: tuple[str, ...]
    flags: tuple[str, ...]
    if array_primary:
        array_relative = base / "wav_arrayMic" / f"{prompt_id}.wav"
        write_wav(repo_root / array_relative, frequency=330.0)
        array_status = AudioStructuralStatus.USABLE
        array_issues = ()
        if unusable_head:
            head_relative = base / "wav_headMic" / f"{prompt_id}.wav"
            head_path = repo_root / head_relative
            head_path.parent.mkdir(parents=True, exist_ok=True)
            head_path.write_bytes(b"")
            head_status = AudioStructuralStatus.UNUSABLE
            head_issues = ("unreadable", "zero_byte")
            selection_reason = PrimarySelectionReason.HEAD_UNUSABLE_ARRAY_FALLBACK
            flags = (
                "head_unusable_array_fallback",
                "head_unusable_unreadable",
                "head_unusable_zero_byte",
            )
        else:
            head_relative = None
            head_status = AudioStructuralStatus.MISSING
            head_issues = ("missing_path",)
            selection_reason = PrimarySelectionReason.ARRAY_STRUCTURALLY_USABLE_HEAD_MISSING
            flags = ()
        primary_relative = array_relative
        alternate_relative = head_relative
        primary_microphone = Microphone.ARRAY
    else:
        head_relative = base / "wav_headMic" / f"{prompt_id}.wav"
        write_wav(repo_root / head_relative, frequency=220.0)
        array_relative = None
        head_status = AudioStructuralStatus.USABLE
        head_issues = ()
        array_status = AudioStructuralStatus.MISSING
        array_issues = ("missing_path",)
        selection_reason = PrimarySelectionReason.HEAD_STRUCTURALLY_USABLE
        flags = ()
        primary_relative = head_relative
        alternate_relative = None
        primary_microphone = Microphone.HEAD

    primary_path = repo_root / primary_relative
    with wave.open(str(primary_path), "rb") as stream:
        frame_count = stream.getnframes()
        sample_rate = stream.getframerate()
        channels = stream.getnchannels()
        sample_width = stream.getsampwidth()
    digest = hashlib.sha256(primary_path.read_bytes()).hexdigest()
    normal = status is not UsabilityStatus.EXCLUDED and prompt_type in {
        PromptType.WORD,
        PromptType.SENTENCE,
    }
    record = ManifestRecord(
        logical_id=f"{speaker_id}/Session1/{prompt_id}",
        speaker_id=speaker_id,
        speaker_group=group,
        condition=condition,
        gender=gender,
        session_id="Session1",
        prompt_id=prompt_id,
        transcript="hello",
        prompt_path=prompt_relative.as_posix(),
        head_mic_path=head_relative.as_posix() if head_relative else None,
        head_audio_status=head_status,
        head_audio_issues=head_issues,
        array_mic_path=array_relative.as_posix() if array_relative else None,
        array_audio_status=array_status,
        array_audio_issues=array_issues,
        primary_audio_path=primary_relative.as_posix(),
        alternate_audio_path=alternate_relative.as_posix() if alternate_relative else None,
        primary_microphone=primary_microphone,
        primary_selection_reason=selection_reason,
        sample_rate=sample_rate,
        channels=channels,
        sample_width_bytes=sample_width,
        bit_depth=sample_width * 8,
        frame_count=frame_count,
        duration_seconds=frame_count / sample_rate,
        file_size=primary_path.stat().st_size,
        primary_audio_sha256=digest,
        prompt_type=prompt_type,
        normal_speech_eligible=normal,
        usability_status=status,
        exclusion_reason=None,
        quality_flags=tuple(sorted(flags)),
    )
    record.validate()
    return record


def make_config(repo_root: Path) -> Module1Config:
    """Construct a fast typed Module 1 test configuration."""

    return Module1Config(
        schema_version="1.0",
        implementation="components_numpy_vmd_v1",
        repo_root=repo_root.resolve(),
        dataset=DatasetConfig(
            manifest_path=(repo_root / "data/manifests/torgo_manifest.jsonl").resolve(),
            module0_summary_path=(
                repo_root / "artifacts/metrics/module0_dataset_summary.json"
            ).resolve(),
            module0_leakage_report_path=(
                repo_root / "artifacts/logs/module0_leakage_report.json"
            ).resolve(),
            raw_root=(repo_root / "data/raw/torgo/extracted").resolve(),
            default_subset="core",
        ),
        audio=AudioConfig(
            target_sample_rate=16_000,
            target_channels=1,
            normalization="rms",
            target_rms=0.1,
            max_gain=10.0,
            peak_limit=0.99,
            silence_epsilon=1.0e-12,
        ),
        vmd=VMDAlgorithmConfig(
            modes_k=5,
            alpha=2_000.0,
            tau=0.0,
            dc=False,
            init=1,
            tolerance=1.0e-7,
            max_iterations=100,
        ),
        reconstruction=ReconstructionConfig(
            default="all_modes",
            selected_modes_ablation=SelectedModesAblationConfig(False, ()),
        ),
        cache=CacheConfig(
            enabled=True,
            resume=True,
            root=(repo_root / "data/processed/vmd").resolve(),
        ),
        execution=ExecutionConfig(progress_every=1, plot_examples=2),
        outputs=OutputConfig(
            summary=(repo_root / "artifacts/metrics/module1_vmd_summary.json").resolve(),
            failures=(repo_root / "artifacts/logs/module1_vmd_failures.csv").resolve(),
            pipeline_log=(repo_root / "artifacts/logs/module1_vmd_pipeline.log").resolve(),
            plots=(repo_root / "artifacts/plots/module1_vmd").resolve(),
        ),
    )


def prepare_tiny_repo(repo_root: Path) -> tuple[Module1Config, tuple[ManifestRecord, ...]]:
    """Create signed-off-like Module 0 inputs for Module 1 smoke tests."""

    records = (
        make_record(
            repo_root,
            speaker_id="F01",
            condition=Condition.DYSARTHRIC,
            gender=Gender.FEMALE,
        ),
        make_record(
            repo_root,
            speaker_id="FC01",
            condition=Condition.CONTROL,
            gender=Gender.FEMALE,
        ),
        make_record(
            repo_root,
            speaker_id="M01",
            condition=Condition.DYSARTHRIC,
            gender=Gender.MALE,
            status=UsabilityStatus.REVIEW,
        ),
        make_record(
            repo_root,
            speaker_id="MC01",
            condition=Condition.CONTROL,
            gender=Gender.MALE,
            prompt_type=PromptType.ARTICULATION_TASK,
        ),
        make_record(
            repo_root,
            speaker_id="F02",
            condition=Condition.DYSARTHRIC,
            gender=Gender.FEMALE,
            array_primary=True,
            unusable_head=True,
        ),
    )
    config = make_config(repo_root)
    write_manifest_jsonl(config.dataset.manifest_path, records)
    source = load_manifest_source(config.dataset.manifest_path)
    raw_snapshot = snapshot_raw_tree(config.dataset.raw_root)
    summary = {
        "source": {"manifest_sha256": source.fingerprint},
        "raw_inventory": raw_snapshot,
        "downstream_subset_policy": {
            "core": {"record_count": len(select_subset(records, "core"))},
            "extended": {"record_count": len(select_subset(records, "extended"))},
        },
    }
    config.dataset.module0_summary_path.parent.mkdir(parents=True, exist_ok=True)
    config.dataset.module0_summary_path.write_text(
        json.dumps(summary), encoding="utf-8"
    )
    config.dataset.module0_leakage_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.dataset.module0_leakage_report_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "violation_count": 0,
                "source": {"manifest_sha256": source.fingerprint},
            }
        ),
        encoding="utf-8",
    )
    return config, records


def write_config_yaml(config: Module1Config, path: Path) -> None:
    """Serialize the tiny config in the production YAML shape for CLI tests."""

    root = config.repo_root

    def relative(item: Path) -> str:
        return item.relative_to(root).as_posix()

    value = {
        "schema_version": config.schema_version,
        "implementation": config.implementation,
        "dataset": {
            "manifest_path": relative(config.dataset.manifest_path),
            "module0_summary_path": relative(config.dataset.module0_summary_path),
            "module0_leakage_report_path": relative(
                config.dataset.module0_leakage_report_path
            ),
            "raw_root": relative(config.dataset.raw_root),
            "default_subset": config.dataset.default_subset,
        },
        "audio": {
            "target_sample_rate": config.audio.target_sample_rate,
            "target_channels": config.audio.target_channels,
            "normalization": config.audio.normalization,
            "target_rms": config.audio.target_rms,
            "max_gain": config.audio.max_gain,
            "peak_limit": config.audio.peak_limit,
            "silence_epsilon": config.audio.silence_epsilon,
        },
        "vmd": {
            "modes_k": config.vmd.modes_k,
            "alpha": config.vmd.alpha,
            "tau": config.vmd.tau,
            "dc": config.vmd.dc,
            "init": config.vmd.init,
            "tolerance": config.vmd.tolerance,
            "max_iterations": config.vmd.max_iterations,
        },
        "reconstruction": {
            "default": config.reconstruction.default,
            "selected_modes_ablation": {
                "enabled": config.reconstruction.selected_modes_ablation.enabled,
                "mode_indices": list(
                    config.reconstruction.selected_modes_ablation.mode_indices
                ),
            },
        },
        "cache": {
            "enabled": config.cache.enabled,
            "resume": config.cache.resume,
            "root": relative(config.cache.root),
        },
        "execution": {
            "progress_every": config.execution.progress_every,
            "plot_examples": config.execution.plot_examples,
        },
        "outputs": {
            "summary": relative(config.outputs.summary),
            "failures": relative(config.outputs.failures),
            "pipeline_log": relative(config.outputs.pipeline_log),
            "plots": relative(config.outputs.plots),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
