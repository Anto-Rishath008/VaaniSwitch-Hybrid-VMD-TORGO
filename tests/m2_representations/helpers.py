"""Small signed-source fixtures for Module 2 tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tests.m1_vmd.helpers import make_record
from vaaniswitch.m0_data.manifest import write_manifest_jsonl
from vaaniswitch.m0_data.schemas import (
    Condition,
    Gender,
    ManifestRecord,
    PromptType,
    UsabilityStatus,
)
from vaaniswitch.m1_vmd.manifest import load_manifest_source, select_subset
from vaaniswitch.m2_representations.config import (
    CacheConfig,
    ExecutionConfig,
    MelConfig,
    Module2Config,
    OutputConfig,
    RecurrenceConfig,
    SourceConfig,
    VisibilityConfig,
)

MODULE1_HASH = "b" * 64


def make_config(repo_root: Path, smoke_ids: tuple[str, ...] = ()) -> Module2Config:
    config = Module2Config(
        schema_version="1.0",
        implementation="vaaniswitch_numpy_representations_v1",
        repo_root=repo_root.resolve(),
        source=SourceConfig(
            manifest_path=(repo_root / "data/manifests/torgo_manifest.jsonl").resolve(),
            module0_summary_path=(
                repo_root / "artifacts/metrics/module0_dataset_summary.json"
            ).resolve(),
            module1_summary_path=(
                repo_root / "artifacts/metrics/module1_vmd_summary.json"
            ).resolve(),
            module1_signoff_path=(
                repo_root / "artifacts/metrics/module1_full_signoff_audit.json"
            ).resolve(),
            module1_cache_root=(repo_root / "data/processed/vmd").resolve(),
            module1_config_hash=MODULE1_HASH,
            default_subset="core",
        ),
        mel=MelConfig(
            sample_rate=16_000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            window="hann_periodic",
            n_mels=80,
            mel_scale="htk",
            filter_normalization="slaney_area",
            f_min_hz=20.0,
            f_max_hz=8_000.0,
            power=2.0,
            log_transform="power_db",
            epsilon=1.0e-10,
            fixed_duration_seconds=8.0,
            crop_policy="center",
            pad_policy="right_zero",
            frequency_resize="linear",
            time_resize="area_average",
            output_height=224,
            output_width=224,
            output_dtype="float32",
        ),
        recurrence=RecurrenceConfig(
            output_size=224,
            scalar_reduction="block_rms",
            scalar_standardization="zscore",
            embedding_dimension=3,
            delay=2,
            distance="euclidean",
            threshold_mode="off_diagonal_recurrence_rate",
            recurrence_rate=0.15,
            include_diagonal=True,
            output_dtype="uint8",
            epsilon=1.0e-12,
        ),
        visibility_graph=VisibilityConfig(
            max_nodes=256,
            scalar_reduction="block_rms",
            scalar_standardization="zscore",
            criterion="natural_visibility_strict",
            node_features=("normalized_time", "standardized_amplitude"),
            store_bidirectional_edges=True,
            store_self_loops=False,
            edge_dtype="int16",
            adjacency_contract="module3_add_self_loops_then_symmetric_degree_normalize",
            epsilon=1.0e-12,
        ),
        cache=CacheConfig(
            enabled=True,
            resume=True,
            root=(repo_root / "data/processed/representations").resolve(),
        ),
        execution=ExecutionConfig(
            progress_every=1,
            diagnostic_examples=2,
            smoke_logical_ids=smoke_ids,
        ),
        outputs=OutputConfig(
            summary=(repo_root / "artifacts/metrics/module2_representation_summary.json").resolve(),
            failures=(repo_root / "artifacts/logs/module2_representation_failures.csv").resolve(),
            pipeline_log=(
                repo_root / "artifacts/logs/module2_representation_pipeline.log"
            ).resolve(),
            plots=(repo_root / "artifacts/plots/module2_representations").resolve(),
        ),
    )
    config.validate()
    return config


def prepare_tiny_repo(
    repo_root: Path,
) -> tuple[Module2Config, tuple[ManifestRecord, ...]]:
    excluded = replace(
        make_record(
            repo_root,
            speaker_id="F02",
            condition=Condition.DYSARTHRIC,
            gender=Gender.FEMALE,
        ),
        usability_status=UsabilityStatus.EXCLUDED,
        normal_speech_eligible=False,
        exclusion_reason="synthetic_exclusion",
    )
    excluded.validate()
    records = (
        make_record(
            repo_root, speaker_id="F01", condition=Condition.DYSARTHRIC, gender=Gender.FEMALE
        ),
        make_record(
            repo_root, speaker_id="FC01", condition=Condition.CONTROL, gender=Gender.FEMALE
        ),
        make_record(
            repo_root, speaker_id="M01", condition=Condition.DYSARTHRIC, gender=Gender.MALE
        ),
        make_record(repo_root, speaker_id="MC01", condition=Condition.CONTROL, gender=Gender.MALE),
        excluded,
        make_record(
            repo_root,
            speaker_id="MC02",
            condition=Condition.CONTROL,
            gender=Gender.MALE,
            prompt_type=PromptType.ARTICULATION_TASK,
        ),
    )
    config = make_config(repo_root, tuple(record.logical_id for record in records[:4]))
    write_manifest_jsonl(config.source.manifest_path, records)
    source = load_manifest_source(config.source.manifest_path)
    core = select_subset(source.records, "core")
    raw_fingerprint = "d" * 64
    _write_json(
        config.source.module0_summary_path,
        {
            "source": {"manifest_sha256": source.fingerprint},
            "downstream_subset_policy": {"core": {"record_count": len(core)}},
            "raw_data_immutability": {"metadata_sha256": raw_fingerprint},
        },
    )
    module1_records = [{"logical_id": record.logical_id} for record in core]
    _write_json(
        config.source.module1_summary_path,
        {
            "status": "full_build_completed",
            "source": {
                "manifest_fingerprint": source.fingerprint,
                "raw_metadata_sha256": raw_fingerprint,
            },
            "selection": {
                "subset": "core",
                "limit": None,
                "subset_record_count": len(core),
                "selected_record_count": len(core),
            },
            "configuration": {
                "config_hash": MODULE1_HASH,
                "all_mode_default": True,
                "selected_mode_ablation_enabled": False,
            },
            "completion": {"successful": len(core), "failed": 0},
            "records": module1_records,
        },
    )
    _write_json(
        config.source.module1_signoff_path,
        {
            "status": "PASS",
            "source": {"manifest_sha256": source.fingerprint, "config_hash": MODULE1_HASH},
            "completion": {
                "expected": len(core),
                "valid_structural_cache_entries": len(core),
                "failed": 0,
            },
            "configuration": {
                "config_hash": MODULE1_HASH,
                "all_mode_default": True,
                "selected_mode_ablation_enabled": False,
            },
        },
    )
    for index, record in enumerate(core):
        phase = np.arange(800, dtype=np.float64)
        waveform = (0.05 * np.sin(2 * np.pi * (180 + index * 20) * phase / 16_000)).astype(
            np.float64
        )
        write_module1_entry(config, source.fingerprint, record, waveform, converged=index != 0)
    return config, records


def write_module1_entry(
    config: Module2Config,
    manifest_fingerprint: str,
    record: ManifestRecord,
    waveform: np.ndarray[Any, np.dtype[np.float64]],
    *,
    converged: bool,
    policy: str = "all_modes",
) -> Path:
    directory = (
        config.source.module1_cache_root
        / MODULE1_HASH
        / record.speaker_id
        / record.session_id
        / record.prompt_id
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "reconstruction.npy"
    np.save(path, waveform, allow_pickle=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    document = {
        "schema_version": "1.0",
        "complete": True,
        "logical_id": record.logical_id,
        "config_hash": MODULE1_HASH,
        "source_manifest_fingerprint": manifest_fingerprint,
        "source_primary_audio_path": record.primary_audio_path,
        "source_audio_sha256": record.primary_audio_sha256,
        "sample_count": waveform.size,
        "artifacts": {
            "reconstruction": {
                "filename": "reconstruction.npy",
                "dtype": "float64",
                "shape": [waveform.size],
                "byte_count": path.stat().st_size,
                "sha256": digest,
            }
        },
        "metadata": {
            "module": "1_shared_vmd",
            "reconstruction_policy": policy,
            "audio": {
                "working_sample_rate": 16_000,
                "working_channels": 1,
                "working_sample_count": waveform.size,
            },
            "source": {
                "speaker_id": record.speaker_id,
                "session_id": record.session_id,
                "prompt_id": record.prompt_id,
                "condition": record.condition.value,
                "primary_microphone": record.primary_microphone.value
                if record.primary_microphone
                else None,
            },
            "vmd": {"converged": converged},
        },
    }
    _write_json(directory / "metadata.json", document)
    return directory


def write_config_yaml(config: Module2Config, path: Path) -> None:
    root = config.repo_root

    def relative(value: Path) -> str:
        return value.relative_to(root).as_posix()

    raw: dict[str, Any] = {
        "schema_version": config.schema_version,
        "implementation": config.implementation,
        "source": {
            "manifest_path": relative(config.source.manifest_path),
            "module0_summary_path": relative(config.source.module0_summary_path),
            "module1_summary_path": relative(config.source.module1_summary_path),
            "module1_signoff_path": relative(config.source.module1_signoff_path),
            "module1_cache_root": relative(config.source.module1_cache_root),
            "module1_config_hash": config.source.module1_config_hash,
            "default_subset": config.source.default_subset,
        },
        "mel": asdict(config.mel),
        "recurrence": asdict(config.recurrence),
        "visibility_graph": {
            **asdict(config.visibility_graph),
            "node_features": list(config.visibility_graph.node_features),
        },
        "cache": {
            "enabled": config.cache.enabled,
            "resume": config.cache.resume,
            "root": relative(config.cache.root),
        },
        "execution": {
            "progress_every": config.execution.progress_every,
            "diagnostic_examples": config.execution.diagnostic_examples,
            "smoke_logical_ids": list(config.execution.smoke_logical_ids),
        },
        "outputs": {
            "summary": relative(config.outputs.summary),
            "failures": relative(config.outputs.failures),
            "pipeline_log": relative(config.outputs.pipeline_log),
            "plots": relative(config.outputs.plots),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
