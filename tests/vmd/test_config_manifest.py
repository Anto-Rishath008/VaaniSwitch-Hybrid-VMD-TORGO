"""Module 1 configuration and Module 0 manifest-interface tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.vmd.helpers import prepare_tiny_repo
from components.data.manifest import write_manifest_jsonl
from components.data.schemas import PromptType, UsabilityStatus
from components.vmd.config import Module1Config, VMDConfigError
from components.vmd.manifest import (
    load_manifest_source,
    resolve_primary_audio,
    select_limited_records,
    select_subset,
)


def test_production_config_locks_initial_all_mode_parameters() -> None:
    config = Module1Config.from_yaml(Path("configs/vmd/default.yaml"))

    assert config.dataset.default_subset == "core"
    assert config.audio.target_sample_rate == 16_000
    assert config.audio.target_channels == 1
    assert config.vmd.modes_k == 5
    assert config.vmd.alpha == 2_000
    assert config.vmd.tau == 0
    assert not config.vmd.dc
    assert config.vmd.init == 1
    assert config.vmd.tolerance == 1.0e-7
    assert config.reconstruction.default == "all_modes"
    assert not config.reconstruction.selected_modes_ablation.enabled
    assert len(config.config_hash) == 64


def test_processing_config_change_produces_new_cache_identity(tmp_path: Path) -> None:
    config, _ = prepare_tiny_repo(tmp_path)
    changed = replace(config, vmd=replace(config.vmd, alpha=2_001.0))

    assert config.config_hash != changed.config_hash


def test_selected_mode_ablation_is_disabled_by_default_and_requires_explicit_indices(
    tmp_path: Path,
) -> None:
    config, _ = prepare_tiny_repo(tmp_path)
    invalid = replace(
        config,
        reconstruction=replace(
            config.reconstruction,
            selected_modes_ablation=replace(
                config.reconstruction.selected_modes_ablation,
                enabled=True,
            ),
        ),
    )

    with pytest.raises(VMDConfigError, match="explicit indices"):
        invalid.validate()


def test_manifest_subset_filtering_matches_core_and_extended_policies(tmp_path: Path) -> None:
    _, records = prepare_tiny_repo(tmp_path)

    core = select_subset(records, "core")
    extended = select_subset(records, "extended")

    assert {record.logical_id for record in core} == {
        "F01/Session1/0001",
        "FC01/Session1/0001",
        "F02/Session1/0001",
    }
    assert {record.logical_id for record in extended} == {
        *(record.logical_id for record in core),
        "M01/Session1/0001",
    }
    assert all(record.prompt_type in {PromptType.WORD, PromptType.SENTENCE} for record in core)
    assert all(record.usability_status is UsabilityStatus.USABLE for record in core)


def test_excluded_record_cannot_enter_either_processing_subset(tmp_path: Path) -> None:
    config, records = prepare_tiny_repo(tmp_path)
    excluded = replace(
        records[0],
        transcript=None,
        prompt_type=PromptType.UNKNOWN,
        normal_speech_eligible=False,
        usability_status=UsabilityStatus.EXCLUDED,
        exclusion_reason="missing_or_unreadable_transcript",
    )
    excluded.validate()
    rewritten = (excluded, *records[1:])
    write_manifest_jsonl(config.dataset.manifest_path, rewritten)

    assert excluded not in select_subset(rewritten, "core")
    assert excluded not in select_subset(rewritten, "extended")


def test_array_primary_fallback_is_used_without_alternate_rediscovery(tmp_path: Path) -> None:
    config, records = prepare_tiny_repo(tmp_path)
    fallback = next(record for record in records if record.speaker_id == "F02")

    resolved = resolve_primary_audio(fallback, config.repo_root)

    assert resolved.name == "0001.wav"
    assert "wav_arrayMic" in resolved.parts
    assert fallback.alternate_audio_path is not None
    assert "wav_headMic" in fallback.alternate_audio_path
    assert resolved.read_bytes() != (config.repo_root / fallback.alternate_audio_path).read_bytes()


def test_limited_selection_is_deterministic_and_condition_balanced(tmp_path: Path) -> None:
    _, records = prepare_tiny_repo(tmp_path)
    core = select_subset(records, "core")

    first = select_limited_records(core, 2)
    second = select_limited_records(core, 2)

    assert first == second
    assert {record.condition.value for record in first} == {"dysarthric", "control"}


def test_manifest_fingerprint_binds_exact_jsonl_bytes(tmp_path: Path) -> None:
    config, _ = prepare_tiny_repo(tmp_path)
    source = load_manifest_source(config.dataset.manifest_path)
    summary = json.loads(config.dataset.module0_summary_path.read_text(encoding="utf-8"))

    assert source.fingerprint == summary["source"]["manifest_sha256"]


def test_unknown_subset_is_rejected(tmp_path: Path) -> None:
    _, records = prepare_tiny_repo(tmp_path)
    with pytest.raises(ValueError, match="unsupported subset"):
        select_subset(records, "training")


def test_config_rejects_cache_or_reports_under_raw_data(tmp_path: Path) -> None:
    config, _ = prepare_tiny_repo(tmp_path)
    raw_cache = replace(config, cache=replace(config.cache, root=config.dataset.raw_root / "vmd"))
    raw_summary = replace(
        config,
        outputs=replace(config.outputs, summary=config.dataset.raw_root / "module1_summary.json"),
    )

    with pytest.raises(VMDConfigError, match="data/processed"):
        raw_cache.validate()
    with pytest.raises(VMDConfigError, match="under artifacts"):
        raw_summary.validate()


def test_config_rejects_overwriting_module0_reports(tmp_path: Path) -> None:
    config, _ = prepare_tiny_repo(tmp_path)
    overwrite = replace(
        config,
        outputs=replace(config.outputs, summary=config.dataset.module0_summary_path),
    )

    with pytest.raises(VMDConfigError, match="overwrite Module 0"):
        overwrite.validate()


def test_config_rejects_normalization_gain_below_one(tmp_path: Path) -> None:
    config, _ = prepare_tiny_repo(tmp_path)
    invalid = replace(config, audio=replace(config.audio, max_gain=0.5))

    with pytest.raises(VMDConfigError, match=r"at least 1\.0"):
        invalid.validate()
