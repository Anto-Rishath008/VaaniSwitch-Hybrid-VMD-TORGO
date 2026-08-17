"""Tiny end-to-end Module 1 cache, report, plot, and CLI tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Never

import pytest

from tests.m1_vmd.helpers import prepare_tiny_repo, write_config_yaml
from vaaniswitch.m0_data.reporting import snapshot_raw_tree
from vaaniswitch.m1_vmd import pipeline
from vaaniswitch.m1_vmd.cli import main
from vaaniswitch.m1_vmd.pipeline import (
    Module1PipelineError,
    run_module1,
    validate_module1,
)


def test_tiny_pipeline_processes_primary_audio_writes_plots_and_preserves_raw(
    tmp_path: Path,
) -> None:
    config, _ = prepare_tiny_repo(tmp_path)
    before = snapshot_raw_tree(config.dataset.raw_root)

    summary = run_module1(config, subset="core", limit=3)

    after = snapshot_raw_tree(config.dataset.raw_root)
    assert before == after
    assert summary["status"] == "smoke_validated"
    assert summary["selection"]["selected_record_count"] == 3
    assert summary["selection"]["conditions"] == ["control", "dysarthric"]
    assert summary["completion"] == {
        "computed": 3,
        "cache_hits": 0,
        "failed": 0,
        "successful": 3,
    }
    assert summary["configuration"]["all_mode_default"] is True
    assert summary["configuration"]["selected_mode_ablation_enabled"] is False
    assert len(summary["outputs"]["plots"]) == 2
    for relative in summary["outputs"]["plots"]:
        content = (config.repo_root / relative).read_text(encoding="utf-8")
        assert "all-mode VMD reconstruction diagnostic" in content
        assert "Relative mode energy" in content
        assert str(tmp_path) not in content
    assert any(
        record["primary_microphone"] == "array" for record in summary["records"]
    )
    validation = validate_module1(config, subset="core", limit=3)
    assert validation["status"] == "PASS"
    assert validation["valid_cache_entries"] == 3


def test_existing_valid_cache_is_skipped_without_rewriting_arrays(tmp_path: Path) -> None:
    config, _ = prepare_tiny_repo(tmp_path)
    first = run_module1(config, subset="core", limit=2)
    first_record = first["records"][0]
    cache_directory = config.cache.root / first_record["cache_directory"]
    modes_path = cache_directory / "modes.npy"
    timestamp = modes_path.stat().st_mtime_ns

    second = run_module1(config, subset="core", limit=2)

    assert second["completion"]["computed"] == 0
    assert second["completion"]["cache_hits"] == 2
    assert modes_path.stat().st_mtime_ns == timestamp


def test_cli_smoke_build_and_validation_work(tmp_path: Path, capsys: object) -> None:
    config, _ = prepare_tiny_repo(tmp_path)
    config_path = tmp_path / "configs/vmd/default.yaml"
    write_config_yaml(config, config_path)

    assert main(["build", "--config", str(config_path), "--subset", "core", "--limit", "2"]) == 0
    build_output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert json.loads(build_output)["status"] == "smoke_validated"

    assert (
        main(
            ["validate", "--config", str(config_path), "--subset", "core", "--limit", "2"]
        )
        == 0
    )
    validation_output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert json.loads(validation_output)["status"] == "PASS"


def test_semantically_incomplete_cache_is_recomputed_on_resume(tmp_path: Path) -> None:
    config, _ = prepare_tiny_repo(tmp_path)
    first = run_module1(config, subset="core", limit=2)
    cache_directory = config.cache.root / first["records"][0]["cache_directory"]
    metadata_path = cache_directory / "metadata.json"
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    del document["metadata"]["timing"]
    metadata_path.write_text(json.dumps(document), encoding="utf-8")

    resumed = run_module1(config, subset="core", limit=2)

    assert resumed["completion"]["computed"] == 1
    assert resumed["completion"]["cache_hits"] == 1
    repaired = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert "timing" in repaired["metadata"]


def test_validation_rejects_missing_referenced_plot(tmp_path: Path) -> None:
    config, _ = prepare_tiny_repo(tmp_path)
    summary = run_module1(config, subset="core", limit=2)
    plot_path = config.repo_root / summary["outputs"]["plots"][0]
    plot_path.unlink()

    with pytest.raises(Module1PipelineError, match="plot is missing"):
        validate_module1(config, subset="core", limit=2)


def test_failed_smoke_report_is_never_labelled_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = prepare_tiny_repo(tmp_path)

    def fail_record(*args: object, **kwargs: object) -> Never:
        raise RuntimeError("synthetic record failure")

    monkeypatch.setattr(pipeline, "_process_record", fail_record)
    with pytest.raises(Module1PipelineError, match="2 failed records"):
        run_module1(config, subset="core", limit=2)

    summary = json.loads(config.outputs.summary.read_text(encoding="utf-8"))
    assert summary["status"] == "smoke_failed"
    assert summary["completion"]["failed"] == 2


def test_explicit_selected_mode_ablation_is_labelled_as_an_ablation(tmp_path: Path) -> None:
    config, _ = prepare_tiny_repo(tmp_path)
    ablation = replace(
        config,
        reconstruction=replace(
            config.reconstruction,
            selected_modes_ablation=replace(
                config.reconstruction.selected_modes_ablation,
                enabled=True,
                mode_indices=(1, 2),
            ),
        ),
        execution=replace(config.execution, plot_examples=1),
    )

    summary = run_module1(ablation, subset="core", limit=1)

    assert summary["scientific_label"] == (
        "explicit selected-mode VMD ablation reconstruction"
    )
    assert summary["reconstruction"]["policy"] == "selected_modes_explicit:1,2"
    plot = config.repo_root / summary["outputs"]["plots"][0]
    assert "selected-mode VMD ablation" in plot.read_text(encoding="utf-8")
