"""Tiny end-to-end Module 0 pipeline test without TORGO test audio."""

import json
from pathlib import Path

import pytest

from tests.m0_data.helpers import add_prompt, make_config
from vaaniswitch.m0_data.pipeline import run_module0, validate_generated_outputs
from vaaniswitch.m0_data.reporting import snapshot_raw_tree


def test_tiny_pipeline_writes_all_outputs_without_mutating_raw(tmp_path: Path) -> None:
    config = make_config(tmp_path, requested_max_folds=2, minimum_folds=2)
    for group, speakers in {
        "F": ("F01", "F02"),
        "M": ("M01", "M02"),
        "FC": ("FC01", "FC02"),
        "MC": ("MC01", "MC02"),
    }.items():
        for speaker in speakers:
            head, _ = add_prompt(
                config,
                group=group,
                speaker=speaker,
                session="Session1",
                prompt_id="0001",
                transcript="hello",
                head=True,
                array=True,
            )
            if speaker == "F01":
                assert head is not None
                head.write_bytes(b"")
    before = snapshot_raw_tree(config.raw_root)

    summary = run_module0(config)

    after = snapshot_raw_tree(config.raw_root)
    assert before == after
    assert summary["logical_utterances"]["total"] == 8
    assert summary["speakers"]["by_group"] == {"F": 2, "FC": 2, "M": 2, "MC": 2}
    assert summary["discovery_warnings"] == {"count": 0, "items": []}
    assert summary["leakage_and_duplicates"]["status"] == "PASS"
    assert summary["logical_utterances"]["excluded_count"] == 0
    assert summary["microphone_coverage"]["selected_head_primary"] == 7
    assert summary["microphone_coverage"]["selected_array_primary"] == 1
    assert summary["microphone_coverage"]["primary_selection_reasons"] == {
        "head_structurally_usable": 7,
        "head_unusable_array_fallback": 1,
    }
    assert summary["downstream_subset_policy"]["core"]["record_count"] == 8
    assert summary["downstream_subset_policy"]["extended"]["record_count"] == 8
    assert all(path.is_file() for path in config.outputs.values())
    pipeline_log = config.outputs["pipeline_log"].read_text(encoding="utf-8")
    assert str(tmp_path) not in pipeline_log
    assert "data/raw/torgo/extracted" in pipeline_log
    validation = validate_generated_outputs(config)
    assert validation["manifest_records"] == 8
    with config.outputs["speaker_folds"].open("r", encoding="utf-8") as stream:
        fold_data = json.load(stream)
    assert fold_data["strategy"]["selected_folds"] == 2

    audio_quality_content = config.outputs["audio_quality"].read_bytes()
    config.outputs["audio_quality"].unlink()
    with pytest.raises(ValueError, match="required Module 0 output is missing"):
        validate_generated_outputs(config)
    config.outputs["audio_quality"].write_bytes(audio_quality_content)

    manifest_csv_content = config.outputs["manifest_csv"].read_text(encoding="utf-8")
    header = manifest_csv_content.splitlines(keepends=True)[0]
    config.outputs["manifest_csv"].write_text(header, encoding="utf-8")
    with pytest.raises(ValueError, match="CSV/JSONL row-count mismatch"):
        validate_generated_outputs(config)
    config.outputs["manifest_csv"].write_text(manifest_csv_content, encoding="utf-8")
    assert validate_generated_outputs(config)["leakage_status"] == "PASS"
