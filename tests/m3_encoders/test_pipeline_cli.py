"""Module 3 validation, smoke artifact, and command-line tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.m3_encoders.helpers import prepare_tiny_m3_repo, write_module3_yaml
from vaaniswitch.m3_encoders import pipeline
from vaaniswitch.m3_encoders.cli import main
from vaaniswitch.m3_encoders.models import ParallelTokenEncoders
from vaaniswitch.m3_encoders.pipeline import Module3PipelineError, validate_module3


def _disable_pretrained_download(monkeypatch: pytest.MonkeyPatch) -> None:
    def build_without_weights(config: object, *, load_pretrained: bool) -> ParallelTokenEncoders:
        del load_pretrained
        return ParallelTokenEncoders(config, load_pretrained=False)  # type: ignore[arg-type]

    monkeypatch.setattr(pipeline, "ParallelTokenEncoders", build_without_weights)


def test_validation_and_cli_consume_only_configured_signed_records(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = prepare_tiny_m3_repo(tmp_path)
    path = tmp_path / "configs/models/encoders.yaml"
    write_module3_yaml(config, path)
    result = validate_module3(config, limit=2)
    assert result["status"] == "PASS"
    assert result["valid_representation_entries"] == 2
    assert main(["validate", "--config", str(path), "--limit", "2"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["encoder_config_hash"] == config.config_hash


def test_smoke_forward_gradient_roundtrip_timing_and_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = prepare_tiny_m3_repo(tmp_path)
    _disable_pretrained_download(monkeypatch)
    summary = pipeline.run_module3_smoke(
        config, limit=2, batch_size=1, timing_repetitions=1
    )

    assert summary["status"] == "PASS"
    assert summary["outputs"] == {
        "mel_tokens": [2, 16, 128],
        "graph_token": [2, 1, 128],
        "recurrence_tokens": [2, 16, 128],
        "nonlinear_tokens": [2, 17, 128],
        "all_finite": True,
    }
    assert all(summary["deterministic_eval"].values())
    assert summary["state_dict_roundtrip"]["exact_match"]
    assert summary["module4_started"] is False
    assert config.outputs.summary.is_file()
    assert config.outputs.parameter_audit.is_file()
    assert config.outputs.pipeline_log.is_file()
    assert config.outputs.failures.read_text(encoding="utf-8").splitlines() == [
        "logical_id,error_type,message"
    ]
    assert not any(path.name.startswith("tokens") for path in tmp_path.rglob("*"))


def test_smoke_cli_works_without_network_in_unit_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = prepare_tiny_m3_repo(tmp_path)
    path = tmp_path / "configs/models/encoders.yaml"
    write_module3_yaml(config, path)
    _disable_pretrained_download(monkeypatch)
    result = main(
        [
            "smoke",
            "--config",
            str(path),
            "--limit",
            "1",
            "--batch-size",
            "1",
            "--timing-repetitions",
            "1",
        ]
    )
    assert result == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"


def test_invalid_limits_and_cli_errors_are_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = prepare_tiny_m3_repo(tmp_path)
    with pytest.raises(Module3PipelineError, match="positive"):
        validate_module3(config, limit=0)
    with pytest.raises(Module3PipelineError, match="too small"):
        validate_module3(config, limit=5)
    assert main(["validate", "--config", str(tmp_path / "missing.yaml")]) == 1
    assert "ERROR:" in capsys.readouterr().err
