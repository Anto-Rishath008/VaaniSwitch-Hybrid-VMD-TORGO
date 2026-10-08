"""Locked configuration and fixed input preprocessing tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from tests.m3_encoders.helpers import make_model_config, write_module3_yaml
from vaaniswitch.m3_encoders.config import EncoderConfigError, Module3Config
from vaaniswitch.m3_encoders.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    preprocess_mel,
    preprocess_recurrence,
)


def test_production_config_locks_128_and_has_stable_hash() -> None:
    root = Path(__file__).resolve().parents[2]
    config = Module3Config.from_yaml(root / "configs/models/encoders.yaml")
    fusion = yaml.safe_load((root / "configs/models/fusion.yaml").read_text(encoding="utf-8"))

    assert config.token_dimension == fusion["token_dim"] == 128
    assert config.graph.hidden_dim == 96
    assert config.recurrence.channels == (16, 32, 64)
    assert config.config_hash == "7c8e44eddb11704d08e45a1b63bc9cc54b28d679f5e3a3ae1a474b61e3a40383"


def test_config_hash_excludes_paths_and_smoke_selection(tmp_path: Path) -> None:
    config = make_model_config(tmp_path)
    moved = replace(
        config,
        smoke_logical_ids=tuple(reversed(config.smoke_logical_ids)),
        outputs=replace(config.outputs, summary=(tmp_path / "artifacts/other.json").resolve()),
    )
    assert config.config_hash == moved.config_hash


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda config: replace(config, token_dimension=192), "token dimension 128"),
        (
            lambda config: replace(config, graph=replace(config.graph, hidden_dim=64)),
            "graph encoder",
        ),
        (
            lambda config: replace(
                config, recurrence=replace(config.recurrence, channels=(32, 64, 128))
            ),
            "recurrence encoder",
        ),
        (
            lambda config: replace(
                config,
                outputs=replace(
                    config.outputs, summary=(config.repo_root / "data/summary.json").resolve()
                ),
            ),
            "cannot overwrite",
        ),
    ],
)
def test_locked_config_variants_are_rejected(
    tmp_path: Path, change: object, message: str
) -> None:
    config = make_model_config(tmp_path)
    assert callable(change)
    invalid = change(config)
    with pytest.raises(EncoderConfigError, match=message):
        invalid._validate()


def test_yaml_schema_and_path_escape_are_rejected(tmp_path: Path) -> None:
    config = make_model_config(tmp_path)
    path = tmp_path / "configs/models/encoders.yaml"
    write_module3_yaml(config, path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["unknown"] = True
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(EncoderConfigError, match="fields do not match"):
        Module3Config.from_yaml(path, repo_root=tmp_path)

    write_module3_yaml(config, path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["source"]["manifest_path"] = "../manifest.jsonl"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(EncoderConfigError, match="repository-relative"):
        Module3Config.from_yaml(path, repo_root=tmp_path)


def test_mel_fixed_clipping_scaling_and_replication() -> None:
    values = torch.tensor([[-120.0, -100.0, 20.0, 40.0]], dtype=torch.float32)
    values = values.reshape(1, 1, 1, 4).repeat(1, 1, 224, 56)
    result = preprocess_mel(values, imagenet_normalization=False)

    assert result.shape == (1, 3, 224, 224)
    assert torch.equal(result[:, 0], result[:, 1])
    assert torch.equal(result[:, 1], result[:, 2])
    assert result[0, 0, 0, 0] == 0
    assert result[0, 0, 0, 1] == 0
    assert result[0, 0, 0, 2] == 1
    assert result[0, 0, 0, 3] == 1


def test_imagenet_normalization_is_fixed_and_corpus_independent() -> None:
    mel = torch.full((2, 1, 224, 224), -40.0)
    result = preprocess_mel(mel)
    scaled = 0.5
    for channel, (mean, std) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD, strict=True)):
        assert result[0, channel, 0, 0].item() == pytest.approx((scaled - mean) / std)
    assert torch.equal(result[0], result[1])


def test_preprocessing_rejects_malformed_and_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        preprocess_mel(torch.zeros((1, 224, 224)))
    mel = torch.zeros((1, 1, 224, 224))
    mel[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        preprocess_mel(mel)
    with pytest.raises(ValueError, match="exceed"):
        preprocess_mel(torch.zeros((1, 1, 224, 224)), db_min=1, db_max=1)


def test_recurrence_uint8_converts_exactly_to_binary_float32() -> None:
    recurrence = torch.eye(224, dtype=torch.uint8).reshape(1, 1, 224, 224)
    result = preprocess_recurrence(recurrence)
    assert result.dtype == torch.float32
    assert torch.equal(result, recurrence.to(torch.float32))
    invalid = recurrence.clone()
    invalid[0, 0, 0, 1] = 2
    with pytest.raises(ValueError, match="binary"):
        preprocess_recurrence(invalid)
