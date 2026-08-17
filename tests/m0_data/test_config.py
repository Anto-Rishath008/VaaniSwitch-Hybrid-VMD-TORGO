"""YAML loading and immutable raw-path safety tests."""

from dataclasses import replace
from pathlib import Path

import pytest

from vaaniswitch.m0_data.config import ConfigError, TorgoConfig

PRODUCTION_CONFIG = Path("configs/data/torgo.yaml")


def test_production_yaml_loads_relative_to_explicit_repo_root(tmp_path: Path) -> None:
    config = TorgoConfig.from_yaml(PRODUCTION_CONFIG, repo_root=tmp_path)

    assert config.raw_root == (tmp_path / "data/raw/torgo/extracted").resolve()
    assert config.microphone_priority == ("head", "array")
    assert config.evaluation.requested_max_folds == 5
    assert config.downstream_subsets["core"].usability_statuses == {"usable"}
    assert config.downstream_subsets["extended"].usability_statuses == {
        "usable",
        "review",
    }
    assert all(path.is_relative_to(tmp_path) for path in config.outputs.values())


def test_config_rejects_output_inside_immutable_raw_tree(tmp_path: Path) -> None:
    config_path = tmp_path / "configs/data/torgo.yaml"
    config_path.parent.mkdir(parents=True)
    text = PRODUCTION_CONFIG.read_text(encoding="utf-8").replace(
        "manifest_jsonl: data/manifests/torgo_manifest.jsonl",
        "manifest_jsonl: data/raw/torgo/extracted/forbidden.jsonl",
    )
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="immutable raw_root"):
        TorgoConfig.from_yaml(config_path)


def test_config_rejects_raw_root_outside_repository(tmp_path: Path) -> None:
    config_path = tmp_path / "configs/data/torgo.yaml"
    config_path.parent.mkdir(parents=True)
    text = PRODUCTION_CONFIG.read_text(encoding="utf-8").replace(
        "raw_root: data/raw/torgo/extracted",
        "raw_root: C:/outside/torgo",
    )
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="raw_root must remain"):
        TorgoConfig.from_yaml(config_path)


def test_config_rejects_tampered_canonical_group_label(tmp_path: Path) -> None:
    config_path = tmp_path / "configs/data/torgo.yaml"
    config_path.parent.mkdir(parents=True)
    text = PRODUCTION_CONFIG.read_text(encoding="utf-8").replace(
        "  F:\n    condition: dysarthric",
        "  F:\n    condition: control",
    )
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="group F must map"):
        TorgoConfig.from_yaml(config_path)


def test_config_requires_content_hashing_for_leakage_audit(tmp_path: Path) -> None:
    config = TorgoConfig.from_yaml(PRODUCTION_CONFIG, repo_root=tmp_path)
    invalid = replace(config, quality=replace(config.quality, compute_sha256=False))

    with pytest.raises(ConfigError, match="compute_sha256 must be true"):
        invalid.validate()
