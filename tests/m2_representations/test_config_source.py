"""Configuration and signed Module 1 source boundary tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tests.m2_representations.helpers import MODULE1_HASH, prepare_tiny_repo
from vaaniswitch.m2_representations.config import (
    Module2Config,
    RepresentationConfigError,
)
from vaaniswitch.m2_representations.mel import generate_log_mel
from vaaniswitch.m2_representations.recurrence import generate_recurrence
from vaaniswitch.m2_representations.source import (
    SourceValidationError,
    load_reconstruction,
    load_source_contract,
)
from vaaniswitch.m2_representations.visibility import generate_visibility_graph


def _metadata_path(config: Module2Config, logical_id: str) -> Path:
    return config.source.module1_cache_root / MODULE1_HASH / Path(logical_id) / "metadata.json"


def _tamper_metadata(config: Module2Config, logical_id: str, change: object) -> None:
    path = _metadata_path(config, logical_id)
    document = json.loads(path.read_text(encoding="utf-8"))
    assert callable(change)
    change(document)
    path.write_text(json.dumps(document), encoding="utf-8")


def test_production_module1_reconstruction_is_loaded_not_raw_wav(tmp_path: Path) -> None:
    config, _ = prepare_tiny_repo(tmp_path)
    contract = load_source_contract(config)
    source = load_reconstruction(config, contract, contract.records[0])

    expected = np.load(config.repo_root / source.relative_path, allow_pickle=False)
    np.testing.assert_array_equal(source.waveform, expected)
    assert source.relative_path.endswith("reconstruction.npy")
    assert "data/raw" not in source.relative_path


def test_experimental_tau_sweep_source_path_is_rejected(tmp_path: Path) -> None:
    config, _ = prepare_tiny_repo(tmp_path)
    invalid = replace(
        config,
        source=replace(
            config.source,
            module1_cache_root=(tmp_path / "data/processed/vmd_fidelity_sweep").resolve(),
        ),
    )
    with pytest.raises(RepresentationConfigError, match="production data/processed/vmd"):
        invalid.validate()


def test_wrong_module1_config_hash_is_rejected(tmp_path: Path) -> None:
    config, _ = prepare_tiny_repo(tmp_path)
    invalid = replace(config, source=replace(config.source, module1_config_hash="c" * 64))
    with pytest.raises(SourceValidationError, match="production VMD hash"):
        load_source_contract(invalid)


def test_selected_mode_reconstruction_is_rejected(tmp_path: Path) -> None:
    config, _ = prepare_tiny_repo(tmp_path)
    contract = load_source_contract(config)
    record = contract.records[0]
    _tamper_metadata(
        config,
        record.logical_id,
        lambda document: document["metadata"].update(
            {"reconstruction_policy": "selected_modes_explicit:1,2"}
        ),
    )
    with pytest.raises(SourceValidationError, match="selected-mode"):
        load_reconstruction(config, contract, record)


def test_nonfinite_reconstruction_is_rejected_even_with_updated_hash(tmp_path: Path) -> None:
    config, _ = prepare_tiny_repo(tmp_path)
    contract = load_source_contract(config)
    record = contract.records[0]
    metadata_path = _metadata_path(config, record.logical_id)
    directory = metadata_path.parent
    array = np.load(directory / "reconstruction.npy", allow_pickle=False)
    array[0] = np.nan
    np.save(directory / "reconstruction.npy", array, allow_pickle=False)
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    artifact = document["artifacts"]["reconstruction"]
    artifact["byte_count"] = (directory / "reconstruction.npy").stat().st_size
    artifact["sha256"] = hashlib.sha256((directory / "reconstruction.npy").read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(SourceValidationError, match="non-finite"):
        load_reconstruction(config, contract, record)


def test_stale_mismatched_logical_id_is_rejected(tmp_path: Path) -> None:
    config, _ = prepare_tiny_repo(tmp_path)
    contract = load_source_contract(config)
    record = contract.records[0]
    _tamper_metadata(
        config,
        record.logical_id,
        lambda document: document.update({"logical_id": "F01/Session1/9999"}),
    )
    with pytest.raises(SourceValidationError, match="identity"):
        load_reconstruction(config, contract, record)


def test_reconstruction_artifact_hash_is_checked(tmp_path: Path) -> None:
    config, _ = prepare_tiny_repo(tmp_path)
    contract = load_source_contract(config)
    record = contract.records[0]
    path = _metadata_path(config, record.logical_id).parent / "reconstruction.npy"
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    with pytest.raises(SourceValidationError, match="SHA-256"):
        load_reconstruction(config, contract, record)


def test_valid_nonconverged_iteration_ceiling_style_record_is_accepted(tmp_path: Path) -> None:
    config, _ = prepare_tiny_repo(tmp_path)
    contract = load_source_contract(config)
    source = load_reconstruction(config, contract, contract.records[0])
    assert source.converged is False
    assert generate_log_mel(source.waveform, config.mel).image.shape == (224, 224)


def test_excluded_and_nonnormal_manifest_records_cannot_enter_core(tmp_path: Path) -> None:
    config, records = prepare_tiny_repo(tmp_path)
    contract = load_source_contract(config)
    selected_ids = {record.logical_id for record in contract.records}
    assert records[4].logical_id not in selected_ids
    assert records[5].logical_id not in selected_ids
    assert len(contract.records) == 4


def test_transform_outputs_are_independent_of_label_and_speaker(tmp_path: Path) -> None:
    config, _ = prepare_tiny_repo(tmp_path)
    waveform = np.linspace(-0.1, 0.1, 800, dtype=np.float64)
    mel_a = generate_log_mel(waveform, config.mel).image
    mel_b = generate_log_mel(waveform.copy(), config.mel).image
    rp_a = generate_recurrence(waveform, config.recurrence).image
    rp_b = generate_recurrence(waveform.copy(), config.recurrence).image
    vg_a = generate_visibility_graph(waveform, config.visibility_graph)
    vg_b = generate_visibility_graph(waveform.copy(), config.visibility_graph)
    np.testing.assert_array_equal(mel_a, mel_b)
    np.testing.assert_array_equal(rp_a, rp_b)
    np.testing.assert_array_equal(vg_a.node_features, vg_b.node_features)
    np.testing.assert_array_equal(vg_a.edge_index, vg_b.edge_index)


def test_actual_signed_core_contract_remains_8673() -> None:
    root = Path(__file__).resolve().parents[2]
    config = Module2Config.from_yaml(root / "configs/representations/default.yaml")
    contract = load_source_contract(config)
    assert contract.signed_record_count == len(contract.records) == 8_673
