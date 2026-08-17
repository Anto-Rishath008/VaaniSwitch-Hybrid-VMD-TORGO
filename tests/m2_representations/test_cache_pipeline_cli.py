"""Cache integrity, resumability, pipeline, protected-input, and CLI tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from tests.m2_representations.helpers import prepare_tiny_repo, write_config_yaml
from vaaniswitch.m2_representations.cache import (
    MEL_FILENAME,
    CacheIdentity,
    RepresentationCache,
)
from vaaniswitch.m2_representations.cli import main
from vaaniswitch.m2_representations.config import Module2Config
from vaaniswitch.m2_representations.mel import generate_log_mel
from vaaniswitch.m2_representations.pipeline import run_module2, validate_module2
from vaaniswitch.m2_representations.plots import write_crop_audit_diagnostic_svg
from vaaniswitch.m2_representations.recurrence import generate_recurrence
from vaaniswitch.m2_representations.source import load_reconstruction, load_source_contract
from vaaniswitch.m2_representations.visibility import generate_visibility_graph


def _identity_and_arrays(
    tmp_path: Path,
) -> tuple[
    Module2Config,
    CacheIdentity,
    npt.NDArray[np.float32],
    npt.NDArray[np.uint8],
    npt.NDArray[np.float32],
    npt.NDArray[np.int16],
    dict[str, Any],
]:
    config, _ = prepare_tiny_repo(tmp_path)
    contract = load_source_contract(config)
    source = load_reconstruction(config, contract, contract.records[0])
    identity = CacheIdentity(
        logical_id=source.logical_id,
        manifest_fingerprint=contract.manifest_fingerprint,
        module1_config_hash=config.source.module1_config_hash,
        source_reconstruction_path=source.relative_path,
        source_reconstruction_sha256=source.artifact_sha256,
        representation_config_hash=config.config_hash,
    )
    mel = generate_log_mel(source.waveform, config.mel)
    recurrence = generate_recurrence(source.waveform, config.recurrence)
    graph = generate_visibility_graph(source.waveform, config.visibility_graph)
    metadata = {
        "source": {"reconstruction_policy": "all_modes"},
        "mel": {},
        "recurrence": {},
        "visibility_graph": {},
        "timing_seconds": {},
    }
    return (
        config,
        identity,
        mel.image,
        recurrence.image,
        graph.node_features,
        graph.edge_index,
        metadata,
    )


def test_cache_write_read_round_trip_and_artifact_hashes(tmp_path: Path) -> None:
    config, identity, mel, recurrence, nodes, edges, metadata = _identity_and_arrays(tmp_path)
    cache = RepresentationCache(config.cache.root)
    written = cache.write(
        identity,
        mel,
        recurrence,
        nodes,
        edges,
        processing_identity=config.processing_identity,
        representation_metadata=metadata,
    )
    loaded = cache.load(identity)
    np.testing.assert_array_equal(written.mel, loaded.mel)
    document = json.loads((loaded.directory / "metadata.json").read_text(encoding="utf-8"))
    assert (
        document["artifacts"]["mel"]["sha256"]
        == hashlib.sha256((loaded.directory / MEL_FILENAME).read_bytes()).hexdigest()
    )


def test_partial_and_corrupt_cache_are_rejected(tmp_path: Path) -> None:
    config, identity, mel, recurrence, nodes, edges, metadata = _identity_and_arrays(tmp_path)
    cache = RepresentationCache(config.cache.root)
    directory = cache.path_for(identity)
    directory.mkdir(parents=True)
    np.save(directory / MEL_FILENAME, mel, allow_pickle=False)
    assert "completion marker" in (cache.validate(identity).reason or "")
    cache.write(
        identity,
        mel,
        recurrence,
        nodes,
        edges,
        processing_identity=config.processing_identity,
        representation_metadata=metadata,
    )
    content = bytearray((directory / MEL_FILENAME).read_bytes())
    content[-1] ^= 1
    (directory / MEL_FILENAME).write_bytes(content)
    assert "SHA-256 mismatch" in (cache.validate(identity).reason or "")


def test_config_change_creates_new_cache_identity(tmp_path: Path) -> None:
    config, identity, *_ = _identity_and_arrays(tmp_path)
    changed = replace(config, mel=replace(config.mel, epsilon=1.0e-9))
    assert config.config_hash != changed.config_hash
    cache = RepresentationCache(config.cache.root)
    changed_identity = replace(identity, representation_config_hash=changed.config_hash)
    assert cache.path_for(identity) != cache.path_for(changed_identity)


def _tree_metadata_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = path.stat()
        digest.update(
            f"{path.relative_to(root).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
        )
    return digest.hexdigest()


def test_pipeline_smoke_preserves_raw_and_module1_and_writes_diagnostics(tmp_path: Path) -> None:
    config, _ = prepare_tiny_repo(tmp_path)
    raw_before = _tree_metadata_fingerprint(tmp_path / "data/raw")
    module1_before = _tree_metadata_fingerprint(config.source.module1_cache_root)

    summary = run_module2(config, limit=4)

    assert _tree_metadata_fingerprint(tmp_path / "data/raw") == raw_before
    assert _tree_metadata_fingerprint(config.source.module1_cache_root) == module1_before
    assert summary["status"] == "smoke_validated"
    assert summary["completion"] == {
        "computed": 4,
        "cache_hits": 0,
        "failed": 0,
        "successful": 4,
    }
    assert len(summary["outputs"]["plots"]) == 2
    assert {record["condition"] for record in summary["records"]} == {
        "dysarthric",
        "control",
    }
    assert summary["records"][0]["source"]["module1_converged"] is False


def test_crop_audit_diagnostic_identifies_full_and_cropped_signal_contracts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "crop.svg"
    write_crop_audit_diagnostic_svg(
        path,
        logical_id="F03/Session1/0118",
        condition="dysarthric",
        duration_seconds=193.828625,
        crop_start_seconds=92.9143125,
        crop_end_seconds=100.9143125,
        mel=np.zeros((224, 224), dtype=np.float32),
        recurrence=np.eye(224, dtype=np.uint8),
        graph_nodes=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        graph_edges=np.asarray([[0, 1], [1, 0]], dtype=np.int16),
    )

    document = path.read_text(encoding="utf-8")
    assert "centered Mel crop: 92.914312-100.914312 s" in document
    assert "RP and NVG summarize the complete recording" in document


def test_valid_cache_is_skipped_on_resume_and_validation_passes(tmp_path: Path) -> None:
    config, _ = prepare_tiny_repo(tmp_path)
    first = run_module2(config, limit=2)
    directory = config.cache.root / first["records"][0]["cache_directory"]
    timestamp = (directory / MEL_FILENAME).stat().st_mtime_ns
    second = run_module2(config, limit=2)
    assert second["completion"]["cache_hits"] == 2
    assert second["completion"]["computed"] == 0
    assert (directory / MEL_FILENAME).stat().st_mtime_ns == timestamp
    assert validate_module2(config, limit=2)["valid_cache_entries"] == 2


def test_cli_smoke_build_and_validation_work(tmp_path: Path, capsys: object) -> None:
    config, _ = prepare_tiny_repo(tmp_path)
    path = tmp_path / "configs/representations/default.yaml"
    write_config_yaml(config, path)
    assert main(["build", "--config", str(path), "--subset", "core", "--limit", "2"]) == 0
    build_output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert json.loads(build_output)["status"] == "smoke_validated"
    assert main(["validate", "--config", str(path), "--subset", "core", "--limit", "2"]) == 0
    validate_output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert json.loads(validate_output)["status"] == "PASS"
