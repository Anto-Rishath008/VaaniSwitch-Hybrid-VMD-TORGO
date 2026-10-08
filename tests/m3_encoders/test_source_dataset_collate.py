"""Signed source rejection, provenance, dataset, and collate regression tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from tests.m3_encoders.helpers import make_sample, prepare_tiny_m3_repo
from vaaniswitch.m2_representations.cache import RepresentationCacheError
from vaaniswitch.m3_encoders.collate import collate_encoder_samples
from vaaniswitch.m3_encoders.config import Module3Config
from vaaniswitch.m3_encoders.dataset import EncoderRepresentationDataset
from vaaniswitch.m3_encoders.source import (
    SignedRepresentationError,
    SignedRepresentationSource,
)


def _entry_directory(config: Module3Config, logical_id: str) -> Path:
    return (
        config.source.representation_cache_root
        / config.source.representation_config_hash
        / Path(logical_id)
    )


def _rewrite_artifact(
    config: Module3Config,
    logical_id: str,
    key: str,
    filename: str,
    array: np.ndarray[Any, Any],
) -> None:
    directory = _entry_directory(config, logical_id)
    path = directory / filename
    np.save(path, array, allow_pickle=False)
    metadata_path = directory / "metadata.json"
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    document["artifacts"][key] = {
        "filename": filename,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "byte_count": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    metadata_path.write_text(json.dumps(document), encoding="utf-8")


def test_signed_source_dataset_retains_logical_speaker_condition_and_fold(
    tmp_path: Path,
) -> None:
    config = prepare_tiny_m3_repo(tmp_path)
    source = SignedRepresentationSource(config)
    dataset = EncoderRepresentationDataset(config, source=source)
    sample = dataset[0]

    assert len(dataset) == 4
    assert sample.provenance.logical_id == source.contract.records[0].logical_id
    assert sample.provenance.speaker_id == source.contract.records[0].speaker_id
    assert sample.provenance.condition == "dysarthric"
    assert sample.provenance.evaluation_fold == 1
    assert sample.mel.shape == (1, 224, 224)
    assert sample.recurrence.dtype == torch.uint8


def test_dataset_uses_manifest_order_and_never_random_splits(tmp_path: Path) -> None:
    config = prepare_tiny_m3_repo(tmp_path)
    dataset = EncoderRepresentationDataset(config)
    assert [record.logical_id for record in dataset.records] == [
        record.logical_id for record in dataset.source.contract.records
    ]
    assert {item.evaluation_fold for item in (dataset[index].provenance for index in range(4))}


def test_collate_stacks_images_pads_nodes_masks_and_preserves_edges() -> None:
    samples = (make_sample(3), make_sample(1))
    batch = collate_encoder_samples(samples)
    assert batch.mel.shape == (2, 1, 224, 224)
    assert batch.recurrence.shape == (2, 1, 224, 224)
    assert batch.graph_nodes.shape == (2, 3, 2)
    assert batch.graph_node_mask.tolist() == [[True, True, True], [True, False, False]]
    assert batch.graph_edges[0].shape[1] == 4
    assert batch.graph_edges[1].shape[1] == 0
    assert batch.to("cpu").provenance == batch.provenance
    with pytest.raises(ValueError, match="empty"):
        collate_encoder_samples([])


def test_wrong_representation_and_module1_hashes_are_rejected(tmp_path: Path) -> None:
    config = prepare_tiny_m3_repo(tmp_path)
    wrong_representation = replace(
        config,
        source=replace(config.source, representation_config_hash="c" * 64),
    )
    with pytest.raises(SignedRepresentationError, match="not production"):
        SignedRepresentationSource(wrong_representation)
    wrong_module1 = replace(
        config, source=replace(config.source, module1_config_hash="c" * 64)
    )
    with pytest.raises(SignedRepresentationError, match="Module 1 provenance"):
        SignedRepresentationSource(wrong_module1)


def test_artifact_hash_corruption_and_unexpected_files_are_rejected(tmp_path: Path) -> None:
    config = prepare_tiny_m3_repo(tmp_path)
    source = SignedRepresentationSource(config)
    logical_id = source.contract.records[0].logical_id
    directory = _entry_directory(config, logical_id)
    mel_path = directory / "mel.npy"
    content = bytearray(mel_path.read_bytes())
    content[-1] ^= 1
    mel_path.write_bytes(content)
    with pytest.raises(RepresentationCacheError, match="SHA-256"):
        source.load(logical_id)

    config = prepare_tiny_m3_repo(tmp_path / "unexpected")
    source = SignedRepresentationSource(config)
    logical_id = source.contract.records[0].logical_id
    (_entry_directory(config, logical_id) / "extra.bin").write_bytes(b"unexpected")
    with pytest.raises(SignedRepresentationError, match="files do not match"):
        source.load(logical_id)


@pytest.mark.parametrize(
    "case",
    [
        "mel_dtype",
        "mel_shape",
        "recurrence_dtype",
        "recurrence_shape",
        "nodes",
        "indices",
        "duplicate",
        "loop",
    ],
)
def test_malformed_or_nonfinite_arrays_are_rejected(tmp_path: Path, case: str) -> None:
    config = prepare_tiny_m3_repo(tmp_path)
    source = SignedRepresentationSource(config)
    logical_id = source.contract.records[0].logical_id
    entry = source.load(logical_id)
    if case == "mel_dtype":
        _rewrite_artifact(config, logical_id, "mel", "mel.npy", entry.mel.astype(np.float64))
        match = "identity"
    elif case == "mel_shape":
        _rewrite_artifact(config, logical_id, "mel", "mel.npy", entry.mel[:-1])
        match = "float32 224x224"
    elif case == "recurrence_dtype":
        _rewrite_artifact(
            config,
            logical_id,
            "recurrence",
            "recurrence.npy",
            entry.recurrence.astype(np.int16),
        )
        match = "identity"
    elif case == "recurrence_shape":
        _rewrite_artifact(
            config,
            logical_id,
            "recurrence",
            "recurrence.npy",
            entry.recurrence[:-1],
        )
        match = "uint8 224x224"
    elif case == "nodes":
        nodes = entry.graph_nodes.copy()
        nodes[0, 0] = np.nan
        _rewrite_artifact(config, logical_id, "graph_nodes", "graph_nodes.npy", nodes)
        match = "finite"
    else:
        if case == "indices":
            edges = np.asarray([[0, 999], [1, 0]], dtype=np.int16)
            match = "outside"
        elif case == "duplicate":
            edges = np.asarray([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.int16)
            match = "duplicate"
        else:
            edges = np.asarray([[0], [0]], dtype=np.int16)
            match = "self-loops"
        _rewrite_artifact(config, logical_id, "graph_edges", "graph_edges.npy", edges)
    with pytest.raises(RepresentationCacheError, match=match):
        source.load(logical_id)


def test_stale_provenance_and_fold_documents_are_rejected(tmp_path: Path) -> None:
    config = prepare_tiny_m3_repo(tmp_path)
    source = SignedRepresentationSource(config)
    logical_id = source.contract.records[0].logical_id
    path = _entry_directory(config, logical_id) / "metadata.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["module1_config_hash"] = "c" * 64
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(SignedRepresentationError, match="provenance"):
        source.load(logical_id)

    config = prepare_tiny_m3_repo(tmp_path / "folds")
    folds_path = config.source.speaker_folds_path
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    folds["source"]["manifest_sha256"] = "c" * 64
    folds_path.write_text(json.dumps(folds), encoding="utf-8")
    with pytest.raises(SignedRepresentationError, match="fold provenance"):
        SignedRepresentationSource(config)


def test_actual_signed_core_and_required_edge_records_are_loadable() -> None:
    root = Path(__file__).resolve().parents[2]
    config = Module3Config.from_yaml(root / "configs/models/encoders.yaml")
    source = SignedRepresentationSource(config)
    assert source.contract.signed_record_count == 8_673
    ceiling = source.load("F03/Session1/0179")
    longest = source.load("F03/Session1/0118")
    assert ceiling.metadata["source"]["module1_converged"] is False
    assert longest.metadata["source"]["sample_count"] == 3_101_258
