"""Atomic, integrity-checked representation cache with metadata as completion marker."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

MEL_FILENAME = "mel.npy"
RECURRENCE_FILENAME = "recurrence.npy"
GRAPH_NODES_FILENAME = "graph_nodes.npy"
GRAPH_EDGES_FILENAME = "graph_edges.npy"
METADATA_FILENAME = "metadata.json"


class RepresentationCacheError(ValueError):
    """Raised for an incomplete, stale, corrupt, or unsafe Module 2 cache."""


@dataclass(frozen=True, slots=True)
class CacheIdentity:
    logical_id: str
    manifest_fingerprint: str
    module1_config_hash: str
    source_reconstruction_path: str
    source_reconstruction_sha256: str
    representation_config_hash: str


@dataclass(frozen=True, slots=True)
class CacheEntry:
    directory: Path
    mel: npt.NDArray[np.float32]
    recurrence: npt.NDArray[np.uint8]
    graph_nodes: npt.NDArray[np.float32]
    graph_edges: npt.NDArray[np.int16]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CacheValidation:
    valid: bool
    reason: str | None


class RepresentationCache:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def path_for(self, identity: CacheIdentity) -> Path:
        speaker, session, prompt = _logical_id_parts(identity.logical_id)
        _validate_hash(identity.representation_config_hash, "representation_config_hash")
        return self.root / identity.representation_config_hash / speaker / session / prompt

    def is_valid(self, identity: CacheIdentity) -> bool:
        return self.validate(identity).valid

    def validate(self, identity: CacheIdentity) -> CacheValidation:
        try:
            self.load(identity)
        except (OSError, RepresentationCacheError, ValueError) as exc:
            return CacheValidation(False, str(exc))
        return CacheValidation(True, None)

    def write(
        self,
        identity: CacheIdentity,
        mel: npt.NDArray[np.float32],
        recurrence: npt.NDArray[np.uint8],
        graph_nodes: npt.NDArray[np.float32],
        graph_edges: npt.NDArray[np.int16],
        *,
        processing_identity: dict[str, Any],
        representation_metadata: dict[str, Any],
    ) -> CacheEntry:
        _validate_arrays(mel, recurrence, graph_nodes, graph_edges)
        directory = self.path_for(identity)
        directory.mkdir(parents=True, exist_ok=True)
        metadata_path = directory / METADATA_FILENAME
        metadata_path.unlink(missing_ok=True)
        artifacts: dict[str, dict[str, Any]] = {}
        arrays = (
            ("mel", directory / MEL_FILENAME, mel),
            ("recurrence", directory / RECURRENCE_FILENAME, recurrence),
            ("graph_nodes", directory / GRAPH_NODES_FILENAME, graph_nodes),
            ("graph_edges", directory / GRAPH_EDGES_FILENAME, graph_edges),
        )
        for name, path, array in arrays:
            _write_npy_atomic(path, array)
            artifacts[name] = _artifact_spec(path, array)
        document = {
            "schema_version": "1.0",
            "complete": True,
            "logical_id": identity.logical_id,
            "manifest_fingerprint": identity.manifest_fingerprint,
            "module1_config_hash": identity.module1_config_hash,
            "source_reconstruction_path": identity.source_reconstruction_path,
            "source_reconstruction_sha256": identity.source_reconstruction_sha256,
            "representation_config_hash": identity.representation_config_hash,
            "processing_identity": processing_identity,
            "artifacts": artifacts,
            "representation_metadata": representation_metadata,
        }
        _reject_absolute_strings(document)
        _write_json_atomic(metadata_path, document)
        return self.load(identity)

    def load(self, identity: CacheIdentity) -> CacheEntry:
        directory = self.path_for(identity)
        metadata_path = directory / METADATA_FILENAME
        if not metadata_path.is_file():
            raise RepresentationCacheError("cache completion marker is missing")
        try:
            document = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RepresentationCacheError("cannot read cache metadata") from exc
        if not isinstance(document, dict):
            raise RepresentationCacheError("cache metadata must be a JSON object")
        required = {
            "schema_version",
            "complete",
            "logical_id",
            "manifest_fingerprint",
            "module1_config_hash",
            "source_reconstruction_path",
            "source_reconstruction_sha256",
            "representation_config_hash",
            "processing_identity",
            "artifacts",
            "representation_metadata",
        }
        if set(document) != required:
            raise RepresentationCacheError("cache metadata fields do not match schema")
        if document.get("schema_version") != "1.0" or document.get("complete") is not True:
            raise RepresentationCacheError("cache metadata is not marked complete")
        expected = {
            "logical_id": identity.logical_id,
            "manifest_fingerprint": identity.manifest_fingerprint,
            "module1_config_hash": identity.module1_config_hash,
            "source_reconstruction_path": identity.source_reconstruction_path,
            "source_reconstruction_sha256": identity.source_reconstruction_sha256,
            "representation_config_hash": identity.representation_config_hash,
        }
        if any(document.get(key) != value for key, value in expected.items()):
            raise RepresentationCacheError("cache identity does not match expected inputs")
        processing_identity = document.get("processing_identity")
        if not isinstance(processing_identity, dict):
            raise RepresentationCacheError("processing_identity must be an object")
        canonical = json.dumps(processing_identity, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        if hashlib.sha256(canonical).hexdigest() != identity.representation_config_hash:
            raise RepresentationCacheError("cache processing identity hash is stale")
        artifacts = document.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != {
            "mel",
            "recurrence",
            "graph_nodes",
            "graph_edges",
        }:
            raise RepresentationCacheError("cache artifacts do not match schema")
        mel = _load_array(directory / MEL_FILENAME, artifacts["mel"], np.dtype(np.float32))
        recurrence = _load_array(
            directory / RECURRENCE_FILENAME, artifacts["recurrence"], np.dtype(np.uint8)
        )
        graph_nodes = _load_array(
            directory / GRAPH_NODES_FILENAME, artifacts["graph_nodes"], np.dtype(np.float32)
        )
        graph_edges = _load_array(
            directory / GRAPH_EDGES_FILENAME, artifacts["graph_edges"], np.dtype(np.int16)
        )
        _validate_arrays(mel, recurrence, graph_nodes, graph_edges)
        metadata = document.get("representation_metadata")
        if not isinstance(metadata, dict):
            raise RepresentationCacheError("representation_metadata must be an object")
        return CacheEntry(
            directory=directory,
            mel=mel,
            recurrence=recurrence,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
            metadata=metadata,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_arrays(
    mel: npt.NDArray[np.float32],
    recurrence: npt.NDArray[np.uint8],
    graph_nodes: npt.NDArray[np.float32],
    graph_edges: npt.NDArray[np.int16],
) -> None:
    if mel.dtype != np.float32 or mel.shape != (224, 224) or not np.all(np.isfinite(mel)):
        raise RepresentationCacheError("Mel artifact must be finite float32 224x224")
    if recurrence.dtype != np.uint8 or recurrence.shape != (224, 224):
        raise RepresentationCacheError("recurrence artifact must be uint8 224x224")
    if not np.array_equal(recurrence, recurrence.T):
        raise RepresentationCacheError("recurrence artifact must be symmetric")
    if not np.all(np.diag(recurrence) == 1) or not np.all(np.isin(recurrence, (0, 1))):
        raise RepresentationCacheError("recurrence artifact values/diagonal are invalid")
    if (
        graph_nodes.dtype != np.float32
        or graph_nodes.ndim != 2
        or graph_nodes.shape[1] != 2
        or not 1 <= graph_nodes.shape[0] <= 256
        or not np.all(np.isfinite(graph_nodes))
    ):
        raise RepresentationCacheError("graph nodes must be finite float32 [N,2], N<=256")
    if graph_edges.dtype != np.int16 or graph_edges.ndim != 2 or graph_edges.shape[0] != 2:
        raise RepresentationCacheError("graph edges must be int16 [2,E]")
    if graph_edges.size:
        if np.min(graph_edges) < 0 or np.max(graph_edges) >= graph_nodes.shape[0]:
            raise RepresentationCacheError("graph edge index is outside node range")
        if np.any(graph_edges[0] == graph_edges[1]):
            raise RepresentationCacheError("graph cache must not store self-loops")
        pairs = {(int(left), int(right)) for left, right in graph_edges.T}
        if len(pairs) != graph_edges.shape[1]:
            raise RepresentationCacheError("graph cache contains duplicate edges")
        if any((right, left) not in pairs for left, right in pairs):
            raise RepresentationCacheError("graph cache edges must be bidirectional")


def _artifact_spec(path: Path, array: np.ndarray[Any, Any]) -> dict[str, Any]:
    return {
        "filename": path.name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "byte_count": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _load_array(path: Path, raw_spec: object, dtype: np.dtype[Any]) -> np.ndarray[Any, Any]:
    if not isinstance(raw_spec, dict):
        raise RepresentationCacheError(f"artifact spec is invalid: {path.name}")
    if raw_spec.get("filename") != path.name or raw_spec.get("dtype") != str(dtype):
        raise RepresentationCacheError(f"artifact identity is invalid: {path.name}")
    if not path.is_file():
        raise RepresentationCacheError(f"cache artifact is missing: {path.name}")
    if path.stat().st_size != raw_spec.get("byte_count"):
        raise RepresentationCacheError(f"cache artifact size mismatch: {path.name}")
    if sha256_file(path) != raw_spec.get("sha256"):
        raise RepresentationCacheError(f"cache artifact SHA-256 mismatch: {path.name}")
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise RepresentationCacheError(f"cannot load cache artifact: {path.name}") from exc
    if not isinstance(array, np.ndarray) or array.dtype != dtype:
        raise RepresentationCacheError(f"cache artifact dtype mismatch: {path.name}")
    if list(array.shape) != raw_spec.get("shape"):
        raise RepresentationCacheError(f"cache artifact shape mismatch: {path.name}")
    return array


def _write_npy_atomic(path: Path, array: np.ndarray[Any, Any]) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, array, allow_pickle=False)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _logical_id_parts(logical_id: str) -> tuple[str, str, str]:
    parts = logical_id.split("/")
    if len(parts) != 3 or any(not part or part in {".", ".."} for part in parts):
        raise RepresentationCacheError(f"invalid logical ID: {logical_id!r}")
    return parts[0], parts[1], parts[2]


def _validate_hash(value: str, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise RepresentationCacheError(f"{name} must be a lowercase SHA-256")


def _reject_absolute_strings(value: object) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _reject_absolute_strings(item)
    elif isinstance(value, list):
        for item in value:
            _reject_absolute_strings(item)
    elif isinstance(value, str) and (Path(value).is_absolute() or ":\\" in value):
        raise RepresentationCacheError("cache metadata contains an absolute path")
