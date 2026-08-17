"""Portable, integrity-checked cache storage for Module 1 VMD results.

The cache deliberately treats ``metadata.json`` as the completion marker.  Mode
and reconstruction arrays are atomically replaced first; metadata is atomically
replaced only after both arrays and their hashes are available.  A crash or
interrupted write therefore leaves a cache entry that validation rejects.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt

CACHE_SCHEMA_VERSION = "1.0"
MODES_FILENAME = "modes.npy"
RECONSTRUCTION_FILENAME = "reconstruction.npy"
METADATA_FILENAME = "metadata.json"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SAFE_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_WINDOWS_ABSOLUTE_PATTERN = re.compile(r"^[A-Za-z]:[/\\]")

JsonValue: TypeAlias = (
    bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"] | None
)
FloatArray: TypeAlias = npt.NDArray[np.float64]


class CacheValidationError(ValueError):
    """Raised when a VMD cache identity or stored entry is invalid."""


@dataclass(frozen=True, slots=True)
class CacheIdentity:
    """Inputs that uniquely identify and constrain one cached decomposition."""

    logical_id: str
    source_primary_audio_path: str
    source_audio_sha256: str
    config_hash: str
    source_manifest_fingerprint: str
    modes_k: int
    sample_count: int

    def validate(self) -> None:
        """Reject unsafe paths, weak identities, and impossible array dimensions."""

        _logical_id_parts(self.logical_id)
        _validate_repository_relative_path(
            self.source_primary_audio_path,
            field_name="source_primary_audio_path",
        )
        _validate_sha256(self.source_audio_sha256, field_name="source_audio_sha256")
        _validate_sha256(self.config_hash, field_name="config_hash")
        _validate_sha256(
            self.source_manifest_fingerprint,
            field_name="source_manifest_fingerprint",
        )
        if not isinstance(self.modes_k, int) or isinstance(self.modes_k, bool) or self.modes_k <= 0:
            raise CacheValidationError("modes_k must be a positive integer")
        if (
            not isinstance(self.sample_count, int)
            or isinstance(self.sample_count, bool)
            or self.sample_count <= 0
        ):
            raise CacheValidationError("sample_count must be a positive integer")


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """A fully validated cache result."""

    directory: Path
    identity: CacheIdentity
    modes: FloatArray
    reconstruction: FloatArray
    metadata: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class CacheValidation:
    """Non-throwing cache validation result used by resumable pipelines."""

    directory: Path
    valid: bool
    reason: str | None


class VMDCache:
    """Read and atomically write VMD arrays beneath a configured cache root."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, identity: CacheIdentity) -> Path:
        """Return ``root/config_hash/speaker/session/prompt`` for an identity."""

        identity.validate()
        speaker_id, session_id, prompt_id = _logical_id_parts(identity.logical_id)
        return self.root / identity.config_hash / speaker_id / session_id / prompt_id

    def validate(self, identity: CacheIdentity) -> CacheValidation:
        """Validate an entry without propagating expected cache-corruption errors."""

        try:
            self.load(identity)
        except CacheValidationError as exc:
            return CacheValidation(
                directory=self.path_for(identity),
                valid=False,
                reason=str(exc),
            )
        return CacheValidation(directory=self.path_for(identity), valid=True, reason=None)

    def is_valid(self, identity: CacheIdentity) -> bool:
        """Return whether an entry is complete and matches the expected identity."""

        return self.validate(identity).valid

    def write(
        self,
        identity: CacheIdentity,
        modes: npt.ArrayLike,
        reconstruction: npt.ArrayLike,
        metadata: Mapping[str, object],
    ) -> CacheEntry:
        """Atomically store arrays, then publish integrity metadata as the marker."""

        identity.validate()
        modes_array, reconstruction_array = _validated_arrays(
            modes,
            reconstruction,
            identity=identity,
        )
        portable_metadata = _to_json_mapping(metadata, field_name="metadata")
        _reject_absolute_strings(portable_metadata, field_name="metadata")

        directory = self.path_for(identity)
        directory.mkdir(parents=True, exist_ok=True)
        modes_path = directory / MODES_FILENAME
        reconstruction_path = directory / RECONSTRUCTION_FILENAME
        metadata_path = directory / METADATA_FILENAME

        # Invalidate any prior completion marker before replacing component files.
        # If the process stops before the final atomic replace, the entry is partial.
        metadata_path.unlink(missing_ok=True)
        _write_npy_atomic(modes_path, modes_array)
        _write_npy_atomic(reconstruction_path, reconstruction_array)

        metadata_document: dict[str, JsonValue] = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "complete": True,
            "logical_id": identity.logical_id,
            "source_primary_audio_path": identity.source_primary_audio_path,
            "source_audio_sha256": identity.source_audio_sha256,
            "config_hash": identity.config_hash,
            "source_manifest_fingerprint": identity.source_manifest_fingerprint,
            "modes_k": identity.modes_k,
            "sample_count": identity.sample_count,
            "artifacts": {
                "modes": _artifact_metadata(modes_path, modes_array),
                "reconstruction": _artifact_metadata(
                    reconstruction_path,
                    reconstruction_array,
                ),
            },
            "metadata": portable_metadata,
        }
        _write_json_atomic(metadata_path, metadata_document)
        return self.load(identity)

    def load(self, identity: CacheIdentity) -> CacheEntry:
        """Load an entry only after validating identity, hashes, shapes, and values."""

        identity.validate()
        directory = self.path_for(identity)
        metadata_path = directory / METADATA_FILENAME
        if not metadata_path.is_file():
            raise CacheValidationError(
                f"cache completion marker is missing: {metadata_path.name}"
            )
        document = _load_json_mapping(metadata_path)
        _validate_document_keys(document)
        if _require_string(document, "schema_version") != CACHE_SCHEMA_VERSION:
            raise CacheValidationError("unsupported cache schema_version")
        if _require_bool(document, "complete") is not True:
            raise CacheValidationError("cache metadata is not marked complete")

        stored_identity = CacheIdentity(
            logical_id=_require_string(document, "logical_id"),
            source_primary_audio_path=_require_string(
                document,
                "source_primary_audio_path",
            ),
            source_audio_sha256=_require_string(document, "source_audio_sha256"),
            config_hash=_require_string(document, "config_hash"),
            source_manifest_fingerprint=_require_string(
                document,
                "source_manifest_fingerprint",
            ),
            modes_k=_require_integer(document, "modes_k"),
            sample_count=_require_integer(document, "sample_count"),
        )
        stored_identity.validate()
        if stored_identity != identity:
            raise CacheValidationError("cache identity does not match expected inputs")

        artifacts = _require_mapping(document, "artifacts")
        if set(artifacts) != {"modes", "reconstruction"}:
            raise CacheValidationError("cache artifacts must contain modes and reconstruction")
        modes_spec = _parse_artifact_spec(
            _require_mapping(artifacts, "modes"),
            expected_filename=MODES_FILENAME,
            expected_shape=(identity.modes_k, identity.sample_count),
        )
        reconstruction_spec = _parse_artifact_spec(
            _require_mapping(artifacts, "reconstruction"),
            expected_filename=RECONSTRUCTION_FILENAME,
            expected_shape=(identity.sample_count,),
        )

        modes = _load_array(directory / MODES_FILENAME, modes_spec)
        reconstruction = _load_array(
            directory / RECONSTRUCTION_FILENAME,
            reconstruction_spec,
        )
        modes, reconstruction = _validated_arrays(
            modes,
            reconstruction,
            identity=identity,
        )

        raw_metadata = document.get("metadata")
        if not isinstance(raw_metadata, dict):
            raise CacheValidationError("metadata must be a JSON object")
        result_metadata = _loaded_json_mapping(raw_metadata, field_name="metadata")
        _reject_absolute_strings(result_metadata, field_name="metadata")
        return CacheEntry(
            directory=directory,
            identity=identity,
            modes=modes,
            reconstruction=reconstruction,
            metadata=result_metadata,
        )


@dataclass(frozen=True, slots=True)
class _ArtifactSpec:
    filename: str
    sha256: str
    shape: tuple[int, ...]
    dtype: str
    byte_count: int


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a lowercase SHA-256 digest without loading the whole file."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise CacheValidationError(f"cannot hash cache artifact {path.name}: {exc}") from exc
    return digest.hexdigest()


def _validated_arrays(
    modes: npt.ArrayLike,
    reconstruction: npt.ArrayLike,
    *,
    identity: CacheIdentity,
) -> tuple[FloatArray, FloatArray]:
    try:
        modes_array = np.asarray(modes, dtype=np.float64)
        reconstruction_array = np.asarray(reconstruction, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CacheValidationError(f"cache arrays must be numeric: {exc}") from exc
    expected_modes_shape = (identity.modes_k, identity.sample_count)
    expected_reconstruction_shape = (identity.sample_count,)
    if modes_array.shape != expected_modes_shape:
        raise CacheValidationError(
            f"modes shape {modes_array.shape} does not match {expected_modes_shape}"
        )
    if reconstruction_array.shape != expected_reconstruction_shape:
        raise CacheValidationError(
            "reconstruction shape "
            f"{reconstruction_array.shape} does not match {expected_reconstruction_shape}"
        )
    if not np.isfinite(modes_array).all():
        raise CacheValidationError("modes contain non-finite values")
    if not np.isfinite(reconstruction_array).all():
        raise CacheValidationError("reconstruction contains non-finite values")
    return modes_array, reconstruction_array


def _artifact_metadata(path: Path, array: FloatArray) -> dict[str, JsonValue]:
    return {
        "filename": path.name,
        "sha256": sha256_file(path),
        "shape": [int(item) for item in array.shape],
        "dtype": str(array.dtype),
        "byte_count": path.stat().st_size,
    }


def _parse_artifact_spec(
    value: Mapping[str, object],
    *,
    expected_filename: str,
    expected_shape: tuple[int, ...],
) -> _ArtifactSpec:
    expected_keys = {"filename", "sha256", "shape", "dtype", "byte_count"}
    if set(value) != expected_keys:
        raise CacheValidationError(
            f"artifact metadata keys do not match schema for {expected_filename}"
        )
    filename = _require_string(value, "filename")
    if filename != expected_filename:
        raise CacheValidationError(f"unexpected artifact filename: {filename!r}")
    digest = _require_string(value, "sha256")
    _validate_sha256(digest, field_name=f"{expected_filename}.sha256")
    raw_shape = value.get("shape")
    if not isinstance(raw_shape, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 0
        for item in raw_shape
    ):
        raise CacheValidationError(f"invalid artifact shape for {expected_filename}")
    shape = tuple(cast(list[int], raw_shape))
    if shape != expected_shape:
        raise CacheValidationError(
            f"stored {expected_filename} shape {shape} does not match {expected_shape}"
        )
    dtype = _require_string(value, "dtype")
    if dtype != "float64":
        raise CacheValidationError(f"unsupported artifact dtype for {expected_filename}: {dtype}")
    byte_count = _require_integer(value, "byte_count")
    if byte_count <= 0:
        raise CacheValidationError(f"invalid artifact byte_count for {expected_filename}")
    return _ArtifactSpec(
        filename=filename,
        sha256=digest,
        shape=shape,
        dtype=dtype,
        byte_count=byte_count,
    )


def _load_array(path: Path, spec: _ArtifactSpec) -> FloatArray:
    if not path.is_file():
        raise CacheValidationError(f"cache artifact is missing: {path.name}")
    try:
        byte_count = path.stat().st_size
    except OSError as exc:
        raise CacheValidationError(f"cannot inspect cache artifact {path.name}: {exc}") from exc
    if byte_count != spec.byte_count:
        raise CacheValidationError(f"cache artifact size mismatch: {path.name}")
    if sha256_file(path) != spec.sha256:
        raise CacheValidationError(f"cache artifact SHA-256 mismatch: {path.name}")
    try:
        loaded: object = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise CacheValidationError(f"cannot load cache artifact {path.name}: {exc}") from exc
    if not isinstance(loaded, np.ndarray):
        raise CacheValidationError(f"cache artifact is not a NumPy array: {path.name}")
    if loaded.dtype != np.dtype(spec.dtype):
        raise CacheValidationError(f"cache artifact dtype mismatch: {path.name}")
    if loaded.shape != spec.shape:
        raise CacheValidationError(f"cache artifact shape mismatch: {path.name}")
    return cast(FloatArray, loaded)


def _write_npy_atomic(target: Path, array: FloatArray) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_json_atomic(target: Path, value: Mapping[str, JsonValue]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _load_json_mapping(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            loaded: object = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CacheValidationError(f"cannot read cache metadata: {exc}") from exc
    if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
        raise CacheValidationError("cache metadata must be a JSON object with string keys")
    return cast(dict[str, object], loaded)


def _validate_document_keys(document: Mapping[str, object]) -> None:
    expected_keys = {
        "schema_version",
        "complete",
        "logical_id",
        "source_primary_audio_path",
        "source_audio_sha256",
        "config_hash",
        "source_manifest_fingerprint",
        "modes_k",
        "sample_count",
        "artifacts",
        "metadata",
    }
    if set(document) != expected_keys:
        missing = sorted(expected_keys - set(document))
        extra = sorted(set(document) - expected_keys)
        raise CacheValidationError(
            f"cache metadata fields do not match schema; missing={missing}, extra={extra}"
        )


def _require_mapping(value: Mapping[str, object], key: str) -> dict[str, object]:
    item = value.get(key)
    if not isinstance(item, dict) or not all(isinstance(item_key, str) for item_key in item):
        raise CacheValidationError(f"{key} must be a JSON object")
    return cast(dict[str, object], item)


def _require_string(value: Mapping[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise CacheValidationError(f"{key} must be a non-empty string")
    return item


def _require_integer(value: Mapping[str, object], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        raise CacheValidationError(f"{key} must be an integer")
    return item


def _require_bool(value: Mapping[str, object], key: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise CacheValidationError(f"{key} must be a boolean")
    return item


def _to_json_mapping(value: Mapping[str, object], *, field_name: str) -> dict[str, JsonValue]:
    if not all(isinstance(key, str) and key for key in value):
        raise CacheValidationError(f"{field_name} keys must be non-empty strings")
    return {
        key: _to_json_value(item, field_name=f"{field_name}.{key}")
        for key, item in value.items()
    }


def _to_json_value(value: object, *, field_name: str) -> JsonValue:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CacheValidationError(f"{field_name} must not contain NaN or infinity")
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        converted = float(value)
        if not math.isfinite(converted):
            raise CacheValidationError(f"{field_name} must not contain NaN or infinity")
        return converted
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) and key for key in value):
            raise CacheValidationError(f"{field_name} keys must be non-empty strings")
        return {
            cast(str, key): _to_json_value(
                item,
                field_name=f"{field_name}.{key}",
            )
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _to_json_value(item, field_name=f"{field_name}[{index}]")
            for index, item in enumerate(value)
        ]
    raise CacheValidationError(f"{field_name} is not JSON serializable")


def _loaded_json_mapping(value: object, *, field_name: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict) or not all(isinstance(key, str) and key for key in value):
        raise CacheValidationError(f"{field_name} must be a JSON object")
    return {
        cast(str, key): _loaded_json_value(item, field_name=f"{field_name}.{key}")
        for key, item in value.items()
    }


def _loaded_json_value(value: object, *, field_name: str) -> JsonValue:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CacheValidationError(f"{field_name} must not contain NaN or infinity")
        return value
    if isinstance(value, list):
        return [
            _loaded_json_value(item, field_name=f"{field_name}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict) and all(isinstance(key, str) and key for key in value):
        return {
            cast(str, key): _loaded_json_value(
                item,
                field_name=f"{field_name}.{key}",
            )
            for key, item in value.items()
        }
    raise CacheValidationError(f"{field_name} contains an invalid JSON value")


def _reject_absolute_strings(value: JsonValue, *, field_name: str) -> None:
    if isinstance(value, str):
        if (
            value.startswith(("/", "\\\\", "//"))
            or _WINDOWS_ABSOLUTE_PATTERN.match(value) is not None
        ):
            raise CacheValidationError(
                f"{field_name} contains a non-portable absolute path: {value!r}"
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_absolute_strings(item, field_name=f"{field_name}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_absolute_strings(item, field_name=f"{field_name}.{key}")


def _logical_id_parts(logical_id: str) -> tuple[str, str, str]:
    if not isinstance(logical_id, str) or not logical_id:
        raise CacheValidationError("logical_id must be a non-empty string")
    if "\\" in logical_id:
        raise CacheValidationError("logical_id must use portable forward slashes")
    parts = PurePosixPath(logical_id).parts
    if len(parts) != 3 or "/".join(parts) != logical_id:
        raise CacheValidationError("logical_id must contain speaker/session/prompt components")
    if not all(_SAFE_COMPONENT_PATTERN.fullmatch(part) is not None for part in parts):
        raise CacheValidationError("logical_id contains an unsafe path component")
    return parts


def _validate_repository_relative_path(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise CacheValidationError(f"{field_name} must be a non-empty string")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or value.startswith(("//", "\\\\"))
        or _WINDOWS_ABSOLUTE_PATTERN.match(value) is not None
    ):
        raise CacheValidationError(f"{field_name} must be a repository-relative safe path")


def _validate_sha256(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise CacheValidationError(f"{field_name} must be a lowercase SHA-256 digest")
