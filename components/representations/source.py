"""Read-only consumption of signed Module 1 production reconstructions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from components.data.schemas import ManifestRecord
from components.vmd.manifest import load_manifest_source, select_subset
from components.representations.config import Module2Config

FloatArray = npt.NDArray[np.float64]


class SourceValidationError(ValueError):
    """Raised when a signed source contract or cache entry is invalid."""


@dataclass(frozen=True, slots=True)
class SourceContract:
    manifest_fingerprint: str
    records: tuple[ManifestRecord, ...]
    record_ids: frozenset[str]
    subset: str
    signed_record_count: int
    raw_metadata_fingerprint: str


@dataclass(frozen=True, slots=True)
class SourceReconstruction:
    logical_id: str
    waveform: FloatArray
    relative_path: str
    artifact_sha256: str
    source_metadata: dict[str, Any]
    converged: bool


def load_source_contract(config: Module2Config, subset: str = "core") -> SourceContract:
    """Cross-check the manifest selection against signed Module 0 and 1 artifacts."""

    if subset != "core":
        raise SourceValidationError("Module 2 production input is restricted to CORE")
    manifest = load_manifest_source(config.source.manifest_path)
    records = select_subset(manifest.records, subset)
    module0 = _read_json(config.source.module0_summary_path, "Module 0 summary")
    module1 = _read_json(config.source.module1_summary_path, "Module 1 summary")
    signoff = _read_json(config.source.module1_signoff_path, "Module 1 sign-off audit")

    module0_source = _mapping(module0, "source")
    module0_subsets = _mapping(module0, "downstream_subset_policy")
    module0_core = _mapping(module0_subsets, "core")
    module0_raw = _mapping(module0, "raw_data_immutability")
    module1_source = _mapping(module1, "source")
    module1_selection = _mapping(module1, "selection")
    module1_config = _mapping(module1, "configuration")
    module1_completion = _mapping(module1, "completion")
    signoff_source = _mapping(signoff, "source")
    signoff_completion = _mapping(signoff, "completion")
    signoff_config = _mapping(signoff, "configuration")

    signed_counts = (
        _integer(module0_core, "record_count"),
        _integer(module1_selection, "subset_record_count"),
        _integer(module1_selection, "selected_record_count"),
        _integer(module1_completion, "successful"),
        _integer(signoff_completion, "expected"),
        _integer(signoff_completion, "valid_structural_cache_entries"),
    )
    if len(set((*signed_counts, len(records)))) != 1:
        raise SourceValidationError(
            "CORE subset count disagrees across manifest and signed Module 0/1 artifacts"
        )
    fingerprints = (
        module0_source.get("manifest_sha256"),
        module1_source.get("manifest_fingerprint"),
        signoff_source.get("manifest_sha256"),
    )
    if any(value != manifest.fingerprint for value in fingerprints):
        raise SourceValidationError("signed artifacts have a stale manifest fingerprint")
    config_hashes = (
        module1_config.get("config_hash"),
        signoff_source.get("config_hash"),
        signoff_config.get("config_hash"),
    )
    if any(value != config.source.module1_config_hash for value in config_hashes):
        raise SourceValidationError("signed artifacts do not identify the production VMD hash")
    if (
        module1.get("status") != "full_build_completed"
        or signoff.get("status") != "PASS"
        or module1_selection.get("subset") != "core"
        or module1_selection.get("limit") is not None
        or module1_completion.get("failed") != 0
        or signoff_completion.get("failed") != 0
        or module1_config.get("all_mode_default") is not True
        or signoff_config.get("all_mode_default") is not True
        or module1_config.get("selected_mode_ablation_enabled") is not False
        or signoff_config.get("selected_mode_ablation_enabled") is not False
    ):
        raise SourceValidationError("Module 1 production sign-off state is invalid")
    summary_records = module1.get("records")
    if not isinstance(summary_records, list):
        raise SourceValidationError("Module 1 summary records are missing")
    summary_ids = [
        item.get("logical_id") if isinstance(item, dict) else None for item in summary_records
    ]
    if summary_ids != [record.logical_id for record in records]:
        raise SourceValidationError("Module 1 summary record order/identity is stale")
    raw_fingerprint = module0_raw.get("metadata_sha256")
    if not isinstance(raw_fingerprint, str) or len(raw_fingerprint) != 64:
        raise SourceValidationError("Module 0 raw metadata fingerprint is invalid")
    if module1_source.get("raw_metadata_sha256") != raw_fingerprint:
        raise SourceValidationError("Module 1 source raw fingerprint is stale")
    return SourceContract(
        manifest_fingerprint=manifest.fingerprint,
        records=records,
        record_ids=frozenset(record.logical_id for record in records),
        subset=subset,
        signed_record_count=signed_counts[0],
        raw_metadata_fingerprint=raw_fingerprint,
    )


def load_reconstruction(
    config: Module2Config,
    contract: SourceContract,
    record: ManifestRecord,
) -> SourceReconstruction:
    """Load and integrity-check one production all-mode reconstruction only."""

    if record.logical_id not in contract.record_ids:
        raise SourceValidationError(
            f"record is outside signed {contract.subset}: {record.logical_id}"
        )
    speaker, session, prompt = _logical_id_parts(record.logical_id)
    namespace = config.source.module1_cache_root / config.source.module1_config_hash
    directory = (namespace / speaker / session / prompt).resolve()
    if "vmd_fidelity_sweep" in directory.parts:
        raise SourceValidationError("experimental tau-sweep reconstruction is forbidden")
    try:
        directory.relative_to(namespace.resolve())
    except ValueError as exc:
        raise SourceValidationError("Module 1 cache path escapes production namespace") from exc
    metadata_path = directory / "metadata.json"
    document = _read_json(metadata_path, f"Module 1 cache metadata for {record.logical_id}")
    if (
        document.get("schema_version") != "1.0"
        or document.get("complete") is not True
        or document.get("logical_id") != record.logical_id
        or document.get("config_hash") != config.source.module1_config_hash
        or document.get("source_manifest_fingerprint") != contract.manifest_fingerprint
    ):
        raise SourceValidationError("Module 1 cache identity does not match expected source")
    if (
        document.get("source_primary_audio_path") != record.primary_audio_path
        or document.get("source_audio_sha256") != record.primary_audio_sha256
    ):
        raise SourceValidationError("Module 1 cache primary-audio binding is stale")
    sample_count = _integer(document, "sample_count")
    metadata = _mapping(document, "metadata")
    if metadata.get("module") != "1_shared_vmd":
        raise SourceValidationError("source cache is not a Module 1 artifact")
    if metadata.get("reconstruction_policy") != "all_modes":
        raise SourceValidationError("selected-mode reconstruction is forbidden in Module 2")
    audio = _mapping(metadata, "audio")
    if (
        audio.get("working_sample_rate") != 16_000
        or audio.get("working_channels") != 1
        or audio.get("working_sample_count") != sample_count
    ):
        raise SourceValidationError("Module 1 reconstruction does not have 16 kHz mono semantics")
    source = _mapping(metadata, "source")
    expected_source = {
        "speaker_id": record.speaker_id,
        "session_id": record.session_id,
        "prompt_id": record.prompt_id,
        "condition": record.condition.value,
        "primary_microphone": record.primary_microphone.value
        if record.primary_microphone
        else None,
    }
    if any(source.get(key) != value for key, value in expected_source.items()):
        raise SourceValidationError("Module 1 cache logical source metadata is stale")
    artifacts = _mapping(document, "artifacts")
    reconstruction_spec = _mapping(artifacts, "reconstruction")
    if (
        reconstruction_spec.get("filename") != "reconstruction.npy"
        or reconstruction_spec.get("dtype") != "float64"
        or reconstruction_spec.get("shape") != [sample_count]
    ):
        raise SourceValidationError("Module 1 reconstruction artifact schema is invalid")
    expected_hash = reconstruction_spec.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise SourceValidationError("Module 1 reconstruction artifact hash is invalid")
    reconstruction_path = directory / "reconstruction.npy"
    if not reconstruction_path.is_file():
        raise SourceValidationError("Module 1 reconstruction.npy is missing")
    if reconstruction_path.stat().st_size != reconstruction_spec.get("byte_count"):
        raise SourceValidationError("Module 1 reconstruction byte count is stale")
    if _sha256_file(reconstruction_path) != expected_hash:
        raise SourceValidationError("Module 1 reconstruction artifact SHA-256 mismatch")
    try:
        waveform = np.load(reconstruction_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise SourceValidationError("cannot load Module 1 reconstruction") from exc
    if not isinstance(waveform, np.ndarray) or waveform.dtype != np.float64:
        raise SourceValidationError("Module 1 reconstruction dtype must be float64")
    if waveform.shape != (sample_count,):
        raise SourceValidationError("Module 1 reconstruction shape is stale")
    if not np.all(np.isfinite(waveform)):
        raise SourceValidationError("Module 1 reconstruction contains non-finite values")
    vmd = _mapping(metadata, "vmd")
    converged = vmd.get("converged")
    if not isinstance(converged, bool):
        raise SourceValidationError("Module 1 convergence flag is invalid")
    relative_path = reconstruction_path.relative_to(config.repo_root).as_posix()
    return SourceReconstruction(
        logical_id=record.logical_id,
        waveform=waveform,
        relative_path=relative_path,
        artifact_sha256=expected_hash,
        source_metadata=metadata,
        converged=converged,
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise SourceValidationError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceValidationError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise SourceValidationError(f"{label} must contain a JSON object")
    return value


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise SourceValidationError(f"{key} must be a JSON object")
    return item


def _integer(value: dict[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        raise SourceValidationError(f"{key} must be an integer")
    return item


def _logical_id_parts(logical_id: str) -> tuple[str, str, str]:
    parts = logical_id.split("/")
    if len(parts) != 3 or any(not part or part in {".", ".."} for part in parts):
        raise SourceValidationError(f"invalid logical ID: {logical_id!r}")
    return parts[0], parts[1], parts[2]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
