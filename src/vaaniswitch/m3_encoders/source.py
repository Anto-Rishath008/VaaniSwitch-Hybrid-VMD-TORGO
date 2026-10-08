"""Strict read-only interface to the signed Module 2 representation cache."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

from vaaniswitch.m0_data.schemas import ManifestRecord
from vaaniswitch.m1_vmd.manifest import load_manifest_source, select_subset
from vaaniswitch.m2_representations.cache import (
    GRAPH_EDGES_FILENAME,
    GRAPH_NODES_FILENAME,
    MEL_FILENAME,
    METADATA_FILENAME,
    RECURRENCE_FILENAME,
    CacheEntry,
    CacheIdentity,
    RepresentationCache,
)
from vaaniswitch.m2_representations.config import Module2Config
from vaaniswitch.m3_encoders.config import Module3Config

EXPECTED_FILES = frozenset(
    {
        MEL_FILENAME,
        RECURRENCE_FILENAME,
        GRAPH_NODES_FILENAME,
        GRAPH_EDGES_FILENAME,
        METADATA_FILENAME,
    }
)


class SignedRepresentationError(ValueError):
    """Raised when signed Module 2 inputs are missing, stale, or corrupt."""


@dataclass(frozen=True, slots=True)
class SignedSourceContract:
    records: tuple[ManifestRecord, ...]
    record_by_id: dict[str, ManifestRecord]
    speaker_to_evaluation_fold: dict[str, int]
    manifest_fingerprint: str
    representation_config_hash: str
    module1_config_hash: str
    signed_record_count: int


class SignedRepresentationSource:
    """Validate global sign-off once, then integrity-check every loaded record."""

    def __init__(self, config: Module3Config) -> None:
        self.config = config
        self.module2_config = Module2Config.from_yaml(
            config.source.module2_config_path, repo_root=config.repo_root
        )
        self.contract = self._load_contract()
        self.cache = RepresentationCache(config.source.representation_cache_root)

    def load(self, logical_id: str) -> CacheEntry:
        record = self.contract.record_by_id.get(logical_id)
        if record is None:
            raise SignedRepresentationError(f"record is outside signed CORE: {logical_id}")
        directory = self._directory(logical_id)
        if not directory.is_dir():
            raise SignedRepresentationError(f"representation directory is missing: {logical_id}")
        entries = list(directory.iterdir())
        filenames = {item.name for item in entries if item.is_file()}
        if filenames != EXPECTED_FILES or any(not item.is_file() for item in entries):
            raise SignedRepresentationError(
                f"representation directory files do not match schema: {logical_id}"
            )
        document = _read_json(directory / METADATA_FILENAME, "representation metadata")
        expected_path = (
            f"data/processed/vmd/{self.contract.module1_config_hash}/"
            f"{logical_id}/reconstruction.npy"
        )
        expected_fields = {
            "logical_id": logical_id,
            "manifest_fingerprint": self.contract.manifest_fingerprint,
            "module1_config_hash": self.contract.module1_config_hash,
            "representation_config_hash": self.contract.representation_config_hash,
            "source_reconstruction_path": expected_path,
        }
        if any(document.get(key) != value for key, value in expected_fields.items()):
            raise SignedRepresentationError(
                f"representation provenance does not match signed inputs: {logical_id}"
            )
        source_hash = document.get("source_reconstruction_sha256")
        if not _is_sha256(source_hash):
            raise SignedRepresentationError("source reconstruction SHA-256 is invalid")
        processing = document.get("processing_identity")
        if (
            not isinstance(processing, dict)
            or processing.get("module1_config_hash") != self.contract.module1_config_hash
        ):
            raise SignedRepresentationError("representation processing provenance is stale")
        entry = self.cache.load(
            CacheIdentity(
                logical_id=logical_id,
                manifest_fingerprint=self.contract.manifest_fingerprint,
                module1_config_hash=self.contract.module1_config_hash,
                source_reconstruction_path=expected_path,
                source_reconstruction_sha256=source_hash,
                representation_config_hash=self.contract.representation_config_hash,
            )
        )
        source_metadata = entry.metadata.get("source")
        if (
            not isinstance(source_metadata, dict)
            or source_metadata.get("reconstruction_policy") != "all_modes"
            or source_metadata.get("sample_rate") != 16_000
        ):
            raise SignedRepresentationError("representation source semantics are stale")
        return entry

    def _load_contract(self) -> SignedSourceContract:
        config = self.config
        if self.module2_config.config_hash != config.source.representation_config_hash:
            raise SignedRepresentationError("Module 2 configuration hash is not production")
        if self.module2_config.source.module1_config_hash != config.source.module1_config_hash:
            raise SignedRepresentationError("Module 2 configuration has wrong Module 1 provenance")
        if self.module2_config.cache.root != config.source.representation_cache_root:
            raise SignedRepresentationError("Module 2 cache root is not the signed production root")
        manifest = load_manifest_source(config.source.manifest_path)
        if manifest.fingerprint != config.source.manifest_fingerprint:
            raise SignedRepresentationError("manifest fingerprint does not match configuration")
        records = select_subset(manifest.records, config.source.subset)
        signoff = _read_json(config.source.module2_signoff_path, "Module 2 sign-off")
        completion = _mapping(signoff, "completion")
        identity = _mapping(signoff, "identity")
        validation = _mapping(signoff, "validation")
        signed_counts = (
            completion.get("expected"),
            completion.get("selected"),
            completion.get("successful"),
            completion.get("validated"),
            validation.get("selected_record_count"),
            validation.get("valid_cache_entries"),
            len(records),
        )
        if (
            signoff.get("status") != "PASS"
            or signoff.get("module2_signed_off") is not True
            or signoff.get("module3_started") is not False
            or completion.get("failed") != 0
            or validation.get("status") != "PASS"
            or len(set(signed_counts)) != 1
        ):
            raise SignedRepresentationError("Module 2 sign-off state is invalid")
        expected_identity = {
            "representation_config_hash": config.source.representation_config_hash,
            "module1_config_hash": config.source.module1_config_hash,
            "manifest_fingerprint": config.source.manifest_fingerprint,
        }
        if any(identity.get(key) != value for key, value in expected_identity.items()):
            raise SignedRepresentationError("Module 2 sign-off identity is stale")
        folds = _read_json(config.source.speaker_folds_path, "speaker folds")
        fold_source = _mapping(folds, "source")
        raw_mapping = folds.get("speaker_to_evaluation_fold")
        if (
            folds.get("schema_version") != "1.0"
            or fold_source.get("manifest_sha256") != manifest.fingerprint
            or not isinstance(raw_mapping, dict)
        ):
            raise SignedRepresentationError("speaker-fold provenance is stale")
        speaker_mapping: dict[str, int] = {}
        for speaker, fold_id in raw_mapping.items():
            if (
                not isinstance(speaker, str)
                or not isinstance(fold_id, int)
                or isinstance(fold_id, bool)
                or fold_id <= 0
            ):
                raise SignedRepresentationError("speaker-fold mapping is malformed")
            speaker_mapping[speaker] = fold_id
        if {record.speaker_id for record in records} != set(speaker_mapping):
            raise SignedRepresentationError("speaker-fold mapping does not cover signed CORE")
        record_by_id = {record.logical_id: record for record in records}
        if len(record_by_id) != len(records):
            raise SignedRepresentationError("signed CORE contains duplicate logical IDs")
        return SignedSourceContract(
            records=records,
            record_by_id=record_by_id,
            speaker_to_evaluation_fold=speaker_mapping,
            manifest_fingerprint=manifest.fingerprint,
            representation_config_hash=config.source.representation_config_hash,
            module1_config_hash=config.source.module1_config_hash,
            signed_record_count=len(records),
        )

    def _directory(self, logical_id: str) -> Path:
        parts = logical_id.split("/")
        if len(parts) != 3 or any(not part or part in {".", ".."} for part in parts):
            raise SignedRepresentationError(f"invalid logical ID: {logical_id!r}")
        namespace = (
            self.config.source.representation_cache_root
            / self.contract.representation_config_hash
        ).resolve()
        directory = (namespace / parts[0] / parts[1] / parts[2]).resolve()
        try:
            directory.relative_to(namespace)
        except ValueError as exc:
            raise SignedRepresentationError(
                "representation path escapes production namespace"
            ) from exc
        return directory


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise SignedRepresentationError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SignedRepresentationError(f"cannot read {label}") from exc
    if not isinstance(value, dict):
        raise SignedRepresentationError(f"{label} must be a JSON object")
    return value


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise SignedRepresentationError(f"{key} must be a JSON object")
    return item


def _is_sha256(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )
