"""Validated Module 2 configuration and numerical cache identity."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


class RepresentationConfigError(ValueError):
    """Raised when Module 2 configuration violates the signed contract."""


@dataclass(frozen=True, slots=True)
class SourceConfig:
    manifest_path: Path
    module0_summary_path: Path
    module1_summary_path: Path
    module1_signoff_path: Path
    module1_cache_root: Path
    module1_config_hash: str
    default_subset: str


@dataclass(frozen=True, slots=True)
class MelConfig:
    sample_rate: int
    n_fft: int
    win_length: int
    hop_length: int
    window: str
    n_mels: int
    mel_scale: str
    filter_normalization: str
    f_min_hz: float
    f_max_hz: float
    power: float
    log_transform: str
    epsilon: float
    fixed_duration_seconds: float
    crop_policy: str
    pad_policy: str
    frequency_resize: str
    time_resize: str
    output_height: int
    output_width: int
    output_dtype: str

    @property
    def fixed_sample_count(self) -> int:
        return round(self.sample_rate * self.fixed_duration_seconds)


@dataclass(frozen=True, slots=True)
class RecurrenceConfig:
    output_size: int
    scalar_reduction: str
    scalar_standardization: str
    embedding_dimension: int
    delay: int
    distance: str
    threshold_mode: str
    recurrence_rate: float
    include_diagonal: bool
    output_dtype: str
    epsilon: float

    @property
    def scalar_state_count(self) -> int:
        return self.output_size + (self.embedding_dimension - 1) * self.delay


@dataclass(frozen=True, slots=True)
class VisibilityConfig:
    max_nodes: int
    scalar_reduction: str
    scalar_standardization: str
    criterion: str
    node_features: tuple[str, ...]
    store_bidirectional_edges: bool
    store_self_loops: bool
    edge_dtype: str
    adjacency_contract: str
    epsilon: float


@dataclass(frozen=True, slots=True)
class CacheConfig:
    enabled: bool
    resume: bool
    root: Path


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    progress_every: int
    diagnostic_examples: int
    smoke_logical_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OutputConfig:
    summary: Path
    failures: Path
    pipeline_log: Path
    plots: Path


@dataclass(frozen=True, slots=True)
class Module2Config:
    schema_version: str
    implementation: str
    repo_root: Path
    source: SourceConfig
    mel: MelConfig
    recurrence: RecurrenceConfig
    visibility_graph: VisibilityConfig
    cache: CacheConfig
    execution: ExecutionConfig
    outputs: OutputConfig

    @property
    def processing_identity(self) -> dict[str, Any]:
        """Return settings that bind or numerically affect representations."""

        identity = {
            "schema_version": self.schema_version,
            "implementation": self.implementation,
            "module1_config_hash": self.source.module1_config_hash,
            "mel": asdict(self.mel),
            "recurrence": asdict(self.recurrence),
            "visibility_graph": asdict(self.visibility_graph),
        }
        serialized = json.loads(json.dumps(identity, sort_keys=True))
        if not isinstance(serialized, dict):
            raise AssertionError("processing identity must serialize to a mapping")
        return serialized

    @property
    def config_hash(self) -> str:
        canonical = json.dumps(
            self.processing_identity, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @classmethod
    def from_yaml(cls, path: Path, *, repo_root: Path | None = None) -> Module2Config:
        resolved = path.resolve()
        if not resolved.is_file():
            raise RepresentationConfigError(f"configuration file does not exist: {path}")
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise RepresentationConfigError("configuration root must be a mapping")
        root = repo_root.resolve() if repo_root is not None else resolved.parents[2]
        source = _mapping(raw, "source")
        mel = _mapping(raw, "mel")
        recurrence = _mapping(raw, "recurrence")
        visibility = _mapping(raw, "visibility_graph")
        cache = _mapping(raw, "cache")
        execution = _mapping(raw, "execution")
        outputs = _mapping(raw, "outputs")
        result = cls(
            schema_version=_string(raw, "schema_version"),
            implementation=_string(raw, "implementation"),
            repo_root=root,
            source=SourceConfig(
                manifest_path=_repo_path(root, _string(source, "manifest_path")),
                module0_summary_path=_repo_path(root, _string(source, "module0_summary_path")),
                module1_summary_path=_repo_path(root, _string(source, "module1_summary_path")),
                module1_signoff_path=_repo_path(root, _string(source, "module1_signoff_path")),
                module1_cache_root=_repo_path(root, _string(source, "module1_cache_root")),
                module1_config_hash=_sha256(source, "module1_config_hash"),
                default_subset=_string(source, "default_subset"),
            ),
            mel=MelConfig(
                sample_rate=_positive_int(mel, "sample_rate"),
                n_fft=_positive_int(mel, "n_fft"),
                win_length=_positive_int(mel, "win_length"),
                hop_length=_positive_int(mel, "hop_length"),
                window=_string(mel, "window"),
                n_mels=_positive_int(mel, "n_mels"),
                mel_scale=_string(mel, "mel_scale"),
                filter_normalization=_string(mel, "filter_normalization"),
                f_min_hz=_nonnegative_float(mel, "f_min_hz"),
                f_max_hz=_positive_float(mel, "f_max_hz"),
                power=_positive_float(mel, "power"),
                log_transform=_string(mel, "log_transform"),
                epsilon=_positive_float(mel, "epsilon"),
                fixed_duration_seconds=_positive_float(mel, "fixed_duration_seconds"),
                crop_policy=_string(mel, "crop_policy"),
                pad_policy=_string(mel, "pad_policy"),
                frequency_resize=_string(mel, "frequency_resize"),
                time_resize=_string(mel, "time_resize"),
                output_height=_positive_int(mel, "output_height"),
                output_width=_positive_int(mel, "output_width"),
                output_dtype=_string(mel, "output_dtype"),
            ),
            recurrence=RecurrenceConfig(
                output_size=_positive_int(recurrence, "output_size"),
                scalar_reduction=_string(recurrence, "scalar_reduction"),
                scalar_standardization=_string(recurrence, "scalar_standardization"),
                embedding_dimension=_positive_int(recurrence, "embedding_dimension"),
                delay=_positive_int(recurrence, "delay"),
                distance=_string(recurrence, "distance"),
                threshold_mode=_string(recurrence, "threshold_mode"),
                recurrence_rate=_unit_float(recurrence, "recurrence_rate"),
                include_diagonal=_boolean(recurrence, "include_diagonal"),
                output_dtype=_string(recurrence, "output_dtype"),
                epsilon=_positive_float(recurrence, "epsilon"),
            ),
            visibility_graph=VisibilityConfig(
                max_nodes=_positive_int(visibility, "max_nodes"),
                scalar_reduction=_string(visibility, "scalar_reduction"),
                scalar_standardization=_string(visibility, "scalar_standardization"),
                criterion=_string(visibility, "criterion"),
                node_features=_string_tuple(visibility, "node_features"),
                store_bidirectional_edges=_boolean(visibility, "store_bidirectional_edges"),
                store_self_loops=_boolean(visibility, "store_self_loops"),
                edge_dtype=_string(visibility, "edge_dtype"),
                adjacency_contract=_string(visibility, "adjacency_contract"),
                epsilon=_positive_float(visibility, "epsilon"),
            ),
            cache=CacheConfig(
                enabled=_boolean(cache, "enabled"),
                resume=_boolean(cache, "resume"),
                root=_repo_path(root, _string(cache, "root")),
            ),
            execution=ExecutionConfig(
                progress_every=_positive_int(execution, "progress_every"),
                diagnostic_examples=_nonnegative_int(execution, "diagnostic_examples"),
                smoke_logical_ids=_string_tuple(execution, "smoke_logical_ids"),
            ),
            outputs=OutputConfig(
                summary=_repo_path(root, _string(outputs, "summary")),
                failures=_repo_path(root, _string(outputs, "failures")),
                pipeline_log=_repo_path(root, _string(outputs, "pipeline_log")),
                plots=_repo_path(root, _string(outputs, "plots")),
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != "1.0":
            raise RepresentationConfigError("schema_version must be 1.0")
        if self.implementation != "components_numpy_representations_v1":
            raise RepresentationConfigError("unsupported Module 2 implementation")
        if self.source.default_subset != "core":
            raise RepresentationConfigError("the default Module 2 subset must be core")
        if self.mel.sample_rate != 16_000:
            raise RepresentationConfigError("Module 2 must consume 16 kHz source semantics")
        if self.mel.n_fft < self.mel.win_length:
            raise RepresentationConfigError("n_fft must be at least win_length")
        if self.mel.f_max_hz > self.mel.sample_rate / 2 or self.mel.f_min_hz >= self.mel.f_max_hz:
            raise RepresentationConfigError("Mel frequency bounds are invalid")
        mel_contract = (
            self.mel.window,
            self.mel.mel_scale,
            self.mel.filter_normalization,
            self.mel.log_transform,
            self.mel.crop_policy,
            self.mel.pad_policy,
            self.mel.frequency_resize,
            self.mel.time_resize,
            self.mel.output_dtype,
        )
        if mel_contract != (
            "hann_periodic",
            "htk",
            "slaney_area",
            "power_db",
            "center",
            "right_zero",
            "linear",
            "area_average",
            "float32",
        ):
            raise RepresentationConfigError("unsupported Mel transform contract")
        if (self.mel.output_height, self.mel.output_width) != (224, 224):
            raise RepresentationConfigError("Mel output must be 224x224")
        recurrence_contract = (
            self.recurrence.output_size,
            self.recurrence.scalar_reduction,
            self.recurrence.scalar_standardization,
            self.recurrence.distance,
            self.recurrence.threshold_mode,
            self.recurrence.include_diagonal,
            self.recurrence.output_dtype,
        )
        if recurrence_contract != (
            224,
            "block_rms",
            "zscore",
            "euclidean",
            "off_diagonal_recurrence_rate",
            True,
            "uint8",
        ):
            raise RepresentationConfigError("unsupported recurrence transform contract")
        if not 0 < self.recurrence.recurrence_rate < 1:
            raise RepresentationConfigError("recurrence_rate must be strictly between 0 and 1")
        vg = self.visibility_graph
        visibility_contract = (
            vg.max_nodes,
            vg.scalar_reduction,
            vg.scalar_standardization,
            vg.criterion,
            vg.node_features,
            vg.store_bidirectional_edges,
            vg.store_self_loops,
            vg.edge_dtype,
        )
        if visibility_contract != (
            256,
            "block_rms",
            "zscore",
            "natural_visibility_strict",
            ("normalized_time", "standardized_amplitude"),
            True,
            False,
            "int16",
        ):
            raise RepresentationConfigError("unsupported visibility-graph contract")
        if len(set(self.execution.smoke_logical_ids)) != len(self.execution.smoke_logical_ids):
            raise RepresentationConfigError("smoke logical IDs must be unique")
        for logical_id in self.execution.smoke_logical_ids:
            _validate_logical_id(logical_id)
        processed = (self.repo_root / "data/processed").resolve()
        artifacts = (self.repo_root / "artifacts").resolve()
        expected_vmd_root = (self.repo_root / "data/processed/vmd").resolve()
        if self.source.module1_cache_root != expected_vmd_root:
            raise RepresentationConfigError(
                "Module 1 source must be the production data/processed/vmd root"
            )
        if "vmd_fidelity_sweep" in self.source.module1_cache_root.parts:
            raise RepresentationConfigError("experimental tau-sweep paths are forbidden")
        if not _is_within(self.cache.root, processed):
            raise RepresentationConfigError("Module 2 cache must remain under data/processed")
        if self.cache.root == self.source.module1_cache_root or _is_within(
            self.cache.root, self.source.module1_cache_root
        ):
            raise RepresentationConfigError("Module 2 cannot write inside the Module 1 cache")
        writable = (
            self.outputs.summary,
            self.outputs.failures,
            self.outputs.pipeline_log,
            self.outputs.plots,
        )
        if any(not _is_within(path, artifacts) for path in writable):
            raise RepresentationConfigError("Module 2 reports must remain under artifacts")
        protected = {
            self.source.manifest_path,
            self.source.module0_summary_path,
            self.source.module1_summary_path,
            self.source.module1_signoff_path,
        }
        if protected.intersection(writable):
            raise RepresentationConfigError("Module 2 outputs cannot overwrite signed inputs")


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise RepresentationConfigError(f"{key} must be a mapping")
    return item


def _string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise RepresentationConfigError(f"{key} must be a non-empty string")
    return item


def _string_tuple(value: dict[str, Any], key: str) -> tuple[str, ...]:
    item = value.get(key)
    if not isinstance(item, list) or any(not isinstance(entry, str) for entry in item):
        raise RepresentationConfigError(f"{key} must be a list of strings")
    return tuple(item)


def _boolean(value: dict[str, Any], key: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise RepresentationConfigError(f"{key} must be a boolean")
    return item


def _integer(value: dict[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        raise RepresentationConfigError(f"{key} must be an integer")
    return item


def _positive_int(value: dict[str, Any], key: str) -> int:
    item = _integer(value, key)
    if item <= 0:
        raise RepresentationConfigError(f"{key} must be positive")
    return item


def _nonnegative_int(value: dict[str, Any], key: str) -> int:
    item = _integer(value, key)
    if item < 0:
        raise RepresentationConfigError(f"{key} must be non-negative")
    return item


def _finite_float(value: dict[str, Any], key: str) -> float:
    item = value.get(key)
    if not isinstance(item, (int, float)) or isinstance(item, bool):
        raise RepresentationConfigError(f"{key} must be numeric")
    result = float(item)
    if not math.isfinite(result):
        raise RepresentationConfigError(f"{key} must be finite")
    return result


def _positive_float(value: dict[str, Any], key: str) -> float:
    result = _finite_float(value, key)
    if result <= 0:
        raise RepresentationConfigError(f"{key} must be positive")
    return result


def _nonnegative_float(value: dict[str, Any], key: str) -> float:
    result = _finite_float(value, key)
    if result < 0:
        raise RepresentationConfigError(f"{key} must be non-negative")
    return result


def _unit_float(value: dict[str, Any], key: str) -> float:
    result = _nonnegative_float(value, key)
    if result > 1:
        raise RepresentationConfigError(f"{key} must be at most 1")
    return result


def _sha256(value: dict[str, Any], key: str) -> str:
    item = _string(value, key)
    if len(item) != 64 or any(char not in "0123456789abcdef" for char in item):
        raise RepresentationConfigError(f"{key} must be a lowercase SHA-256")
    return item


def _repo_path(root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RepresentationConfigError(f"path must be repository-relative: {value}")
    return (root / candidate).resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _validate_logical_id(logical_id: str) -> None:
    parts = logical_id.split("/")
    if len(parts) != 3 or any(not part or part in {".", ".."} for part in parts):
        raise RepresentationConfigError(f"invalid logical ID: {logical_id!r}")
