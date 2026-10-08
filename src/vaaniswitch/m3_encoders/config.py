"""Strict Module 3 configuration and content-addressed model identity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


class EncoderConfigError(ValueError):
    """Raised when Module 3 configuration violates the locked architecture."""


@dataclass(frozen=True, slots=True)
class SourceConfig:
    manifest_path: Path
    speaker_folds_path: Path
    module2_config_path: Path
    module2_signoff_path: Path
    representation_cache_root: Path
    representation_config_hash: str
    module1_config_hash: str
    manifest_fingerprint: str
    subset: str


@dataclass(frozen=True, slots=True)
class MelEncoderConfig:
    backbone: str
    input_channels: int
    pretrained_weights: str
    replicate_single_channel: bool
    fixed_db_min: float
    fixed_db_max: float
    imagenet_normalization: bool
    adaptive_pool: tuple[int, int]
    tokens: int
    freeze_backbone_initially: bool


@dataclass(frozen=True, slots=True)
class GraphEncoderConfig:
    implementation: str
    input_dim: int
    layers: int
    hidden_dim: int
    graph_feature_dim: int
    activation: str
    pooling: str
    tokens: int
    add_self_loops: bool
    adjacency_normalization: str


@dataclass(frozen=True, slots=True)
class RecurrenceEncoderConfig:
    implementation: str
    input_channels: int
    channels: tuple[int, ...]
    normalization: str
    activation: str
    adaptive_pool: tuple[int, int]
    tokens: int


@dataclass(frozen=True, slots=True)
class OutputConfig:
    summary: Path
    failures: Path
    pipeline_log: Path
    parameter_audit: Path


@dataclass(frozen=True, slots=True)
class Module3Config:
    schema_version: str
    implementation: str
    repo_root: Path
    source: SourceConfig
    token_dimension: int
    mel: MelEncoderConfig
    graph: GraphEncoderConfig
    recurrence: RecurrenceEncoderConfig
    dropout: float
    smoke_logical_ids: tuple[str, ...]
    outputs: OutputConfig

    @property
    def processing_identity(self) -> dict[str, Any]:
        """Return source/model settings that define encoder semantics."""

        identity = {
            "schema_version": self.schema_version,
            "implementation": self.implementation,
            "representation_config_hash": self.source.representation_config_hash,
            "module1_config_hash": self.source.module1_config_hash,
            "manifest_fingerprint": self.source.manifest_fingerprint,
            "token_dimension": self.token_dimension,
            "mel": asdict(self.mel),
            "graph": asdict(self.graph),
            "recurrence": asdict(self.recurrence),
            "regularization": {"dropout": self.dropout},
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
    def from_yaml(cls, path: Path, *, repo_root: Path | None = None) -> Module3Config:
        resolved = path.resolve()
        if not resolved.is_file():
            raise EncoderConfigError(f"configuration file does not exist: {path}")
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise EncoderConfigError("configuration root must be a mapping")
        _require_keys(
            raw,
            {
                "schema_version",
                "implementation",
                "source",
                "common_token_dim",
                "mel",
                "graph",
                "recurrence",
                "regularization",
                "execution",
                "outputs",
            },
            "configuration",
        )
        root = repo_root.resolve() if repo_root is not None else resolved.parents[2]
        source = _mapping(raw, "source")
        mel = _mapping(raw, "mel")
        graph = _mapping(raw, "graph")
        recurrence = _mapping(raw, "recurrence")
        regularization = _mapping(raw, "regularization")
        execution = _mapping(raw, "execution")
        outputs = _mapping(raw, "outputs")
        _require_keys(
            source,
            {
                "manifest_path",
                "speaker_folds_path",
                "module2_config_path",
                "module2_signoff_path",
                "representation_cache_root",
                "representation_config_hash",
                "module1_config_hash",
                "manifest_fingerprint",
                "subset",
            },
            "source",
        )
        _require_keys(
            mel,
            {
                "backbone",
                "input_channels",
                "pretrained_weights",
                "replicate_single_channel",
                "fixed_db_min",
                "fixed_db_max",
                "imagenet_normalization",
                "adaptive_pool",
                "tokens",
                "freeze_backbone_initially",
            },
            "mel",
        )
        _require_keys(
            graph,
            {
                "implementation",
                "input_dim",
                "layers",
                "hidden_dim",
                "graph_feature_dim",
                "activation",
                "pooling",
                "tokens",
                "add_self_loops",
                "adjacency_normalization",
            },
            "graph",
        )
        _require_keys(
            recurrence,
            {
                "implementation",
                "input_channels",
                "channels",
                "normalization",
                "activation",
                "adaptive_pool",
                "tokens",
            },
            "recurrence",
        )
        _require_keys(regularization, {"dropout"}, "regularization")
        _require_keys(execution, {"smoke_logical_ids"}, "execution")
        _require_keys(
            outputs,
            {"summary", "failures", "pipeline_log", "parameter_audit"},
            "outputs",
        )
        result = cls(
            schema_version=_string(raw, "schema_version"),
            implementation=_string(raw, "implementation"),
            repo_root=root,
            source=SourceConfig(
                manifest_path=_repo_path(root, _string(source, "manifest_path")),
                speaker_folds_path=_repo_path(root, _string(source, "speaker_folds_path")),
                module2_config_path=_repo_path(root, _string(source, "module2_config_path")),
                module2_signoff_path=_repo_path(root, _string(source, "module2_signoff_path")),
                representation_cache_root=_repo_path(
                    root, _string(source, "representation_cache_root")
                ),
                representation_config_hash=_sha256(source, "representation_config_hash"),
                module1_config_hash=_sha256(source, "module1_config_hash"),
                manifest_fingerprint=_sha256(source, "manifest_fingerprint"),
                subset=_string(source, "subset"),
            ),
            token_dimension=_positive_int(raw, "common_token_dim"),
            mel=MelEncoderConfig(
                backbone=_string(mel, "backbone"),
                input_channels=_positive_int(mel, "input_channels"),
                pretrained_weights=_string(mel, "pretrained_weights"),
                replicate_single_channel=_boolean(mel, "replicate_single_channel"),
                fixed_db_min=_number(mel, "fixed_db_min"),
                fixed_db_max=_number(mel, "fixed_db_max"),
                imagenet_normalization=_boolean(mel, "imagenet_normalization"),
                adaptive_pool=_pair(mel, "adaptive_pool"),
                tokens=_positive_int(mel, "tokens"),
                freeze_backbone_initially=_boolean(mel, "freeze_backbone_initially"),
            ),
            graph=GraphEncoderConfig(
                implementation=_string(graph, "implementation"),
                input_dim=_positive_int(graph, "input_dim"),
                layers=_positive_int(graph, "layers"),
                hidden_dim=_positive_int(graph, "hidden_dim"),
                graph_feature_dim=_positive_int(graph, "graph_feature_dim"),
                activation=_string(graph, "activation"),
                pooling=_string(graph, "pooling"),
                tokens=_positive_int(graph, "tokens"),
                add_self_loops=_boolean(graph, "add_self_loops"),
                adjacency_normalization=_string(graph, "adjacency_normalization"),
            ),
            recurrence=RecurrenceEncoderConfig(
                implementation=_string(recurrence, "implementation"),
                input_channels=_positive_int(recurrence, "input_channels"),
                channels=_integer_tuple(recurrence, "channels"),
                normalization=_string(recurrence, "normalization"),
                activation=_string(recurrence, "activation"),
                adaptive_pool=_pair(recurrence, "adaptive_pool"),
                tokens=_positive_int(recurrence, "tokens"),
            ),
            dropout=_probability(regularization, "dropout"),
            smoke_logical_ids=_string_tuple(execution, "smoke_logical_ids"),
            outputs=OutputConfig(
                summary=_repo_path(root, _string(outputs, "summary")),
                failures=_repo_path(root, _string(outputs, "failures")),
                pipeline_log=_repo_path(root, _string(outputs, "pipeline_log")),
                parameter_audit=_repo_path(root, _string(outputs, "parameter_audit")),
            ),
        )
        result._validate()
        return result

    def _validate(self) -> None:
        if self.schema_version != "1.0" or self.implementation != "vaaniswitch_pytorch_encoders_v1":
            raise EncoderConfigError("unsupported Module 3 schema or implementation")
        if self.source.subset != "core" or self.token_dimension != 128:
            raise EncoderConfigError("Module 3 is locked to CORE and token dimension 128")
        if (
            self.mel.backbone != "resnet18"
            or self.mel.input_channels != 3
            or self.mel.pretrained_weights != "IMAGENET1K_V1"
            or not self.mel.replicate_single_channel
            or self.mel.fixed_db_min != -100.0
            or self.mel.fixed_db_max != 20.0
            or not self.mel.imagenet_normalization
            or self.mel.adaptive_pool != (4, 4)
            or self.mel.tokens != 16
            or not self.mel.freeze_backbone_initially
        ):
            raise EncoderConfigError("Mel encoder settings violate the locked contract")
        if (
            self.graph.implementation != "pure_pytorch_gcn"
            or self.graph.input_dim != 2
            or self.graph.layers != 2
            or self.graph.hidden_dim != 96
            or self.graph.graph_feature_dim != 128
            or self.graph.activation != "gelu"
            or self.graph.pooling != "mean_plus_max"
            or self.graph.tokens != 1
            or not self.graph.add_self_loops
            or self.graph.adjacency_normalization != "symmetric"
        ):
            raise EncoderConfigError("graph encoder settings violate the locked contract")
        if (
            self.recurrence.implementation != "small_cnn"
            or self.recurrence.input_channels != 1
            or self.recurrence.channels != (16, 32, 64)
            or self.recurrence.normalization != "groupnorm"
            or self.recurrence.activation != "gelu"
            or self.recurrence.adaptive_pool != (4, 4)
            or self.recurrence.tokens != 16
        ):
            raise EncoderConfigError("recurrence encoder settings violate the locked contract")
        if not self.smoke_logical_ids or len(set(self.smoke_logical_ids)) != len(
            self.smoke_logical_ids
        ):
            raise EncoderConfigError("smoke logical IDs must be non-empty and unique")
        if (
            "F03/Session1/0179" not in self.smoke_logical_ids
            or "F03/Session1/0118" not in self.smoke_logical_ids
        ):
            raise EncoderConfigError("smoke panel must contain both required F03 records")
        processed = (self.repo_root / "data" / "processed" / "representations").resolve()
        if self.source.representation_cache_root != processed:
            raise EncoderConfigError("representation cache root must be the signed production root")
        protected = {
            self.source.manifest_path,
            self.source.speaker_folds_path,
            self.source.module2_config_path,
            self.source.module2_signoff_path,
        }
        for path in asdict(self.outputs).values():
            if path in protected or _is_relative_to(path, self.repo_root / "data"):
                raise EncoderConfigError("Module 3 outputs cannot overwrite signed inputs or data")


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise EncoderConfigError(f"{key} must be a mapping")
    return item


def _require_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise EncoderConfigError(f"{label} fields do not match schema")


def _string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise EncoderConfigError(f"{key} must be a non-empty string")
    return item


def _boolean(value: dict[str, Any], key: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise EncoderConfigError(f"{key} must be a boolean")
    return item


def _positive_int(value: dict[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
        raise EncoderConfigError(f"{key} must be a positive integer")
    return item


def _number(value: dict[str, Any], key: str) -> float:
    item = value.get(key)
    if not isinstance(item, (int, float)) or isinstance(item, bool):
        raise EncoderConfigError(f"{key} must be numeric")
    return float(item)


def _probability(value: dict[str, Any], key: str) -> float:
    item = _number(value, key)
    if not 0.0 <= item < 1.0:
        raise EncoderConfigError(f"{key} must be in [0,1)")
    return item


def _pair(value: dict[str, Any], key: str) -> tuple[int, int]:
    item = value.get(key)
    if (
        not isinstance(item, list)
        or len(item) != 2
        or any(not isinstance(part, int) or isinstance(part, bool) or part <= 0 for part in item)
    ):
        raise EncoderConfigError(f"{key} must contain two positive integers")
    return item[0], item[1]


def _integer_tuple(value: dict[str, Any], key: str) -> tuple[int, ...]:
    item = value.get(key)
    if not isinstance(item, list) or not item or any(
        not isinstance(part, int) or isinstance(part, bool) or part <= 0 for part in item
    ):
        raise EncoderConfigError(f"{key} must contain positive integers")
    return tuple(item)


def _string_tuple(value: dict[str, Any], key: str) -> tuple[str, ...]:
    item = value.get(key)
    if not isinstance(item, list) or any(not isinstance(part, str) or not part for part in item):
        raise EncoderConfigError(f"{key} must contain strings")
    return tuple(item)


def _sha256(value: dict[str, Any], key: str) -> str:
    item = _string(value, key)
    if len(item) != 64 or any(char not in "0123456789abcdef" for char in item):
        raise EncoderConfigError(f"{key} must be a lowercase SHA-256")
    return item


def _repo_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise EncoderConfigError(f"path must be repository-relative: {value}")
    resolved = (root / path).resolve()
    if not _is_relative_to(resolved, root):
        raise EncoderConfigError(f"path escapes repository: {value}")
    return resolved


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True
