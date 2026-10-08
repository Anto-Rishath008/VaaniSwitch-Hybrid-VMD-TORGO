"""Validated baseline semantics; output paths cannot target signed inputs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml


class FusionConfigError(ValueError):
    """The requested configuration violates the Module 4 baseline."""


ARCHITECTURE: dict[str, Any] = {
    "token_dim": 128,
    "attention": {
        "type": "cross_attention",
        "query": "mel_tokens",
        "key": "nonlinear_tokens",
        "value": "nonlinear_tokens",
        "query_tokens": 16,
        "key_value_tokens": 17,
        "heads": 4,
        "blocks": 1,
        "batch_first": True,
        "dropout": 0.10,
        "residual": True,
        "normalization": "layernorm",
    },
    "pooling": {"mel": "mean", "nonlinear": "mean"},
    "classifier": {
        "input_dim": 256,
        "hidden_dim": 128,
        "output_dim": 1,
        "activation": "gelu",
        "dropout": 0.30,
    },
    "loss": {
        "type": "bce_with_logits",
        "label_positive": "dysarthric",
        "training_pos_weight": "train_split_only",
    },
}


def semantic_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class FusionSourceConfig:
    encoder_config_path: Path
    module3_signoff_path: Path
    encoder_config_hash: str
    representation_config_hash: str
    module1_config_hash: str
    manifest_fingerprint: str
    core_record_count: int

    @property
    def hashes(self) -> dict[str, str]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key.endswith(("_hash", "_fingerprint"))
        }


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    max_epochs: int = 30
    patience: int = 5
    min_delta: float = 1e-4
    selection_metric: str = "validation_macro_f1"
    threshold: float = 0.5
    seed: int = 42
    inner_split: str = "class_stratified_speaker_sha256_v1"
    augmentation: str = "none"
    resampling: str = "none"
    scheduler: str = "none"
    freeze_mel_backbone: bool = True

    def __post_init__(self) -> None:
        for key in ("batch_size", "max_epochs", "patience", "seed"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise FusionConfigError(f"{key} must be a positive integer")
        if self.max_epochs > 30 or self.patience > self.max_epochs or self.seed != 42:
            raise FusionConfigError("baseline uses seed 42, at most 30 epochs, bounded patience")
        for key in ("learning_rate", "weight_decay", "min_delta", "threshold"):
            value = getattr(self, key)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise FusionConfigError(f"{key} must be a finite nonnegative number")
        if self.learning_rate == 0 or self.threshold != 0.5:
            raise FusionConfigError("positive learning rate and fixed threshold 0.5 required")
        locked = {
            "optimizer": "adamw",
            "selection_metric": "validation_macro_f1",
            "inner_split": "class_stratified_speaker_sha256_v1",
            "augmentation": "none",
            "resampling": "none",
            "scheduler": "none",
            "freeze_mel_backbone": True,
        }
        if any(getattr(self, key) != value for key, value in locked.items()):
            raise FusionConfigError("training policy violates the frozen Module 4 baseline")
        if self.freeze_mel_backbone is not True:
            raise FusionConfigError("backbone freeze must be boolean true")


@dataclass(frozen=True, slots=True)
class Module4Config:
    repo_root: Path
    source: FusionSourceConfig
    training: TrainingConfig
    cpu_threads: int = 2

    @property
    def processing_identity(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "implementation": "vaaniswitch_pytorch_fusion_v1",
            **json.loads(json.dumps(ARCHITECTURE)),
            "source": {**self.source.hashes, "core_record_count": self.source.core_record_count},
            "training": asdict(self.training),
            "label_mapping": {"control": 0, "dysarthric": 1},
        }

    @property
    def config_hash(self) -> str:
        return semantic_hash(self.processing_identity)

    @property
    def hashes(self) -> dict[str, str]:
        return {**self.source.hashes, "fusion_config_hash": self.config_hash}

    def with_batch_size(self, value: int | None) -> Module4Config:
        return (
            self
            if value is None
            else replace(self, training=replace(self.training, batch_size=value))
        )

    def output(self, kind: str, name: str) -> Path:
        """Only Module 4 filenames in dedicated artifact categories are writable."""
        if kind not in {"metrics", "logs", "checkpoints"} or not name.startswith("module4"):
            raise FusionConfigError("outputs must belong to Module 4")
        if Path(name).name != name:
            raise FusionConfigError("output name must be a basename")
        expected_parent = self.repo_root.resolve() / "artifacts" / kind
        parent = expected_parent.resolve()
        if parent != expected_parent:
            raise FusionConfigError(
                "output category cannot redirect outside its canonical location"
            )
        result = (parent / name).resolve()
        if (
            not result.is_relative_to(self.repo_root.resolve())
            or result.parent != parent
            or result.name != name
        ):
            raise FusionConfigError("output escapes its artifact directory")
        return result

    @classmethod
    def from_yaml(cls, path: Path, *, repo_root: Path | None = None) -> Module4Config:
        resolved = path.resolve()
        root = repo_root.resolve() if repo_root else resolved.parents[2]
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        keys = set(ARCHITECTURE) | {
            "schema_version",
            "implementation",
            "source",
            "training",
            "execution",
        }
        if not isinstance(raw, dict) or set(raw) != keys:
            raise FusionConfigError("Module 4 configuration fields do not match schema")
        if raw["schema_version"] != "1.0" or raw["implementation"] != (
            "vaaniswitch_pytorch_fusion_v1"
        ):
            raise FusionConfigError("unsupported Module 4 implementation/schema")
        for key, value in ARCHITECTURE.items():
            if semantic_hash(raw[key]) != semantic_hash(value):
                raise FusionConfigError(f"{key} violates the locked fusion contract")
        source = raw["source"]
        if not isinstance(source, dict) or set(source) != set(
            FusionSourceConfig.__dataclass_fields__
        ):
            raise FusionConfigError("source fields do not match schema")
        source = dict(source)
        for key in ("encoder_config_path", "module3_signoff_path"):
            value = source[key]
            if not isinstance(value, str) or Path(value).is_absolute() or ".." in Path(value).parts:
                raise FusionConfigError("source path must be repository relative")
            source[key] = (root / value).resolve()
            if not source[key].is_relative_to(root):
                raise FusionConfigError("source path escapes repository")
        for key in (
            "encoder_config_hash",
            "representation_config_hash",
            "module1_config_hash",
            "manifest_fingerprint",
        ):
            value = source[key]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise FusionConfigError(f"{key} must be a lowercase SHA-256")
        if type(source["core_record_count"]) is not int or source["core_record_count"] <= 0:
            raise FusionConfigError("signed CORE count must be positive")
        training = raw["training"]
        if not isinstance(training, dict) or set(training) != set(
            TrainingConfig.__dataclass_fields__
        ):
            raise FusionConfigError("training fields do not match schema")
        execution = raw["execution"]
        if not isinstance(execution, dict) or set(execution) != {"cpu_threads"}:
            raise FusionConfigError("execution fields do not match schema")
        threads = execution["cpu_threads"]
        if type(threads) is not int or threads <= 0:
            raise FusionConfigError("cpu_threads must be a positive integer")
        return cls(root, FusionSourceConfig(**source), TrainingConfig(**training), threads)
