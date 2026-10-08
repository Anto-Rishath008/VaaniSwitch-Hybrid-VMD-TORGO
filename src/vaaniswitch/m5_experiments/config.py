"""Frozen scientific protocol, bounded candidates, and canonical Module 5 paths."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

ABLATIONS = ("A0", "A1", "B1", "B2", "B3", "C1", "C2")
SIGNED_HASHES = {
    "manifest_fingerprint": "91c3327ea1f6a6b2057ffa02198f86d682668d22f247beef27f1466e959a9061",
    "module1_config_hash": "bfd7aec36d99a88cfacc72df85e124d3f4b99b76ed47d71c3ced9a153fa92685",
    "representation_config_hash": (
        "d8a36c6ab8c48b15e61ecd63194ffa30ede4"
        "e7e7e7ab0a30c197002652f9d65d"
    ),
    "encoder_config_hash": "7c8e44eddb11704d08e45a1b63bc9cc54b28d679f5e3a3ae1a474b61e3a40383",
    "fusion_config_hash": "6fc69cf0ad558258a4b51196e82fee8fc9f2bc6cd25349babd6a99144e68f6de",
}


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_path(root: Path, relative: str) -> Path:
    """Reject redirected ancestors as well as traversal before every output access."""
    root = root.resolve()
    path = root / relative
    if path.is_absolute() and not path.is_relative_to(root):
        raise ValueError("path escapes repository")
    resolved = path.resolve()
    if resolved != path or not resolved.is_relative_to(root):
        raise ValueError("canonical Module 5 path was redirected")
    return resolved


@dataclass(frozen=True)
class Candidate:
    id: str
    learning_rate: float
    weight_decay: float
    classifier_dropout: float


CANDIDATES = (
    Candidate("baseline", 3e-4, 1e-4, 0.30),
    Candidate("lower_lr", 1e-4, 1e-4, 0.30),
    Candidate("stronger_decay", 3e-4, 1e-3, 0.30),
    Candidate("stronger_dropout", 3e-4, 1e-4, 0.40),
)
TRAINING = {
    "optimizer": "adamw",
    "batch_size": 16,
    "max_epochs": 30,
    "patience": 5,
    "min_delta": 1e-4,
    "threshold": 0.5,
    "augmentation": "none",
    "backbone": "frozen",
}
SELECTION = {
    "metric": "macro_f1",
    "tie_breakers": ["balanced_accuracy", "roc_auc", "lower_loss", "fewer_trainable_parameters"],
    "ablation_policy": "share_outer_fold_selected_c2_candidate",
}


@dataclass(frozen=True)
class ExperimentConfig:
    repo_root: Path
    cpu_threads: int = 2

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "implementation": "vaaniswitch_nested_experiments_v1",
            "seed": 42,
            "training": TRAINING,
            "selection": SELECTION,
            "candidates": [asdict(c) for c in CANDIDATES],
            "ablations": list(ABLATIONS),
            "c3": "disabled_design_only",
            "signed_hashes": SIGNED_HASHES,
            "architecture": "signed_D128_heads4_all_modes_K5_nodes256_canvas8s",
            "inner_split": "class_stratified_speaker_sha256_v1",
            "epoch_selection": "lexicographic_metrics; patience resets on macro_f1 > best + 1e-4",
            "aggregation": "unweighted_three_fold_mean_and_sample_std_ddof1",
        }

    @property
    def config_hash(self) -> str:
        return digest(self.identity)

    @property
    def artifact_root(self) -> Path:
        return canonical_path(self.repo_root, f"artifacts/module5/{self.config_hash}")

    def output(self, relative: str) -> Path:
        return canonical_path(self.repo_root, f"artifacts/module5/{self.config_hash}/{relative}")

    def candidate(self, candidate_id: str) -> Candidate:
        for candidate in CANDIDATES:
            if candidate.id == candidate_id:
                return candidate
        raise ValueError("unknown candidate")

    @classmethod
    def from_yaml(cls, path: Path, *, repo_root: Path) -> ExperimentConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = cls(repo_root.resolve())
        expected = {
            k: config.identity[k]
            for k in (
                "schema_version",
                "implementation",
                "seed",
                "training",
                "selection",
                "candidates",
                "ablations",
                "c3",
            )
        }
        if not isinstance(raw, dict) or set(raw) != {*expected, "execution"}:
            raise ValueError("Module 5 protocol fields differ from the frozen schema")
        if any(raw[k] != value for k, value in expected.items()):
            raise ValueError("Module 5 primary protocol is frozen; unsupported configuration")
        execution = raw["execution"]
        if not isinstance(execution, dict) or set(execution) != {"cpu_threads"}:
            raise ValueError("invalid execution settings")
        threads = execution["cpu_threads"]
        if type(threads) is not int or threads < 1:
            raise ValueError("cpu_threads must be positive")
        return cls(repo_root.resolve(), threads)
