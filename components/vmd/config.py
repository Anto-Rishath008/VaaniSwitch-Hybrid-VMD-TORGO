"""Configuration loading and cache identity for Module 1."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


class VMDConfigError(ValueError):
    """Raised when the Module 1 configuration is incomplete or unsafe."""


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Module 0 interface consumed by Module 1."""

    manifest_path: Path
    module0_summary_path: Path
    module0_leakage_report_path: Path
    raw_root: Path
    default_subset: str


@dataclass(frozen=True, slots=True)
class AudioConfig:
    """Working-waveform standardization settings."""

    target_sample_rate: int
    target_channels: int
    normalization: str
    target_rms: float
    max_gain: float
    peak_limit: float
    silence_epsilon: float


@dataclass(frozen=True, slots=True)
class VMDAlgorithmConfig:
    """Numerical VMD settings; these are initial, not claimed optimal."""

    modes_k: int
    alpha: float
    tau: float
    dc: bool
    init: int
    tolerance: float
    max_iterations: int


@dataclass(frozen=True, slots=True)
class SelectedModesAblationConfig:
    """Explicit-only selected-mode ablation interface."""

    enabled: bool
    mode_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ReconstructionConfig:
    """Reconstruction policy. All-mode reconstruction is the locked default."""

    default: str
    selected_modes_ablation: SelectedModesAblationConfig


@dataclass(frozen=True, slots=True)
class CacheConfig:
    """Portable, resumable cache controls."""

    enabled: bool
    resume: bool
    root: Path


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Small-run observability settings."""

    progress_every: int
    plot_examples: int


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Module 1 report destinations."""

    summary: Path
    failures: Path
    pipeline_log: Path
    plots: Path


@dataclass(frozen=True, slots=True)
class Module1Config:
    """Validated, repository-relative Module 1 configuration."""

    schema_version: str
    implementation: str
    repo_root: Path
    dataset: DatasetConfig
    audio: AudioConfig
    vmd: VMDAlgorithmConfig
    reconstruction: ReconstructionConfig
    cache: CacheConfig
    execution: ExecutionConfig
    outputs: OutputConfig

    @property
    def processing_identity(self) -> dict[str, Any]:
        """Return only settings that affect per-record numerical cache contents."""

        identity = {
            "schema_version": self.schema_version,
            "implementation": self.implementation,
            "audio": asdict(self.audio),
            "vmd": asdict(self.vmd),
            "reconstruction": asdict(self.reconstruction),
        }
        serialized = json.loads(json.dumps(identity, sort_keys=True))
        if not isinstance(serialized, dict):
            raise AssertionError("processing identity must serialize to a mapping")
        return serialized

    @property
    def config_hash(self) -> str:
        """Return the full stable SHA-256 cache identity."""

        canonical = json.dumps(
            self.processing_identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @classmethod
    def from_yaml(
        cls,
        path: Path,
        *,
        repo_root: Path | None = None,
    ) -> Module1Config:
        """Load YAML and resolve production paths relative to the repository."""

        resolved = path.resolve()
        if not resolved.is_file():
            raise VMDConfigError(f"configuration file does not exist: {path}")
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise VMDConfigError("configuration root must be a mapping")
        root = repo_root.resolve() if repo_root is not None else resolved.parents[2]

        dataset_raw = _mapping(raw, "dataset")
        audio_raw = _mapping(raw, "audio")
        vmd_raw = _mapping(raw, "vmd")
        reconstruction_raw = _mapping(raw, "reconstruction")
        ablation_raw = _mapping(reconstruction_raw, "selected_modes_ablation")
        cache_raw = _mapping(raw, "cache")
        execution_raw = _mapping(raw, "execution")
        outputs_raw = _mapping(raw, "outputs")

        result = cls(
            schema_version=_string(raw, "schema_version"),
            implementation=_string(raw, "implementation"),
            repo_root=root,
            dataset=DatasetConfig(
                manifest_path=_repo_path(root, _string(dataset_raw, "manifest_path")),
                module0_summary_path=_repo_path(
                    root, _string(dataset_raw, "module0_summary_path")
                ),
                module0_leakage_report_path=_repo_path(
                    root, _string(dataset_raw, "module0_leakage_report_path")
                ),
                raw_root=_repo_path(root, _string(dataset_raw, "raw_root")),
                default_subset=_string(dataset_raw, "default_subset"),
            ),
            audio=AudioConfig(
                target_sample_rate=_positive_int(audio_raw, "target_sample_rate"),
                target_channels=_positive_int(audio_raw, "target_channels"),
                normalization=_string(audio_raw, "normalization"),
                target_rms=_positive_float(audio_raw, "target_rms"),
                max_gain=_positive_float(audio_raw, "max_gain"),
                peak_limit=_unit_float(audio_raw, "peak_limit"),
                silence_epsilon=_positive_float(audio_raw, "silence_epsilon"),
            ),
            vmd=VMDAlgorithmConfig(
                modes_k=_positive_int(vmd_raw, "modes_k"),
                alpha=_positive_float(vmd_raw, "alpha"),
                tau=_nonnegative_float(vmd_raw, "tau"),
                dc=_boolean(vmd_raw, "dc"),
                init=_integer(vmd_raw, "init"),
                tolerance=_positive_float(vmd_raw, "tolerance"),
                max_iterations=_positive_int(vmd_raw, "max_iterations"),
            ),
            reconstruction=ReconstructionConfig(
                default=_string(reconstruction_raw, "default"),
                selected_modes_ablation=SelectedModesAblationConfig(
                    enabled=_boolean(ablation_raw, "enabled"),
                    mode_indices=_integer_tuple(ablation_raw, "mode_indices"),
                ),
            ),
            cache=CacheConfig(
                enabled=_boolean(cache_raw, "enabled"),
                resume=_boolean(cache_raw, "resume"),
                root=_repo_path(root, _string(cache_raw, "root")),
            ),
            execution=ExecutionConfig(
                progress_every=_positive_int(execution_raw, "progress_every"),
                plot_examples=_nonnegative_int(execution_raw, "plot_examples"),
            ),
            outputs=OutputConfig(
                summary=_repo_path(root, _string(outputs_raw, "summary")),
                failures=_repo_path(root, _string(outputs_raw, "failures")),
                pipeline_log=_repo_path(root, _string(outputs_raw, "pipeline_log")),
                plots=_repo_path(root, _string(outputs_raw, "plots")),
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        """Enforce locked scientific and portability constraints."""

        if self.schema_version != "1.0":
            raise VMDConfigError("schema_version must be 1.0")
        if self.implementation != "components_numpy_vmd_v1":
            raise VMDConfigError("unsupported VMD implementation identity")
        if self.dataset.default_subset != "core":
            raise VMDConfigError("the default Module 1 subset must be core")
        if self.audio.target_sample_rate != 16_000 or self.audio.target_channels != 1:
            raise VMDConfigError("Module 1 working audio must be 16 kHz mono")
        if self.audio.normalization != "rms":
            raise VMDConfigError("only conservative RMS normalization is supported")
        if self.audio.max_gain < 1.0:
            raise VMDConfigError("audio.max_gain must be at least 1.0")
        if self.audio.target_rms > self.audio.peak_limit:
            raise VMDConfigError("target_rms cannot exceed peak_limit")
        if self.vmd.init not in {0, 1}:
            raise VMDConfigError("VMD init must be 0 or deterministic uniform init 1")
        if self.reconstruction.default != "all_modes":
            raise VMDConfigError("all_modes must remain the default reconstruction")
        ablation = self.reconstruction.selected_modes_ablation
        if ablation.enabled:
            if not ablation.mode_indices:
                raise VMDConfigError("enabled selected-mode ablation requires explicit indices")
            if any(index < 1 or index > self.vmd.modes_k for index in ablation.mode_indices):
                raise VMDConfigError("selected mode indices must use one-based indices within K")
        elif ablation.mode_indices:
            raise VMDConfigError("disabled selected-mode ablation cannot select modes")
        if len(set(ablation.mode_indices)) != len(ablation.mode_indices):
            raise VMDConfigError("selected mode indices must be unique")
        for path in (
            self.dataset.manifest_path,
            self.dataset.module0_summary_path,
            self.dataset.module0_leakage_report_path,
            self.dataset.raw_root,
            self.cache.root,
            self.outputs.summary,
            self.outputs.failures,
            self.outputs.pipeline_log,
            self.outputs.plots,
        ):
            try:
                path.relative_to(self.repo_root)
            except ValueError as exc:
                raise VMDConfigError(f"configured path escapes repository: {path}") from exc

        raw_namespace = (self.repo_root / "data/raw").resolve()
        processed_namespace = (self.repo_root / "data/processed").resolve()
        artifacts_namespace = (self.repo_root / "artifacts").resolve()
        if not _is_within(self.dataset.raw_root, raw_namespace):
            raise VMDConfigError("dataset.raw_root must remain under data/raw")
        if not _is_within(self.cache.root, processed_namespace):
            raise VMDConfigError("cache.root must remain under data/processed")
        writable_outputs = (
            self.outputs.summary,
            self.outputs.failures,
            self.outputs.pipeline_log,
            self.outputs.plots,
        )
        if any(not _is_within(path, artifacts_namespace) for path in writable_outputs):
            raise VMDConfigError("Module 1 report outputs must remain under artifacts")
        protected_inputs = {
            self.dataset.manifest_path,
            self.dataset.module0_summary_path,
            self.dataset.module0_leakage_report_path,
        }
        if protected_inputs.intersection(writable_outputs):
            raise VMDConfigError("Module 1 outputs cannot overwrite Module 0 inputs")
        if not self.outputs.summary.name.startswith("module1_"):
            raise VMDConfigError("Module 1 summary filename must start with module1_")
        if not self.outputs.failures.name.startswith("module1_"):
            raise VMDConfigError("Module 1 failure filename must start with module1_")
        if not self.outputs.pipeline_log.name.startswith("module1_"):
            raise VMDConfigError("Module 1 log filename must start with module1_")


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise VMDConfigError(f"{key} must be a mapping")
    return item


def _string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise VMDConfigError(f"{key} must be a non-empty string")
    return item


def _boolean(value: dict[str, Any], key: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise VMDConfigError(f"{key} must be a boolean")
    return item


def _integer(value: dict[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        raise VMDConfigError(f"{key} must be an integer")
    return item


def _positive_int(value: dict[str, Any], key: str) -> int:
    item = _integer(value, key)
    if item <= 0:
        raise VMDConfigError(f"{key} must be positive")
    return item


def _nonnegative_int(value: dict[str, Any], key: str) -> int:
    item = _integer(value, key)
    if item < 0:
        raise VMDConfigError(f"{key} must be non-negative")
    return item


def _finite_float(value: dict[str, Any], key: str) -> float:
    item = value.get(key)
    if not isinstance(item, (int, float)) or isinstance(item, bool):
        raise VMDConfigError(f"{key} must be numeric")
    result = float(item)
    if not (-float("inf") < result < float("inf")):
        raise VMDConfigError(f"{key} must be finite")
    return result


def _positive_float(value: dict[str, Any], key: str) -> float:
    result = _finite_float(value, key)
    if result <= 0:
        raise VMDConfigError(f"{key} must be positive")
    return result


def _nonnegative_float(value: dict[str, Any], key: str) -> float:
    result = _finite_float(value, key)
    if result < 0:
        raise VMDConfigError(f"{key} must be non-negative")
    return result


def _unit_float(value: dict[str, Any], key: str) -> float:
    result = _positive_float(value, key)
    if result > 1:
        raise VMDConfigError(f"{key} must be at most 1")
    return result


def _integer_tuple(value: dict[str, Any], key: str) -> tuple[int, ...]:
    item = value.get(key)
    if not isinstance(item, list) or any(
        not isinstance(entry, int) or isinstance(entry, bool) for entry in item
    ):
        raise VMDConfigError(f"{key} must be a list of integers")
    return tuple(item)


def _repo_path(root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise VMDConfigError(f"path must be repository-relative: {value}")
    return (root / candidate).resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True
