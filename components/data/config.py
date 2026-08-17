"""Configuration loading and validation for Module 0."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from components.data.schemas import (
    TORGO_GROUP_RULES,
    Condition,
    Gender,
    GroupMetadata,
    PromptType,
    UsabilityStatus,
)


class ConfigError(ValueError):
    """Raised when the TORGO YAML is incomplete or unsafe."""


@dataclass(frozen=True, slots=True)
class QualityConfig:
    """Signal audit thresholds and execution controls."""

    chunk_frames: int
    max_workers: int
    compute_sha256: bool
    extremely_short_seconds: float
    near_silent_dbfs: float
    clipping_amplitude_ratio: float
    clipping_fraction_threshold: float
    review_flags: frozenset[str]


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Deterministic grouped cross-validation policy."""

    group_by: str
    requested_max_folds: int
    minimum_folds: int
    hard_strata: tuple[str, ...]
    balance_statuses: frozenset[UsabilityStatus]
    seed: int


@dataclass(frozen=True, slots=True)
class DownstreamSubsetConfig:
    """Declarative record-selection policy exposed to later modules."""

    normal_speech_eligible: bool
    usability_statuses: frozenset[UsabilityStatus]
    prompt_types: frozenset[PromptType]
    experiment_role: str


@dataclass(frozen=True, slots=True)
class TorgoConfig:
    """Validated, path-aware configuration used throughout Module 0."""

    schema_version: str
    dataset_id: str
    language: str
    repo_root: Path
    raw_root: Path
    logical_key: tuple[str, ...]
    groups: dict[str, GroupMetadata]
    session_pattern: str
    prompt_id_pattern: str
    prompt_directory: str
    prompt_extension: str
    transcript_encodings: tuple[str, ...]
    microphone_directories: dict[str, str]
    microphone_priority: tuple[str, ...]
    quality: QualityConfig
    evaluation: EvaluationConfig
    downstream_subsets: dict[str, DownstreamSubsetConfig]
    outputs: dict[str, Path]
    manual_audit_reference: dict[str, int]

    @classmethod
    def from_yaml(
        cls,
        config_path: Path,
        *,
        repo_root: Path | None = None,
    ) -> TorgoConfig:
        """Load YAML and resolve every production path relative to the repository."""

        resolved_config = config_path.resolve()
        if not resolved_config.is_file():
            raise ConfigError(f"configuration file does not exist: {config_path}")
        raw_data = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
        if not isinstance(raw_data, dict):
            raise ConfigError("configuration root must be a mapping")

        if repo_root is None:
            try:
                inferred_root = resolved_config.parents[2]
            except IndexError as exc:
                raise ConfigError("cannot infer repository root from config path") from exc
        else:
            inferred_root = repo_root.resolve()

        groups_raw = _mapping(raw_data, "groups")
        groups: dict[str, GroupMetadata] = {}
        for group_name, group_value in groups_raw.items():
            if not isinstance(group_name, str) or not isinstance(group_value, dict):
                raise ConfigError("each groups entry must be a named mapping")
            groups[group_name] = GroupMetadata(
                condition=Condition(_string(group_value, "condition")),
                gender=Gender(_string(group_value, "gender")),
                speaker_pattern=_string(group_value, "speaker_pattern"),
            )

        discovery = _mapping(raw_data, "discovery")
        microphones_raw = _mapping(discovery, "microphones")
        microphone_directories = {
            str(name): _string(_ensure_mapping(value, f"microphones.{name}"), "directory")
            for name, value in microphones_raw.items()
        }
        priority = _string_tuple(discovery, "microphone_priority")
        if priority != ("head", "array"):
            raise ConfigError("microphone_priority must be exactly [head, array]")
        if set(priority) != set(microphone_directories):
            raise ConfigError("microphone definitions must match microphone_priority")

        quality_raw = _mapping(raw_data, "quality")
        quality = QualityConfig(
            chunk_frames=_positive_int(quality_raw, "chunk_frames"),
            max_workers=_positive_int(quality_raw, "max_workers"),
            compute_sha256=_boolean(quality_raw, "compute_sha256"),
            extremely_short_seconds=_nonnegative_float(quality_raw, "extremely_short_seconds"),
            near_silent_dbfs=float(quality_raw["near_silent_dbfs"]),
            clipping_amplitude_ratio=_unit_float(quality_raw, "clipping_amplitude_ratio"),
            clipping_fraction_threshold=_unit_float(quality_raw, "clipping_fraction_threshold"),
            review_flags=frozenset(_string_tuple(quality_raw, "review_flags")),
        )

        evaluation_raw = _mapping(raw_data, "evaluation")
        evaluation = EvaluationConfig(
            group_by=_string(evaluation_raw, "group_by"),
            requested_max_folds=_positive_int(evaluation_raw, "requested_max_folds"),
            minimum_folds=_positive_int(evaluation_raw, "minimum_folds"),
            hard_strata=_string_tuple(evaluation_raw, "hard_strata"),
            balance_statuses=frozenset(
                UsabilityStatus(item) for item in _string_tuple(evaluation_raw, "balance_statuses")
            ),
            seed=int(evaluation_raw["seed"]),
        )
        if evaluation.minimum_folds > evaluation.requested_max_folds:
            raise ConfigError("minimum_folds cannot exceed requested_max_folds")
        if evaluation.group_by != "speaker_id":
            raise ConfigError("Module 0 evaluation must group by speaker_id")
        if evaluation.hard_strata != ("condition", "gender"):
            raise ConfigError("hard_strata must be [condition, gender] for TORGO")

        subsets_raw = _mapping(raw_data, "downstream_subsets")
        downstream_subsets: dict[str, DownstreamSubsetConfig] = {}
        for name, raw_subset in subsets_raw.items():
            subset = _ensure_mapping(raw_subset, f"downstream_subsets.{name}")
            downstream_subsets[str(name)] = DownstreamSubsetConfig(
                normal_speech_eligible=_boolean(subset, "normal_speech_eligible"),
                usability_statuses=frozenset(
                    UsabilityStatus(item) for item in _string_tuple(subset, "usability_statuses")
                ),
                prompt_types=frozenset(
                    PromptType(item) for item in _string_tuple(subset, "prompt_types")
                ),
                experiment_role=_string(subset, "experiment_role"),
            )

        output_raw = _mapping(raw_data, "outputs")
        outputs = {
            str(name): (inferred_root / _string(output_raw, str(name))).resolve()
            for name in output_raw
        }
        raw_root = (inferred_root / _string(raw_data, "raw_root")).resolve()
        if not raw_root.is_relative_to(inferred_root):
            raise ConfigError("raw_root must remain inside the repository")
        for name, output_path in outputs.items():
            if output_path == raw_root or output_path.is_relative_to(raw_root):
                raise ConfigError(f"output {name!r} cannot be inside immutable raw_root")
            if not output_path.is_relative_to(inferred_root):
                raise ConfigError(f"output {name!r} must remain inside the repository")

        manual_reference_raw = _mapping(raw_data, "manual_audit_reference")
        manual_reference = {str(name): int(value) for name, value in manual_reference_raw.items()}

        config = cls(
            schema_version=_string(raw_data, "schema_version"),
            dataset_id=_string(raw_data, "dataset_id"),
            language=_string(raw_data, "language"),
            repo_root=inferred_root,
            raw_root=raw_root,
            logical_key=_string_tuple(raw_data, "logical_key"),
            groups=groups,
            session_pattern=_string(discovery, "session_pattern"),
            prompt_id_pattern=_string(discovery, "prompt_id_pattern"),
            prompt_directory=_string(discovery, "prompt_directory"),
            prompt_extension=_string(discovery, "prompt_extension"),
            transcript_encodings=_string_tuple(discovery, "transcript_encodings"),
            microphone_directories=microphone_directories,
            microphone_priority=priority,
            quality=quality,
            evaluation=evaluation,
            downstream_subsets=downstream_subsets,
            outputs=outputs,
            manual_audit_reference=manual_reference,
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Validate scientific and path invariants not covered during parsing."""

        if self.logical_key != ("speaker_id", "session_id", "prompt_id"):
            raise ConfigError("logical_key must be [speaker_id, session_id, prompt_id]")
        required_groups = {"F", "M", "FC", "MC"}
        if set(self.groups) != required_groups:
            raise ConfigError(f"groups must be exactly {sorted(required_groups)}")
        for group_name, (condition, gender, pattern) in TORGO_GROUP_RULES.items():
            configured = self.groups[group_name]
            if (
                configured.condition is not condition
                or configured.gender is not gender
                or configured.speaker_pattern != pattern
            ):
                raise ConfigError(
                    f"group {group_name} must map to condition={condition.value}, "
                    f"gender={gender.value}, speaker_pattern={pattern!r}"
                )
        if not self.prompt_extension.startswith("."):
            raise ConfigError("prompt_extension must begin with a dot")
        if not self.transcript_encodings:
            raise ConfigError("at least one transcript encoding is required")
        if not self.quality.compute_sha256:
            raise ConfigError("compute_sha256 must be true for duplicate-content leakage auditing")
        expected_subsets = {
            "core": frozenset({UsabilityStatus.USABLE}),
            "extended": frozenset({UsabilityStatus.USABLE, UsabilityStatus.REVIEW}),
        }
        if set(self.downstream_subsets) != set(expected_subsets):
            raise ConfigError("downstream_subsets must contain exactly core and extended")
        for name, expected_statuses in expected_subsets.items():
            subset = self.downstream_subsets[name]
            if not subset.normal_speech_eligible:
                raise ConfigError(f"downstream subset {name} must require normal_speech_eligible")
            if subset.usability_statuses != expected_statuses:
                raise ConfigError(f"downstream subset {name} has invalid usability_statuses")
            if subset.prompt_types != frozenset({PromptType.WORD, PromptType.SENTENCE}):
                raise ConfigError(
                    f"downstream subset {name} must contain only word and sentence prompts"
                )
        required_outputs = {
            "manifest_jsonl",
            "manifest_csv",
            "speaker_folds",
            "dataset_summary",
            "exclusions",
            "audio_quality",
            "microphone_coverage",
            "leakage_report",
            "pipeline_log",
        }
        missing = required_outputs - self.outputs.keys()
        if missing:
            raise ConfigError(f"missing output paths: {sorted(missing)}")

    def relative_path(self, path: Path) -> str:
        """Return a repository-relative POSIX path or reject the path."""

        resolved = path.resolve()
        if not resolved.is_relative_to(self.repo_root):
            raise ConfigError(f"path is outside repository: {path}")
        return resolved.relative_to(self.repo_root).as_posix()


def _ensure_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a mapping")
    return value


def _mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in data:
        raise ConfigError(f"missing configuration key: {key}")
    return _ensure_mapping(data[key], key)


def _string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{key} must be a non-empty string")
    return value


def _string_tuple(data: dict[str, Any], key: str) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{key} must be a list of strings")
    return tuple(value)


def _positive_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{key} must be a positive integer")
    return value


def _boolean(data: dict[str, Any], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be a boolean")
    return value


def _nonnegative_float(data: dict[str, Any], key: str) -> float:
    value = float(data[key])
    if value < 0:
        raise ConfigError(f"{key} cannot be negative")
    return value


def _unit_float(data: dict[str, Any], key: str) -> float:
    value = float(data[key])
    if not 0 <= value <= 1:
        raise ConfigError(f"{key} must be between 0 and 1")
    return value
