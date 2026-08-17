"""Typed records shared by the Module 0 data pipeline."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Final

SCHEMA_VERSION = "1.0"


class Condition(StrEnum):
    """Binary TORGO speaker condition."""

    DYSARTHRIC = "dysarthric"
    CONTROL = "control"


class Gender(StrEnum):
    """Gender convention encoded by the TORGO group directory."""

    FEMALE = "female"
    MALE = "male"


TORGO_GROUP_RULES: Final[dict[str, tuple[Condition, Gender, str]]] = {
    "F": (Condition.DYSARTHRIC, Gender.FEMALE, r"^F\d+$"),
    "M": (Condition.DYSARTHRIC, Gender.MALE, r"^M\d+$"),
    "FC": (Condition.CONTROL, Gender.FEMALE, r"^FC\d+$"),
    "MC": (Condition.CONTROL, Gender.MALE, r"^MC\d+$"),
}


class Microphone(StrEnum):
    """Supported TORGO microphone roles."""

    HEAD = "head"
    ARRAY = "array"


class AudioStructuralStatus(StrEnum):
    """Whether an available microphone recording can serve as primary audio."""

    MISSING = "missing"
    USABLE = "usable"
    UNUSABLE = "unusable"


class PrimarySelectionReason(StrEnum):
    """Auditable reason for the selected primary microphone."""

    HEAD_STRUCTURALLY_USABLE = "head_structurally_usable"
    ARRAY_STRUCTURALLY_USABLE_HEAD_MISSING = "array_structurally_usable_head_missing"
    HEAD_UNUSABLE_ARRAY_FALLBACK = "head_unusable_array_fallback"
    NO_STRUCTURALLY_USABLE_AUDIO = "no_structurally_usable_audio"


class PromptType(StrEnum):
    """Conservative prompt categories used by later experiments."""

    WORD = "word"
    SENTENCE = "sentence"
    ARTICULATION_TASK = "articulation_task"
    INSTRUCTION_OR_NONSTANDARD = "instruction_or_nonstandard"
    UNKNOWN = "unknown"


class UsabilityStatus(StrEnum):
    """Module 0 disposition for a logical utterance."""

    USABLE = "usable"
    REVIEW = "review"
    EXCLUDED = "excluded"


class ManifestValidationError(ValueError):
    """Raised when a manifest record violates its schema or mic policy."""


@dataclass(frozen=True, slots=True)
class GroupMetadata:
    """Condition and gender supplied by a TORGO top-level group."""

    condition: Condition
    gender: Gender
    speaker_pattern: str


@dataclass(frozen=True, slots=True)
class TranscriptResult:
    """Decoded prompt text and any non-destructive quality findings."""

    transcript: str | None
    flags: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True, slots=True)
class DiscoveredUtterance:
    """Prompt-driven logical utterance before its primary WAV is audited."""

    speaker_id: str
    speaker_group: str
    condition: Condition
    gender: Gender
    session_id: str
    prompt_id: str
    prompt_path: Path
    transcript: str | None
    transcript_quality_flags: tuple[str, ...]
    transcript_error: str | None
    head_mic_path: Path | None
    array_mic_path: Path | None

    @property
    def logical_id(self) -> str:
        """Return the stable canonical key in readable path-like form."""

        return f"{self.speaker_id}/{self.session_id}/{self.prompt_id}"


@dataclass(frozen=True, slots=True)
class OrphanAudio:
    """A direct microphone WAV that has no exact prompt-file match."""

    speaker_id: str
    speaker_group: str
    session_id: str
    prompt_id: str
    microphone: Microphone
    audio_path: Path
    reason: str


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Complete prompt-driven discovery result."""

    utterances: tuple[DiscoveredUtterance, ...]
    orphan_audio: tuple[OrphanAudio, ...]
    audio_paths: tuple[Path, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AudioAuditRecord:
    """Header, signal-quality, and file-hash audit for one direct WAV."""

    audio_path: str
    file_size: int | None
    readable: bool
    error: str | None
    zero_byte: bool
    sample_rate: int | None
    channels: int | None
    sample_width_bytes: int | None
    bit_depth: int | None
    frame_count: int | None
    duration_seconds: float | None
    compression_type: str | None
    sample_count: int | None
    rms_dbfs: float | None
    peak_amplitude_ratio: float | None
    full_scale_sample_count: int | None
    clipped_sample_count: int | None
    clipped_sample_fraction: float | None
    extremely_short: bool
    near_silent: bool
    possible_clipping: bool
    sha256: str | None
    quality_flags: tuple[str, ...]

    CSV_FIELDS: ClassVar[tuple[str, ...]] = (
        "audio_path",
        "file_size",
        "readable",
        "error",
        "zero_byte",
        "sample_rate",
        "channels",
        "sample_width_bytes",
        "bit_depth",
        "frame_count",
        "duration_seconds",
        "compression_type",
        "sample_count",
        "rms_dbfs",
        "peak_amplitude_ratio",
        "full_scale_sample_count",
        "clipped_sample_count",
        "clipped_sample_fraction",
        "extremely_short",
        "near_silent",
        "possible_clipping",
        "sha256",
        "quality_flags",
        "structurally_usable",
        "structural_issues",
    )

    @property
    def structural_issues(self) -> tuple[str, ...]:
        """Return normalized structural failures used for primary selection."""

        issues: set[str] = set()
        if self.file_size is None:
            issues.add("file_size_unavailable")
        elif self.file_size <= 0 or self.zero_byte:
            issues.add("zero_byte")

        if not self.readable:
            issues.add("unreadable")
            return tuple(sorted(issues))

        if self.sample_rate is None or self.sample_rate <= 0:
            issues.add("invalid_sample_rate")
        if self.channels is None or self.channels <= 0:
            issues.add("invalid_channels")
        if self.sample_width_bytes is None or self.sample_width_bytes <= 0:
            issues.add("invalid_sample_width")
        if self.bit_depth is None or self.bit_depth <= 0:
            issues.add("invalid_bit_depth")
        elif self.sample_width_bytes is not None and self.bit_depth != self.sample_width_bytes * 8:
            issues.add("inconsistent_bit_depth")
        if self.frame_count is None:
            issues.add("invalid_frame_count")
        elif self.frame_count <= 0:
            issues.add("empty_audio")
        if self.compression_type != "NONE":
            issues.add("unsupported_compression")

        if (
            self.frame_count is not None
            and self.frame_count > 0
            and self.sample_rate is not None
            and self.sample_rate > 0
        ):
            expected_duration = self.frame_count / self.sample_rate
            if (
                self.duration_seconds is None
                or not math.isfinite(self.duration_seconds)
                or self.duration_seconds <= 0
                or not math.isclose(
                    self.duration_seconds,
                    expected_duration,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            ):
                issues.add("invalid_duration")
        if (
            self.frame_count is not None
            and self.frame_count >= 0
            and self.channels is not None
            and self.channels > 0
            and self.sample_count != self.frame_count * self.channels
        ):
            issues.add("invalid_sample_count")
        return tuple(sorted(issues))

    @property
    def structurally_usable(self) -> bool:
        """Return whether this audited WAV is eligible for primary selection."""

        return not self.structural_issues

    def to_dict(self, *, csv_safe: bool = False) -> dict[str, Any]:
        """Serialize the audit record with stable field ordering."""

        result = asdict(self)
        result["quality_flags"] = (
            "|".join(self.quality_flags) if csv_safe else list(self.quality_flags)
        )
        result["structurally_usable"] = self.structurally_usable
        result["structural_issues"] = (
            "|".join(self.structural_issues) if csv_safe else list(self.structural_issues)
        )
        return result


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    """Canonical one-row-per-prompt TORGO record."""

    logical_id: str
    speaker_id: str
    speaker_group: str
    condition: Condition
    gender: Gender
    session_id: str
    prompt_id: str
    transcript: str | None
    prompt_path: str
    head_mic_path: str | None
    head_audio_status: AudioStructuralStatus
    head_audio_issues: tuple[str, ...]
    array_mic_path: str | None
    array_audio_status: AudioStructuralStatus
    array_audio_issues: tuple[str, ...]
    primary_audio_path: str | None
    alternate_audio_path: str | None
    primary_microphone: Microphone | None
    primary_selection_reason: PrimarySelectionReason
    sample_rate: int | None
    channels: int | None
    sample_width_bytes: int | None
    bit_depth: int | None
    frame_count: int | None
    duration_seconds: float | None
    file_size: int | None
    primary_audio_sha256: str | None
    prompt_type: PromptType
    normal_speech_eligible: bool
    usability_status: UsabilityStatus
    exclusion_reason: str | None
    quality_flags: tuple[str, ...]

    CSV_FIELDS: ClassVar[tuple[str, ...]] = (
        "logical_id",
        "speaker_id",
        "speaker_group",
        "condition",
        "gender",
        "session_id",
        "prompt_id",
        "transcript",
        "prompt_path",
        "head_mic_path",
        "head_audio_status",
        "head_audio_issues",
        "array_mic_path",
        "array_audio_status",
        "array_audio_issues",
        "primary_audio_path",
        "alternate_audio_path",
        "primary_microphone",
        "primary_selection_reason",
        "sample_rate",
        "channels",
        "sample_width_bytes",
        "bit_depth",
        "frame_count",
        "duration_seconds",
        "file_size",
        "primary_audio_sha256",
        "prompt_type",
        "normal_speech_eligible",
        "usability_status",
        "exclusion_reason",
        "quality_flags",
    )

    def validate(self) -> None:
        """Fail loudly on schema, path, or microphone-policy inconsistencies."""

        required = {
            "logical_id": self.logical_id,
            "speaker_id": self.speaker_id,
            "speaker_group": self.speaker_group,
            "session_id": self.session_id,
            "prompt_id": self.prompt_id,
            "prompt_path": self.prompt_path,
        }
        for field_name, value in required.items():
            if not isinstance(value, str) or not value:
                raise ManifestValidationError(f"{field_name} must be a non-empty string")

        if self.transcript is not None and (
            not isinstance(self.transcript, str) or not self.transcript.strip()
        ):
            raise ManifestValidationError("transcript must be null or a non-empty string")
        if self.exclusion_reason is not None and (
            not isinstance(self.exclusion_reason, str) or not self.exclusion_reason
        ):
            raise ManifestValidationError("exclusion_reason must be null or a non-empty string")
        if not isinstance(self.normal_speech_eligible, bool):
            raise ManifestValidationError("normal_speech_eligible must be a boolean")
        if not isinstance(self.quality_flags, tuple) or not all(
            isinstance(flag, str) and flag for flag in self.quality_flags
        ):
            raise ManifestValidationError("quality_flags must be a tuple of non-empty strings")
        if tuple(sorted(set(self.quality_flags))) != self.quality_flags:
            raise ManifestValidationError("quality_flags must be sorted and unique")
        if not isinstance(self.condition, Condition):
            raise ManifestValidationError("condition must be a Condition value")
        if not isinstance(self.gender, Gender):
            raise ManifestValidationError("gender must be a Gender value")
        if not isinstance(self.prompt_type, PromptType):
            raise ManifestValidationError("prompt_type must be a PromptType value")
        if not isinstance(self.usability_status, UsabilityStatus):
            raise ManifestValidationError("usability_status must be a UsabilityStatus value")
        if not isinstance(self.head_audio_status, AudioStructuralStatus):
            raise ManifestValidationError(
                "head_audio_status must be an AudioStructuralStatus value"
            )
        if not isinstance(self.array_audio_status, AudioStructuralStatus):
            raise ManifestValidationError(
                "array_audio_status must be an AudioStructuralStatus value"
            )
        if not isinstance(self.primary_selection_reason, PrimarySelectionReason):
            raise ManifestValidationError(
                "primary_selection_reason must be a PrimarySelectionReason value"
            )

        expected_id = f"{self.speaker_id}/{self.session_id}/{self.prompt_id}"
        if self.logical_id != expected_id:
            raise ManifestValidationError(
                f"logical_id {self.logical_id!r} does not match {expected_id!r}"
            )

        try:
            expected_condition, expected_gender, speaker_pattern = TORGO_GROUP_RULES[
                self.speaker_group
            ]
        except KeyError as exc:
            raise ManifestValidationError(
                f"unsupported TORGO speaker group: {self.speaker_group!r}"
            ) from exc
        if re.fullmatch(speaker_pattern, self.speaker_id) is None:
            raise ManifestValidationError(
                f"speaker {self.speaker_id!r} does not match group {self.speaker_group!r}"
            )
        if self.condition is not expected_condition or self.gender is not expected_gender:
            raise ManifestValidationError(
                f"canonical TORGO label mismatch for group {self.speaker_group}: "
                f"expected {expected_condition.value}/{expected_gender.value}"
            )

        for field_name in (
            "prompt_path",
            "head_mic_path",
            "array_mic_path",
            "primary_audio_path",
            "alternate_audio_path",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            if not isinstance(value, str) or not value:
                raise ManifestValidationError(f"{field_name} must be null or a non-empty string")
            path = PurePosixPath(value)
            if (
                path.is_absolute()
                or ".." in path.parts
                or re.match(r"^[A-Za-z]:[/\\]", value) is not None
                or value.startswith(("\\\\", "//"))
                or "\\" in value
            ):
                raise ManifestValidationError(
                    f"{field_name} must be a repository-relative safe path: {value!r}"
                )

        _validate_microphone_evidence(
            "head",
            self.head_mic_path,
            self.head_audio_status,
            self.head_audio_issues,
        )
        _validate_microphone_evidence(
            "array",
            self.array_mic_path,
            self.array_audio_status,
            self.array_audio_issues,
        )

        if self.head_audio_status is AudioStructuralStatus.USABLE:
            expected_primary = self.head_mic_path
            expected_alternate = self.array_mic_path
            expected_microphone: Microphone | None = Microphone.HEAD
            expected_selection_reason = PrimarySelectionReason.HEAD_STRUCTURALLY_USABLE
        elif self.array_audio_status is AudioStructuralStatus.USABLE:
            expected_primary = self.array_mic_path
            expected_alternate = self.head_mic_path
            expected_microphone = Microphone.ARRAY
            expected_selection_reason = (
                PrimarySelectionReason.HEAD_UNUSABLE_ARRAY_FALLBACK
                if self.head_audio_status is AudioStructuralStatus.UNUSABLE
                else PrimarySelectionReason.ARRAY_STRUCTURALLY_USABLE_HEAD_MISSING
            )
        else:
            expected_primary = None
            expected_alternate = None
            expected_microphone = None
            expected_selection_reason = PrimarySelectionReason.NO_STRUCTURALLY_USABLE_AUDIO

        if (
            self.primary_audio_path != expected_primary
            or self.alternate_audio_path != expected_alternate
            or self.primary_microphone != expected_microphone
            or self.primary_selection_reason is not expected_selection_reason
        ):
            raise ManifestValidationError(
                f"microphone policy violation for {self.logical_id}: valid head is mandatory, "
                "otherwise only an audit-proven valid array may be primary"
            )

        for microphone, status, issues in (
            ("head", self.head_audio_status, self.head_audio_issues),
            ("array", self.array_audio_status, self.array_audio_issues),
        ):
            if status is AudioStructuralStatus.UNUSABLE:
                required_flags = {f"{microphone}_unusable_{issue}" for issue in issues}
                missing_flags = required_flags - set(self.quality_flags)
                if missing_flags:
                    raise ManifestValidationError(
                        f"{microphone} structural evidence is missing quality flags: "
                        f"{sorted(missing_flags)}"
                    )
        fallback_flag = PrimarySelectionReason.HEAD_UNUSABLE_ARRAY_FALLBACK.value
        if (fallback_flag in self.quality_flags) != (
            self.primary_selection_reason is PrimarySelectionReason.HEAD_UNUSABLE_ARRAY_FALLBACK
        ):
            raise ManifestValidationError(
                "head-unusable array fallback requires the matching provenance flag"
            )

        if expected_primary is None and (
            self.usability_status is not UsabilityStatus.EXCLUDED
            or not self.exclusion_reason
            or (
                "missing_primary_audio"
                if self.head_audio_status is AudioStructuralStatus.MISSING
                and self.array_audio_status is AudioStructuralStatus.MISSING
                else "no_structurally_usable_audio"
            )
            not in self.exclusion_reason.split("|")
        ):
            raise ManifestValidationError(
                "records without structurally usable audio must be excluded with the "
                "matching structural reason"
            )

        if self.transcript is None and (
            self.usability_status is not UsabilityStatus.EXCLUDED
            or not self.exclusion_reason
            or "missing_or_unreadable_transcript" not in self.exclusion_reason.split("|")
        ):
            raise ManifestValidationError(
                "records without a transcript must be excluded as missing_or_unreadable_transcript"
            )

        if self.usability_status is UsabilityStatus.EXCLUDED:
            if not self.exclusion_reason:
                raise ManifestValidationError("excluded records require exclusion_reason")
            if self.normal_speech_eligible:
                raise ManifestValidationError("excluded records cannot be normal-speech eligible")
        elif self.exclusion_reason is not None:
            raise ManifestValidationError("non-excluded records cannot have exclusion_reason")

        expected_normal_speech_eligible = (
            self.usability_status is not UsabilityStatus.EXCLUDED
            and self.prompt_type in {PromptType.WORD, PromptType.SENTENCE}
        )
        if self.normal_speech_eligible is not expected_normal_speech_eligible:
            raise ManifestValidationError(
                "normal_speech_eligible must follow prompt_type and usability_status"
            )

        integer_fields = (
            "sample_rate",
            "channels",
            "sample_width_bytes",
            "bit_depth",
            "frame_count",
            "file_size",
        )
        for field_name in integer_fields:
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ManifestValidationError(
                    f"{field_name} must be null or a non-negative integer"
                )
        if self.duration_seconds is not None and (
            not isinstance(self.duration_seconds, (int, float))
            or isinstance(self.duration_seconds, bool)
            or not math.isfinite(self.duration_seconds)
            or self.duration_seconds < 0
        ):
            raise ManifestValidationError(
                "duration_seconds must be null or a finite non-negative number"
            )
        if self.primary_audio_sha256 is not None and (
            not isinstance(self.primary_audio_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.primary_audio_sha256) is None
        ):
            raise ManifestValidationError(
                "primary_audio_sha256 must be null or a lowercase SHA-256 digest"
            )

        if expected_primary is None:
            unexpected_primary_metadata = {
                name: getattr(self, name)
                for name in (
                    "sample_rate",
                    "channels",
                    "sample_width_bytes",
                    "bit_depth",
                    "frame_count",
                    "duration_seconds",
                    "file_size",
                    "primary_audio_sha256",
                )
                if getattr(self, name) is not None
            }
            if unexpected_primary_metadata:
                raise ManifestValidationError(
                    "records without primary audio cannot carry primary metadata"
                )

        if self.usability_status is not UsabilityStatus.EXCLUDED:
            required_positive = {
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "sample_width_bytes": self.sample_width_bytes,
                "bit_depth": self.bit_depth,
                "frame_count": self.frame_count,
                "duration_seconds": self.duration_seconds,
                "file_size": self.file_size,
            }
            missing_or_nonpositive = [
                name for name, value in required_positive.items() if value is None or value <= 0
            ]
            if missing_or_nonpositive:
                raise ManifestValidationError(
                    "non-excluded primary audio requires positive readable metadata: "
                    f"{missing_or_nonpositive}"
                )
            if self.primary_audio_sha256 is None:
                raise ManifestValidationError(
                    "non-excluded primary audio requires a SHA-256 digest"
                )
            assert self.sample_width_bytes is not None
            assert self.bit_depth is not None
            assert self.frame_count is not None
            assert self.sample_rate is not None
            assert self.duration_seconds is not None
            if self.bit_depth != self.sample_width_bytes * 8:
                raise ManifestValidationError(
                    "bit_depth must equal sample_width_bytes multiplied by 8"
                )
            expected_duration = self.frame_count / self.sample_rate
            if not math.isclose(
                self.duration_seconds,
                expected_duration,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ManifestValidationError(
                    "duration_seconds must match frame_count divided by sample_rate"
                )

    def to_dict(self, *, csv_safe: bool = False) -> dict[str, Any]:
        """Serialize enums and tuple flags for JSONL or CSV."""

        result = asdict(self)
        result["condition"] = self.condition.value
        result["gender"] = self.gender.value
        result["primary_microphone"] = (
            self.primary_microphone.value if self.primary_microphone else None
        )
        result["head_audio_status"] = self.head_audio_status.value
        result["head_audio_issues"] = (
            "|".join(self.head_audio_issues) if csv_safe else list(self.head_audio_issues)
        )
        result["array_audio_status"] = self.array_audio_status.value
        result["array_audio_issues"] = (
            "|".join(self.array_audio_issues) if csv_safe else list(self.array_audio_issues)
        )
        result["primary_selection_reason"] = self.primary_selection_reason.value
        result["prompt_type"] = self.prompt_type.value
        result["usability_status"] = self.usability_status.value
        result["quality_flags"] = (
            "|".join(self.quality_flags) if csv_safe else list(self.quality_flags)
        )
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ManifestRecord:
        """Construct and validate a record loaded from JSON."""

        expected_fields = set(cls.CSV_FIELDS)
        actual_fields = set(value)
        if actual_fields != expected_fields:
            missing = sorted(expected_fields - actual_fields)
            extra = sorted(actual_fields - expected_fields)
            raise ManifestValidationError(
                f"manifest fields do not match schema; missing={missing}, extra={extra}"
            )
        flags = value["quality_flags"]
        if not isinstance(flags, list) or not all(isinstance(item, str) for item in flags):
            raise ManifestValidationError("quality_flags must be a JSON array of strings")
        normal_speech_eligible = value["normal_speech_eligible"]
        if not isinstance(normal_speech_eligible, bool):
            raise ManifestValidationError("normal_speech_eligible must be a JSON boolean")
        head_audio_issues = _json_string_tuple(value, "head_audio_issues")
        array_audio_issues = _json_string_tuple(value, "array_audio_issues")
        record = cls(
            logical_id=_required_json_string(value, "logical_id"),
            speaker_id=_required_json_string(value, "speaker_id"),
            speaker_group=_required_json_string(value, "speaker_group"),
            condition=Condition(_required_json_string(value, "condition")),
            gender=Gender(_required_json_string(value, "gender")),
            session_id=_required_json_string(value, "session_id"),
            prompt_id=_required_json_string(value, "prompt_id"),
            transcript=_optional_json_string(value, "transcript"),
            prompt_path=_required_json_string(value, "prompt_path"),
            head_mic_path=_optional_json_string(value, "head_mic_path"),
            head_audio_status=AudioStructuralStatus(
                _required_json_string(value, "head_audio_status")
            ),
            head_audio_issues=head_audio_issues,
            array_mic_path=_optional_json_string(value, "array_mic_path"),
            array_audio_status=AudioStructuralStatus(
                _required_json_string(value, "array_audio_status")
            ),
            array_audio_issues=array_audio_issues,
            primary_audio_path=_optional_json_string(value, "primary_audio_path"),
            alternate_audio_path=_optional_json_string(value, "alternate_audio_path"),
            primary_microphone=(
                Microphone(primary_microphone)
                if (primary_microphone := _optional_json_string(value, "primary_microphone"))
                else None
            ),
            primary_selection_reason=PrimarySelectionReason(
                _required_json_string(value, "primary_selection_reason")
            ),
            sample_rate=_optional_json_integer(value, "sample_rate"),
            channels=_optional_json_integer(value, "channels"),
            sample_width_bytes=_optional_json_integer(value, "sample_width_bytes"),
            bit_depth=_optional_json_integer(value, "bit_depth"),
            frame_count=_optional_json_integer(value, "frame_count"),
            duration_seconds=_optional_json_number(value, "duration_seconds"),
            file_size=_optional_json_integer(value, "file_size"),
            primary_audio_sha256=_optional_json_string(value, "primary_audio_sha256"),
            prompt_type=PromptType(_required_json_string(value, "prompt_type")),
            normal_speech_eligible=normal_speech_eligible,
            usability_status=UsabilityStatus(_required_json_string(value, "usability_status")),
            exclusion_reason=_optional_json_string(value, "exclusion_reason"),
            quality_flags=tuple(flags),
        )
        record.validate()
        return record


def _required_json_string(value: dict[str, Any], key: str) -> str:
    item = value[key]
    if not isinstance(item, str) or not item:
        raise ManifestValidationError(f"{key} must be a non-empty JSON string")
    return item


def _optional_json_string(value: dict[str, Any], key: str) -> str | None:
    item = value[key]
    if item is None:
        return None
    if not isinstance(item, str) or not item:
        raise ManifestValidationError(f"{key} must be null or a non-empty JSON string")
    return item


def _optional_json_integer(value: dict[str, Any], key: str) -> int | None:
    item = value[key]
    if item is None:
        return None
    if not isinstance(item, int) or isinstance(item, bool):
        raise ManifestValidationError(f"{key} must be null or a JSON integer")
    return item


def _optional_json_number(value: dict[str, Any], key: str) -> float | None:
    item = value[key]
    if item is None:
        return None
    if not isinstance(item, (int, float)) or isinstance(item, bool):
        raise ManifestValidationError(f"{key} must be null or a JSON number")
    return float(item)


def _json_string_tuple(value: dict[str, Any], key: str) -> tuple[str, ...]:
    item = value[key]
    if not isinstance(item, list) or not all(
        isinstance(element, str) and element for element in item
    ):
        raise ManifestValidationError(f"{key} must be a JSON array of non-empty strings")
    return tuple(item)


def _validate_microphone_evidence(
    microphone: str,
    path: str | None,
    status: AudioStructuralStatus,
    issues: tuple[str, ...],
) -> None:
    if not isinstance(issues, tuple) or not all(
        isinstance(issue, str) and issue for issue in issues
    ):
        raise ManifestValidationError(
            f"{microphone}_audio_issues must be a tuple of non-empty strings"
        )
    if tuple(sorted(set(issues))) != issues:
        raise ManifestValidationError(f"{microphone}_audio_issues must be sorted and unique")
    if path is None:
        if status is not AudioStructuralStatus.MISSING or issues != ("missing_path",):
            raise ManifestValidationError(
                f"missing {microphone} path requires status=missing and issue=missing_path"
            )
        return
    if status is AudioStructuralStatus.MISSING:
        raise ManifestValidationError(
            f"existing {microphone} path cannot have structural status missing"
        )
    if status is AudioStructuralStatus.USABLE and issues:
        raise ManifestValidationError(
            f"structurally usable {microphone} audio cannot have structural issues"
        )
    if status is AudioStructuralStatus.UNUSABLE and not issues:
        raise ManifestValidationError(
            f"structurally unusable {microphone} audio requires rejection evidence"
        )
