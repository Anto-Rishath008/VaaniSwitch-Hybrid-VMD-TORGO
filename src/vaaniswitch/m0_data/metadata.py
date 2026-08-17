"""TORGO path metadata and prompt-text parsing."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from vaaniswitch.m0_data.config import TorgoConfig
from vaaniswitch.m0_data.schemas import GroupMetadata, TranscriptResult


class MetadataError(ValueError):
    """Raised for malformed TORGO structural metadata."""


def parse_speaker(
    speaker_group: str,
    speaker_id: str,
    config: TorgoConfig,
) -> GroupMetadata:
    """Validate a speaker against its exact TORGO group."""

    try:
        metadata = config.groups[speaker_group]
    except KeyError as exc:
        raise MetadataError(f"unsupported TORGO speaker group: {speaker_group!r}") from exc
    if re.fullmatch(metadata.speaker_pattern, speaker_id) is None:
        raise MetadataError(f"speaker {speaker_id!r} does not match group {speaker_group!r}")
    return metadata


def parse_session_id(name: str, config: TorgoConfig) -> str:
    """Validate and preserve a TORGO session directory name verbatim."""

    if re.fullmatch(config.session_pattern, name) is None:
        raise MetadataError(f"invalid TORGO session ID: {name!r}")
    return name


def parse_prompt_id(path: Path, config: TorgoConfig) -> str:
    """Read a numeric prompt ID from the prompt file stem."""

    if path.suffix.casefold() != config.prompt_extension.casefold():
        raise MetadataError(f"unexpected prompt extension: {path.name!r}")
    prompt_id = path.stem
    if re.fullmatch(config.prompt_id_pattern, prompt_id) is None:
        raise MetadataError(f"invalid TORGO prompt ID: {path.name!r}")
    return prompt_id


def read_transcript(path: Path, encodings: tuple[str, ...]) -> TranscriptResult:
    """Decode the real prompt file without synthesizing text from its filename."""

    try:
        payload = path.read_bytes()
    except OSError as exc:
        return TranscriptResult(
            transcript=None,
            error=f"prompt_read_error:{type(exc).__name__}:{exc}",
        )
    if not payload:
        return TranscriptResult(transcript=None, error="empty_prompt_file")

    transcript: str | None = None
    decode_errors: list[str] = []
    for encoding in encodings:
        try:
            transcript = payload.decode(encoding)
            break
        except (LookupError, UnicodeDecodeError) as exc:
            decode_errors.append(f"{encoding}:{type(exc).__name__}")
    if transcript is None:
        return TranscriptResult(
            transcript=None,
            error="prompt_decode_error:" + ",".join(decode_errors),
        )

    transcript = transcript.strip()
    if not transcript:
        return TranscriptResult(transcript=None, error="blank_prompt_text")

    flags: list[str] = []
    if transcript.casefold() == "xxx":
        flags.append("placeholder_transcript")
    if any(_is_disallowed_control(character) for character in transcript):
        flags.append("malformed_transcript_control_characters")
    return TranscriptResult(transcript=transcript, flags=tuple(flags))


def _is_disallowed_control(character: str) -> bool:
    """Return whether a decoded prompt contains an unsafe control character."""

    return character not in {"\n", "\t"} and unicodedata.category(character) == "Cc"
