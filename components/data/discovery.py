"""Prompt-driven discovery of canonical TORGO utterances."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path

from components.data.config import TorgoConfig
from components.data.metadata import (
    MetadataError,
    parse_prompt_id,
    parse_session_id,
    parse_speaker,
    read_transcript,
)
from components.data.schemas import (
    DiscoveredUtterance,
    DiscoveryResult,
    Microphone,
    OrphanAudio,
)

LOGGER = logging.getLogger(__name__)


class DiscoveryError(RuntimeError):
    """Raised when raw TORGO structure cannot be mapped unambiguously."""


def discover_torgo(config: TorgoConfig) -> DiscoveryResult:
    """Discover logical utterances from direct session prompt files.

    Prompt files are the source of truth. Direct microphone WAVs without an
    exact prompt stem are returned as orphan audit findings and are never used
    to invent transcripts or logical records.
    """

    if not config.raw_root.is_dir():
        raise DiscoveryError(f"TORGO raw root does not exist: {config.raw_root}")

    utterances: list[DiscoveredUtterance] = []
    orphans: list[OrphanAudio] = []
    all_audio_paths: set[Path] = set()
    warnings: list[str] = []

    for speaker_group in config.groups:
        group_path = config.raw_root / speaker_group
        if not group_path.is_dir():
            warnings.append(f"missing_group_directory:{speaker_group}")
            continue
        for speaker_path in _sorted_directories(group_path):
            try:
                group_metadata = parse_speaker(
                    speaker_group,
                    speaker_path.name,
                    config,
                )
            except MetadataError as exc:
                raise DiscoveryError(str(exc)) from exc

            for malformed_session in _malformed_session_directories(speaker_path, config):
                warnings.append(
                    "ignored_malformed_session_directory:"
                    f"{speaker_group}/{speaker_path.name}/{malformed_session.name}"
                )
            for session_path in _session_directories(speaker_path, config):
                session_id = parse_session_id(session_path.name, config)
                prompt_directory = session_path / config.prompt_directory
                if not prompt_directory.is_dir():
                    warnings.append(
                        f"missing_prompt_directory:{speaker_group}/{speaker_path.name}/{session_id}"
                    )
                prompt_paths = _files_with_suffix(
                    prompt_directory,
                    config.prompt_extension,
                )
                prompt_by_id: dict[str, Path] = {}
                for prompt_path in prompt_paths:
                    try:
                        prompt_id = parse_prompt_id(prompt_path, config)
                    except MetadataError as exc:
                        raise DiscoveryError(str(exc)) from exc
                    if prompt_id in prompt_by_id:
                        raise DiscoveryError(
                            "duplicate prompt ID in session "
                            f"{speaker_path.name}/{session_id}/{prompt_id}"
                        )
                    prompt_by_id[prompt_id] = prompt_path

                mic_files: dict[Microphone, dict[str, Path]] = {}
                for microphone in (Microphone.HEAD, Microphone.ARRAY):
                    directory_name = config.microphone_directories[microphone.value]
                    files = _files_with_suffix(session_path / directory_name, ".wav")
                    all_audio_paths.update(files)
                    by_stem: dict[str, Path] = {}
                    for audio_path in files:
                        if audio_path.stem in by_stem:
                            raise DiscoveryError(
                                "duplicate microphone stem in session: "
                                f"{speaker_path.name}/{session_id}/{microphone.value}/"
                                f"{audio_path.stem}"
                            )
                        by_stem[audio_path.stem] = audio_path
                    mic_files[microphone] = by_stem

                prompt_ids = set(prompt_by_id)
                for microphone, by_stem in mic_files.items():
                    for orphan_id in sorted(set(by_stem) - prompt_ids, key=_natural_key):
                        reason = (
                            "missing_prompt_metadata"
                            if re.fullmatch(config.prompt_id_pattern, orphan_id)
                            else "malformed_audio_filename_or_prompt_id"
                        )
                        orphans.append(
                            OrphanAudio(
                                speaker_id=speaker_path.name,
                                speaker_group=speaker_group,
                                session_id=session_id,
                                prompt_id=orphan_id,
                                microphone=microphone,
                                audio_path=by_stem[orphan_id],
                                reason=reason,
                            )
                        )

                for prompt_id in sorted(prompt_by_id, key=_natural_key):
                    prompt_path = prompt_by_id[prompt_id]
                    transcript = read_transcript(prompt_path, config.transcript_encodings)
                    utterances.append(
                        DiscoveredUtterance(
                            speaker_id=speaker_path.name,
                            speaker_group=speaker_group,
                            condition=group_metadata.condition,
                            gender=group_metadata.gender,
                            session_id=session_id,
                            prompt_id=prompt_id,
                            prompt_path=prompt_path,
                            transcript=transcript.transcript,
                            transcript_quality_flags=transcript.flags,
                            transcript_error=transcript.error,
                            head_mic_path=mic_files[Microphone.HEAD].get(prompt_id),
                            array_mic_path=mic_files[Microphone.ARRAY].get(prompt_id),
                        )
                    )

    ordered_utterances = tuple(sorted(utterances, key=_utterance_sort_key))
    _reject_duplicate_logical_ids(ordered_utterances)
    ordered_orphans = tuple(
        sorted(
            orphans,
            key=lambda item: (
                item.speaker_id,
                _natural_key(item.session_id),
                _natural_key(item.prompt_id),
                item.microphone.value,
                item.audio_path.name,
            ),
        )
    )
    ordered_audio = tuple(sorted(all_audio_paths, key=lambda path: path.as_posix()))
    LOGGER.info(
        "Discovered %d prompt records, %d direct microphone WAVs, and %d orphan WAVs",
        len(ordered_utterances),
        len(ordered_audio),
        len(ordered_orphans),
    )
    return DiscoveryResult(
        utterances=ordered_utterances,
        orphan_audio=ordered_orphans,
        audio_paths=ordered_audio,
        warnings=tuple(sorted(warnings)),
    )


def _sorted_directories(path: Path) -> list[Path]:
    return sorted(
        (item for item in path.iterdir() if item.is_dir()),
        key=lambda item: _natural_key(item.name),
    )


def _session_directories(speaker_path: Path, config: TorgoConfig) -> list[Path]:
    return [
        item
        for item in _sorted_directories(speaker_path)
        if re.fullmatch(config.session_pattern, item.name) is not None
    ]


def _malformed_session_directories(
    speaker_path: Path,
    config: TorgoConfig,
) -> list[Path]:
    return [
        item
        for item in _sorted_directories(speaker_path)
        if item.name.casefold().startswith("session")
        and re.fullmatch(config.session_pattern, item.name) is None
    ]


def _files_with_suffix(path: Path, suffix: str) -> list[Path]:
    if not path.is_dir():
        return []
    expected = suffix.casefold()
    return sorted(
        (
            item
            for item in path.iterdir()
            if item.is_file() and item.name.casefold().endswith(expected)
        ),
        key=lambda item: (_natural_key(item.stem), item.name),
    )


def _natural_key(value: str) -> tuple[tuple[int, int | str], ...]:
    parts: list[tuple[int, int | str]] = []
    for part in re.split(r"(\d+)", value):
        if not part:
            continue
        if part.isdigit():
            parts.append((0, int(part)))
        else:
            parts.append((1, part.casefold()))
    return tuple(parts)


def _utterance_sort_key(
    utterance: DiscoveredUtterance,
) -> tuple[str, tuple[tuple[int, int | str], ...], tuple[tuple[int, int | str], ...]]:
    return (
        utterance.speaker_id,
        _natural_key(utterance.session_id),
        _natural_key(utterance.prompt_id),
    )


def _reject_duplicate_logical_ids(utterances: Iterable[DiscoveredUtterance]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for utterance in utterances:
        if utterance.logical_id in seen:
            duplicates.add(utterance.logical_id)
        seen.add(utterance.logical_id)
    if duplicates:
        raise DiscoveryError(f"duplicate logical IDs: {sorted(duplicates)}")
