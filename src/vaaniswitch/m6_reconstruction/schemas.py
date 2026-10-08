"""Disjoint audio-job and transcript states for Module 6."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any

RAW = "RAW_ASR_TRANSCRIPT"
CORRECTED = "OPTIONAL_CORRECTED_TRANSCRIPT"
REFERENCE = "REFERENCE_TRANSCRIPT"
DEVELOPMENT_IDS = (
    "M03/Session2/0002",
    "M03/Session2/0013",
    "MC02/Session1/0007",
    "MC02/Session1/0006",
)
DEMO_IDS = ("M03/Session2/0013", "MC02/Session1/0006")
PANEL_SHA = "15703d20490c83ac814664c48c3d8c8bf1197d2054d7d9d5ce495c68804a536b"


def split_id(logical_id: str) -> tuple[str, str, str]:
    if not re.fullmatch(r"(?:M|F|MC|FC)\d{2}/Session\d+(?:_\d+)*/\d{4}", logical_id):
        raise ValueError("invalid speaker/session/prompt identity")
    speaker, session, prompt = logical_id.split("/")
    return speaker, session, prompt


@dataclass(frozen=True)
class AudioJob:
    logical_id: str
    primary_audio_path: str
    primary_audio_sha256: str
    frame_count: int

    def __post_init__(self) -> None:
        speaker, session, prompt = split_id(self.logical_id)
        path = PurePosixPath(self.primary_audio_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in self.primary_audio_path
            or path.parts[:4] != ("data", "raw", "torgo", "extracted")
            or len(path.parts) != 9
            or path.parts[5:7] != (speaker, session)
            or path.parts[7] not in ("wav_headMic", "wav_arrayMic")
            or path.name != f"{prompt}.wav"
        ):
            raise ValueError("audio path does not match composite identity")
        if not re.fullmatch("[0-9a-f]{64}", self.primary_audio_sha256):
            raise ValueError("invalid source SHA-256")
        if type(self.frame_count) is not int or self.frame_count <= 0:
            raise ValueError("invalid complete frame count")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AudioJob:
        if set(value) != {
            "logical_id",
            "primary_audio_path",
            "primary_audio_sha256",
            "frame_count",
        }:
            raise ValueError("recognition jobs may contain audio bindings only")
        return cls(**value)
