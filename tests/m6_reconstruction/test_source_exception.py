"""Narrow frozen-source exception checks; generic/generated validation stays strict."""

import hashlib
import io
import json
import struct
import wave
from dataclasses import replace
from pathlib import Path

import pytest

from vaaniswitch.m6_reconstruction import source_audio
from vaaniswitch.m6_reconstruction.artifacts import write_json
from vaaniswitch.m6_reconstruction.audio import decode_pcm
from vaaniswitch.m6_reconstruction.schemas import AudioJob


def pcm(*, channels: int = 1, delta: int = 8, missing: int = 0) -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setparams((channels, 2, 16000, 12, "NONE", "not compressed"))
        wav.writeframes(b"\x01\x02" * 12 * channels)
    payload = bytearray(stream.getvalue())
    if missing:
        del payload[-missing:]
    struct.pack_into("<I", payload, 4, len(payload) - 8 + delta)
    return bytes(payload)


def bound_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    *,
    frames: int = 12,
    key: str = "M01/Session1/0007",
) -> tuple[source_audio.SourceReader, AudioJob, Path]:
    relative = f"data/raw/torgo/extracted/M/{key.rsplit('/', 1)[0]}/wav_headMic/{key[-4:]}.wav"
    job = AudioJob(key, relative, hashlib.sha256(payload).hexdigest(), frames)
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    # Synthetic trusted metadata fixture; production loader verifies immutable Phase-0/manifest SHA.
    monkeypatch.setattr(source_audio, "frozen_jobs", lambda root: {key: job})
    return source_audio.SourceReader(tmp_path), job, path


def test_exact_exception_only_explicit_path_and_provenance_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = pcm()
    with pytest.raises(ValueError):
        decode_pcm(payload, source=True)
    with pytest.raises(ValueError):
        decode_pcm(payload, source=False)
    reader, job, path = bound_reader(tmp_path, monkeypatch, payload)
    audio, metadata = reader.load(job)
    assert len(audio) == 12
    assert metadata["exception_id"] == source_audio.EXCEPTION_ID
    assert metadata["physical_bytes"] == metadata["riff_chunk_size"] == len(payload)
    assert metadata["data_offset"] == 44 and metadata["data_bytes"] == 24
    assert metadata["overdeclared_bytes"] == 8
    assert metadata["source_bytes_modified"] is False
    output = write_json(tmp_path, "artifacts/module6/test/source_validation.json", metadata)
    assert json.loads(output.read_bytes()) == metadata
    assert path.read_bytes() == payload


@pytest.mark.parametrize("delta", [6, 10])
def test_other_overdeclarations_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delta: int
) -> None:
    reader, job, _ = bound_reader(tmp_path, monkeypatch, pcm(delta=delta))
    with pytest.raises(ValueError, match="exact"):
        reader.load(job)


@pytest.mark.parametrize("kind", ["missing", "format", "frames", "sha", "job", "unapproved"])
def test_exception_rejects_failed_requirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    payload = pcm(missing=2 if kind == "missing" else 0, channels=2 if kind == "format" else 1)
    reader, job, path = bound_reader(
        tmp_path,
        monkeypatch,
        payload,
        frames=13 if kind == "frames" else 12,
        key="M01/Session1/0099" if kind == "unapproved" else "M01/Session1/0007",
    )
    if kind == "sha":
        path.write_bytes(payload[:-1] + b"\xff")
    if kind == "job":
        job = replace(job, frame_count=13)
    with pytest.raises(ValueError):
        reader.load(job)


def test_all_five_real_sources_match_frozen_metadata(project_root: Path) -> None:
    reader = source_audio.SourceReader(project_root)
    for key in source_audio.EXCEPTION_IDS:
        job = reader.jobs[key]
        before = (project_root / job.primary_audio_path).read_bytes()
        audio, metadata = reader.load(job)
        assert len(audio) == job.frame_count
        assert metadata["exception_id"] == source_audio.EXCEPTION_ID
        assert metadata["complete_file_sha256"] == job.primary_audio_sha256
        assert (project_root / job.primary_audio_path).read_bytes() == before


def test_unfrozen_tree_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        source_audio.SourceReader(tmp_path)
