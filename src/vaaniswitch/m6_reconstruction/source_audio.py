"""Frozen TORGO source validation, including one explicitly authorized container exception."""

from __future__ import annotations

import hashlib
import io
import json
import struct
import wave
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .artifacts import verify_checksum
from .audio import decode_pcm
from .schemas import AudioJob

PHASE0_SHA = "0dd875290726e83a92eefc9f5c234785631034b411226239effad1beee78c8b8"
EXCEPTION_ID = "TORGO_RIFF_CHUNKSIZE_PLUS8_EXCEPTION"
EXCEPTION_IDS = frozenset(f"M01/Session1/{p}" for p in ("0007", "0008", "0014", "0017", "0020"))


def frozen_jobs(root: Path) -> dict[str, AudioJob]:
    path = root / "docs/modules/6_english_asr_tts/PHASE0_EVIDENCE.json"
    verify_checksum(path, PHASE0_SHA)
    phase0 = json.loads(path.read_bytes())
    manifest = root / "data/manifests/torgo_manifest.jsonl"
    verify_checksum(
        manifest,
        phase0["upstream_verification"]["protected_sha256"]["data/manifests/torgo_manifest.jsonl"],
    )
    hashes = phase0["panel_audio_verification"]["primary_audio_hashes"]
    jobs = {}
    for line in manifest.read_bytes().splitlines():
        row = json.loads(line)
        key = row["logical_id"]
        if key in hashes:
            job = AudioJob(
                **{
                    k: row[k]
                    for k in (
                        "logical_id",
                        "primary_audio_path",
                        "primary_audio_sha256",
                        "frame_count",
                    )
                }
            )
            if job.primary_audio_sha256 != hashes[key] or key in jobs:
                raise ValueError("frozen source binding mismatch")
            jobs[key] = job
    if set(jobs) != set(hashes):
        raise ValueError("incomplete frozen source inventory")
    return jobs


def _exception_pcm(payload: bytes, job: AudioJob) -> tuple[NDArray[np.float32], dict[str, Any]]:
    """Inspect ORIGINAL bytes. No header rewriting, even in a temporary buffer."""
    if job.logical_id not in EXCEPTION_IDS:
        raise ValueError("source not on the approved TORGO exception list")
    if len(payload) < 44 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise ValueError("invalid RIFF/WAVE signature")
    size = struct.unpack_from("<I", payload, 4)[0]
    if size != len(payload):
        raise ValueError("only the exact +8 RIFF declaration is authorized")
    position = 12
    chunks: dict[bytes, tuple[int, bytes, int]] = {}
    while position < len(payload):
        if position + 8 > len(payload):
            raise ValueError("truncated chunk header")
        tag, length = struct.unpack_from("<4sI", payload, position)
        start = position + 8
        end = start + length
        padded_end = end + length % 2
        if padded_end > len(payload) or tag in chunks:
            raise ValueError("truncated or duplicate source chunk")
        chunks[tag] = (start, payload[start:end], padded_end)
        position = padded_end
    if b"fmt " not in chunks or b"data" not in chunks:
        raise ValueError("missing fmt or data chunk")
    fmt = chunks[b"fmt "][1]
    if len(fmt) != 16:
        raise ValueError("exception requires the verified 16-byte PCM fmt chunk")
    encoding, channels, rate, byte_rate, align, bits = struct.unpack("<HHIIHH", fmt)
    if (encoding, channels, rate, byte_rate, align, bits) != (1, 1, 16000, 32000, 2, 16):
        raise ValueError("exception requires valid PCM16 mono 16000 Hz fmt")
    offset, data, data_end = chunks[b"data"]
    if data_end != len(payload) or not data or len(data) % align:
        raise ValueError("data must contain complete frames ending at physical EOF")
    frames = len(data) // align
    if frames != job.frame_count:
        raise ValueError("frozen manifest frame count mismatch")
    with wave.open(io.BytesIO(payload), "rb") as wav:
        if wav.getparams()[:5] != (1, 2, 16000, frames, "NONE"):
            raise ValueError("independent wave metadata mismatch")
        if wav.readframes(frames + 1) != data:
            raise ValueError("independent wave PCM completeness mismatch")
    audio = np.frombuffer(data, dtype="<i2").astype(np.float32) / np.float32(32768)
    if len(audio) != job.frame_count or not np.isfinite(audio).all():
        raise ValueError("invalid complete decoded samples")
    return audio, {
        "format_status": EXCEPTION_ID,
        "exception_id": EXCEPTION_ID,
        "physical_bytes": len(payload),
        "riff_chunk_size": size,
        "declared_riff_total": size + 8,
        "overdeclared_bytes": 8,
        "fmt": {
            "encoding_tag": encoding,
            "channels": channels,
            "sample_rate": rate,
            "byte_rate": byte_rate,
            "block_align": align,
            "bit_depth": bits,
        },
        "data_offset": offset,
        "data_bytes": len(data),
        "data_padded_end": data_end,
        "decoded_frame_count": frames,
        "manifest_frame_count": job.frame_count,
        "python_wave_complete_pcm_verified": True,
        "source_bytes_modified": False,
        "explanation": "Complete PCM payload; malformed outer RIFF ChunkSize. Accepted under "
        "a narrow source-container exception after verifying complete PCM payload and frozen "
        "source identity.",
        "standards_conformant_container": False,
    }


class SourceReader:
    """Bind audio-only jobs to the immutable Phase-0 manifest and complete source hashes."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.jobs = frozen_jobs(self.root)

    def load(self, job: AudioJob) -> tuple[NDArray[np.float32], dict[str, Any]]:
        if self.jobs.get(job.logical_id) != job:
            raise ValueError("job differs from frozen SHA/frame/path binding")
        path = self.root / job.primary_audio_path
        raw_root = self.root / "data/raw/torgo/extracted"
        if not path.resolve().is_relative_to(raw_root.resolve()):
            raise ValueError("source outside frozen TORGO raw tree")
        if path.resolve() != path.absolute():
            raise ValueError("redirected TORGO source path")
        payload = path.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != job.primary_audio_sha256:
            raise ValueError("source SHA mismatch")
        if len(payload) >= 8 and struct.unpack_from("<I", payload, 4)[0] + 8 > len(payload):
            audio, metadata = _exception_pcm(payload, job)
        else:
            audio, decoded = decode_pcm(payload, source=True)
            if len(audio) != job.frame_count:
                raise ValueError("frozen manifest frame count mismatch")
            metadata = {
                "format_status": "STRICT_PASS",
                "exception_id": None,
                "source_bytes_modified": False,
                "decoded": decoded,
                "decoded_frame_count": len(audio),
                "manifest_frame_count": job.frame_count,
            }
            if decoded["source_trailing_bytes"]:
                raise ValueError("frozen panel requires exact source lengths outside approved +8")
        metadata.update(
            logical_id=job.logical_id,
            path=job.primary_audio_path,
            complete_file_sha256=actual,
            decoded_float32_sha256=hashlib.sha256(audio.tobytes()).hexdigest(),
        )
        return audio, metadata
