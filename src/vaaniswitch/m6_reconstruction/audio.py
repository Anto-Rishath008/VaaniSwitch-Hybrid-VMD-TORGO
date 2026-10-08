"""Complete PCM audio loading, without FFmpeg, resampling or truncation."""

from __future__ import annotations

import io
import struct
import wave
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def decode_pcm(payload: bytes, *, source: bool) -> tuple[NDArray[np.float32], dict[str, Any]]:
    try:
        if len(payload) < 44 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
            raise ValueError("not RIFF/WAVE")
        declared = struct.unpack_from("<I", payload, 4)[0] + 8
        if declared > len(payload):
            raise ValueError("truncated RIFF: declared length exceeds physical bytes")
        if not source and declared != len(payload):
            raise ValueError("inconsistent generated RIFF length")
        if declared < 44:
            raise ValueError("malformed RIFF declared region")
        region = payload[:declared]
        position = 12
        data_sizes: list[int] = []
        while position < declared:
            if position + 8 > declared:
                raise ValueError("truncated RIFF chunk header")
            chunk_size = struct.unpack_from("<I", region, position + 4)[0]
            chunk_end = position + 8 + chunk_size
            if chunk_end + (chunk_size % 2) > declared:
                raise ValueError("truncated RIFF chunk payload")
            if region[position : position + 4] == b"data":
                data_sizes.append(chunk_size)
            position = chunk_end + (chunk_size % 2)
        if len(data_sizes) != 1:
            raise ValueError("expected one complete PCM data chunk")
        with wave.open(io.BytesIO(region), "rb") as wav:
            channels, width, rate, frames, compression, _ = wav.getparams()
            if compression != "NONE" or channels not in (1, 2) or width not in (1, 2, 3, 4):
                raise ValueError("unsupported PCM format")
            if not 8000 <= rate <= 192000 or frames <= 0:
                raise ValueError("invalid rate or empty WAV")
            if source and (rate, channels, width) != (16000, 1, 2):
                raise ValueError("ASR requires complete PCM16 mono 16000 Hz audio")
            data = wav.readframes(frames)
        if len(data) != frames * channels * width or data_sizes[0] != len(data):
            raise ValueError("truncated PCM frames")
        if width == 1:
            values = np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128
        elif width == 3:
            octets = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            integers = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
            values = ((integers ^ 0x800000) - 0x800000).astype(np.float32)
        else:
            values = np.frombuffer(data, dtype=f"<i{width}").astype(np.float32)
        values /= float(2 ** (8 * width - 1))
        if not np.isfinite(values).all():
            raise ValueError("nonfinite samples")
        nonzero = int(np.count_nonzero(values))
        if not source and nonzero == 0:
            raise ValueError("silent generated WAV")
        metadata = {
            "container": "RIFF/WAVE",
            "encoding": "PCM",
            "sample_rate": rate,
            "channels": channels,
            "bit_depth": width * 8,
            "frame_count": frames,
            "duration_seconds": frames / rate,
            "bytes": len(payload),
            "nonzero_samples": nonzero,
            "finite_samples": True,
        }
        if source:
            metadata.update(
                declared_riff_bytes=declared, source_trailing_bytes=len(payload) - declared
            )
        return values, metadata
    except (wave.Error, EOFError, struct.error) as exc:
        raise ValueError("malformed or unsupported WAV") from exc


def validate_generated_wav(path: Path) -> dict[str, Any]:
    return decode_pcm(path.read_bytes(), source=False)[1]
