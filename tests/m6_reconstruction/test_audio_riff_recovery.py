"""Regression coverage for RIFF boundaries without weakening truncation rejection."""

import io
import struct
import wave

import pytest

from vaaniswitch.m6_reconstruction.audio import decode_pcm


def pcm_source(frames: int = 32123) -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setparams((1, 2, 16000, frames, "NONE", "not compressed"))
        wav.writeframes(b"\x01\x02" * frames)
    return stream.getvalue()


def test_strict_complete_source() -> None:
    audio, meta = decode_pcm(pcm_source(), source=True)
    assert len(audio) == meta["frame_count"] == 32123
    assert (meta["sample_rate"], meta["channels"], meta["bit_depth"]) == (16000, 1, 16)


def test_source_trailing_bytes_are_recorded_without_modification() -> None:
    container = pcm_source()
    original = container + b"harmless trailing bytes"
    audio, meta = decode_pcm(original, source=True)
    assert len(audio) == 32123
    assert meta["declared_riff_bytes"] == len(container)
    assert meta["source_trailing_bytes"] == 23
    assert meta["bytes"] == len(original)
    assert original == container + b"harmless trailing bytes"


def test_overdeclared_riff_is_rejected_even_with_complete_pcm() -> None:
    payload = bytearray(pcm_source())
    # Exact observed TORGO pattern: ChunkSize equals file length, instead of length - 8.
    struct.pack_into("<I", payload, 4, len(payload))
    with pytest.raises(ValueError, match="RIFF"):
        decode_pcm(bytes(payload), source=True)


def test_truncated_data_rejected_with_adjusted_outer_header() -> None:
    payload = bytearray(pcm_source()[:-2])
    struct.pack_into("<I", payload, 4, len(payload) - 8)
    with pytest.raises(ValueError):
        decode_pcm(bytes(payload), source=True)


def test_malformed_declared_region_not_hidden_by_trailing_bytes() -> None:
    payload = bytearray(pcm_source())
    struct.pack_into("<I", payload, 4, 20)
    with pytest.raises(ValueError):
        decode_pcm(bytes(payload), source=True)


def test_generated_file_stays_strict() -> None:
    payload = pcm_source()
    assert decode_pcm(payload, source=False)[1]["nonzero_samples"] > 0
    with pytest.raises(ValueError, match="RIFF"):
        decode_pcm(payload + b"trailing", source=False)
    with pytest.raises(ValueError):
        decode_pcm(b"RIFF malformed", source=True)
