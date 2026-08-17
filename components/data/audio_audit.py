"""Efficient, non-mutating WAV header and signal-quality audit."""

from __future__ import annotations

import hashlib
import logging
import math
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from components.data.config import QualityConfig, TorgoConfig
from components.data.schemas import AudioAuditRecord

LOGGER = logging.getLogger(__name__)


def audit_audio_paths(
    paths: tuple[Path, ...] | list[Path],
    config: TorgoConfig,
) -> tuple[AudioAuditRecord, ...]:
    """Audit unique WAV paths concurrently and return path-sorted results."""

    unique_paths = sorted(set(paths), key=lambda path: path.as_posix())

    def run(path: Path) -> AudioAuditRecord:
        return audit_wav(path, config)

    with ThreadPoolExecutor(max_workers=config.quality.max_workers) as executor:
        records = list(executor.map(run, unique_paths))
    records.sort(key=lambda item: item.audio_path)
    LOGGER.info("Audited %d direct microphone WAV files", len(records))
    return tuple(records)


def audit_wav(path: Path, config: TorgoConfig) -> AudioAuditRecord:
    """Read one WAV without modifying, resampling, or preprocessing it."""

    relative_path = config.relative_path(path)
    try:
        file_size = path.stat().st_size
    except OSError as exc:
        return _failed_record(
            relative_path,
            None,
            error=f"file_stat_error:{type(exc).__name__}:{exc}",
        )

    try:
        sha256 = _sha256(path) if config.quality.compute_sha256 else None
    except OSError as exc:
        return _failed_record(
            relative_path,
            file_size,
            error=f"file_hash_error:{type(exc).__name__}:{exc}",
        )
    if file_size == 0:
        return _failed_record(
            relative_path,
            file_size,
            error="zero_byte_file",
            zero_byte=True,
            sha256=sha256,
        )

    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            compression_type = wav_file.getcomptype()
            if channels <= 0 or sample_width not in {1, 2, 3, 4} or sample_rate <= 0:
                raise ValueError(
                    "invalid WAV parameters: "
                    f"channels={channels}, width={sample_width}, rate={sample_rate}"
                )
            if compression_type != "NONE":
                raise ValueError(f"unsupported WAV compression: {compression_type}")
            signal = _scan_signal(
                wav_file,
                sample_width=sample_width,
                quality=config.quality,
            )
            expected_sample_count = frame_count * channels
            if signal.sample_count != expected_sample_count:
                raise ValueError(
                    "truncated WAV payload: "
                    f"expected {expected_sample_count} samples, read {signal.sample_count}"
                )
    except (EOFError, OSError, ValueError, wave.Error) as exc:
        return _failed_record(
            relative_path,
            file_size,
            error=f"wav_read_error:{type(exc).__name__}:{exc}",
            sha256=sha256,
        )

    duration = frame_count / sample_rate
    extremely_short = duration < config.quality.extremely_short_seconds
    near_silent = signal.rms_dbfs is not None and signal.rms_dbfs <= config.quality.near_silent_dbfs
    possible_clipping = (
        signal.clipped_sample_fraction is not None
        and signal.clipped_sample_fraction >= config.quality.clipping_fraction_threshold
    )
    flags: list[str] = []
    if frame_count == 0:
        flags.append("empty_audio")
    if extremely_short:
        flags.append("extremely_short")
    if near_silent:
        flags.append("near_silent")
    if possible_clipping:
        flags.append("possible_clipping")

    return AudioAuditRecord(
        audio_path=relative_path,
        file_size=file_size,
        readable=True,
        error=None,
        zero_byte=False,
        sample_rate=sample_rate,
        channels=channels,
        sample_width_bytes=sample_width,
        bit_depth=sample_width * 8,
        frame_count=frame_count,
        duration_seconds=duration,
        compression_type=compression_type,
        sample_count=signal.sample_count,
        rms_dbfs=signal.rms_dbfs,
        peak_amplitude_ratio=signal.peak_amplitude_ratio,
        full_scale_sample_count=signal.full_scale_sample_count,
        clipped_sample_count=signal.clipped_sample_count,
        clipped_sample_fraction=signal.clipped_sample_fraction,
        extremely_short=extremely_short,
        near_silent=near_silent,
        possible_clipping=possible_clipping,
        sha256=sha256,
        quality_flags=tuple(flags),
    )


class _SignalStats:
    __slots__ = (
        "clipped_sample_count",
        "clipped_sample_fraction",
        "full_scale_sample_count",
        "peak_amplitude_ratio",
        "rms_dbfs",
        "sample_count",
    )

    def __init__(
        self,
        *,
        sample_count: int,
        rms_dbfs: float | None,
        peak_amplitude_ratio: float | None,
        full_scale_sample_count: int,
        clipped_sample_count: int,
        clipped_sample_fraction: float | None,
    ) -> None:
        self.sample_count = sample_count
        self.rms_dbfs = rms_dbfs
        self.peak_amplitude_ratio = peak_amplitude_ratio
        self.full_scale_sample_count = full_scale_sample_count
        self.clipped_sample_count = clipped_sample_count
        self.clipped_sample_fraction = clipped_sample_fraction


def _scan_signal(
    wav_file: wave.Wave_read,
    *,
    sample_width: int,
    quality: QualityConfig,
) -> _SignalStats:
    full_scale = float(1 << (sample_width * 8 - 1))
    positive_full_scale = full_scale - 1.0
    clipping_level = positive_full_scale * quality.clipping_amplitude_ratio
    sum_squares = 0.0
    sample_count = 0
    peak = 0.0
    full_scale_count = 0
    clipped_count = 0

    while True:
        payload = wav_file.readframes(quality.chunk_frames)
        if not payload:
            break
        samples = _decode_pcm(payload, sample_width)
        if samples.size == 0:
            continue
        floating = samples.astype(np.float64, copy=False)
        absolute = np.abs(floating)
        sum_squares += float(np.dot(floating, floating))
        sample_count += int(samples.size)
        peak = max(peak, float(np.max(absolute)))
        full_scale_count += int(np.count_nonzero(absolute >= positive_full_scale))
        clipped_count += int(np.count_nonzero(absolute >= clipping_level))

    if sample_count == 0:
        return _SignalStats(
            sample_count=0,
            rms_dbfs=None,
            peak_amplitude_ratio=None,
            full_scale_sample_count=0,
            clipped_sample_count=0,
            clipped_sample_fraction=None,
        )

    rms_ratio = math.sqrt(sum_squares / sample_count) / full_scale
    rms_dbfs = 20.0 * math.log10(rms_ratio) if rms_ratio > 0 else float("-inf")
    return _SignalStats(
        sample_count=sample_count,
        rms_dbfs=rms_dbfs,
        peak_amplitude_ratio=peak / full_scale,
        full_scale_sample_count=full_scale_count,
        clipped_sample_count=clipped_count,
        clipped_sample_fraction=clipped_count / sample_count,
    )


def _decode_pcm(payload: bytes, sample_width: int) -> npt.NDArray[np.signedinteger[Any]]:
    if len(payload) % sample_width:
        raise ValueError("PCM payload is not aligned to sample width")
    if sample_width == 1:
        return np.frombuffer(payload, dtype=np.uint8).astype(np.int16) - 128
    if sample_width == 2:
        return np.frombuffer(payload, dtype="<i2")
    if sample_width == 4:
        return np.frombuffer(payload, dtype="<i4")

    octets = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3)
    values = (
        octets[:, 0].astype(np.int32)
        | (octets[:, 1].astype(np.int32) << 8)
        | (octets[:, 2].astype(np.int32) << 16)
    )
    return (values ^ 0x800000) - 0x800000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _failed_record(
    relative_path: str,
    file_size: int | None,
    *,
    error: str,
    zero_byte: bool = False,
    sha256: str | None = None,
) -> AudioAuditRecord:
    flags = ["unreadable"]
    if zero_byte:
        flags.insert(0, "zero_byte")
    return AudioAuditRecord(
        audio_path=relative_path,
        file_size=file_size,
        readable=False,
        error=error,
        zero_byte=zero_byte,
        sample_rate=None,
        channels=None,
        sample_width_bytes=None,
        bit_depth=None,
        frame_count=None,
        duration_seconds=None,
        compression_type=None,
        sample_count=None,
        rms_dbfs=None,
        peak_amplitude_ratio=None,
        full_scale_sample_count=None,
        clipped_sample_count=None,
        clipped_sample_fraction=None,
        extremely_short=False,
        near_silent=False,
        possible_clipping=False,
        sha256=sha256,
        quality_flags=tuple(flags),
    )
