"""Conservative WAV loading and waveform standardization for Module 1.

This module is deliberately read-only: it decodes the Module 0 primary WAV into
memory and never rewrites either the source file or a processed audio copy.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
NormalizationStrategy = Literal["none", "rms"]


class AudioLoadError(ValueError):
    """Raised when a WAV cannot be decoded as supported integer PCM."""


class AudioNormalizationError(ValueError):
    """Raised when waveform normalization cannot be performed safely."""


@dataclass(frozen=True)
class AudioLoadConfig:
    """Target layout for an in-memory Module 1 working waveform."""

    target_sample_rate: int = 16_000
    target_channels: int = 1

    def __post_init__(self) -> None:
        if self.target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive")
        if self.target_channels != 1:
            raise ValueError("Module 1 currently supports only mono target audio")


@dataclass(frozen=True)
class NormalizationConfig:
    """Parameters for conservative RMS normalization.

    ``max_gain`` prevents excessive amplification of quiet recordings and
    ``peak_limit`` guarantees headroom without hard clipping. Signals at or
    below ``silence_epsilon`` RMS are left unchanged.
    """

    strategy: NormalizationStrategy = "rms"
    target_rms: float = 0.1
    max_gain: float = 10.0
    peak_limit: float = 0.99
    silence_epsilon: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.strategy not in {"none", "rms"}:
            raise ValueError(f"unsupported normalization strategy: {self.strategy}")
        if not math.isfinite(self.target_rms) or self.target_rms <= 0.0:
            raise ValueError("target_rms must be finite and positive")
        if not math.isfinite(self.max_gain) or self.max_gain < 1.0:
            raise ValueError("max_gain must be finite and at least 1.0")
        if not math.isfinite(self.peak_limit) or not 0.0 < self.peak_limit <= 1.0:
            raise ValueError("peak_limit must be finite and in (0, 1]")
        if not math.isfinite(self.silence_epsilon) or self.silence_epsilon < 0.0:
            raise ValueError("silence_epsilon must be finite and non-negative")


@dataclass(frozen=True)
class LoadedAudio:
    """Decoded and layout-standardized waveform plus source-header evidence."""

    waveform: FloatArray
    sample_rate: int
    channels: int
    source_sample_rate: int
    source_channels: int
    sample_width_bytes: int
    frame_count: int
    file_size: int
    downmixed: bool
    resampled: bool
    resample_method: str | None

    @property
    def source_duration_seconds(self) -> float:
        """Duration declared by the source WAV header."""

        return self.frame_count / self.source_sample_rate


@dataclass(frozen=True)
class NormalizationStats:
    """Auditable signal-level measurements from normalization."""

    strategy: NormalizationStrategy
    rms_before: float
    rms_after: float
    peak_before: float
    peak_after: float
    applied_gain: float
    is_silent: bool
    gain_limited: bool
    peak_limited: bool


@dataclass(frozen=True)
class NormalizedAudio:
    """Normalized waveform and its measurements."""

    waveform: FloatArray
    stats: NormalizationStats


@dataclass(frozen=True)
class StandardizedAudio:
    """Final in-memory signal consumed by VMD and its provenance."""

    waveform: FloatArray
    sample_rate: int
    channels: int
    source_sample_rate: int
    source_channels: int
    sample_width_bytes: int
    frame_count: int
    file_size: int
    downmixed: bool
    resampled: bool
    resample_method: str | None
    normalization: NormalizationStats


DEFAULT_AUDIO_LOAD_CONFIG = AudioLoadConfig()
DEFAULT_NORMALIZATION_CONFIG = NormalizationConfig()


def _decode_pcm(raw_frames: bytes, sample_width_bytes: int) -> FloatArray:
    """Decode little-endian integer PCM to float64 in approximately [-1, 1)."""

    if sample_width_bytes == 1:
        values_u8 = np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float64)
        return (values_u8 - 128.0) / 128.0
    if sample_width_bytes == 2:
        values_i16 = np.frombuffer(raw_frames, dtype="<i2").astype(np.float64)
        return values_i16 / 32_768.0
    if sample_width_bytes == 3:
        byte_values = np.frombuffer(raw_frames, dtype=np.uint8).reshape(-1, 3)
        widened = byte_values.astype(np.int32)
        values_i24 = widened[:, 0] | (widened[:, 1] << 8) | (widened[:, 2] << 16)
        values_i24 = np.where(
            values_i24 & 0x80_0000,
            values_i24 - 0x100_0000,
            values_i24,
        )
        return values_i24.astype(np.float64) / 8_388_608.0
    if sample_width_bytes == 4:
        values_i32 = np.frombuffer(raw_frames, dtype="<i4").astype(np.float64)
        return values_i32 / 2_147_483_648.0
    raise AudioLoadError(
        f"unsupported PCM sample width {sample_width_bytes}; expected 1, 2, 3, or 4 bytes"
    )


def resample_waveform(
    waveform: FloatArray,
    source_sample_rate: int,
    target_sample_rate: int,
) -> FloatArray:
    """Deterministically resample a mono waveform using linear interpolation.

    TORGO already satisfies the 16 kHz target, so this explicit NumPy-only path
    is principally a safe fallback for nonstandard input rather than a routine
    transformation of the corpus.
    """

    samples = _validate_waveform(waveform)
    if source_sample_rate <= 0 or target_sample_rate <= 0:
        raise AudioLoadError("source and target sample rates must be positive")
    if source_sample_rate == target_sample_rate or samples.size == 0:
        return samples.copy()

    target_length = max(
        1,
        (int(samples.size) * target_sample_rate + source_sample_rate // 2)
        // source_sample_rate,
    )
    if samples.size == 1:
        return np.full(target_length, samples[0], dtype=np.float64)

    source_positions = np.arange(samples.size, dtype=np.float64)
    target_positions = (
        np.arange(target_length, dtype=np.float64)
        * float(source_sample_rate)
        / float(target_sample_rate)
    )
    return np.asarray(
        np.interp(target_positions, source_positions, samples),
        dtype=np.float64,
    )


def load_wav(
    path: Path,
    config: AudioLoadConfig = DEFAULT_AUDIO_LOAD_CONFIG,
) -> LoadedAudio:
    """Load integer-PCM WAV audio and standardize it to the requested layout."""

    source_path = Path(path)
    try:
        file_size = source_path.stat().st_size
        with wave.open(str(source_path), "rb") as wav_file:
            source_channels = wav_file.getnchannels()
            source_sample_rate = wav_file.getframerate()
            sample_width_bytes = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
            compression_type = wav_file.getcomptype()
            raw_frames = wav_file.readframes(frame_count)
    except (EOFError, OSError, ValueError, wave.Error) as exc:
        raise AudioLoadError(f"could not read WAV {source_path}: {exc}") from exc

    if compression_type != "NONE":
        raise AudioLoadError(
            f"unsupported compressed WAV {source_path}: compression={compression_type}"
        )
    if source_channels <= 0:
        raise AudioLoadError(f"invalid channel count in {source_path}: {source_channels}")
    if source_sample_rate <= 0:
        raise AudioLoadError(f"invalid sample rate in {source_path}: {source_sample_rate}")
    if frame_count < 0:
        raise AudioLoadError(f"invalid frame count in {source_path}: {frame_count}")
    if sample_width_bytes not in {1, 2, 3, 4}:
        raise AudioLoadError(
            f"unsupported PCM sample width in {source_path}: {sample_width_bytes} bytes"
        )

    expected_bytes = frame_count * source_channels * sample_width_bytes
    if len(raw_frames) != expected_bytes:
        raise AudioLoadError(
            f"truncated WAV data in {source_path}: expected {expected_bytes} bytes, "
            f"read {len(raw_frames)}"
        )

    decoded = _decode_pcm(raw_frames, sample_width_bytes)
    expected_samples = frame_count * source_channels
    if decoded.size != expected_samples:
        raise AudioLoadError(
            f"decoded sample-count mismatch in {source_path}: expected {expected_samples}, "
            f"decoded {decoded.size}"
        )

    frames = decoded.reshape(frame_count, source_channels)
    if source_channels == 1:
        mono = frames[:, 0].copy()
    else:
        mono = np.asarray(frames.mean(axis=1, dtype=np.float64), dtype=np.float64)

    resampled = source_sample_rate != config.target_sample_rate
    waveform = resample_waveform(mono, source_sample_rate, config.target_sample_rate)
    if not bool(np.isfinite(waveform).all()):
        raise AudioLoadError(f"decoded waveform contains non-finite samples: {source_path}")

    return LoadedAudio(
        waveform=waveform,
        sample_rate=config.target_sample_rate,
        channels=config.target_channels,
        source_sample_rate=source_sample_rate,
        source_channels=source_channels,
        sample_width_bytes=sample_width_bytes,
        frame_count=frame_count,
        file_size=file_size,
        downmixed=source_channels != config.target_channels,
        resampled=resampled,
        resample_method="linear_interpolation" if resampled else None,
    )


def normalize_waveform(
    waveform: FloatArray,
    config: NormalizationConfig = DEFAULT_NORMALIZATION_CONFIG,
) -> NormalizedAudio:
    """Apply finite, silence-safe, gain-capped RMS normalization."""

    samples = _validate_waveform(waveform)
    rms_before = _rms(samples)
    peak_before = _peak(samples)
    is_silent = samples.size == 0 or rms_before <= config.silence_epsilon

    desired_gain = 1.0
    gain_limited = False
    peak_limited = False
    if config.strategy == "rms" and not is_silent:
        desired_gain = config.target_rms / rms_before
        gain = min(desired_gain, config.max_gain)
        gain_limited = desired_gain > config.max_gain
        if peak_before > 0.0:
            peak_safe_gain = config.peak_limit / peak_before
            if peak_safe_gain < gain:
                gain = peak_safe_gain
                peak_limited = True
    else:
        gain = 1.0

    normalized = np.asarray(samples * gain, dtype=np.float64)
    if not bool(np.isfinite(normalized).all()):
        raise AudioNormalizationError("normalization produced non-finite samples")

    stats = NormalizationStats(
        strategy=config.strategy,
        rms_before=rms_before,
        rms_after=_rms(normalized),
        peak_before=peak_before,
        peak_after=_peak(normalized),
        applied_gain=float(gain),
        is_silent=is_silent,
        gain_limited=gain_limited,
        peak_limited=peak_limited,
    )
    return NormalizedAudio(waveform=normalized, stats=stats)


def standardize_audio(
    path: Path,
    audio_config: AudioLoadConfig = DEFAULT_AUDIO_LOAD_CONFIG,
    normalization_config: NormalizationConfig = DEFAULT_NORMALIZATION_CONFIG,
) -> StandardizedAudio:
    """Load, standardize, and normalize one primary audio path in memory."""

    loaded = load_wav(path, audio_config)
    normalized = normalize_waveform(loaded.waveform, normalization_config)
    return StandardizedAudio(
        waveform=normalized.waveform,
        sample_rate=loaded.sample_rate,
        channels=loaded.channels,
        source_sample_rate=loaded.source_sample_rate,
        source_channels=loaded.source_channels,
        sample_width_bytes=loaded.sample_width_bytes,
        frame_count=loaded.frame_count,
        file_size=loaded.file_size,
        downmixed=loaded.downmixed,
        resampled=loaded.resampled,
        resample_method=loaded.resample_method,
        normalization=normalized.stats,
    )


def _validate_waveform(waveform: FloatArray) -> FloatArray:
    samples = np.asarray(waveform, dtype=np.float64)
    if samples.ndim != 1:
        raise AudioNormalizationError(
            f"expected a one-dimensional mono waveform, got shape {samples.shape}"
        )
    if not bool(np.isfinite(samples).all()):
        raise AudioNormalizationError("waveform contains non-finite samples")
    return samples


def _rms(waveform: FloatArray) -> float:
    if waveform.size == 0:
        return 0.0
    scale = float(np.max(np.abs(waveform)))
    if scale == 0.0:
        return 0.0
    scaled = waveform / scale
    return scale * float(np.sqrt(np.mean(np.square(scaled), dtype=np.float64)))


def _peak(waveform: FloatArray) -> float:
    if waveform.size == 0:
        return 0.0
    return float(np.max(np.abs(waveform)))
