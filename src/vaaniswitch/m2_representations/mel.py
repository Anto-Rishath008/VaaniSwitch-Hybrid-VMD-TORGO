"""Project-owned deterministic log-Mel generation with a fixed physical time axis."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import numpy.typing as npt

from vaaniswitch.m2_representations.config import MelConfig
from vaaniswitch.m2_representations.schemas import MelResult


def generate_log_mel(waveform: npt.NDArray[np.float64], config: MelConfig) -> MelResult:
    """Generate a finite float32 224x224 power-dB Mel image."""

    _validate_waveform(waveform)
    native_frames = frame_count(waveform.size, config.win_length, config.hop_length)
    fixed, handling, padded_seconds, cropped_seconds = _fixed_time_waveform(waveform, config)
    frames = np.lib.stride_tricks.sliding_window_view(fixed, config.win_length)[
        :: config.hop_length
    ]
    window = _periodic_hann(config.win_length)
    spectrum = np.fft.rfft(frames * window[None, :], n=config.n_fft, axis=1)
    power = np.square(np.abs(spectrum), dtype=np.float64) / float(np.sum(window**2))
    filterbank = mel_filterbank(
        config.sample_rate,
        config.n_fft,
        config.n_mels,
        config.f_min_hz,
        config.f_max_hz,
    )
    mel_power = filterbank @ power.T
    log_mel = 10.0 * np.log10(np.maximum(mel_power, config.epsilon))
    time_resized = log_mel @ area_resize_matrix(log_mel.shape[1], config.output_width)
    frequency_resized = linear_resize_matrix(config.n_mels, config.output_height) @ time_resized
    image = np.asarray(frequency_resized, dtype=np.float32)
    if image.shape != (config.output_height, config.output_width):
        raise AssertionError("Mel resize produced the wrong shape")
    if not np.all(np.isfinite(image)):
        raise ValueError("Mel representation contains non-finite values")
    return MelResult(
        image=image,
        native_frame_count=native_frames,
        fixed_frame_count=frames.shape[0],
        time_handling=handling,
        padded_seconds=padded_seconds,
        cropped_seconds=cropped_seconds,
    )


def frame_count(sample_count: int, win_length: int, hop_length: int) -> int:
    """Return non-centered STFT frame count, zero for signals shorter than a window."""

    if sample_count < win_length:
        return 0
    return 1 + (sample_count - win_length) // hop_length


@lru_cache(maxsize=8)
def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min_hz: float,
    f_max_hz: float,
) -> npt.NDArray[np.float64]:
    """Build HTK-spaced triangular filters with Slaney area normalization."""

    mel_min = 2595.0 * np.log10(1.0 + f_min_hz / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + f_max_hz / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (np.power(10.0, mel_points / 2595.0) - 1.0)
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    bank = np.zeros((n_mels, frequencies.size), dtype=np.float64)
    for index in range(n_mels):
        left, center, right = hz_points[index : index + 3]
        rising = (frequencies - left) / max(center - left, np.finfo(np.float64).eps)
        falling = (right - frequencies) / max(right - center, np.finfo(np.float64).eps)
        triangle = np.maximum(0.0, np.minimum(rising, falling))
        bank[index] = triangle * (2.0 / max(right - left, np.finfo(np.float64).eps))
    if np.any(np.sum(bank, axis=1) == 0):
        raise ValueError("Mel configuration creates an empty filter at the selected FFT resolution")
    return bank


@lru_cache(maxsize=16)
def area_resize_matrix(source_size: int, target_size: int) -> npt.NDArray[np.float64]:
    """Return weights for duration-preserving one-dimensional area averaging."""

    if source_size <= 0 or target_size <= 0:
        raise ValueError("resize dimensions must be positive")
    scale = source_size / target_size
    matrix = np.zeros((source_size, target_size), dtype=np.float64)
    for target in range(target_size):
        left = target * scale
        right = (target + 1) * scale
        first = int(np.floor(left))
        last = min(int(np.ceil(right)), source_size)
        for source in range(first, last):
            overlap = max(0.0, min(right, source + 1.0) - max(left, float(source)))
            matrix[source, target] = overlap / scale
    return matrix


@lru_cache(maxsize=16)
def linear_resize_matrix(source_size: int, target_size: int) -> npt.NDArray[np.float64]:
    """Return a matrix for endpoint-aligned linear interpolation."""

    if source_size <= 0 or target_size <= 0:
        raise ValueError("resize dimensions must be positive")
    matrix = np.zeros((target_size, source_size), dtype=np.float64)
    positions = np.linspace(0.0, source_size - 1.0, target_size)
    for target, position in enumerate(positions):
        lower = int(np.floor(position))
        upper = min(lower + 1, source_size - 1)
        weight = position - lower
        matrix[target, lower] += 1.0 - weight
        matrix[target, upper] += weight
    return matrix


def _fixed_time_waveform(
    waveform: npt.NDArray[np.float64], config: MelConfig
) -> tuple[npt.NDArray[np.float64], str, float, float]:
    target = config.fixed_sample_count
    if waveform.size < target:
        padded = np.pad(waveform, (0, target - waveform.size), mode="constant")
        return padded, "padded", (target - waveform.size) / config.sample_rate, 0.0
    if waveform.size > target:
        start = (waveform.size - target) // 2
        cropped = np.array(waveform[start : start + target], dtype=np.float64, copy=True)
        return cropped, "cropped", 0.0, (waveform.size - target) / config.sample_rate
    return np.array(waveform, dtype=np.float64, copy=True), "exact", 0.0, 0.0


@lru_cache(maxsize=8)
def _periodic_hann(length: int) -> npt.NDArray[np.float64]:
    return np.hanning(length + 1)[:-1].astype(np.float64)


def _validate_waveform(waveform: npt.NDArray[np.float64]) -> None:
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("waveform must be a non-empty one-dimensional array")
    if waveform.dtype != np.float64:
        raise ValueError("source reconstruction must be float64")
    if not np.all(np.isfinite(waveform)):
        raise ValueError("waveform contains non-finite values")
