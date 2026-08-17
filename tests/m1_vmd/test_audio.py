from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from vaaniswitch.m1_vmd.audio import (
    AudioLoadConfig,
    AudioLoadError,
    AudioNormalizationError,
    NormalizationConfig,
    load_wav,
    normalize_waveform,
    standardize_audio,
)


def _encode_integer_pcm(samples: NDArray[np.int64], sample_width_bytes: int) -> bytes:
    flattened = samples.reshape(-1)
    if sample_width_bytes == 1:
        return (flattened + 128).astype(np.uint8).tobytes()
    if sample_width_bytes == 2:
        return flattened.astype("<i2").tobytes()
    if sample_width_bytes == 3:
        unsigned = np.where(flattened < 0, flattened + (1 << 24), flattened)
        packed = np.empty((unsigned.size, 3), dtype=np.uint8)
        packed[:, 0] = unsigned & 0xFF
        packed[:, 1] = (unsigned >> 8) & 0xFF
        packed[:, 2] = (unsigned >> 16) & 0xFF
        return packed.tobytes()
    if sample_width_bytes == 4:
        return flattened.astype("<i4").tobytes()
    raise AssertionError(f"unsupported test sample width: {sample_width_bytes}")


def _write_pcm_wav(
    path: Path,
    samples: NDArray[np.int64],
    *,
    sample_rate: int = 16_000,
    sample_width_bytes: int = 2,
) -> None:
    if samples.ndim != 2:
        raise AssertionError("test samples must have shape (frames, channels)")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(samples.shape[1])
        wav_file.setsampwidth(sample_width_bytes)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(_encode_integer_pcm(samples, sample_width_bytes))


def test_loader_preserves_valid_16khz_mono_samples_and_source_file(tmp_path: Path) -> None:
    path = tmp_path / "valid.wav"
    integer_samples = np.array([-32768, -12345, 0, 23456, 32767], dtype=np.int64)[:, None]
    _write_pcm_wav(path, integer_samples)
    bytes_before = path.read_bytes()

    loaded = load_wav(path)

    np.testing.assert_array_equal(loaded.waveform, integer_samples[:, 0] / 32768.0)
    assert loaded.waveform.dtype == np.float64
    assert loaded.sample_rate == 16_000
    assert loaded.channels == 1
    assert loaded.source_sample_rate == 16_000
    assert loaded.source_channels == 1
    assert loaded.sample_width_bytes == 2
    assert loaded.frame_count == integer_samples.shape[0]
    assert loaded.file_size == len(bytes_before)
    assert not loaded.downmixed
    assert not loaded.resampled
    assert loaded.resample_method is None
    assert path.read_bytes() == bytes_before


def test_loader_downmixes_and_resamples_nonstandard_audio_deterministically(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stereo_8khz.wav"
    stereo = np.array(
        [[0, 0], [8192, 0], [16384, 0], [24576, 0]],
        dtype=np.int64,
    )
    _write_pcm_wav(path, stereo, sample_rate=8_000)

    first = load_wav(path)
    second = load_wav(path)

    source_mono = stereo.mean(axis=1) / 32768.0
    expected = np.interp(np.arange(8) * 0.5, np.arange(4), source_mono)
    np.testing.assert_allclose(first.waveform, expected, atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(first.waveform, second.waveform)
    assert first.source_channels == 2
    assert first.downmixed
    assert first.resampled
    assert first.resample_method == "linear_interpolation"
    assert first.waveform.shape == (8,)


@pytest.mark.parametrize(
    ("sample_width_bytes", "denominator"),
    [(1, 128), (2, 32768), (3, 8388608), (4, 2147483648)],
)
def test_loader_decodes_supported_integer_pcm_widths(
    tmp_path: Path,
    sample_width_bytes: int,
    denominator: int,
) -> None:
    path = tmp_path / f"pcm_{sample_width_bytes * 8}.wav"
    signed_samples = np.array(
        [-denominator, -(denominator // 2), 0, denominator // 2, denominator - 1],
        dtype=np.int64,
    )[:, None]
    _write_pcm_wav(path, signed_samples, sample_width_bytes=sample_width_bytes)

    loaded = load_wav(path)

    np.testing.assert_array_equal(loaded.waveform, signed_samples[:, 0] / denominator)
    assert loaded.sample_width_bytes == sample_width_bytes
    assert bool(np.isfinite(loaded.waveform).all())


def test_normalization_is_finite_and_reaches_target_when_not_limited() -> None:
    waveform = np.array([0.1, -0.1, 0.1, -0.1], dtype=np.float64)

    result = normalize_waveform(
        waveform,
        NormalizationConfig(target_rms=0.2, max_gain=3.0, peak_limit=0.9),
    )

    assert bool(np.isfinite(result.waveform).all())
    assert result.stats.rms_before == pytest.approx(0.1)
    assert result.stats.rms_after == pytest.approx(0.2)
    assert result.stats.applied_gain == pytest.approx(2.0)
    assert not result.stats.gain_limited
    assert not result.stats.peak_limited


def test_normalization_obeys_gain_and_peak_caps() -> None:
    quiet = np.full(8, 1.0e-4, dtype=np.float64)
    gain_capped = normalize_waveform(
        quiet,
        NormalizationConfig(target_rms=0.1, max_gain=5.0, peak_limit=0.99),
    )
    peaky = np.array([0.8, 0.01, -0.01, 0.01], dtype=np.float64)
    peak_capped = normalize_waveform(
        peaky,
        NormalizationConfig(target_rms=0.5, max_gain=10.0, peak_limit=0.9),
    )

    assert gain_capped.stats.applied_gain == pytest.approx(5.0)
    assert gain_capped.stats.gain_limited
    assert gain_capped.stats.peak_after <= 0.99
    assert peak_capped.stats.peak_after == pytest.approx(0.9)
    assert peak_capped.stats.peak_limited


@pytest.mark.parametrize(
    "waveform",
    [np.array([], dtype=np.float64), np.zeros(16, dtype=np.float64)],
)
def test_silence_and_empty_signal_are_unchanged_and_safe(
    waveform: NDArray[np.float64],
) -> None:
    result = normalize_waveform(waveform)

    np.testing.assert_array_equal(result.waveform, waveform)
    assert bool(np.isfinite(result.waveform).all())
    assert result.stats.is_silent
    assert result.stats.rms_before == 0.0
    assert result.stats.rms_after == 0.0
    assert result.stats.applied_gain == 1.0


def test_one_frame_audio_is_standardized_safely(tmp_path: Path) -> None:
    path = tmp_path / "one_frame.wav"
    _write_pcm_wav(path, np.array([[8192]], dtype=np.int64), sample_rate=8_000)

    standardized = standardize_audio(path)

    assert standardized.waveform.shape == (2,)
    assert standardized.sample_rate == 16_000
    assert standardized.resampled
    assert bool(np.isfinite(standardized.waveform).all())
    assert standardized.normalization.rms_after == pytest.approx(0.1)


def test_empty_frame_wav_loads_as_safe_empty_signal(tmp_path: Path) -> None:
    path = tmp_path / "empty.wav"
    _write_pcm_wav(path, np.empty((0, 1), dtype=np.int64))

    standardized = standardize_audio(path)

    assert standardized.frame_count == 0
    assert standardized.waveform.shape == (0,)
    assert standardized.normalization.is_silent


def test_nonfinite_normalization_input_is_rejected() -> None:
    with pytest.raises(AudioNormalizationError, match="non-finite"):
        normalize_waveform(np.array([0.0, np.nan], dtype=np.float64))


def test_truncated_wav_payload_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "truncated.wav"
    samples = np.arange(100, dtype=np.int64)[:, None]
    _write_pcm_wav(path, samples)
    path.write_bytes(path.read_bytes()[:-20])

    with pytest.raises(AudioLoadError, match="truncated WAV data"):
        load_wav(path)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AudioLoadConfig(target_sample_rate=0),
        lambda: AudioLoadConfig(target_channels=2),
        lambda: NormalizationConfig(target_rms=0.0),
        lambda: NormalizationConfig(max_gain=0.5),
        lambda: NormalizationConfig(peak_limit=1.1),
        lambda: NormalizationConfig(silence_epsilon=-1.0),
    ],
)
def test_invalid_audio_configuration_is_rejected(factory: object) -> None:
    if not callable(factory):
        raise AssertionError("parameter must be callable")
    with pytest.raises(ValueError):
        factory()
