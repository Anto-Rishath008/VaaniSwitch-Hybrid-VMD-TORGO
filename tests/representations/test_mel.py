"""Mel transform determinism, semantics, and fixed-time policy tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from tests.representations.helpers import make_config
from components.representations.config import RepresentationConfigError
from components.representations.mel import (
    area_resize_matrix,
    frame_count,
    generate_log_mel,
    linear_resize_matrix,
    mel_filterbank,
)


def _waveform(seconds: float = 1.0) -> npt.NDArray[np.float64]:
    samples = np.arange(round(16_000 * seconds), dtype=np.float64)
    return cast(
        npt.NDArray[np.float64],
        (0.1 * np.sin(2 * np.pi * 220 * samples / 16_000)).astype(np.float64),
    )


def test_mel_is_deterministic_finite_float32_and_224_square(tmp_path: Path) -> None:
    config = make_config(tmp_path).mel
    first = generate_log_mel(_waveform(), config)
    second = generate_log_mel(_waveform(), config)

    np.testing.assert_array_equal(first.image, second.image)
    assert first.image.shape == (224, 224)
    assert first.image.dtype == np.float32
    assert np.all(np.isfinite(first.image))


def test_mel_native_frames_use_25ms_window_10ms_hop_without_centering(tmp_path: Path) -> None:
    config = make_config(tmp_path).mel
    result = generate_log_mel(_waveform(1.0), config)

    assert config.sample_rate == 16_000
    assert config.win_length == 400
    assert config.hop_length == 160
    assert result.native_frame_count == 98
    assert result.fixed_frame_count == 798
    assert frame_count(16_000, 400, 160) == 98


def test_short_signal_is_right_padded_on_fixed_eight_second_axis(tmp_path: Path) -> None:
    result = generate_log_mel(_waveform(1.0), make_config(tmp_path).mel)

    assert result.time_handling == "padded"
    assert result.padded_seconds == 7.0
    assert result.cropped_seconds == 0.0
    assert np.all(result.image[:, -1] == np.float32(-100.0))


def test_long_signal_is_deterministically_center_cropped(tmp_path: Path) -> None:
    config = make_config(tmp_path).mel
    waveform = _waveform(9.0)
    result = generate_log_mel(waveform, config)
    center = waveform[8_000:-8_000]
    centered_result = generate_log_mel(center, config)

    assert center.size == 128_000
    assert result.time_handling == "cropped"
    assert result.cropped_seconds == 1.0
    assert result.fixed_frame_count == 798
    assert result.image.shape == (224, 224)
    np.testing.assert_array_equal(result.image, centered_result.image)


def test_exact_eight_second_signal_is_neither_padded_nor_cropped(tmp_path: Path) -> None:
    result = generate_log_mel(_waveform(8.0), make_config(tmp_path).mel)
    assert result.time_handling == "exact"
    assert result.padded_seconds == result.cropped_seconds == 0.0


def test_mel_rejects_wrong_dtype_and_nonfinite_input(tmp_path: Path) -> None:
    config = make_config(tmp_path).mel
    with pytest.raises(ValueError, match="float64"):
        generate_log_mel(_waveform().astype(np.float32), config)
    waveform = _waveform()
    waveform[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        generate_log_mel(waveform, config)


def test_resize_weights_preserve_constants() -> None:
    area = area_resize_matrix(798, 224)
    production_linear = linear_resize_matrix(80, 224)
    generic_linear = linear_resize_matrix(128, 224)
    np.testing.assert_allclose(area.sum(axis=0), 1.0)
    np.testing.assert_allclose(production_linear.sum(axis=1), 1.0)
    np.testing.assert_allclose(generic_linear.sum(axis=1), 1.0)
    assert np.all(production_linear >= 0.0)
    assert np.all(np.count_nonzero(production_linear, axis=1) <= 2)
    np.testing.assert_allclose(production_linear @ np.ones(80), np.ones(224))


def test_mel_filterbank_is_nonnegative_and_bounded_to_rfft_bins() -> None:
    bank = mel_filterbank(16_000, 512, 80, 20.0, 8_000.0)
    assert bank.shape == (80, 257)
    assert np.all(bank >= 0)
    assert np.all(np.sum(bank, axis=1) > 0)


def test_mel_filterbank_rejects_empty_native_band_configuration() -> None:
    with pytest.raises(ValueError, match="empty filter"):
        mel_filterbank(16_000, 512, 128, 20.0, 8_000.0)


def test_non_16khz_semantics_are_rejected_by_config(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    invalid = replace(config, mel=replace(config.mel, sample_rate=8_000, f_max_hz=4_000.0))
    with pytest.raises(RepresentationConfigError, match="16 kHz"):
        invalid.validate()
