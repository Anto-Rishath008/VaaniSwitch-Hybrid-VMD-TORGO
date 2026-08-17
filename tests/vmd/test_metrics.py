"""Mode and reconstruction metric regression tests."""

import math

import numpy as np
import pytest

from components.vmd.metrics import compute_mode_metrics, compute_reconstruction_metrics


def test_mode_energies_and_relative_energies_are_well_formed() -> None:
    time = np.arange(1_600, dtype=np.float64) / 16_000
    modes = np.vstack(
        (
            np.sin(2 * np.pi * 200 * time),
            0.5 * np.sin(2 * np.pi * 2_000 * time),
            np.zeros_like(time),
        )
    )

    metrics = compute_mode_metrics(modes, sample_rate=16_000)

    assert len(metrics) == 3
    assert [metric.mode_index for metric in metrics] == [1, 2, 3]
    assert all(metric.sample_count == 1_600 for metric in metrics)
    assert all(
        math.isfinite(metric.absolute_energy) and metric.absolute_energy >= 0 for metric in metrics
    )
    assert sum(metric.relative_energy for metric in metrics) == pytest.approx(1.0)
    assert all(0.0 <= metric.center_frequency_hz <= 8_000.0 for metric in metrics)
    assert metrics[0].center_frequency_hz == pytest.approx(200.0, abs=1.0)
    assert metrics[1].center_frequency_hz == pytest.approx(2_000.0, abs=1.0)


def test_silent_modes_have_safe_zero_metadata() -> None:
    metrics = compute_mode_metrics(np.zeros((5, 31)), sample_rate=16_000)

    assert all(metric.absolute_energy == 0.0 for metric in metrics)
    assert all(metric.relative_energy == 0.0 for metric in metrics)
    assert all(metric.center_frequency_hz == 0.0 for metric in metrics)
    assert all(metric.spectral_centroid_hz == 0.0 for metric in metrics)
    assert all(metric.rms == 0.0 for metric in metrics)


def test_reconstruction_metrics_for_exact_signal_are_ideal() -> None:
    signal = np.linspace(-1.0, 1.0, 101)
    metrics = compute_reconstruction_metrics(signal, signal.copy())

    assert metrics.original_length == metrics.reconstruction_length == 101
    assert metrics.mse == 0.0
    assert metrics.rmse == 0.0
    assert metrics.normalized_reconstruction_error == 0.0
    assert metrics.correlation_coefficient == pytest.approx(1.0)
    assert metrics.energy_difference == 0.0
    assert metrics.energy_ratio == pytest.approx(1.0)
    assert all(math.isfinite(value) for value in metrics.to_dict().values())


def test_reconstruction_metrics_are_safe_for_silence() -> None:
    signal = np.zeros(19)
    metrics = compute_reconstruction_metrics(signal, signal)

    assert metrics.normalized_reconstruction_error == 0.0
    assert metrics.correlation_coefficient == 1.0
    assert metrics.energy_ratio == 1.0


def test_reconstruction_metric_input_validation() -> None:
    with pytest.raises(ValueError, match="lengths"):
        compute_reconstruction_metrics(np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        compute_reconstruction_metrics(np.array([0.0, np.nan]), np.zeros(2))
    with pytest.raises(ValueError, match="one-dimensional"):
        compute_reconstruction_metrics(np.zeros((1, 2)), np.zeros(2))
    with pytest.raises(ValueError, match="must not be empty"):
        compute_reconstruction_metrics(np.array([]), np.array([]))
