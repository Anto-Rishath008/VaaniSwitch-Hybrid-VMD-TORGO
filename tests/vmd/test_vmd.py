"""Synthetic regression tests for the pure-NumPy VMD core."""

import numpy as np
import pytest
from numpy.typing import NDArray

from components.vmd.reconstruction import (
    reconstruct_all_modes,
    reconstruct_selected_modes,
)
from components.vmd.vmd import VMDParameters, variational_mode_decomposition


def _multitone(
    sample_count: int = 2_001, sample_rate: int = 16_000
) -> NDArray[np.float64]:
    time = np.arange(sample_count, dtype=np.float64) / sample_rate
    signal: NDArray[np.float64] = (
        0.7 * np.sin(2 * np.pi * 220 * time) + 0.3 * np.sin(2 * np.pi * 1_100 * time)
    )
    return signal


def test_vmd_returns_exactly_k_finite_length_preserving_modes() -> None:
    signal = _multitone()
    result = variational_mode_decomposition(
        signal,
        sample_rate=16_000,
        parameters=VMDParameters(modes_k=5, max_iterations=300),
    )

    assert result.modes.shape == (5, signal.size)
    assert result.center_frequencies_hz.shape == (5,)
    assert np.all(np.isfinite(result.modes))
    assert np.all((result.center_frequencies_hz >= 0.0) & (result.center_frequencies_hz <= 8_000))
    assert result.iterations <= 300


@pytest.mark.parametrize("sample_count", [1, 2, 3, 7, 16, 17])
def test_short_and_odd_inputs_preserve_length(sample_count: int) -> None:
    signal = np.linspace(-0.25, 0.25, sample_count, dtype=np.float64)
    result = variational_mode_decomposition(
        signal,
        sample_rate=16_000,
        parameters=VMDParameters(max_iterations=40),
    )

    assert result.modes.shape == (5, sample_count)
    assert reconstruct_all_modes(result.modes).shape == signal.shape
    assert np.all(np.isfinite(result.modes))


def test_zero_signal_is_safe_and_reconstructs_exactly() -> None:
    signal = np.zeros(257, dtype=np.float64)
    result = variational_mode_decomposition(signal, sample_rate=16_000)

    assert result.converged
    assert result.iterations == 0
    assert np.array_equal(result.modes, np.zeros((5, 257)))
    assert np.array_equal(reconstruct_all_modes(result.modes), signal)


def test_init_one_is_deterministic() -> None:
    signal = _multitone(sample_count=512)
    parameters = VMDParameters(init=1, max_iterations=100)

    first = variational_mode_decomposition(signal, sample_rate=16_000, parameters=parameters)
    second = variational_mode_decomposition(signal, sample_rate=16_000, parameters=parameters)

    np.testing.assert_array_equal(first.modes, second.modes)
    np.testing.assert_array_equal(first.center_frequencies_hz, second.center_frequencies_hz)
    assert first.iterations == second.iterations
    assert first.final_delta == second.final_delta


def test_first_iteration_convergence_diagnostic_does_not_overflow() -> None:
    with np.errstate(over="raise", invalid="raise"):
        result = variational_mode_decomposition(
            _multitone(sample_count=128),
            sample_rate=16_000,
            parameters=VMDParameters(max_iterations=1),
        )

    assert np.isfinite(result.final_delta)


def test_all_mode_reconstruction_is_close_to_multitone_input() -> None:
    signal = _multitone(sample_count=2_000)
    result = variational_mode_decomposition(
        signal,
        sample_rate=16_000,
        parameters=VMDParameters(max_iterations=500),
    )
    reconstruction = reconstruct_all_modes(result.modes)
    relative_error = np.linalg.norm(reconstruction - signal) / np.linalg.norm(signal)

    # VMD with tau=0 permits a small residual by design.  A 2% relative-L2
    # bound is tight enough to detect material signal loss while retaining the
    # specified noise-tolerant dual setting (this deterministic case is ~1%).
    assert relative_error < 0.02


def test_multitone_dominant_modes_locate_known_frequency_components() -> None:
    signal = _multitone(sample_count=2_000)
    result = variational_mode_decomposition(
        signal,
        sample_rate=16_000,
        parameters=VMDParameters(max_iterations=500),
    )
    energies = np.sum(result.modes * result.modes, axis=1)
    dominant_centers = result.center_frequencies_hz[np.argsort(energies)[-2:]]

    assert min(abs(dominant_centers - 220.0)) < 10.0
    assert min(abs(dominant_centers - 1_100.0)) < 10.0


def test_selected_mode_reconstruction_requires_explicit_valid_mode_numbers() -> None:
    modes = np.vstack((np.ones(8), np.full(8, 2.0), np.full(8, 4.0)))

    np.testing.assert_array_equal(reconstruct_selected_modes(modes, [1, 3]), np.full(8, 5.0))
    with pytest.raises(ValueError, match="must not be empty"):
        reconstruct_selected_modes(modes, [])
    with pytest.raises(ValueError, match="duplicates"):
        reconstruct_selected_modes(modes, [1, 1])
    with pytest.raises(ValueError, match="between"):
        reconstruct_selected_modes(modes, [0])


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        (VMDParameters(modes_k=0), "modes_k"),
        (VMDParameters(alpha=0.0), "alpha"),
        (VMDParameters(tau=-1.0), "tau"),
        (VMDParameters(init=3), "init"),
        (VMDParameters(tolerance=0.0), "tolerance"),
        (VMDParameters(max_iterations=0), "max_iterations"),
    ],
)
def test_invalid_vmd_parameters_are_rejected(parameters: VMDParameters, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        variational_mode_decomposition([0.0, 1.0], sample_rate=16_000, parameters=parameters)


def test_invalid_signal_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        variational_mode_decomposition([], sample_rate=16_000)
    with pytest.raises(ValueError, match="finite"):
        variational_mode_decomposition([0.0, np.nan], sample_rate=16_000)
    with pytest.raises(ValueError, match="one-dimensional"):
        variational_mode_decomposition([[0.0], [1.0]], sample_rate=16_000)
    with pytest.raises(ValueError, match="sample_rate"):
        variational_mode_decomposition([0.0], sample_rate=0)
