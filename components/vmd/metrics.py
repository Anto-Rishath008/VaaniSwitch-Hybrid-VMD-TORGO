"""Mode metadata and all-mode reconstruction-quality metrics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class ModeMetrics:
    """Signal statistics for one one-based VMD mode."""

    mode_index: int
    sample_count: int
    center_frequency_hz: float
    absolute_energy: float
    relative_energy: float
    rms: float
    spectral_centroid_hz: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReconstructionMetrics:
    """Length, error, correlation, and energy comparison for reconstruction."""

    original_length: int
    reconstruction_length: int
    mse: float
    rmse: float
    normalized_reconstruction_error: float
    correlation_coefficient: float
    original_energy: float
    reconstruction_energy: float
    energy_difference: float
    energy_ratio: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def _one_sided_frequency_statistics(
    signal: NDArray[np.float64], sample_rate: int
) -> tuple[float, float]:
    """Return power center frequency and magnitude spectral centroid in hertz."""

    spectrum = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    magnitudes = np.abs(spectrum)
    power = magnitudes * magnitudes
    power_sum = float(np.sum(power))
    magnitude_sum = float(np.sum(magnitudes))
    center = 0.0 if power_sum == 0.0 else float(np.dot(frequencies, power) / power_sum)
    centroid = (
        0.0 if magnitude_sum == 0.0 else float(np.dot(frequencies, magnitudes) / magnitude_sum)
    )
    nyquist = sample_rate / 2.0
    return float(np.clip(center, 0.0, nyquist)), float(np.clip(centroid, 0.0, nyquist))


def compute_mode_metrics(modes: ArrayLike, *, sample_rate: int) -> tuple[ModeMetrics, ...]:
    """Compute finite energy and frequency metadata for every VMD mode."""

    array = np.asarray(modes, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("modes must have shape (K, sample_count) with non-zero dimensions")
    if not np.all(np.isfinite(array)):
        raise ValueError("modes must contain only finite values")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero")

    energies = np.sum(array * array, axis=1, dtype=np.float64)
    total_energy = float(np.sum(energies))
    relative_energies = (
        np.zeros_like(energies) if total_energy == 0.0 else energies / total_energy
    )

    results: list[ModeMetrics] = []
    for index, mode in enumerate(array):
        center, centroid = _one_sided_frequency_statistics(mode, sample_rate)
        energy = float(energies[index])
        results.append(
            ModeMetrics(
                mode_index=index + 1,
                sample_count=int(mode.size),
                center_frequency_hz=center,
                absolute_energy=energy,
                relative_energy=float(relative_energies[index]),
                rms=float(np.sqrt(np.mean(mode * mode, dtype=np.float64))),
                spectral_centroid_hz=centroid,
            )
        )
    return tuple(results)


def compute_reconstruction_metrics(
    original: ArrayLike,
    reconstruction: ArrayLike,
) -> ReconstructionMetrics:
    """Compare a reconstruction against its same-length working waveform."""

    reference = np.asarray(original, dtype=np.float64)
    candidate = np.asarray(reconstruction, dtype=np.float64)
    if reference.ndim != 1 or candidate.ndim != 1:
        raise ValueError("original and reconstruction must be one-dimensional")
    if reference.size == 0 or candidate.size == 0:
        raise ValueError("original and reconstruction must not be empty")
    if reference.size != candidate.size:
        raise ValueError("original and reconstruction lengths must match")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("original and reconstruction must contain only finite values")

    error = candidate - reference
    mse = float(np.mean(error * error, dtype=np.float64))
    rmse = math.sqrt(mse)
    original_energy = float(np.dot(reference, reference))
    reconstruction_energy = float(np.dot(candidate, candidate))
    error_norm = float(np.linalg.norm(error))
    original_norm = math.sqrt(original_energy)
    epsilon = float(np.finfo(np.float64).eps)
    if original_norm == 0.0:
        normalized_error = 0.0 if error_norm == 0.0 else error_norm / epsilon
    else:
        normalized_error = error_norm / original_norm

    reference_centered = reference - float(np.mean(reference))
    candidate_centered = candidate - float(np.mean(candidate))
    reference_spread = float(np.linalg.norm(reference_centered))
    candidate_spread = float(np.linalg.norm(candidate_centered))
    if reference_spread == 0.0 or candidate_spread == 0.0:
        correlation = 1.0 if np.array_equal(reference, candidate) else 0.0
    else:
        correlation = float(
            np.clip(
                np.dot(reference_centered, candidate_centered)
                / (reference_spread * candidate_spread),
                -1.0,
                1.0,
            )
        )

    if original_energy == 0.0:
        energy_ratio = 1.0 if reconstruction_energy == 0.0 else reconstruction_energy / epsilon
    else:
        energy_ratio = reconstruction_energy / original_energy

    values = (
        mse,
        rmse,
        normalized_error,
        correlation,
        original_energy,
        reconstruction_energy,
        reconstruction_energy - original_energy,
        energy_ratio,
    )
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("reconstruction metric computation produced non-finite output")

    return ReconstructionMetrics(
        original_length=int(reference.size),
        reconstruction_length=int(candidate.size),
        mse=mse,
        rmse=rmse,
        normalized_reconstruction_error=normalized_error,
        correlation_coefficient=correlation,
        original_energy=original_energy,
        reconstruction_energy=reconstruction_energy,
        energy_difference=reconstruction_energy - original_energy,
        energy_ratio=energy_ratio,
    )
