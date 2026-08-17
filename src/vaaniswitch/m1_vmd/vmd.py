"""Pure-NumPy Variational Mode Decomposition for Module 1.

The implementation follows the VMD variational objective directly: each mode is
updated in the frequency domain with a quadratic bandwidth penalty, followed by
an update of its power-weighted center frequency.  It is an independent,
project-local implementation and deliberately exposes no third-party VMD API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class VMDParameters:
    """Numerical parameters for a VMD decomposition.

    Frequencies inside the solver are expressed in cycles per sample.  ``tau``
    is the dual-ascent step; zero is the conventional noise-tolerant setting.
    """

    modes_k: int = 5
    alpha: float = 2000.0
    tau: float = 0.0
    dc: bool = False
    init: int = 1
    tolerance: float = 1.0e-7
    max_iterations: int = 500
    random_seed: int = 0

    def validate(self) -> None:
        """Raise ``ValueError`` when a solver parameter is invalid."""

        if self.modes_k < 1:
            raise ValueError("modes_k must be at least 1")
        if not np.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError("alpha must be finite and greater than zero")
        if not np.isfinite(self.tau) or self.tau < 0.0:
            raise ValueError("tau must be finite and non-negative")
        if self.init not in {0, 1, 2}:
            raise ValueError("init must be 0, 1, or 2")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be finite and greater than zero")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")


@dataclass(frozen=True, slots=True)
class VMDResult:
    """A length-preserving VMD result and convergence diagnostics."""

    modes: NDArray[np.float64]
    center_frequencies_hz: NDArray[np.float64]
    iterations: int
    converged: bool
    final_delta: float


def _initial_center_frequencies(parameters: VMDParameters) -> NDArray[np.float64]:
    """Return initial center frequencies in cycles per sample."""

    if parameters.init == 0:
        centers = np.zeros(parameters.modes_k, dtype=np.float64)
    elif parameters.init == 1:
        centers = np.arange(parameters.modes_k, dtype=np.float64) / (
            2.0 * parameters.modes_k
        )
    else:
        generator = np.random.default_rng(parameters.random_seed)
        # A log-uniform spread avoids clustering every randomized center near
        # Nyquist while remaining deterministic for a fixed seed.
        lower = np.finfo(np.float64).eps
        centers = np.sort(np.exp(generator.uniform(np.log(lower), np.log(0.5), parameters.modes_k)))

    if parameters.dc:
        centers[0] = 0.0
    return centers


def _mirror_extend(signal: NDArray[np.float64]) -> tuple[NDArray[np.float64], int]:
    """Mirror the boundaries and return the crop offset for the input region."""

    if signal.size == 1:
        return signal.copy(), 0
    flank = max(1, signal.size // 2)
    extended = np.concatenate((signal[:flank][::-1], signal, signal[-flank:][::-1]))
    return extended, flank


def variational_mode_decomposition(
    signal: ArrayLike,
    *,
    sample_rate: int,
    parameters: VMDParameters | None = None,
) -> VMDResult:
    """Decompose a one-dimensional real signal into exactly ``K`` VMD modes.

    The returned modes always match the original sample count, including odd
    and very short inputs.  With ``init=1`` the computation is deterministic.
    Silence is handled explicitly to avoid undefined frequency estimates.
    """

    params = parameters or VMDParameters()
    params.validate()
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero")

    waveform = np.asarray(signal, dtype=np.float64)
    if waveform.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if waveform.size == 0:
        raise ValueError("signal must contain at least one sample")
    if not np.all(np.isfinite(waveform)):
        raise ValueError("signal must contain only finite values")

    if not np.any(waveform):
        return VMDResult(
            modes=np.zeros((params.modes_k, waveform.size), dtype=np.float64),
            center_frequencies_hz=np.zeros(params.modes_k, dtype=np.float64),
            iterations=0,
            converged=True,
            final_delta=0.0,
        )

    extended, crop_start = _mirror_extend(waveform)
    frequencies = np.fft.rfftfreq(extended.size)
    source_spectrum = np.fft.rfft(extended)
    mode_spectra = np.zeros((params.modes_k, source_spectrum.size), dtype=np.complex128)
    centers = _initial_center_frequencies(params)
    dual = np.zeros_like(source_spectrum)

    convergence = float("inf")
    converged = False
    iterations = 0
    tiny = float(np.finfo(np.float64).tiny)
    source_energy = max(float(np.sum(np.abs(source_spectrum) ** 2)), tiny)

    for iteration in range(1, params.max_iterations + 1):
        previous = mode_spectra.copy()
        mode_sum = np.sum(mode_spectra, axis=0)

        for mode_index in range(params.modes_k):
            other_modes = mode_sum - mode_spectra[mode_index]
            residual = source_spectrum - other_modes - 0.5 * dual
            frequency_offset = frequencies - centers[mode_index]
            denominator = 1.0 + params.alpha * frequency_offset * frequency_offset
            updated = residual / denominator
            mode_spectra[mode_index] = updated
            mode_sum = other_modes + updated

            if not (params.dc and mode_index == 0):
                power = np.abs(updated) ** 2
                power_sum = float(np.sum(power))
                if power_sum > tiny:
                    centers[mode_index] = float(np.dot(frequencies, power) / power_sum)
            else:
                centers[mode_index] = 0.0

        if params.tau > 0.0:
            dual += params.tau * (mode_sum - source_spectrum)

        change_energy = float(np.sum(np.abs(mode_spectra - previous) ** 2))
        # A fixed source-energy denominator is stable on the first iteration,
        # when every previous mode is exactly zero, and makes the stopping
        # diagnostic comparable across iterations for one utterance.
        convergence = change_energy / source_energy
        iterations = iteration
        if convergence <= params.tolerance:
            converged = True
            break

    extended_modes = np.fft.irfft(mode_spectra, n=extended.size, axis=1)
    modes = np.asarray(
        extended_modes[:, crop_start : crop_start + waveform.size],
        dtype=np.float64,
    )
    centers_hz = np.clip(centers * float(sample_rate), 0.0, sample_rate / 2.0)

    if not np.all(np.isfinite(modes)) or not np.all(np.isfinite(centers_hz)):
        raise FloatingPointError("VMD produced non-finite output")

    return VMDResult(
        modes=modes,
        center_frequencies_hz=np.asarray(centers_hz, dtype=np.float64),
        iterations=iterations,
        converged=converged,
        final_delta=float(convergence),
    )


decompose_vmd = variational_mode_decomposition
