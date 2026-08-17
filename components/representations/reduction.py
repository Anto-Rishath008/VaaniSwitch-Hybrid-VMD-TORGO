"""Deterministic anti-aliasing scalar reductions used by RP and NVG."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def block_rms(signal: npt.NDArray[np.float64], count: int) -> npt.NDArray[np.float64]:
    """Aggregate contiguous time bins by RMS, retaining exactly ``count`` states."""

    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional array")
    if count <= 0:
        raise ValueError("count must be positive")
    boundaries = np.linspace(0, signal.size, count + 1)
    result = np.empty(count, dtype=np.float64)
    squared = np.square(signal, dtype=np.float64)
    for index in range(count):
        start = int(np.floor(boundaries[index]))
        stop = int(np.floor(boundaries[index + 1]))
        if stop <= start:
            sample = min(
                int(np.floor((boundaries[index] + boundaries[index + 1]) / 2)), signal.size - 1
            )
            result[index] = abs(float(signal[sample]))
        else:
            result[index] = float(np.sqrt(np.mean(squared[start:stop], dtype=np.float64)))
    return result


def zscore(series: npt.NDArray[np.float64], epsilon: float) -> npt.NDArray[np.float64]:
    """Standardize one scalar series without using corpus, class, or speaker statistics."""

    mean = float(np.mean(series, dtype=np.float64))
    standard_deviation = float(np.std(series, dtype=np.float64))
    if standard_deviation <= epsilon:
        return np.zeros_like(series, dtype=np.float64)
    return (series - mean) / standard_deviation
