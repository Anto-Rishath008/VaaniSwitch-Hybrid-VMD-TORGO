"""Deterministic binary recurrence plots from bounded embedded state sequences."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from vaaniswitch.m2_representations.config import RecurrenceConfig
from vaaniswitch.m2_representations.reduction import block_rms, zscore
from vaaniswitch.m2_representations.schemas import RecurrenceResult


def generate_recurrence(
    waveform: npt.NDArray[np.float64], config: RecurrenceConfig
) -> RecurrenceResult:
    """Create a 224-state, Euclidean, fixed-recurrence-rate binary RP."""

    if waveform.ndim != 1 or waveform.size == 0 or not np.all(np.isfinite(waveform)):
        raise ValueError("waveform must be non-empty, one-dimensional, and finite")
    scalar = zscore(block_rms(waveform, config.scalar_state_count), config.epsilon)
    states = np.stack(
        [
            scalar[offset : offset + config.output_size]
            for offset in range(0, config.embedding_dimension * config.delay, config.delay)
        ],
        axis=1,
    )
    differences = states[:, None, :] - states[None, :, :]
    distances = np.sqrt(np.sum(np.square(differences), axis=2, dtype=np.float64))
    upper = distances[np.triu_indices(config.output_size, k=1)]
    threshold = float(np.quantile(upper, config.recurrence_rate, method="higher"))
    image = np.asarray(distances <= threshold, dtype=np.uint8)
    np.fill_diagonal(image, 1)
    density = float(np.mean(image, dtype=np.float64))
    return RecurrenceResult(
        image=image,
        scalar_state_count=scalar.size,
        embedded_state_count=states.shape[0],
        threshold=threshold,
        recurrence_density=density,
    )
