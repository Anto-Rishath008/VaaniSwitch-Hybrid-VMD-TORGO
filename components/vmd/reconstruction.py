"""Explicit all-mode and optional selected-mode VMD reconstruction."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validated_modes(modes: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(modes, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("modes must have shape (K, sample_count) with non-zero dimensions")
    if not np.all(np.isfinite(array)):
        raise ValueError("modes must contain only finite values")
    return array


def reconstruct_all_modes(modes: ArrayLike) -> NDArray[np.float64]:
    """Return the default reconstruction formed by summing every VMD mode."""

    array = _validated_modes(modes)
    reconstruction: NDArray[np.float64] = np.sum(array, axis=0, dtype=np.float64)
    return reconstruction


def reconstruct_selected_modes(
    modes: ArrayLike,
    selected_mode_numbers: tuple[int, ...] | list[int],
) -> NDArray[np.float64]:
    """Reconstruct an explicit future ablation from one-based mode numbers.

    This function intentionally defines no automatic selection or deletion
    rule.  Callers must opt in and state every selected one-based mode number.
    """

    array = _validated_modes(modes)
    if not selected_mode_numbers:
        raise ValueError("selected_mode_numbers must not be empty")
    if len(set(selected_mode_numbers)) != len(selected_mode_numbers):
        raise ValueError("selected_mode_numbers must not contain duplicates")
    if any(number < 1 or number > array.shape[0] for number in selected_mode_numbers):
        raise ValueError(f"selected mode numbers must be between 1 and {array.shape[0]}")
    indices = np.asarray(selected_mode_numbers, dtype=np.intp) - 1
    reconstruction: NDArray[np.float64] = np.sum(array[indices], axis=0, dtype=np.float64)
    return reconstruction
