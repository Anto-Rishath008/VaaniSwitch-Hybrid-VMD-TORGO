"""Regression tests for scientifically faithful Module 1 SVG scaling."""

import numpy as np
import pytest

from vaaniswitch.m1_vmd.plots import _polyline


def _vertical_excursion(points: str, *, middle: float) -> float:
    ys = [float(point.split(",")[1]) for point in points.split()]
    return max(abs(y - middle) for y in ys)


def test_overlay_traces_preserve_relative_amplitude_with_shared_scale() -> None:
    """A half-amplitude reconstruction must not be independently peak-normalized."""

    original = np.array([-1.0, 1.0], dtype=np.float64)
    reconstruction = original * 0.5
    original_points = _polyline(original, 0, 100, 0, 100, scale=1.0)
    reconstruction_points = _polyline(reconstruction, 0, 100, 0, 100, scale=1.0)

    original_excursion = _vertical_excursion(original_points, middle=50.0)
    reconstruction_excursion = _vertical_excursion(reconstruction_points, middle=50.0)

    assert reconstruction_excursion == pytest.approx(original_excursion * 0.5)
