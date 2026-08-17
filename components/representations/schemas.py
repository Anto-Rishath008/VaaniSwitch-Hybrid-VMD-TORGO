"""Typed numerical results shared by Module 2 components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class MelResult:
    image: npt.NDArray[np.float32]
    native_frame_count: int
    fixed_frame_count: int
    time_handling: str
    padded_seconds: float
    cropped_seconds: float


@dataclass(frozen=True, slots=True)
class RecurrenceResult:
    image: npt.NDArray[np.uint8]
    scalar_state_count: int
    embedded_state_count: int
    threshold: float
    recurrence_density: float


@dataclass(frozen=True, slots=True)
class GraphResult:
    node_features: npt.NDArray[np.float32]
    edge_index: npt.NDArray[np.int16]
    graph_density: float
    isolated_node_count: int
    connected_component_count: int
