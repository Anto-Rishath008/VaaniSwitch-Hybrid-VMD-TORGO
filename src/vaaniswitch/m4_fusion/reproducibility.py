"""CPU baseline seeding and resumable Python, NumPy, and PyTorch RNG state."""

import random
from typing import Any, cast

import numpy as np
import torch


def seed_everything(seed: int, cpu_threads: int) -> None:
    if seed != 42 or cpu_threads < 1:
        raise ValueError("Module 4 requires seed 42 and positive CPU threads")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)


def capture_rng() -> dict[str, Any]:
    numpy_state = cast(tuple[Any, ...], np.random.get_state(legacy=True))
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "numpy": [
            numpy_state[0],
            numpy_state[1].tolist(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ],
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    value = state["numpy"]
    np.random.set_state(
        (value[0], np.asarray(value[1], dtype=np.uint32), value[2], value[3], value[4])
    )
