"""Keep CPU fixtures deterministic and bounded."""

from collections.abc import Iterator

import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_cpu() -> Iterator[None]:
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)
