"""The unchanged recurrence subset of the signed Module 3 configuration."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RecurrenceEncoderConfig:
    implementation: str = "small_cnn"
    input_channels: int = 1
    channels: tuple[int, ...] = (16, 32, 64)
    normalization: str = "groupnorm"
    activation: str = "gelu"
    adaptive_pool: tuple[int, int] = (4, 4)
    tokens: int = 16
