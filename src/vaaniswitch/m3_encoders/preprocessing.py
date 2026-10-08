"""Fixed, label-independent Module 3 input transforms."""

from __future__ import annotations

import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_mel(
    mel: torch.Tensor,
    *,
    db_min: float = -100.0,
    db_max: float = 20.0,
    imagenet_normalization: bool = True,
) -> torch.Tensor:
    """Clamp power-dB Mel images, map to [0,1], replicate, and normalize."""

    if mel.ndim != 4 or mel.shape[1:] != (1, 224, 224):
        raise ValueError("Mel input must have shape [B,1,224,224]")
    if not mel.is_floating_point() or not torch.isfinite(mel).all():
        raise ValueError("Mel input must be finite floating point")
    if db_max <= db_min:
        raise ValueError("Mel dB maximum must exceed minimum")
    scaled = torch.clamp(mel.to(torch.float32), db_min, db_max)
    scaled = (scaled - db_min) / (db_max - db_min)
    replicated = scaled.repeat(1, 3, 1, 1)
    if not imagenet_normalization:
        return replicated
    mean = replicated.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = replicated.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (replicated - mean) / std


def preprocess_recurrence(recurrence: torch.Tensor) -> torch.Tensor:
    """Convert a binary recurrence batch to float32 without corpus statistics."""

    if recurrence.ndim != 4 or recurrence.shape[1:] != (1, 224, 224):
        raise ValueError("recurrence input must have shape [B,1,224,224]")
    if recurrence.is_floating_point() and not torch.isfinite(recurrence).all():
        raise ValueError("recurrence input must be finite")
    if not torch.logical_or(recurrence == 0, recurrence == 1).all():
        raise ValueError("recurrence input must be binary")
    return recurrence.to(torch.float32)
