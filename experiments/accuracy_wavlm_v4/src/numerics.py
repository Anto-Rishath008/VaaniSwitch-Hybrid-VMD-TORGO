"""Fail on non-finite tensors; no scaler, silent batch skip, or numerical retry."""
import numpy as np
import torch
from .common import require


def configure_numerics(seed=42):
    from experiments.accuracy_hubert_v2.src.training import seed_all
    seed_all(seed)
    # V2 needed warn-only for its RP network; V4 has no such operation.
    # Use one deterministic backend for forward and checkpoint recomputation.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)


def finite(value, where):
    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), 'Non-finite '+where)
    elif isinstance(value, np.ndarray):
        require(np.isfinite(value).all(), 'Non-finite '+where)
    elif isinstance(value, dict):
        for k, v in value.items():
            finite(v, where+'/'+str(k))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            finite(v, where+'/'+str(i))
    elif isinstance(value, (float, np.floating)):
        require(np.isfinite(value), 'Non-finite '+where)
    return value


def state_check(model, optimizer=None):
    for name, p in model.named_parameters():
        finite(p, 'parameter/'+name)
    if optimizer is not None:
        finite(optimizer.state_dict(), 'optimizer')


def step(model, optimizer, clip):
    trainable = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            require(p.grad is not None, 'Missing intended gradient/'+name)
            finite(p.grad, 'gradient/'+name)
            trainable.append(p)
        else:
            require(p.grad is None, 'Frozen parameter received gradient/'+name)
    norm = torch.nn.utils.clip_grad_norm_(trainable, clip, error_if_nonfinite=True)
    finite(norm, 'gradient_norm')
    optimizer.step()
    state_check(model, optimizer)
    optimizer.zero_grad(set_to_none=True)
    return float(norm)
