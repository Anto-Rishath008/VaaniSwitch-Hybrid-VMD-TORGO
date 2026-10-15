"""Fail-closed FP32 classifier optimization with record-level diagnostics."""
from __future__ import annotations

import torch


class NumericalError(RuntimeError):
    def __init__(self, stage, tensor, context=None):
        self.details = {"stage": stage, "tensor": tensor, **(context or {})}
        super().__init__(f"Non-finite numerical state: {self.details}")


def finite_tensor(value, name, context=None):
    if not bool(torch.isfinite(value).all()):
        raise NumericalError("finite_check", name, context)


def finite_state(value, name, context=None):
    if torch.is_tensor(value):
        finite_tensor(value, name, context)
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_state(item, f"{name}.{key}", context)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            finite_state(item, f"{name}.{index}", context)


def checked_forward(model, batch, context=None):
    for name, tensor in batch.items():
        finite_tensor(tensor, f"input.{name}", context)
    # Explicitly disable autocast even if the caller entered an autocast context.
    with torch.autocast("cuda", enabled=False):
        logits = model(batch["speech"].float(), batch.get("rp"))
    finite_tensor(logits, "forward.logits", context)
    return logits


def optimizer_step(model, optimizer, batch, context=None):
    """No gradient skipping/retry. A failed step cannot publish a checkpoint."""
    optimizer.zero_grad(set_to_none=True)
    finite_state(model.state_dict(), "parameters.before", context)
    finite_state(optimizer.state_dict()["state"], "optimizer.before", context)
    logits = checked_forward(model, batch, context)
    loss = (torch.nn.functional.binary_cross_entropy_with_logits(
        logits, batch["labels"], reduction="none") * batch["weights"]).mean()
    finite_tensor(loss, "loss", context)
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is None:
            raise RuntimeError(f"Missing gradient for {name}: {context}")
        if parameter.grad is not None:
            finite_tensor(parameter.grad, f"gradient.{name}", context)
    try:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    except torch.cuda.OutOfMemoryError:
        raise  # Preserve the caller's bounded, whole-epoch OOM rollback.
    except RuntimeError as exc:
        raise NumericalError("gradient_norm", "total_norm", context) from exc
    finite_tensor(norm, "gradient_norm", context)
    optimizer.step()
    finite_state(model.state_dict(), "parameters.after", context)
    finite_state(optimizer.state_dict()["state"], "optimizer.after", context)
    return float(loss.detach()), float(norm.detach())
