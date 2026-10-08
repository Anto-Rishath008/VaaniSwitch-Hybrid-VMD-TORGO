"""Numerical, parameter, gradient, timing, and state diagnostics for Module 3."""

from __future__ import annotations

import io
import statistics
import time
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import nn

from vaaniswitch.m3_encoders.config import Module3Config
from vaaniswitch.m3_encoders.models import ParallelTokenEncoders
from vaaniswitch.m3_encoders.schemas import EncoderBatch


def parameter_audit(model: ParallelTokenEncoders) -> dict[str, Any]:
    mel_backbone = _parameters(model.mel_encoder.backbone.parameters())
    mel_projection = _parameters(
        list(model.mel_encoder.projection.parameters())
        + list(model.mel_encoder.layer_norm.parameters())
    )
    recurrence = _parameters(model.recurrence_encoder.parameters())
    graph = _parameters(model.graph_encoder.parameters())
    all_parameters = list(model.parameters())
    total = sum(parameter.numel() for parameter in all_parameters)
    trainable = sum(parameter.numel() for parameter in all_parameters if parameter.requires_grad)
    return {
        "mel_backbone": mel_backbone,
        "mel_projection_and_norm": mel_projection,
        "recurrence_cnn": recurrence,
        "visibility_graph_gcn": graph,
        "total_parameters": total,
        "initial_trainable_parameters": trainable,
        "initial_frozen_parameters": total - trainable,
    }


def token_statistics(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.detach().to(torch.float64).cpu()
    norms = torch.linalg.vector_norm(values, dim=-1).flatten().tolist()
    flattened = values.flatten()
    return {
        "minimum": float(flattened.min()),
        "mean": float(flattened.mean()),
        "std": float(flattened.std(unbiased=False)),
        "maximum": float(flattened.max()),
        "l2_norm": _distribution(norms),
    }


def gradient_sanity(
    model: ParallelTokenEncoders, batch: EncoderBatch
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    model.train()
    output = model(batch)
    objective = (
        output.mel_tokens.square().mean()
        + output.graph_token.square().mean()
        + output.recurrence_tokens.square().mean()
    )
    objective.backward()
    result = {
        "objective": float(objective.detach()),
        "mel_projection": _gradient_group(
            list(model.mel_encoder.projection.named_parameters())
            + list(model.mel_encoder.layer_norm.named_parameters())
        ),
        "recurrence_cnn": _gradient_group(model.recurrence_encoder.named_parameters()),
        "visibility_graph_gcn": _gradient_group(model.graph_encoder.named_parameters()),
        "frozen_mel_backbone": {
            "all_requires_grad_false": all(
                not parameter.requires_grad for parameter in model.mel_encoder.backbone.parameters()
            ),
            "all_gradients_absent": all(
                parameter.grad is None for parameter in model.mel_encoder.backbone.parameters()
            ),
        },
        "optimizer_step_performed": False,
    }
    model.zero_grad(set_to_none=True)
    model.eval()
    return result


def state_dict_roundtrip(
    model: ParallelTokenEncoders,
    config: Module3Config,
    batch: EncoderBatch,
) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        expected = model(batch)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = ParallelTokenEncoders(config, load_pretrained=False)
    restored.load_state_dict(torch.load(stream, map_location="cpu", weights_only=True))
    restored.eval()
    with torch.inference_mode():
        actual = restored(batch)
    comparisons = {
        "mel": torch.equal(expected.mel_tokens, actual.mel_tokens),
        "graph": torch.equal(expected.graph_token, actual.graph_token),
        "recurrence": torch.equal(expected.recurrence_tokens, actual.recurrence_tokens),
    }
    return {"exact_match": all(comparisons.values()), "branches": comparisons}


def measure_cpu_timings(
    model: ParallelTokenEncoders,
    batch: EncoderBatch,
    *,
    repetitions: int,
) -> dict[str, Any]:
    if repetitions <= 0:
        raise ValueError("timing repetitions must be positive")
    model.eval()
    calls: dict[str, Callable[[], object]] = {
        "mel": lambda: model.mel_encoder(batch.mel),
        "recurrence": lambda: model.recurrence_encoder(batch.recurrence),
        "graph": lambda: model.graph_encoder(
            batch.graph_nodes, batch.graph_edges, batch.graph_node_mask
        ),
        "combined": lambda: model(batch),
    }
    result: dict[str, Any] = {"device": "cpu", "batch_size": batch.mel.shape[0]}
    with torch.inference_mode():
        for name, call in calls.items():
            call()
            values: list[float] = []
            for _ in range(repetitions):
                started = time.perf_counter()
                call()
                values.append(time.perf_counter() - started)
            result[name] = _distribution(values)
    return result


def deterministic_eval(model: ParallelTokenEncoders, batch: EncoderBatch) -> dict[str, bool]:
    model.eval()
    with torch.inference_mode():
        first = model(batch)
        second = model(batch)
    return {
        "mel": torch.equal(first.mel_tokens, second.mel_tokens),
        "graph": torch.equal(first.graph_token, second.graph_token),
        "recurrence": torch.equal(first.recurrence_tokens, second.recurrence_tokens),
        "nonlinear": torch.equal(first.nonlinear_tokens, second.nonlinear_tokens),
    }


def _parameters(parameters: Iterable[nn.Parameter]) -> dict[str, int]:
    values = list(parameters)
    total = sum(parameter.numel() for parameter in values)
    trainable = sum(parameter.numel() for parameter in values if parameter.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def _gradient_group(parameters: Iterable[tuple[str, nn.Parameter]]) -> dict[str, Any]:
    relevant = [(name, parameter) for name, parameter in parameters if parameter.requires_grad]
    missing = [name for name, parameter in relevant if parameter.grad is None]
    nonfinite = [
        name
        for name, parameter in relevant
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    return {
        "trainable_parameter_tensors": len(relevant),
        "all_gradients_present": not missing,
        "all_gradients_finite": not nonfinite,
        "missing": missing,
        "nonfinite": nonfinite,
    }


def _distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }
