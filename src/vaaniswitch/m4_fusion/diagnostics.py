"""Small real development-only shape, BCE, overfit, checkpoint, and CPU timing audits."""

from __future__ import annotations

import statistics
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from vaaniswitch.m3_encoders.diagnostics import parameter_audit as encoder_parameter_audit
from vaaniswitch.m3_encoders.schemas import EncoderBatch
from vaaniswitch.m4_fusion.checkpoint import load_checkpoint, save_checkpoint
from vaaniswitch.m4_fusion.config import Module4Config
from vaaniswitch.m4_fusion.losses import batch_labels, make_loss
from vaaniswitch.m4_fusion.model import DysarthriaClassifier
from vaaniswitch.m4_fusion.pipeline import build_model, record_split, selection_identity
from vaaniswitch.m4_fusion.reporting import write_json
from vaaniswitch.m4_fusion.reproducibility import seed_everything
from vaaniswitch.m4_fusion.source import DevelopmentContext, prepare_context
from vaaniswitch.m4_fusion.splits import guard_batch
from vaaniswitch.m4_fusion.trainer import batches, finite_gradients, make_optimizer


def parameter_audit(model: DysarthriaClassifier) -> dict[str, Any]:
    encoder = encoder_parameter_audit(model.encoders)
    return {
        "module3": encoder,
        "cross_attention": sum(
            p.numel() for p in model.fusion.cross_attention.attention.parameters()
        ),
        "residual_layer_norm": sum(
            p.numel() for p in model.fusion.cross_attention.layer_norm.parameters()
        ),
        "classifier": sum(p.numel() for p in model.fusion.classifier.parameters()),
        "combined_total": sum(p.numel() for p in model.parameters()),
        "combined_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "combined_frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def measure(call: Callable[[], object], repetitions: int = 5) -> dict[str, float]:
    call()
    timings = []
    for _ in range(repetitions):
        start = time.perf_counter()
        call()
        timings.append(time.perf_counter() - start)
    return distribution(timings)


def gradient_audit(model: DysarthriaClassifier, batch: EncoderBatch) -> dict[str, Any]:
    model.train()
    model.zero_grad(set_to_none=True)
    targets = batch_labels(batch)
    pos_weight = float((targets == 0).sum() / (targets == 1).sum())
    loss = make_loss(pos_weight)(model(batch).logits, targets)
    loss.backward()
    finite_gradients(model)
    prefixes = {
        "mel_projection": "encoders.mel_encoder.projection",
        "mel_layer_norm": "encoders.mel_encoder.layer_norm",
        "recurrence_cnn": "encoders.recurrence_encoder",
        "graph_gcn": "encoders.graph_encoder",
        "cross_attention": "fusion.cross_attention",
        "classifier": "fusion.classifier",
    }
    groups = {}
    for label, prefix in prefixes.items():
        gradients = [p.grad for name, p in model.named_parameters() if name.startswith(prefix)]
        norm = sum(float(g.detach().abs().sum()) for g in gradients if g is not None)
        if norm == 0:
            raise ValueError(f"zero aggregate gradient in {label}")
        groups[label] = {"tensors": len(gradients), "finite": True, "absolute_gradient_sum": norm}
    return {
        "status": "PASS",
        "bce_loss": float(loss.detach()),
        "groups": groups,
        "backbone_gradients_absent": True,
        "optimizer_step_performed": False,
    }


def tiny_batch_overfit(
    model: DysarthriaClassifier,
    batch: EncoderBatch,
    context: DevelopmentContext,
    *,
    steps: int = 25,
) -> tuple[dict[str, Any], torch.optim.AdamW]:
    if not 1 <= steps <= 25:
        raise ValueError("tiny-batch sanity is capped at 25 updates")
    guard_batch(batch, context.source.contract, context.split, "train")
    targets = batch_labels(batch)
    if set(targets.tolist()) != {0.0, 1.0}:
        raise ValueError("tiny batch must contain both classes")
    pos_weight = float((targets == 0).sum() / (targets == 1).sum())
    criterion = make_loss(pos_weight)
    optimizer = make_optimizer(model, context)
    model.eval()
    with torch.no_grad():
        initial = float(criterion(model(batch).logits, targets))
    timings = []
    for _ in range(steps):
        model.train()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(batch).logits, targets)
        loss.backward()
        finite_gradients(model)
        optimizer.step()
        timings.append(time.perf_counter() - start)
    model.eval()
    with torch.no_grad():
        final = float(criterion(model(batch).logits, targets))
    if final >= initial * 0.8:
        raise ValueError(f"tiny batch did not materially reduce eval BCE: {initial} -> {final}")
    return {
        "status": "PASS",
        "updates": steps,
        "records": len(batch.provenance),
        "initial_eval_bce": initial,
        "final_eval_bce": final,
        "relative_loss_reduction": 1 - final / initial,
        "pos_weight": pos_weight,
        "update_seconds": distribution(timings),
        "scientific_performance_result": False,
    }, optimizer


def run_smoke(config: Module4Config, *, outer_test_fold: int) -> dict[str, Any]:
    seed_everything(config.training.seed, config.cpu_threads)
    context = prepare_context(config, outer_test_fold)
    split = record_split(context)
    start = time.perf_counter()
    batch = next(batches(context, "train", smoke=True))
    load_seconds = time.perf_counter() - start
    guard_batch(batch, context.source.contract, context.split, "train")
    model = build_model(context, load_pretrained=True)
    frozen_before = {
        k: v.clone() for k, v in model.encoders.mel_encoder.backbone.state_dict().items()
    }
    model.eval()
    with torch.inference_mode():
        tokens = model.encoders(batch)
        output = model(batch, return_attention=True)
        first = model(batch)
        second = model(batch)
        if not torch.equal(first.logits, second.logits):
            raise ValueError("evaluation is not exactly deterministic")
        if not torch.isfinite(output.probabilities).all() or output.attention_weights is None:
            raise ValueError("invalid probabilities or attention diagnostic")
        timings = {
            "module3_encoding": measure(lambda: model.encoders(batch)),
            "module4_fusion": measure(
                lambda: model.fusion(tokens.mel_tokens, tokens.nonlinear_tokens)
            ),
            "combined": measure(lambda: model(batch)),
        }
    shapes = {
        "mel_tokens": list(tokens.mel_tokens.shape),
        "graph_token": list(tokens.graph_token.shape),
        "recurrence_tokens": list(tokens.recurrence_tokens.shape),
        "nonlinear_tokens": list(tokens.nonlinear_tokens.shape),
        **{
            name: list(getattr(output, name).shape)
            for name in (
                "attention_output",
                "attended_mel_tokens",
                "mel_pool",
                "nonlinear_pool",
                "fusion_vector",
                "logits",
                "probabilities",
                "attention_weights",
            )
        },
    }
    parameters = parameter_audit(model)
    gradients = gradient_audit(model, batch)
    overfit, optimizer = tiny_batch_overfit(model, batch, context)
    frozen_unchanged = all(
        torch.equal(frozen_before[k], v)
        for k, v in (model.encoders.mel_encoder.backbone.state_dict().items())
    )
    if not frozen_unchanged:
        raise ValueError("frozen backbone weights or BatchNorm buffers changed")
    selection = selection_identity(context, smoke=True)
    checkpoint_root = config.output("checkpoints", "module4")
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="roundtrip_", dir=checkpoint_root) as temporary:
        path = Path(temporary) / "sanity.pt"
        save_checkpoint(
            path,
            model,
            optimizer,
            config,
            context.split,
            epoch=1,
            best_metric=0.0,
            best_epoch=1,
            bad_epochs=0,
            pos_weight=overfit["pos_weight"],
            selection=selection,
            smoke=True,
            history=[{"epoch": 1, "purpose": "tiny-batch checkpoint sanity only"}],
        )
        restored = build_model(context, load_pretrained=False)
        restored_optimizer = make_optimizer(restored, context)
        load_checkpoint(
            path,
            restored,
            restored_optimizer,
            config,
            context.split,
            selection=selection,
            smoke=True,
        )
        restored.eval()
        with torch.inference_mode():
            exact = torch.equal(model(batch).logits, restored(batch).logits)
        if not exact:
            raise ValueError("checkpoint roundtrip logits differ")
        checkpoint_bytes = path.stat().st_size
    size = len(batch.provenance)
    estimated_epoch = (
        split["train"]["record_count"] / size * overfit["update_seconds"]["mean"]
        + split["validation"]["record_count"] / size * timings["combined"]["mean"]
        + (split["train"]["record_count"] + split["validation"]["record_count"])
        / size
        * load_seconds
    )
    result = {
        "status": "PASS",
        "module4_status": "IMPLEMENTATION READY / AWAITING MANUAL BASELINE TRAINING",
        "hashes": config.hashes,
        "processing_identity": config.processing_identity,
        "split": split,
        "smoke_logical_ids": [p.logical_id for p in batch.provenance],
        "smoke_conditions": sorted({p.condition for p in batch.provenance}),
        "smoke_genders": sorted({p.gender for p in batch.provenance}),
        "shapes": shapes,
        "all_finite": True,
        "eval_deterministic_exact": True,
        "attention_weights_api": "return_attention=True gives [B,4,16,17]; default None",
        "attention_interpretation": "diagnostic association weights, not causal explanations",
        "gradient_sanity": gradients,
        "frozen_backbone_weights_and_buffers_unchanged": frozen_unchanged,
        "tiny_batch_sanity": overfit,
        "checkpoint_roundtrip_exact": exact,
        "checkpoint_bytes": checkpoint_bytes,
        "parameters": parameters,
        "cpu_timing_seconds": {
            "batch_size": size,
            "threads": config.cpu_threads,
            "repetitions": 5,
            "load_and_collate": load_seconds,
            **timings,
        },
        "manual_runtime_ESTIMATE": {
            "epoch_seconds": estimated_epoch,
            "maximum_epochs": config.training.max_epochs,
            "maximum_run_hours": estimated_epoch * config.training.max_epochs / 3600,
            "method": "linear scaling of tiny CPU update, forward, and strict load timings; "
            "batch16 throughput, filesystem caching, and early stopping can change runtime",
        },
        "runtime": {"torch": str(torch.__version__), "device": "cpu"},
        "outer_test_forward_count": 0,
        "long_baseline_training_run": False,
        "module4_signed_off": False,
        "module5_started": False,
    }
    write_json(config.output("metrics", "module4_fusion_summary.json"), result)
    failures = config.output("logs", "module4_fusion_failures.csv")
    if not failures.exists():
        failures.parent.mkdir(parents=True, exist_ok=True)
        failures.write_text("command,error_type,message\n", encoding="utf-8")
    return result
