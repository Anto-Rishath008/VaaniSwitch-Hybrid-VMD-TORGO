"""Development-only entry points, checkpoint selection, and bounded baseline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vaaniswitch.m4_fusion.checkpoint import (
    checkpoint_directory,
    load_checkpoint,
    save_checkpoint,
)
from vaaniswitch.m4_fusion.config import Module4Config, semantic_hash
from vaaniswitch.m4_fusion.losses import training_pos_weight
from vaaniswitch.m4_fusion.model import DysarthriaClassifier
from vaaniswitch.m4_fusion.reporting import log_event, write_json
from vaaniswitch.m4_fusion.reproducibility import restore_rng, seed_everything
from vaaniswitch.m4_fusion.source import DevelopmentContext, prepare_context
from vaaniswitch.m4_fusion.splits import split_audit
from vaaniswitch.m4_fusion.trainer import make_optimizer, run_epoch, selected_records


def build_model(context: DevelopmentContext, *, load_pretrained: bool) -> DysarthriaClassifier:
    return DysarthriaClassifier(context.encoder_config, load_pretrained=load_pretrained)


def selection_identity(context: DevelopmentContext, *, smoke: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for partition in ("train", "validation"):
        records = selected_records(context, partition, smoke=smoke)
        result[partition] = {
            "count": len(records),
            "logical_ids_sha256": semantic_hash([r.logical_id for r in records]),
        }
    return result


def audit_development_split(config: Module4Config, outer_test_fold: int) -> dict[str, Any]:
    context = prepare_context(config, outer_test_fold)
    return record_split(context)


def record_split(context: DevelopmentContext) -> dict[str, Any]:
    result = split_audit(context.source.contract, context.split)
    records = selected_records(context, "train", smoke=False)
    result.update(
        {
            "hashes": context.config.hashes,
            "training_pos_weight": training_pos_weight(records, context.split.train_speakers),
            "baseline_training_run": False,
        }
    )
    write_json(context.config.output("metrics", "module4_split_audit.json"), result)
    return result


def run_training(
    config: Module4Config,
    *,
    outer_test_fold: int,
    smoke: bool = False,
    dry_run: bool = False,
    resume: Path | None = None,
) -> dict[str, Any]:
    seed_everything(config.training.seed, config.cpu_threads)
    context = prepare_context(config, outer_test_fold)
    audit = record_split(context)  # Stored before model initialization or training.
    directory = checkpoint_directory(config, context.split, smoke=smoke)
    if dry_run:
        return {
            "status": "DRY_RUN_PASS",
            "split": audit,
            "checkpoint_directory": str(directory),
            "model_forwards": 0,
            "baseline_training_run": False,
        }
    if resume is None and any((directory / name).exists() for name in ("best.pt", "last.pt")):
        raise ValueError("run already has checkpoints; use --resume with its last.pt")
    if resume is not None and resume.resolve() != (directory / "last.pt").resolve():
        raise ValueError("resume requires this exact run's last.pt")
    model = build_model(context, load_pretrained=resume is None)
    optimizer = make_optimizer(model, context)
    train_records = selected_records(context, "train", smoke=smoke)
    pos_weight = training_pos_weight(train_records, context.split.train_speakers)
    selection = selection_identity(context, smoke=smoke)
    best_metric, best_epoch, bad_epochs, start_epoch = -1.0, 0, 0, 1
    history: list[dict[str, Any]] = []
    if resume is not None:
        state = load_checkpoint(
            resume,
            model,
            optimizer,
            config,
            context.split,
            selection=selection,
            smoke=smoke,
        )
        if state["pos_weight"] != pos_weight:
            raise ValueError("checkpoint training pos_weight mismatch")
        best_metric, best_epoch = state["best_metric"], state["best_epoch"]
        bad_epochs, start_epoch = state["bad_epochs"], state["epoch"] + 1
        history = state["history"]
        restore_rng(state["rng"])
    log_path = config.output("logs", "module4_training.log")
    log_event(
        log_path,
        {
            "event": "run_start",
            "mode": "smoke" if smoke else "baseline",
            "config_hash": config.config_hash,
            "split": context.split.identity,
            "selection": selection,
            "pos_weight": pos_weight,
            "start_epoch": start_epoch,
            "outer_test_forward_count": 0,
        },
    )
    max_epochs = 1 if smoke else config.training.max_epochs
    for epoch in range(start_epoch, max_epochs + 1):
        if bad_epochs >= config.training.patience:
            break
        train = run_epoch(
            model,
            context,
            "train",
            pos_weight=pos_weight,
            smoke=smoke,
            epoch=epoch,
            optimizer=optimizer,
        )
        validation = run_epoch(model, context, "validation", pos_weight=pos_weight, smoke=smoke)
        score = float(validation["macro_f1"])
        improved = score > best_metric + config.training.min_delta
        if improved:
            best_metric, best_epoch, bad_epochs = score, epoch, 0
        else:
            bad_epochs += 1
        row = {
            "epoch": epoch,
            "train": train,
            "validation": validation,
            "learning_rate": config.training.learning_rate,
            "best_so_far": improved,
            "best_validation_macro_f1": best_metric,
            "bad_epochs": bad_epochs,
        }
        history.append(row)
        log_event(log_path, {"event": "epoch", "mode": "smoke" if smoke else "baseline", **row})
        kwargs: dict[str, Any] = {
            "epoch": epoch,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "pos_weight": pos_weight,
            "selection": selection,
            "smoke": smoke,
            "history": history,
        }
        if improved:
            save_checkpoint(
                directory / "best.pt", model, optimizer, config, context.split, **kwargs
            )
        save_checkpoint(directory / "last.pt", model, optimizer, config, context.split, **kwargs)
    result = {
        "status": "SMOKE_TRAIN_PASS" if smoke else "BASELINE_TRAINING_FINISHED_PENDING_AUDIT",
        "mode": "smoke" if smoke else "baseline",
        "hashes": config.hashes,
        "split": context.split.identity,
        "split_hash": context.split.split_hash,
        "selection": selection,
        "training_pos_weight": pos_weight,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_metric,
        "history": history,
        "best_checkpoint": str(directory / "best.pt"),
        "last_checkpoint": str(directory / "last.pt"),
        "early_stopped": bad_epochs >= config.training.patience,
        "outer_test_forward_count": 0,
        "module4_signed_off": False,
        "module5_started": False,
        "scientific_note": "smoke metrics are pipeline diagnostics only"
        if smoke
        else ("development validation only; not final outer-fold performance"),
    }
    filename = "module4_training_smoke.json" if smoke else "module4_baseline_training.json"
    write_json(config.output("metrics", filename), result)
    return result


def validate_checkpoint(
    config: Module4Config,
    checkpoint: Path,
    *,
    outer_test_fold: int,
    smoke: bool = False,
) -> dict[str, Any]:
    seed_everything(config.training.seed, config.cpu_threads)
    context = prepare_context(config, outer_test_fold)
    model = build_model(context, load_pretrained=False)
    optimizer = make_optimizer(model, context)
    selection = selection_identity(context, smoke=smoke)
    state = load_checkpoint(
        checkpoint, model, optimizer, config, context.split, selection=selection, smoke=smoke
    )
    pos_weight = training_pos_weight(
        selected_records(context, "train", smoke=smoke), context.split.train_speakers
    )
    if state["pos_weight"] != pos_weight:
        raise ValueError("checkpoint training pos_weight mismatch")
    metrics = run_epoch(model, context, "validation", pos_weight=pos_weight, smoke=smoke)
    saved = state["history"][-1]["validation"]
    for key in (
        "loss",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "precision",
        "sensitivity",
        "specificity",
        "roc_auc",
        "confusion_matrix",
        "record_count",
    ):
        actual, expected = metrics[key], saved[key]
        if isinstance(actual, float) and isinstance(expected, float):
            matches = abs(actual - expected) <= 1e-6
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(f"recomputed validation metric differs from checkpoint: {key}")
    result = {
        "status": "PASS",
        "mode": "smoke" if smoke else "baseline",
        "checkpoint": str(checkpoint),
        "epoch": state["epoch"],
        "is_best_epoch": state["epoch"] == state["best_epoch"],
        "hashes": config.hashes,
        "split": context.split.identity,
        "validation": metrics,
        "saved_validation_metrics_match": True,
        "outer_test_forward_count": 0,
        "module4_signed_off": False,
        "module5_started": False,
    }
    filename = (
        "module4_checkpoint_smoke_validation.json"
        if smoke
        else ("module4_baseline_validation.json")
    )
    write_json(config.output("metrics", filename), result)
    return result
