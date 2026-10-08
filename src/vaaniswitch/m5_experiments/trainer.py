"""Inner selection and fresh fixed-epoch final training, with resumable RNG/AdamW state."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import torch

from vaaniswitch.m4_fusion.losses import batch_labels, make_loss
from vaaniswitch.m4_fusion.metrics import binary_metrics
from vaaniswitch.m4_fusion.reproducibility import restore_rng, seed_everything
from vaaniswitch.m5_experiments.ablation_models import AblationModel
from vaaniswitch.m5_experiments.checkpoint import load_checkpoint, save_checkpoint
from vaaniswitch.m5_experiments.config import SIGNED_HASHES, Candidate
from vaaniswitch.m5_experiments.data import ExperimentDataset, batches, pos_weight, selected_records
from vaaniswitch.m5_experiments.reporting import read_json
from vaaniswitch.m5_experiments.search import InnerResult, metric_key
from vaaniswitch.m5_experiments.source import ExperimentContext
from vaaniswitch.m5_experiments.splits import NestedSplit, Role, record_hash


def build_model(
    context: ExperimentContext, ablation: str, candidate: Candidate, *, load_pretrained: bool
) -> AblationModel:
    return AblationModel(
        context.predecessor.encoder_config, ablation, candidate, load_pretrained=load_pretrained
    )


def make_optimizer(model: AblationModel, candidate: Candidate) -> torch.optim.AdamW:
    model.verify_frozen_backbone()
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=candidate.learning_rate,
        weight_decay=candidate.weight_decay,
    )


def run_epoch(
    model: AblationModel,
    dataset: ExperimentDataset,
    *,
    weight: float,
    epoch: int,
    smoke: bool,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    fitting = dataset.role in {"inner_train", "development"}
    if dataset.role == "outer_test" or fitting != (optimizer is not None):
        raise ValueError("training/inner-validation epoch cannot accept outer evaluation")
    model.train(fitting)
    criterion = make_loss(weight)
    total_loss = 0.0
    seen: list[str] = []
    labels: list[int] = []
    scores: list[float] = []
    start = time.perf_counter()
    for batch in batches(dataset, epoch=epoch, smoke=smoke):
        dataset.guard(batch)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(fitting):
            output = model(batch)
            targets = batch_labels(batch)
            loss = criterion(output.logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            if optimizer is not None:
                loss.backward()
                model.verify_frozen_backbone()
                for name, p in model.named_parameters():
                    if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):
                        raise ValueError(f"missing/non-finite gradient: {name}")
                optimizer.step()
        total_loss += float(loss.detach()) * len(batch.provenance)
        seen.extend(p.logical_id for p in batch.provenance)
        if not fitting:
            labels.extend(int(v) for v in targets.tolist())
            scores.extend(output.probabilities.detach().tolist())
    if sorted(seen) != sorted(r.logical_id for r in dataset.records):
        raise ValueError("epoch must consume assigned records exactly once")
    result = {
        "loss": total_loss / len(seen),
        "record_count": len(seen),
        "record_hash": record_hash(dataset.records),
        "role": dataset.role,
        "elapsed_seconds": time.perf_counter() - start,
    }
    if not fitting:
        result.update(binary_metrics(labels, scores))
    return result


def run_identity(
    context: ExperimentContext,
    split: NestedSplit,
    ablation: str,
    candidate: Candidate,
    run_id: str,
    *,
    smoke: bool,
) -> dict[str, Any]:
    return {
        "experiment_id": run_id,
        "ablation": ablation,
        "outer_fold": split.outer_fold,
        "candidate": asdict(candidate),
        "seed": 42,
        "config_hash": context.config.config_hash,
        "code_hash": context.code_hash,
        "signed_hashes": SIGNED_HASHES,
        "split_hash": split.split_hash,
        "split": split.identity,
        "mode": "smoke" if smoke else "production",
        "threshold": 0.5,
        "torch": str(torch.__version__),
        "cpu_threads": context.config.cpu_threads,
        "pretrained_start": "ImageNet1K_V1_for_Mel;independent_branch_seeds_42_43_44_head45",
        "record_hashes": {
            role: record_hash(selected_records(context, split, role, smoke=smoke))
            for role in ("inner_train", "inner_validation", "development")
        },
    }


def train_stage(
    context: ExperimentContext,
    split: NestedSplit,
    ablation: str,
    candidate: Candidate,
    run_id: str,
    *,
    smoke: bool,
    resume: bool,
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final = selection is not None
    stage = "final" if final else "inner"
    prefix = "smoke" if smoke else "experiments"
    directory = context.config.output(f"{prefix}/runs/{run_id}/{stage}")
    exists = (directory / "latest.json").exists()
    if exists and not resume:
        raise ValueError("checkpoint exists; explicit resume required")
    identity = {
        **run_identity(context, split, ablation, candidate, run_id, smoke=smoke),
        "stage": stage,
        "selection_status": "SELECTED" if final else "RUNNING",
        "selected_inner_configuration": selection,
        "all_development_speakers": list(split.speakers("development")),
        "outer_test_speakers": list(split.speakers("outer_test")),
    }
    seed_everything(42, context.config.cpu_threads)
    # No inner checkpoint is ever loaded here when final=True. Resume loads only final/.
    model = build_model(context, ablation, candidate, load_pretrained=not exists)
    optimizer = make_optimizer(model, candidate)
    role: Role = "development" if final else "inner_train"
    weight = pos_weight(context, split, role, smoke=smoke)
    dataset = ExperimentDataset(context, split, role, ablation, smoke=smoke)
    validation = (
        None
        if final
        else ExperimentDataset(context, split, "inner_validation", ablation, smoke=smoke)
    )
    progress: dict[str, Any] = {
        "epoch": 0,
        "history": [],
        "bad_epochs": 0,
        "best_inner": None,
        "patience_best": 0.0,
        "pos_weight": weight,
    }
    if exists:
        saved = load_checkpoint(directory, model, optimizer, identity)
        progress = saved["progress"]
        if progress["pos_weight"] != weight:
            raise ValueError("resume class weight mismatch")
        restore_rng(saved["rng"])
    maximum = int(selection["epoch"]) if selection else (1 if smoke else 30)
    if not 1 <= maximum <= 30 or progress["epoch"] > maximum:
        raise ValueError("invalid selected final epoch count")
    counts = model.parameter_counts()
    for epoch in range(progress["epoch"] + 1, maximum + 1):
        if not final and progress["bad_epochs"] >= 5:
            break
        training = run_epoch(
            model, dataset, weight=weight, epoch=epoch, smoke=smoke, optimizer=optimizer
        )
        history = {"epoch": epoch, "training": training}
        if validation is not None:
            metrics = run_epoch(model, validation, weight=weight, epoch=epoch, smoke=smoke)
            if metrics["role"] != "inner_validation" or metrics["record_hash"] != record_hash(
                validation.records
            ):
                raise ValueError("early stopping requires inner-validation evidence only")
            result = InnerResult(
                run_id,
                candidate.id,
                split.split_hash,
                record_hash(validation.records),
                epoch,
                metrics,
                counts["trainable"],
            )
            best = progress["best_inner"]
            if best is None or metric_key(metrics, counts["trainable"]) > metric_key(
                best["metrics"], best["trainable_parameters"]
            ):
                progress["best_inner"] = result.identity
            significant = epoch == 1 or metrics["macro_f1"] > progress["patience_best"] + 1e-4
            progress["bad_epochs"] = 0 if significant else progress["bad_epochs"] + 1
            if significant:
                progress["patience_best"] = metrics["macro_f1"]
            history["validation"] = metrics
        progress["epoch"] = epoch
        progress["history"].append(history)
        save_checkpoint(directory, model, optimizer, identity, progress)
        print(f"{run_id} {stage} epoch {epoch}/{maximum} saved", flush=True)
    return {
        "identity": identity,
        "progress": progress,
        "parameter_counts": counts,
        "checkpoint": read_json(directory / "latest.json"),
        "training_seconds": sum(h["training"]["elapsed_seconds"] for h in progress["history"]),
    }
