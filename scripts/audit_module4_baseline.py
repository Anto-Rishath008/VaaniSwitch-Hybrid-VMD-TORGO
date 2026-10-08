"""Reproduce the completed Module 4 baseline on development validation only.

This read-only model audit never calls training, optimizer.step, or a test loader.
Original training/validation reports and checkpoints are preserved. Predictions
stay in memory; the output contains aggregate evidence and validation provenance.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision.models import resnet18

from vaaniswitch.m4_fusion.checkpoint import load_checkpoint
from vaaniswitch.m4_fusion.config import Module4Config, semantic_hash
from vaaniswitch.m4_fusion.losses import batch_labels, make_loss
from vaaniswitch.m4_fusion.pipeline import build_model, selection_identity
from vaaniswitch.m4_fusion.reporting import read_json, write_json
from vaaniswitch.m4_fusion.reproducibility import seed_everything
from vaaniswitch.m4_fusion.source import prepare_context
from vaaniswitch.m4_fusion.splits import guard_batch, split_audit
from vaaniswitch.m4_fusion.trainer import batches, make_optimizer, run_epoch

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(path: Path) -> dict[str, Any]:
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
    }


def confusion_metrics(tn: int, fp: int, fn: int, tp: int) -> dict[str, Any]:
    """Independent direct formulas, without the production metric implementation."""
    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "record_count": tn + fp + fn + tp,
        "accuracy": (tn + tp) / (tn + fp + fn + tp),
        "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2,
        "macro_f1": (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn)) / 2,
        "precision": tp / (tp + fp),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
    }


def compare(actual: dict[str, Any], expected: dict[str, Any]) -> float:
    differences = []
    for key, value in actual.items():
        if isinstance(value, float):
            differences.append(abs(value - expected[key]))
            assert differences[-1] <= 1e-6, (key, value, expected[key])
        else:
            assert value == expected[key], (key, value, expected[key])
    return max(differences, default=0.0)


def check_finite(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        assert bool(torch.isfinite(value).all())
    elif isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            check_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            check_finite(item)


def main() -> None:
    config = Module4Config.from_yaml(ROOT / "configs/models/fusion.yaml")
    seed_everything(42, config.cpu_threads)
    context = prepare_context(config, 1)
    training = read_json(ROOT / "artifacts/metrics/module4_baseline_training.json")
    previous_validation = read_json(ROOT / "artifacts/metrics/module4_baseline_validation.json")
    audit = split_audit(context.source.contract, context.split)
    assert context.split.identity == training["split"] == previous_validation["split"]
    assert config.hashes == training["hashes"] == previous_validation["hashes"]
    assert context.split.split_hash == training["split_hash"]
    assert previous_validation["status"] == "PASS"
    records = context.source.contract.records
    partitions = {
        role: [r for r in records if r.speaker_id in context.split.identity[f"{role}_speakers"]]
        for role in ("train", "validation", "test")
    }
    assert [len(partitions[r]) for r in partitions] == [4760, 1094, 2819]
    assert len({(r.speaker_id, r.session_id, r.prompt_id) for r in records}) == len(records)
    overlaps = {}
    for field in ("logical_id", "primary_audio_path", "primary_audio_sha256"):
        groups = [{getattr(r, field) for r in rows} - {None} for rows in partitions.values()]
        overlaps[field] = sum(len(groups[i] & groups[j]) for i, j in ((0, 1), (0, 2), (1, 2)))
        assert overlaps[field] == 0
    counts = Counter(r.condition.value for r in partitions["train"])
    pos_weight = counts["control"] / counts["dysarthric"]
    assert counts == {"control": 3306, "dysarthric": 1454}
    assert pos_weight == training["training_pos_weight"]
    audit.update(training_pos_weight=pos_weight, logical_record_overlap=overlaps)
    history = training["history"]
    assert len(history) == training["epochs_completed"] == 6
    check_finite(history)
    best_metric, bad_epochs = -1.0, 0
    for epoch, row in enumerate(history, 1):
        assert row["epoch"] == epoch and row["train"]["record_count"] == 4760
        metric = row["validation"]
        assert metric["TN"] + metric["FP"] == 684 and metric["TP"] + metric["FN"] == 410
        compare(confusion_metrics(*(metric[k] for k in ("TN", "FP", "FN", "TP"))), metric)
        assert 0 <= metric["roc_auc"] <= 1 and metric["threshold"] == 0.5
        assert row["train"]["loss"] >= 0 and metric["loss"] >= 0
        improved = metric["macro_f1"] > best_metric + config.training.min_delta
        bad_epochs = 0 if improved else bad_epochs + 1
        best_metric = metric["macro_f1"] if improved else best_metric
        assert row["best_so_far"] == improved and row["bad_epochs"] == bad_epochs
        assert row["best_validation_macro_f1"] == best_metric
        assert row["learning_rate"] == config.training.learning_rate
    assert training["early_stopped"] and bad_epochs == config.training.patience == 5
    assert training["best_epoch"] == 1 and training["best_validation_macro_f1"] == best_metric
    logs = [
        json.loads(line)
        for line in (ROOT / "artifacts/logs/module4_training.log").read_text().splitlines()
    ]
    baseline = [r for r in logs if r["mode"] == "baseline"]
    assert len(baseline) == 7 and baseline[0]["event"] == "run_start"
    assert baseline[0]["start_epoch"] == 1 and baseline[0]["split"] == context.split.identity
    assert baseline[0]["selection"] == selection_identity(context, smoke=False)
    assert baseline[0]["pos_weight"] == pos_weight
    assert [
        {k: v for k, v in r.items() if k not in ("event", "mode")} for r in baseline[1:]
    ] == history
    for row in logs:
        if row["event"] == "run_start":
            assert row["split"] == context.split.identity
            assert row["selection"] == selection_identity(context, smoke=row["mode"] == "smoke")
            assert row["outer_test_forward_count"] == 0
    with (ROOT / "artifacts/logs/module4_fusion_failures.csv").open(newline="") as stream:
        failures = list(csv.DictReader(stream))
    assert failures == []
    print("PASS: live split, train-only weighting, six epochs, early stopping, logs", flush=True)

    # Local, hash-verified original ImageNet file; no network and no reference-model forward.
    original = Path(torch.hub.get_dir()) / "checkpoints/resnet18-f37072fd.pth"
    original_fingerprint = fingerprint(original)
    assert original_fingerprint["sha256"].startswith("f37072fd")
    reference = resnet18(weights=None)
    reference.load_state_dict(torch.load(original, map_location="cpu", weights_only=True))
    reference_backbone = torch.nn.Sequential(*list(reference.children())[:-2]).state_dict()
    selection = selection_identity(context, smoke=False)
    checkpoint_audits = {}
    states = {}
    for name, expected_epoch in (("last", 6), ("best", 1)):
        path = Path(training[f"{name}_checkpoint"])
        model = build_model(context, load_pretrained=False)
        optimizer = make_optimizer(model, context)
        state = load_checkpoint(
            path, model, optimizer, config, context.split, selection=selection, smoke=False
        )
        check_finite(state)
        assert state["epoch"] == expected_epoch and state["best_epoch"] == 1
        assert state["history"] == history[:expected_epoch] and state["pos_weight"] == pos_weight
        assert state["best_metric"] == best_metric and state["seed"] == 42
        backbone = model.encoders.mel_encoder.backbone
        assert all(torch.equal(v, reference_backbone[k]) for k, v in backbone.state_dict().items())
        names = {id(p): n for n, p in model.named_parameters()}
        optimizer_names = [names[id(p)] for g in optimizer.param_groups for p in g["params"]]
        assert all(not n.startswith("encoders.mel_encoder.backbone.") for n in optimizer_names)
        steps = sorted({int(v["step"].item()) for v in state["optimizer_state"]["state"].values()})
        assert steps == [math.ceil(4760 / 16) * expected_epoch]
        # Tensor locations are exact model keys, exact AdamW fields, or the RNG state.
        assert all(
            not isinstance(v, torch.Tensor)
            for k, v in state.items()
            if k not in ("model_state", "optimizer_state", "rng")
        )
        checkpoint_audits[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            **fingerprint(path),
            "size_mib": path.stat().st_size / 2**20,
            "fields": sorted(state),
            "epoch": state["epoch"],
            "best_epoch": state["best_epoch"],
            "model_state_tensors": len(state["model_state"]),
            "backbone_state_tensors_equal_to_imagenet": len(reference_backbone),
            "optimizer_parameter_names": optimizer_names,
            "optimizer_steps": steps,
            "frozen_parameters": sum(p.numel() for p in backbone.parameters()),
            "metadata_and_all_tensors_valid": True,
            "embedded_dataset_arrays": False,
            "requires_grad_false_and_grad_none": all(
                not p.requires_grad and p.grad is None for p in backbone.parameters()
            ),
        }
        states[name] = state
    print(
        "PASS: both checksummed checkpoints; frozen ImageNet weights and buffers exact", flush=True
    )

    # The loop is the production validation path; hooks independently enforce and measure scope.
    seen: list[str] = []
    loaded: list[str] = []
    labels: list[int] = []
    probabilities: list[float] = []
    logits: list[float] = []
    original_load = context.source.load

    def guarded_load(logical_id: str) -> Any:
        assert context.source.contract.record_by_id[logical_id].speaker_id in {"M03", "MC02"}
        loaded.append(logical_id)
        return original_load(logical_id)

    def pre_forward(module: Any, args: Any) -> None:
        batch = args[0]
        guard_batch(batch, context.source.contract, context.split, "validation")
        seen.extend(p.logical_id for p in batch.provenance)
        assert all(p.speaker_id in {"M03", "MC02"} for p in batch.provenance)
        assert all(p.evaluation_fold != 1 for p in batch.provenance)

    def post_forward(module: Any, args: Any, output: Any) -> None:
        labels.extend(int(v) for v in batch_labels(args[0]).tolist())
        probabilities.extend(output.probabilities.detach().tolist())
        logits.extend(output.logits.detach().tolist())

    context.source.load = guarded_load
    pre = model.register_forward_pre_hook(pre_forward)
    post = model.register_forward_hook(post_forward)
    validation = run_epoch(model, context, "validation", pos_weight=pos_weight, smoke=False)
    post.remove()
    expected_ids = sorted(r.logical_id for r in partitions["validation"])
    assert seen == loaded == expected_ids
    y, p, z = np.asarray(labels), np.asarray(probabilities), np.asarray(logits)
    assert np.isfinite(p).all() and np.isfinite(z).all() and ((p >= 0) & (p <= 1)).all()
    predicted = p >= 0.5
    independent = confusion_metrics(
        int(((y == 0) & ~predicted).sum()),
        int(((y == 0) & predicted).sum()),
        int(((y == 1) & ~predicted).sum()),
        int(((y == 1) & predicted).sum()),
    )
    positive, negative = p[y == 1, None], p[y == 0][None, :]
    independent["roc_auc"] = float(((positive > negative) + 0.5 * (positive == negative)).mean())
    # Stable double-precision BCE computed independently, with the actual float32 loss weight.
    used_weight = float(make_loss(pos_weight).pos_weight.item())
    independent["loss"] = float(
        np.where(y == 1, used_weight * np.logaddexp(0, -z), np.logaddexp(0, z)).mean()
    )
    differences = {
        "independent_vs_saved_max_abs": compare(independent, history[0]["validation"]),
        "independent_vs_production_max_abs": compare(independent, validation),
        "independent_vs_user_validation_max_abs": compare(
            independent, previous_validation["validation"]
        ),
    }
    fixed = next(batches(context, "validation", smoke=True))
    with torch.no_grad():
        first, second = model(fixed), model(fixed)
    assert torch.equal(first.logits, second.logits)
    assert torch.equal(first.probabilities, second.probabilities)
    assert torch.isfinite(first.logits).all() and torch.isfinite(first.probabilities).all()
    assert ((first.probabilities >= 0) & (first.probabilities <= 1)).all()
    pre.remove()
    train_times = [r["train"]["elapsed_seconds"] for r in history]
    val_times = [r["validation"]["elapsed_seconds"] for r in history]
    result = {
        "status": "PASS",
        "scope": "Module 4 completed baseline audit; development only",
        "module4_signed_off": False,
        "pending": "regression gates and final closure",
        "hashes": config.hashes,
        "split": audit,
        "history": history,
        "epochs_completed": 6,
        "best_epoch": 1,
        "early_stopped": True,
        "pos_weight_stored": pos_weight,
        "pos_weight_float32_used": used_weight,
        "checkpoint_audit": checkpoint_audits,
        "imagenet_reference": original_fingerprint,
        "saved_validation": history[0]["validation"],
        "reproduced_validation": validation,
        "independent_validation": independent,
        "metric_agreement": differences,
        "probability_range": [float(p.min()), float(p.max())],
        "all_history_checkpoint_logits_probabilities_finite": True,
        "deterministic_reload": {
            "exact": True,
            "batch_size": len(fixed.provenance),
            "logical_ids": [p.logical_id for p in fixed.provenance],
        },
        "no_peek": {
            "outer_test_forward_count": 0,
            "validation_unique_records": len(expected_ids),
            "validation_batches": math.ceil(len(expected_ids) / 16),
            "source_loads_including_fixed_batch": len(loaded),
            "forwarded_records_including_repeat_batch": len(seen),
            "validation_record_ids_sha256": semantic_hash(expected_ids),
            "forward_speakers": sorted({i.split("/")[0] for i in seen}),
            "baseline_counters_are_policy_assertions": True,
            "audit_uses_actual_source_and_model_hooks": True,
        },
        "failures": {
            "production_failure_rows": len(failures),
            "baseline_run_starts": 1,
            "resume_recovery_events": 0,
            "checkpoint_load_failures": 0,
            "audit_failed_records": 0,
            "audit_failed_batches": 0,
        },
        "timing_seconds": {
            "train_total": sum(train_times),
            "validation_total": sum(val_times),
            "measured_stage_total": sum(train_times) + sum(val_times),
            "train_mean": statistics.mean(train_times),
            "train_median": statistics.median(train_times),
            "train_min": min(train_times),
            "train_max": max(train_times),
            "validation_mean": statistics.mean(val_times),
            "full_command_overhead_measured": False,
        },
        "module5_started": False,
    }
    write_json(ROOT / "artifacts/metrics/module4_baseline_audit.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "independent_validation",
                    "metric_agreement",
                    "no_peek",
                    "timing_seconds",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
