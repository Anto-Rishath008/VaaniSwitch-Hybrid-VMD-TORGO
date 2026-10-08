"""Report a completed C2 panel from retained evidence; no training or model construction.

This external helper is outside ExperimentContext.code_hash, which hashes only
src/vaaniswitch/m[012345]*/*.py. It never changes the frozen package, protocol,
ledger, selections, or checkpoints. --save publishes only a separate report.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import torch

from vaaniswitch.m5_experiments.checkpoint import checked_generation, validate_progress
from vaaniswitch.m5_experiments.config import CANDIDATES, ExperimentConfig
from vaaniswitch.m5_experiments.data import pos_weight
from vaaniswitch.m5_experiments.ledger import experiment_id
from vaaniswitch.m5_experiments.reporting import read_json, write_json
from vaaniswitch.m5_experiments.search import InnerResult, choose_candidate, metric_key
from vaaniswitch.m5_experiments.source import ExperimentContext, prepare_experiments, require_frozen
from vaaniswitch.m5_experiments.splits import Role, record_hash
from vaaniswitch.m5_experiments.trainer import run_identity

ROOT = Path(__file__).resolve().parents[1]
RULE = [
    "macro_f1",
    "balanced_accuracy",
    "valid_roc_auc",
    "lower_validation_loss",
    "fewer_trainable_parameters",
    "ascending_candidate_id",
]
METRICS = [
    "loss",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "precision",
    "sensitivity",
    "specificity",
    "roc_auc",
    "TN",
    "FP",
    "FN",
    "TP",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_runs(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise ValueError("Phase 5B has no production ledger")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        rows = list(connection.execute("SELECT id,identity,state,data,claim FROM runs"))
        outer_events = connection.execute(
            "SELECT COUNT(*) FROM events WHERE event LIKE '%OUTER%'"
        ).fetchone()[0]
    finally:
        connection.close()
    require(outer_events == 0, "Outer activity exists; this is a Phase 5B-only report")
    return {
        row[0]: {
            "identity": json.loads(row[1]),
            "state": row[2],
            "data": json.loads(row[3]),
            "claim": row[4],
        }
        for row in rows
    }


def decisive_comparisons(winner: InnerResult, results: list[InnerResult]) -> list[dict[str, Any]]:
    comparisons = []
    winning_key = metric_key(winner.metrics, winner.trainable_parameters)
    for other in results:
        if other.candidate_id == winner.candidate_id:
            continue
        other_key = metric_key(other.metrics, other.trainable_parameters)
        decisive = next(
            (i for i, (a, b) in enumerate(zip(winning_key, other_key, strict=True)) if a != b), 5
        )
        comparisons.append(
            {
                "compared_experiment_id": other.experiment_id,
                "candidate_id": other.candidate_id,
                "first_decisive_field": RULE[decisive],
                "winner_ordering_key": list(winning_key),
                "other_ordering_key": list(other_key),
                "note": "maximize numeric key; exact numeric ties use ascending candidate ID",
            }
        )
    return comparisons


def panel_report(context: ExperimentContext, fold: int) -> dict[str, Any]:
    require_frozen(context)
    config, split = context.config, context.split(fold)
    runs = read_runs(config.output("experiments/ledger.sqlite3"))
    selected_path = config.output(f"experiments/selection/fold{fold}_c2.json")
    require(selected_path.exists(), "Fold selection missing; finish the existing search command")
    selected = read_json(selected_path)
    results, candidates = [], []
    expected_weight = pos_weight(context, split, "inner_train", smoke=False)
    for candidate in CANDIDATES:
        run_id = experiment_id(config, fold, "C2", candidate.id)
        require(run_id in runs, f"Missing candidate: {run_id}")
        run = runs[run_id]
        require(
            run["state"] in {"INNER_COMPLETE", "SELECTED"} and run["claim"] is None,
            f"Candidate is incomplete or beyond Phase 5B: {run_id}",
        )
        expected_identity = run_identity(context, split, "C2", candidate, run_id, smoke=False)
        require(run["identity"] == expected_identity, f"Candidate identity mismatch: {run_id}")
        directory = config.output(f"experiments/runs/{run_id}/inner")
        checkpoint = checked_generation(directory, read_json(directory / "latest.json"))
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        identity, progress = payload["identity"], payload["progress"]
        require(payload["schema"] == "module5_checkpoint_v1", "Checkpoint schema mismatch")
        require(
            all(identity[k] == value for k, value in expected_identity.items())
            and identity["stage"] == "inner"
            and identity["selected_inner_configuration"] is None,
            "Checkpoint identity mismatch",
        )
        validate_progress(progress)
        require(progress["pos_weight"] == expected_weight, "pos_weight is not inner-train-only")
        result = InnerResult(**run["data"]["inner_result"])
        require(result.identity == progress["best_inner"], "Ledger/checkpoint best result mismatch")
        require(
            result.experiment_id == run_id and result.candidate_id == candidate.id,
            "Cross-candidate result mismatch",
        )
        epoch_roles: tuple[tuple[str, Role], ...] = (
            ("training", "inner_train"), ("validation", "inner_validation")
        )
        for epoch in progress["history"]:
            for key, role in epoch_roles:
                data = epoch[key]
                expected_records = split.records(context.predecessor.source.contract, role)
                require(
                    data["role"] == role
                    and data["record_hash"] == record_hash(expected_records)
                    and data["record_count"] == len(expected_records),
                    "Unexpected epoch role/records",
                )
            require(epoch["validation"]["threshold"] == 0.5, "Threshold drift")
        require(
            result.metrics == progress["history"][result.epoch - 1]["validation"],
            "Selected epoch metrics mismatch",
        )
        best_epoch = max(
            progress["history"],
            key=lambda h: metric_key(h["validation"], result.trainable_parameters),
        )
        require(best_epoch["epoch"] == result.epoch, "Best epoch differs from frozen metric rule")
        require(
            all(key in result.metrics for key in METRICS), "Required validation metrics missing"
        )
        training_seconds = sum(h["training"]["elapsed_seconds"] for h in progress["history"])
        validation_seconds = sum(h["validation"]["elapsed_seconds"] for h in progress["history"])
        require(
            training_seconds == run["data"]["inner_training_seconds"], "Training timing mismatch"
        )
        results.append(result)
        candidates.append(
            {
                "experiment_id": run_id,
                "candidate_id": candidate.id,
                "candidate": expected_identity["candidate"],
                "outer_fold": fold,
                "protocol_hash": config.config_hash,
                "code_hash": context.code_hash,
                "split_hash": split.split_hash,
                "seed": 42,
                "inner_training_speakers": list(split.speakers("inner_train")),
                "inner_validation_speakers": list(split.speakers("inner_validation")),
                "train_only_pos_weight": expected_weight,
                "epochs_completed": progress["epoch"],
                "best_epoch": result.epoch,
                "best_validation": {k: result.metrics[k] for k in METRICS},
                "training_seconds": training_seconds,
                "validation_seconds": validation_seconds,
                "measured_epoch_wall_seconds": training_seconds + validation_seconds,
                "timing_scope": (
                    "sum of recorded epoch stages; excludes setup and interrupted unsaved work"
                ),
                "checkpoint_path": checkpoint.relative_to(config.repo_root).as_posix(),
                "checkpoint_kind": (
                    "latest resumable state; selected best epoch is retained as metadata"
                ),
                "ledger_status": run["state"],
                "outer_model_forward_count": 0,
                "outer_count_evidence": (
                    "frozen inner-only code, matching epoch roles/record hashes, "
                    "no outer claim/events; derived audit, "
                    "not a separately instrumented runtime counter"
                ),
            }
        )
    winner = choose_candidate(
        results,
        split_hash=split.split_hash,
        validation_record_hash=record_hash(
            split.records(context.predecessor.source.contract, "inner_validation")
        ),
        candidate_ids={c.id for c in CANDIDATES},
    )
    require(
        selected == winner.identity, "Selection file differs from recomputed four-candidate winner"
    )
    winner_run = runs[winner.experiment_id]
    require(
        winner_run["state"] == "SELECTED" and winner_run["data"]["selection"] == selected,
        "Selection ledger mismatch",
    )
    require_frozen(context)
    return {
        "status": "PASS",
        "phase": "5B",
        "outer_fold": fold,
        "protocol_hash": config.config_hash,
        "frozen_code_hash": context.code_hash,
        "candidates": candidates,
        "selection": selected,
        "compared_experiment_ids": [r.experiment_id for r in results],
        "tie_break_order": RULE,
        "tie_break_reasoning": decisive_comparisons(winner, results),
        "outer_model_forward_count": 0,
        "module5_signed_off": False,
        "module6_started": False,
        "helper_scope": (
            "read-only retained-evidence comparison; no model instantiation/forward/training"
        ),
    }


def publish(path: Path, report: dict[str, Any]) -> None:
    if path.exists():
        require(read_json(path) == report, "Existing report differs; preserve it for review")
    else:
        write_json(path, report, immutable=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-fold", required=True, type=int, choices=[1, 2, 3])
    parser.add_argument("--save", action="store_true", help="publish a separate immutable report")
    args = parser.parse_args(argv)
    try:
        config = ExperimentConfig.from_yaml(
            ROOT / "configs/experiments/module5/protocol.yaml", repo_root=ROOT
        )
        report = panel_report(prepare_experiments(config), args.outer_fold)
        if args.save:
            publish(
                config.output(f"experiments/reports/fold{args.outer_fold}_phase5b.json"), report
            )
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(json.dumps({"status": "NOT READY", "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
