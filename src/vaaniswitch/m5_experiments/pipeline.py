"""Explicit manual phases; completed experiments are reused without further training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vaaniswitch.m5_experiments.config import ABLATIONS, CANDIDATES, Candidate
from vaaniswitch.m5_experiments.ledger import Ledger, experiment_id
from vaaniswitch.m5_experiments.reporting import read_json, write_json
from vaaniswitch.m5_experiments.search import InnerResult, choose_candidate
from vaaniswitch.m5_experiments.source import ExperimentContext, require_frozen
from vaaniswitch.m5_experiments.splits import record_hash
from vaaniswitch.m5_experiments.trainer import run_identity, train_stage


def run_inner(
    context: ExperimentContext,
    ledger: Ledger,
    fold: int,
    ablation: str,
    candidate: Candidate,
    *,
    smoke: bool = False,
    resume: bool = False,
    retry_reason: str | None = None,
) -> InnerResult:
    if not smoke:
        require_frozen(context)
    split = context.split(fold)
    run_id = experiment_id(context.config, fold, ablation, candidate.id, smoke=smoke)
    with ledger.lock(run_id):
        ledger.plan(run_id, run_identity(context, split, ablation, candidate, run_id, smoke=smoke))
        if retry_reason:
            if not resume:
                raise ValueError("failure recovery requires explicit --resume")
            ledger.retry_failed(run_id, retry_reason)
        run = ledger.get(run_id)
        if run["state"] in {
            "INNER_COMPLETE",
            "SELECTED",
            "FINAL_TRAIN_COMPLETE",
            "OUTER_EVALUATED",
        }:
            return InnerResult(**run["data"]["inner_result"])
        if run["state"] not in {"PLANNED", "RUNNING"}:
            raise ValueError("failed/invalidated experiment requires explicit disposition")
        if run["state"] == "RUNNING" and not resume:
            raise ValueError("interrupted experiment requires explicit --resume")
        if run["state"] == "PLANNED":
            ledger.transition(run_id, "RUNNING")
        try:
            report = train_stage(
                context, split, ablation, candidate, run_id, smoke=smoke, resume=resume
            )
            result = InnerResult(**report["progress"]["best_inner"])
            ledger.transition(
                run_id,
                "INNER_COMPLETE",
                {
                    "inner_result": result.identity,
                    "inner_training_seconds": report["training_seconds"],
                    "parameter_counts": report["parameter_counts"],
                },
            )
            return result
        except Exception as exc:
            ledger.transition(run_id, "FAILED", {"reason": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            ledger.export()


def selection_path(context: ExperimentContext, fold: int) -> Path:
    return context.config.output(f"experiments/selection/fold{fold}_c2.json")


def search_c2(
    context: ExperimentContext, fold: int, *, resume: bool = False, retry_reason: str | None = None
) -> dict[str, Any]:
    require_frozen(context)
    ledger = Ledger(context.config)
    split = context.split(fold)
    with ledger.lock(f"search_fold{fold}"):
        path = selection_path(context, fold)
        if path.exists():
            selection = read_json(path)
            retained = []
            for candidate in CANDIDATES:
                candidate_run = ledger.get(experiment_id(context.config, fold, "C2", candidate.id))
                if candidate_run["state"] not in {
                    "INNER_COMPLETE",
                    "SELECTED",
                    "FINAL_TRAIN_COMPLETE",
                    "OUTER_EVALUATED",
                }:
                    raise ValueError("selection recovery requires all completed inner candidates")
                retained.append(InnerResult(**candidate_run["data"]["inner_result"]))
            expected = choose_candidate(
                retained,
                split_hash=split.split_hash,
                validation_record_hash=record_hash(
                    split.records(context.predecessor.source.contract, "inner_validation")
                ),
                candidate_ids={c.id for c in CANDIDATES},
            )
            if selection != expected.identity:
                raise ValueError("selection recovery differs from the retained inner winner")
            run = ledger.get(selection["experiment_id"])
            if run["state"] == "INNER_COMPLETE":
                ledger.transition(run["id"], "SELECTED", {"selection": selection})
            elif run["data"].get("selection") != selection or run["state"] in {
                "FAILED",
                "INVALIDATED",
            }:
                raise ValueError("C2 selection/ledger mismatch")
            return selection
        results = []
        for candidate in CANDIDATES:
            run_id = experiment_id(context.config, fold, "C2", candidate.id)
            failed = any(r["id"] == run_id and r["state"] == "FAILED" for r in ledger.all_runs())
            results.append(
                run_inner(
                    context,
                    ledger,
                    fold,
                    "C2",
                    candidate,
                    resume=resume,
                    retry_reason=retry_reason if failed else None,
                )
            )
        winner = choose_candidate(
            results,
            split_hash=split.split_hash,
            validation_record_hash=record_hash(
                split.records(context.predecessor.source.contract, "inner_validation")
            ),
            candidate_ids={c.id for c in CANDIDATES},
        )
        selection = winner.identity
        write_json(path, selection, immutable=True)
        ledger.transition(winner.experiment_id, "SELECTED", {"selection": selection})
        ledger.export()
        return selection


def prepare_ablations(
    context: ExperimentContext, fold: int, *, resume: bool = False, retry_reason: str | None = None
) -> dict[str, Any]:
    require_frozen(context)
    selected = read_json(selection_path(context, fold))
    candidate = context.config.candidate(selected["candidate_id"])
    ledger = Ledger(context.config)
    c2 = ledger.get(selected["experiment_id"])
    if c2["data"].get("selection") != selected:
        raise ValueError("C2 selection file differs from the frozen ledger selection")
    if c2["state"] not in {"SELECTED", "FINAL_TRAIN_COMPLETE", "OUTER_EVALUATED"}:
        raise ValueError("complete C2 selection first")
    selections = {"C2": selected}
    for ablation in ABLATIONS:
        if ablation == "C2":
            continue
        run_id = experiment_id(context.config, fold, ablation, candidate.id)
        failed = any(r["id"] == run_id and r["state"] == "FAILED" for r in ledger.all_runs())
        result = run_inner(
            context,
            ledger,
            fold,
            ablation,
            candidate,
            resume=resume,
            retry_reason=retry_reason if failed else None,
        )
        run = ledger.get(result.experiment_id)
        if run["state"] == "INNER_COMPLETE":
            ledger.transition(result.experiment_id, "SELECTED", {"selection": result.identity})
        selections[ablation] = result.identity
    path = context.config.output(f"experiments/selection/fold{fold}_ablations.json")
    document = {
        "outer_fold": fold,
        "config_hash": context.config.config_hash,
        "common_candidate": candidate.id,
        "selections": selections,
    }
    if path.exists():
        if read_json(path) != document:
            raise ValueError("ablation selections changed after freeze")
    else:
        write_json(path, document, immutable=True)
    ledger.export()
    return document


def retrain_fold(
    context: ExperimentContext, fold: int, *, resume: bool = False, retry_reason: str | None = None
) -> dict[str, Any]:
    require_frozen(context)
    bundle = read_json(context.config.output(f"experiments/selection/fold{fold}_ablations.json"))
    ledger = Ledger(context.config)
    for ablation in ABLATIONS:
        selection = bundle["selections"][ablation]
        run_id = selection["experiment_id"]
        candidate = context.config.candidate(selection["candidate_id"])
        with ledger.lock(run_id):
            run = ledger.get(run_id)
            if run["state"] == "FAILED" and retry_reason:
                if not resume:
                    raise ValueError("failure recovery requires explicit --resume")
                ledger.retry_failed(run_id, retry_reason)
                run = ledger.get(run_id)
            if run["state"] in {"FINAL_TRAIN_COMPLETE", "OUTER_EVALUATED"}:
                continue
            if run["state"] != "SELECTED" or run["data"]["selection"] != selection:
                raise ValueError("retraining requires immutable inner selection")
            try:
                report = train_stage(
                    context,
                    context.split(fold),
                    ablation,
                    candidate,
                    run_id,
                    smoke=False,
                    resume=resume,
                    selection=selection,
                )
                if report["progress"]["epoch"] != selection["epoch"]:
                    raise ValueError("final retrain did not use selected epoch count")
                ledger.transition(
                    run_id,
                    "FINAL_TRAIN_COMPLETE",
                    {
                        "final_checkpoint": report["checkpoint"],
                        "final_identity": report["identity"],
                        "final_epoch": report["progress"]["epoch"],
                        "final_training_seconds": report["training_seconds"],
                        "final_pos_weight": report["progress"]["pos_weight"],
                        "parameter_counts": report["parameter_counts"],
                    },
                )
            except Exception as exc:
                ledger.transition(run_id, "FAILED", {"reason": f"{type(exc).__name__}: {exc}"})
                raise
            finally:
                ledger.export()
    return {"outer_fold": fold, "status": "FINAL_TRAIN_COMPLETE", "outer_evaluations": 0}
