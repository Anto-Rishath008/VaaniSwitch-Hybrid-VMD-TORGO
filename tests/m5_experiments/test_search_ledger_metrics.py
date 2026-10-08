from __future__ import annotations

import sqlite3
from dataclasses import replace
from typing import Any

import pytest

from vaaniswitch.m4_fusion.losses import label_for_condition
from vaaniswitch.m4_fusion.metrics import binary_metrics
from vaaniswitch.m5_experiments.aggregation import aggregate
from vaaniswitch.m5_experiments.config import ABLATIONS
from vaaniswitch.m5_experiments.ledger import Ledger, experiment_id
from vaaniswitch.m5_experiments.search import InnerResult, choose_candidate, metric_key
from vaaniswitch.m5_experiments.source import ExperimentContext


def inner_result(candidate: str = "baseline", **metrics: Any) -> InnerResult:
    return InnerResult(
        candidate,
        candidate,
        "split",
        "records",
        3,
        {"macro_f1": 0.7, "balanced_accuracy": 0.7, "roc_auc": 0.8, "loss": 0.4, **metrics},
        100,
    )


@pytest.mark.parametrize(
    "metric,better",
    [("macro_f1", 0.71), ("balanced_accuracy", 0.71), ("roc_auc", 0.81), ("loss", 0.39)],
)
def test_lexicographic_tie_break_order(metric: str, better: float) -> None:
    first = inner_result()
    second = inner_result("lower_lr", **{metric: better})
    winner = choose_candidate(
        [first, second],
        split_hash="split",
        validation_record_hash="records",
        candidate_ids={"baseline", "lower_lr"},
    )
    assert winner == second
    poorer_primary = replace(second, metrics={**second.metrics, "macro_f1": 0.69})
    assert metric_key(first.metrics, 100) > metric_key(poorer_primary.metrics, 100)


def test_simpler_model_and_epoch_identity() -> None:
    first = inner_result()
    second = replace(first, candidate_id="lower_lr", trainable_parameters=99, epoch=7)
    selected = choose_candidate(
        [first, second],
        split_hash="split",
        validation_record_hash="records",
        candidate_ids={"baseline", "lower_lr"},
    )
    assert selected == second and selected.identity["epoch"] == 7
    assert selected.selection_hash != first.selection_hash
    assert metric_key({**first.metrics, "roc_auc": None}, 1) < metric_key(first.metrics, 100)


@pytest.mark.parametrize(
    "change",
    [
        {"role": "outer_test"},
        {"split_hash": "outer"},
        {"validation_record_hash": "outer"},
        {"epoch": 0},
        {"epoch": 31},
    ],
)
def test_outer_evidence_or_invalid_epoch_cannot_select(change: dict[str, Any]) -> None:
    result = replace(inner_result(), **change)
    with pytest.raises(ValueError, match="inner-validation"):
        choose_candidate(
            [result],
            split_hash="split",
            validation_record_hash="records",
            candidate_ids={"baseline"},
        )


@pytest.mark.parametrize("values", [[], [inner_result(), inner_result()]])
def test_search_requires_complete_unique_candidates(values: list[InnerResult]) -> None:
    with pytest.raises(ValueError, match="exactly once"):
        choose_candidate(
            values,
            split_hash="split",
            validation_record_hash="records",
            candidate_ids={"baseline", "lower_lr"},
        )


def test_metrics_hand_computable_and_fixed_threshold() -> None:
    metrics = binary_metrics([0, 0, 0, 0, 1, 1], [0.1, 0.2, 0.6, 0.8, 0.3, 0.9])
    assert metrics["confusion_matrix"] == [[2, 2], [1, 1]]
    assert metrics["macro_f1"] == pytest.approx(17 / 35)
    assert metrics["roc_auc"] == 0.75
    assert metrics["accuracy"] == metrics["balanced_accuracy"] == 0.5
    assert label_for_condition("control") == 0 and label_for_condition("dysarthric") == 1
    assert binary_metrics([1, 1], [0.2, 0.7])["roc_auc"] is None
    assert binary_metrics([0, 1], [0.5, 0.5])["roc_auc"] == 0.5
    with pytest.raises(ValueError, match="fixed"):
        binary_metrics([0, 1], [0.2, 0.9], threshold=0.4)


def test_aggregation_sample_std_equal_fold_weight() -> None:
    results: list[dict[str, Any]] = []
    for ablation in ABLATIONS:
        for fold, score in enumerate((0.2, 0.4, 0.6), 1):
            results.append(
                {
                    "ablation": ablation,
                    "outer_fold": fold,
                    "metrics": {**binary_metrics([0, 1], [0.2, 0.9]), "accuracy": score},
                }
            )
    report = aggregate(results)
    summary = report["ablations"]["C2"]["summary"]["accuracy"]
    assert summary["mean"] == pytest.approx(0.4)
    assert summary["sample_std"] == pytest.approx(0.2)
    assert report["ablations"]["C2"]["pooled_confusion_matrix_secondary"] == [[3, 0], [0, 3]]
    assert report["c2_macro_f1_differences"]["C1"]["positive_folds"] == 0
    results[0]["metrics"]["roc_auc"] = None
    assert aggregate(results)["ablations"]["A0"]["summary"]["roc_auc"]["mean"] is None
    with pytest.raises(ValueError, match="exactly seven"):
        aggregate(results[:-1])
    with pytest.raises(ValueError, match="exactly seven"):
        aggregate([*results, results[0]])


def test_ledger_all_states_and_one_outer_claim(context: ExperimentContext) -> None:
    ledger = Ledger(context.config)
    run_id = experiment_id(context.config, 1, "C2", "baseline")
    ledger.plan(run_id, {"id": run_id})
    ledger.plan(run_id, {"id": run_id})
    assert ledger.get(run_id)["state"] == "PLANNED"
    with pytest.raises(ValueError, match="identity changed"):
        ledger.plan(run_id, {"id": "changed"})
    with pytest.raises(ValueError):
        ledger.claim_outer(run_id)
    ledger.transition(run_id, "RUNNING")
    ledger.transition(run_id, "INNER_COMPLETE")
    ledger.transition(run_id, "SELECTED", {"selection": {"epoch": 1}})
    with pytest.raises(ValueError):
        ledger.claim_outer(run_id)
    ledger.transition(run_id, "FINAL_TRAIN_COMPLETE", {"final_checkpoint": {"sha256": "a"}})
    token = ledger.claim_outer(run_id)
    ledger.verify_claim(run_id, token)
    with pytest.raises(ValueError):
        ledger.claim_outer(run_id)
    ledger.transition(run_id, "OUTER_EVALUATED", {"outer_result_sha256": "result"})
    before = ledger.get(run_id)
    for target in ("RUNNING", "FAILED", "PLANNED", "OUTER_EVALUATED"):
        with pytest.raises(ValueError):
            ledger.transition(run_id, target, {"reason": "test"})
    assert ledger.get(run_id) == before
    ledger.transition(run_id, "INVALIDATED", {"reason": "explicit documented defect"})
    assert ledger.get(run_id)["data"]["outer_result_sha256"] == "result"
    assert ledger.export().is_file()


@pytest.mark.parametrize(
    "state", ["INNER_COMPLETE", "SELECTED", "FINAL_TRAIN_COMPLETE", "OUTER_EVALUATED", "unknown"]
)
def test_invalid_ledger_transition(context: ExperimentContext, state: str) -> None:
    ledger = Ledger(context.config)
    ledger.plan("fixture", {})
    with pytest.raises(ValueError):
        ledger.transition("fixture", state)


def test_failed_resume_requires_reason_and_preserves_identity(context: ExperimentContext) -> None:
    ledger = Ledger(context.config)
    ledger.plan("fixture", {"seed": 42})
    ledger.transition("fixture", "RUNNING")
    ledger.transition("fixture", "FAILED", {"reason": "interruption"})
    with pytest.raises(ValueError):
        ledger.retry_failed("fixture", "")
    ledger.retry_failed("fixture", "power restored; unchanged implementation")
    assert ledger.get("fixture")["state"] == "RUNNING"
    assert ledger.get("fixture")["identity"] == {"seed": 42}
    with ledger.lock("fixture"), pytest.raises(ValueError, match="locked"), ledger.lock("fixture"):
        pass


def test_experiment_ids_stable_and_separate(context: ExperimentContext) -> None:
    config = context.config
    first = experiment_id(config, 1, "C2", "baseline")
    assert first == experiment_id(config, 1, "C2", "baseline")
    assert (
        len(
            {
                first,
                experiment_id(config, 2, "C2", "baseline"),
                experiment_id(config, 1, "C1", "baseline"),
                experiment_id(config, 1, "C2", "lower_lr"),
                experiment_id(config, 1, "C2", "baseline", smoke=True),
            }
        )
        == 5
    )


def test_ledger_connection_closes_after_transaction(context: ExperimentContext) -> None:
    ledger = Ledger(context.config)
    with ledger.connect() as connection:
        connection.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
