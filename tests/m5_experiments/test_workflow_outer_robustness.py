from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from tests.m4_fusion.helpers import MemorySource
from vaaniswitch.m4_fusion.metrics import binary_metrics
from vaaniswitch.m5_experiments import evaluator, pipeline
from vaaniswitch.m5_experiments.aggregation import load_outer_results
from vaaniswitch.m5_experiments.config import ABLATIONS, CANDIDATES
from vaaniswitch.m5_experiments.ledger import Ledger, experiment_id
from vaaniswitch.m5_experiments.reporting import write_json
from vaaniswitch.m5_experiments.robustness import robustness
from vaaniswitch.m5_experiments.search import InnerResult
from vaaniswitch.m5_experiments.source import ExperimentContext, freeze_protocol
from vaaniswitch.m5_experiments.splits import record_hash
from vaaniswitch.m5_experiments.trainer import run_identity, train_stage


def fake_training(
    context: ExperimentContext,
    split: Any,
    ablation: str,
    candidate: Any,
    run_id: str,
    *,
    smoke: bool,
    resume: bool,
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = InnerResult(
        run_id,
        candidate.id,
        split.split_hash,
        record_hash(split.records(context.predecessor.source.contract, "inner_validation")),
        2,
        {
            "macro_f1": 0.8 if candidate.id == "lower_lr" else 0.7,
            "balanced_accuracy": 0.7,
            "roc_auc": 0.8,
            "loss": 0.5,
        },
        100,
    )
    return {
        "progress": {
            "best_inner": result.identity,
            "epoch": selection["epoch"] if selection else 3,
            "pos_weight": 1.0,
        },
        "training_seconds": 1.0,
        "parameter_counts": {"trainable": 100, "classifier": 50, "total": 100},
        "checkpoint": {"sha256": "fixture"},
        "identity": {"selection": selection},
    }


def test_manual_search_ablations_and_final_ledger(
    context: ExperimentContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze_protocol(context)
    monkeypatch.setattr(pipeline, "train_stage", fake_training)
    selected = pipeline.search_c2(context, 1)
    assert selected["candidate_id"] == "lower_lr"
    assert pipeline.search_c2(context, 1) == selected
    bundle = pipeline.prepare_ablations(context, 1)
    assert set(bundle["selections"]) == set(ABLATIONS)
    assert {s["candidate_id"] for s in bundle["selections"].values()} == {"lower_lr"}
    assert all(s["epoch"] == 2 for s in bundle["selections"].values())
    report = pipeline.retrain_fold(context, 1)
    assert report["status"] == "FINAL_TRAIN_COMPLETE" and report["outer_evaluations"] == 0
    runs = Ledger(context.config).all_runs()
    assert len(runs) == 10
    assert sum(r["state"] == "FINAL_TRAIN_COMPLETE" for r in runs) == 7
    assert sum(r["state"] == "INNER_COMPLETE" for r in runs) == 3
    assert pipeline.retrain_fold(context, 1) == report


def test_tampered_selection_cannot_change_ablation_candidate(
    context: ExperimentContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze_protocol(context)
    monkeypatch.setattr(pipeline, "train_stage", fake_training)
    selected = pipeline.search_c2(context, 1)
    selected["candidate_id"] = "stronger_dropout"
    write_json(pipeline.selection_path(context, 1), selected)
    with pytest.raises(ValueError, match="selection"):
        pipeline.prepare_ablations(context, 1)


def test_selection_recovery_cannot_promote_a_losing_candidate(
    context: ExperimentContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze_protocol(context)
    monkeypatch.setattr(pipeline, "train_stage", fake_training)
    selected = pipeline.search_c2(context, 1)
    assert selected["candidate_id"] == "lower_lr"
    ledger = Ledger(context.config)
    losing_id = experiment_id(context.config, 1, "C2", "baseline")
    losing_result = ledger.get(losing_id)["data"]["inner_result"]
    write_json(pipeline.selection_path(context, 1), losing_result)
    with pytest.raises(ValueError, match="selection"):
        pipeline.search_c2(context, 1, resume=True)


def ready_outer(context: ExperimentContext) -> tuple[Ledger, dict[str, Any]]:
    freeze_protocol(context)
    ledger = Ledger(context.config)
    split, candidate = context.split(1), CANDIDATES[0]
    selections = {}
    for ablation in ABLATIONS:
        run_id = experiment_id(context.config, 1, ablation, candidate.id)
        identity = run_identity(context, split, ablation, candidate, run_id, smoke=False)
        ledger.plan(run_id, identity)
        ledger.transition(run_id, "RUNNING")
        result = InnerResult(
            run_id,
            candidate.id,
            split.split_hash,
            record_hash(split.records(context.predecessor.source.contract, "inner_validation")),
            1,
            {"macro_f1": 0.8, "balanced_accuracy": 0.8, "roc_auc": 0.9, "loss": 0.4},
            100,
        )
        ledger.transition(
            run_id,
            "INNER_COMPLETE",
            {"inner_result": result.identity, "inner_training_seconds": 0.1},
        )
        ledger.transition(run_id, "SELECTED", {"selection": result.identity})
        selections[ablation] = result.identity
        if ablation == "B1":
            report = train_stage(
                context,
                split,
                ablation,
                candidate,
                run_id,
                smoke=False,
                resume=False,
                selection=result.identity,
            )
            data = {
                "final_checkpoint": report["checkpoint"],
                "final_identity": report["identity"],
                "final_epoch": 1,
                "parameter_counts": report["parameter_counts"],
                "final_training_seconds": report["training_seconds"],
            }
        else:
            data = {"final_checkpoint": {"synthetic_unconsumed_fixture": True}}
        ledger.transition(run_id, "FINAL_TRAIN_COMPLETE", data)
    bundle = {
        "outer_fold": 1,
        "config_hash": context.config.config_hash,
        "common_candidate": candidate.id,
        "selections": selections,
    }
    write_json(context.config.output("experiments/selection/fold1_ablations.json"), bundle)
    assert isinstance(context.predecessor.source, MemorySource)
    context.predecessor.source.forbidden_speakers.clear()
    return ledger, selections["B1"]


def test_actual_synthetic_outer_once_and_immutable(context: ExperimentContext) -> None:
    ledger, selection = ready_outer(context)
    source = context.predecessor.source
    assert isinstance(source, MemorySource)
    before = len(source.loaded)
    result = evaluator.evaluate_outer(context, 1, "B1")
    assert len(source.loaded) - before == 5
    assert result["source_load_count"] == result["forward_count"] == 5
    assert {p["speaker_id"] for p in result["predictions"]} == set(
        context.split(1).speakers("outer_test")
    )
    assert ledger.get(selection["experiment_id"])["state"] == "OUTER_EVALUATED"
    size = len(source.loaded)
    assert evaluator.evaluate_outer(context, 1, "B1") == result
    assert len(source.loaded) == size
    assert load_outer_results(context.config) == [result]
    path = context.config.output(f"experiments/runs/{selection['experiment_id']}/outer_result.json")
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_json(path, {"replacement": True}, immutable=True)
    assert path.read_bytes() == original
    result["metrics"]["accuracy"] = 0.1234
    write_json(path, result)
    with pytest.raises(ValueError, match="modified"):
        evaluator.evaluate_outer(context, 1, "B1")
    with pytest.raises(ValueError, match="checksum"):
        load_outer_results(context.config)


def test_outer_blocked_until_all_comparisons_retrained(context: ExperimentContext) -> None:
    ledger, _ = ready_outer(context)
    a0 = experiment_id(context.config, 1, "A0", "baseline")
    ledger.transition(a0, "INVALIDATED", {"reason": "synthetic incomplete comparison"})
    with pytest.raises(ValueError, match="all ablations"):
        evaluator.evaluate_outer(context, 1, "B1")


def test_failure_after_outer_claim_cannot_retry(
    context: ExperimentContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, selection = ready_outer(context)

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("failure after outer claim")

    monkeypatch.setattr(evaluator, "batches", fail)
    with pytest.raises(RuntimeError, match="after outer claim"):
        evaluator.evaluate_outer(context, 1, "B1")
    assert ledger.get(selection["experiment_id"])["state"] == "FAILED"
    with pytest.raises(ValueError, match="pre-outer"):
        ledger.retry_failed(selection["experiment_id"], "try again")
    with pytest.raises(ValueError):
        evaluator.evaluate_outer(context, 1, "B1")


def subgroup_results(context: ExperimentContext) -> list[dict[str, Any]]:
    results = []
    for split in context.splits:
        predictions: list[dict[str, Any]] = []
        for record in split.records(context.predecessor.source.contract, "outer_test"):
            label = int(record.condition.value == "dysarthric")
            predictions.append(
                {
                    "logical_id": record.logical_id,
                    "speaker_id": record.speaker_id,
                    "label": label,
                    "probability": 0.8 if label else 0.2,
                }
            )
        results.append(
            {
                "ablation": "C2",
                "outer_fold": split.outer_fold,
                "selection": {"never_change": True},
                "predictions": predictions,
                "metrics": binary_metrics(
                    [p["label"] for p in predictions], [p["probability"] for p in predictions]
                ),
            }
        )
    return results


def test_robustness_counts_single_class_and_no_selection_feedback(
    context: ExperimentContext,
) -> None:
    results = subgroup_results(context)
    before = deepcopy(results)
    report = robustness(context, results)
    assert results == before
    assert report["record_count"] == 15
    assert report["groups"]["pooled_descriptive/duration/<=8s"]["sample_count"] == 15
    speaker_groups = [
        v for k, v in report["groups"].items() if k.startswith("pooled_descriptive/speaker/")
    ]
    assert len(speaker_groups) == 15
    assert all(v["sample_count"] == 1 and v["metrics"]["roc_auc"] is None for v in speaker_groups)
    assert all(v["small_subgroup"] for v in speaker_groups)
    assert all(
        v["control_count"] + v["dysarthric_count"] == v["sample_count"]
        for v in report["groups"].values()
    )
    with pytest.raises(ValueError, match="all three"):
        robustness(context, results[:-1])
    results[0]["predictions"].append(results[0]["predictions"][0])
    with pytest.raises(ValueError, match="complete"):
        robustness(context, results)


def test_preserves_raw_and_predecessor_sentinels(
    context: ExperimentContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = []
    for relative in (
        "data/processed/vmd/signed/modes.npy",
        "data/processed/representations/signed/mel.npy",
        "src/vaaniswitch/m3_encoders/signed.py",
        "src/vaaniswitch/m4_fusion/signed.py",
        "artifacts/checkpoints/module4/best.pt",
        "artifacts/metrics/module4_full_signoff_audit.json",
    ):
        path = context.config.repo_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"signed sentinel")
        paths.append(path)
    paths.extend((context.config.repo_root / "data/raw").rglob("*.wav"))
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
    freeze_protocol(context)
    monkeypatch.setattr(pipeline, "train_stage", fake_training)
    pipeline.search_c2(context, 1)
    pipeline.prepare_ablations(context, 1)
    pipeline.retrain_fold(context, 1)
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
