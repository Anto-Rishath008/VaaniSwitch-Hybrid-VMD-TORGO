"""Metadata-only audit and mocked handoff tests: no model forward or training."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from scripts.audit_module5_phase5b import validate_history
from vaaniswitch.m4_fusion.metrics import binary_metrics
from vaaniswitch.m5_experiments import pipeline
from vaaniswitch.m5_experiments.config import ABLATIONS
from vaaniswitch.m5_experiments.ledger import Ledger
from vaaniswitch.m5_experiments.search import InnerResult
from vaaniswitch.m5_experiments.source import ExperimentContext, freeze_protocol
from vaaniswitch.m5_experiments.splits import record_hash


def progress_fixture() -> dict[str, Any]:
    metrics = {**binary_metrics([0, 1], [0.2, 0.8]), "loss": 0.2}
    return {
        "history": [{"epoch": e, "validation": deepcopy(metrics)} for e in range(1, 7)],
        "epoch": 6,
        "bad_epochs": 5,
        "patience_best": 1.0,
    }


def test_history_accepts_complete_consistent_inner_evidence() -> None:
    validate_history(progress_fixture(), 1, 1)


@pytest.mark.parametrize("defect", ["metric", "labels", "patience", "premature", "extra_epoch"])
def test_history_rejects_inconsistent_evidence(defect: str) -> None:
    progress = progress_fixture()
    if defect == "metric":
        progress["history"][1]["validation"]["accuracy"] = 0.8
    elif defect == "labels":
        progress["history"][1]["validation"]["TP"] = 2
    elif defect == "patience":
        progress["bad_epochs"] = 4
    elif defect == "premature":
        progress["history"].pop()
        progress.update(epoch=5, bad_epochs=4)
    else:
        progress["history"].append({"epoch": 7, "validation": progress["history"][0]["validation"]})
    with pytest.raises(ValueError):
        validate_history(progress, 1, 1)


@pytest.mark.parametrize("interrupt", [False, True])
def test_phase5c_reuses_c2_and_completed_ablations_without_final_or_outer(
    context: ExperimentContext, monkeypatch: pytest.MonkeyPatch, interrupt: bool
) -> None:
    calls: list[str] = []
    interrupted = False

    def metadata_training(
        ctx: ExperimentContext,
        split: Any,
        ablation: str,
        candidate: Any,
        run_id: str,
        *,
        smoke: bool,
        resume: bool,
        **kwargs: Any,
    ) -> dict[str, Any]:
        nonlocal interrupted
        assert not kwargs and not smoke  # A final selection argument is forbidden.
        calls.append(ablation)
        if interrupt and ablation == "A1" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("synthetic process interruption")
        result = InnerResult(
            run_id,
            candidate.id,
            split.split_hash,
            record_hash(split.records(ctx.predecessor.source.contract, "inner_validation")),
            ABLATIONS.index(ablation) + 1,
            {
                "macro_f1": 0.8 if candidate.id == "lower_lr" else 0.7,
                "balanced_accuracy": 0.7,
                "roc_auc": 0.8,
                "loss": 0.5,
            },
            100,
        )
        return {
            "progress": {"best_inner": result.identity},
            "training_seconds": 1.0,
            "parameter_counts": {"trainable": 100, "classifier": 50, "total": 100},
        }

    freeze_protocol(context)
    monkeypatch.setattr(pipeline, "train_stage", metadata_training)
    selected = pipeline.search_c2(context, 1)
    calls.clear()
    selected_path = pipeline.selection_path(context, 1)
    selection_bytes = selected_path.read_bytes()
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            pipeline.prepare_ablations(context, 1, resume=True)
        assert calls == ["A0", "A1"]
    bundle = pipeline.prepare_ablations(context, 1, resume=True)
    assert bundle["common_candidate"] == selected["candidate_id"] == "lower_lr"
    assert bundle["selections"]["C2"] == selected
    assert set(bundle["selections"]) == set(ABLATIONS)
    assert {s["candidate_id"] for s in bundle["selections"].values()} == {"lower_lr"}
    assert len({s["epoch"] for s in bundle["selections"].values()}) == 7
    assert calls.count("A0") == 1 and "C2" not in calls
    assert len(calls) == (7 if interrupt else 6)
    before = list(calls)
    assert pipeline.prepare_ablations(context, 1, resume=True) == bundle
    assert calls == before and selected_path.read_bytes() == selection_bytes
    runs = Ledger(context.config).all_runs()
    assert len(runs) == 10
    assert sum(r["state"] == "SELECTED" for r in runs) == 7
    assert sum(r["state"] == "INNER_COMPLETE" for r in runs) == 3
    assert all(r["claim"] is None for r in runs)
    assert not list(context.config.output("experiments").rglob("outer_result.json"))
    assert not list(context.config.output("experiments").rglob("final"))
