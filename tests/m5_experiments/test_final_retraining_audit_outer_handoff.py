"""Metadata-only final audit/outer handoff tests; never call the outer evaluator."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import module5_outer_evidence as evidence
from scripts.audit_module5_final_retraining import validate_final_history
from vaaniswitch.m4_fusion.metrics import binary_metrics
from vaaniswitch.m5_experiments.config import file_hash
from vaaniswitch.m5_experiments.ledger import Ledger
from vaaniswitch.m5_experiments.reporting import write_json
from vaaniswitch.m5_experiments.source import ExperimentContext


def final_progress() -> dict[str, Any]:
    return {
        "epoch": 2,
        "pos_weight": 2.0,
        "bad_epochs": 0,
        "patience_best": 0.0,
        "best_inner": None,
        "history": [
            {
                "epoch": e,
                "training": {
                    "role": "development",
                    "record_count": 10,
                    "record_hash": "records",
                    "loss": 0.4,
                    "elapsed_seconds": 0.1,
                },
            }
            for e in (1, 2)
        ],
    }


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "epoch",
        "role",
        "count",
        "hash",
        "weight",
        "validation",
        "metrics",
        "early_stop",
        "missing_epoch",
    ],
)
def test_final_history_enforces_development_only_fixed_epochs(damage: str | None) -> None:
    progress = final_progress()
    training = progress["history"][0]["training"]
    if damage == "epoch":
        progress["epoch"] = 3
    elif damage == "role":
        training["role"] = "outer_test"
    elif damage == "count":
        training["record_count"] = 8
    elif damage == "hash":
        training["record_hash"] = "inner"
    elif damage == "weight":
        progress["pos_weight"] = 3.0
    elif damage == "validation":
        progress["history"][0]["validation"] = {}
    elif damage == "metrics":
        training["accuracy"] = 1.0
    elif damage == "early_stop":
        progress["bad_epochs"] = 5
    elif damage == "missing_epoch":
        progress["history"].pop()
    if damage:
        with pytest.raises(ValueError):
            validate_final_history(progress, {"epoch": 2}, 10, "records", 2.0)
    else:
        validate_final_history(progress, {"epoch": 2}, 10, "records", 2.0)


@pytest.mark.parametrize("damage", [None, "state", "claim", "event", "result"])
def test_preflight_refuses_consumed_or_orphaned_outer_evidence(damage: str | None) -> None:
    run = {"state": "FINAL_TRAIN_COMPLETE", "claim": None}
    events = []
    if damage == "state":
        run["state"] = "SELECTED"
    elif damage == "claim":
        run["claim"] = "consumed"
    elif damage == "event":
        events = [(1, "run", "time", "OUTER_CLAIMED", "{}")]
    if damage:
        with pytest.raises(ValueError):
            evidence.validate_unclaimed(run, events, damage == "result")
    else:
        evidence.validate_unclaimed(run, events, False)


def test_one_use_claim_transaction_and_no_post_claim_retry(context: ExperimentContext) -> None:
    ledger = Ledger(context.config)
    run_id = "metadata_fixture"
    ledger.plan(run_id, {"fixture": True})
    ledger.transition(run_id, "RUNNING")
    ledger.transition(run_id, "INNER_COMPLETE")
    ledger.transition(run_id, "SELECTED", {"selection": {"epoch": 1}})
    ledger.transition(run_id, "FINAL_TRAIN_COMPLETE", {"final_checkpoint": {"fixture": True}})
    token = ledger.claim_outer(run_id)
    ledger.verify_claim(run_id, token)
    with pytest.raises(ValueError, match="once"):
        ledger.claim_outer(run_id)
    ledger.transition(run_id, "FAILED", {"reason": "synthetic post-claim interruption"})
    with pytest.raises(ValueError, match="pre-outer"):
        ledger.retry_failed(run_id, "must not retry consumed claim")
    assert ledger.get(run_id)["claim"] == token


@pytest.mark.parametrize("damage", [None, "checksum", "metrics", "record_order", "claim", "events"])
def test_completed_prediction_report_is_read_only_and_binds_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str | None,
) -> None:
    monkeypatch.setattr(evidence, "ROOT", tmp_path)
    records = tuple(
        SimpleNamespace(
            logical_id=f"speaker/{i}",
            speaker_id=f"S{i}",
            condition=SimpleNamespace(value=condition),
        )
        for i, condition in enumerate(("control", "dysarthric"))
    )
    predictions = [
        {
            "logical_id": r.logical_id,
            "speaker_id": r.speaker_id,
            "label": i,
            "probability": 0.2 + 0.6 * i,
        }
        for i, r in enumerate(records)
    ]
    identity = {"ablation": "B1", "outer_fold": 1}
    selection, checkpoint = {"epoch": 2}, {"sha256": "synthetic"}
    result = {
        "status": "OUTER_EVALUATED",
        "experiment_id": "fixture",
        "identity": identity,
        "selection": selection,
        "checkpoint": checkpoint,
        "ablation": "B1",
        "outer_fold": 1,
        "claim": "one",
        "threshold": 0.5,
        "predictions": predictions,
        "forward_count": 2,
        "source_load_count": 2,
        "metrics": binary_metrics([0, 1], [0.2, 0.8]),
    }
    if damage == "metrics":
        result["metrics"]["accuracy"] = 0.0
    elif damage == "record_order":
        result["predictions"].reverse()
    elif damage == "claim":
        result["claim"] = "different"
    path = tmp_path / "runs/fixture/outer_result.json"
    write_json(path, result, immutable=True)
    checksum = file_hash(path)
    run = {
        "state": "OUTER_EVALUATED",
        "claim": "one",
        "identity": identity,
        "data": {
            "selection": selection,
            "final_checkpoint": checkpoint,
            "final_identity": {"stage": "final"},
            "outer_result_sha256": checksum,
        },
    }
    events = [
        (1, "fixture", "t", "OUTER_CLAIMED", json.dumps({"claim": "one"})),
        (2, "fixture", "t", "OUTER_EVALUATED", json.dumps({"outer_result_sha256": checksum})),
    ]
    if damage == "events":
        events.append(deepcopy(events[0]))
    elif damage == "checksum":
        run["data"]["outer_result_sha256"] = "incorrect"
    before = path.read_bytes()
    if damage:
        with pytest.raises(ValueError):
            evidence.completed_result(tmp_path, "fixture", run, events, records)
    else:
        report = evidence.completed_result(tmp_path, "fixture", run, events, records)
        assert report["record_count"] == 2 and report["record_hash"] == evidence.record_hash(
            records
        )
        assert evidence.completed_result(tmp_path, "fixture", run, events, records) == report
    assert path.read_bytes() == before
