"""Reporting fixtures contain metadata only; no model is built or trained."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import pytest
import torch

from scripts import report_module5_phase5b as helper
from vaaniswitch.m4_fusion.metrics import binary_metrics
from vaaniswitch.m5_experiments import trainer
from vaaniswitch.m5_experiments.config import CANDIDATES
from vaaniswitch.m5_experiments.data import pos_weight
from vaaniswitch.m5_experiments.ledger import Ledger, experiment_id
from vaaniswitch.m5_experiments.reporting import read_json, write_json
from vaaniswitch.m5_experiments.search import InnerResult, choose_candidate
from vaaniswitch.m5_experiments.source import ExperimentContext, freeze_protocol
from vaaniswitch.m5_experiments.splits import record_hash


def fixture_panel(context: ExperimentContext) -> tuple[Ledger, dict[str, Any]]:
    freeze_protocol(context)
    split, config = context.split(1), context.config
    ledger = Ledger(config)
    results = []
    for candidate in CANDIDATES:
        run_id = experiment_id(config, 1, "C2", candidate.id)
        identity = trainer.run_identity(context, split, "C2", candidate, run_id, smoke=False)
        training_records = split.records(context.predecessor.source.contract, "inner_train")
        validation_records = split.records(context.predecessor.source.contract, "inner_validation")
        validation = {
            **binary_metrics([0, 1], [0.1, 0.9]),
            "loss": 0.1 if candidate.id == "lower_lr" else 0.5,
            "role": "inner_validation",
            "record_count": len(validation_records),
            "record_hash": record_hash(validation_records),
            "elapsed_seconds": 0.2,
        }
        result = InnerResult(
            run_id,
            candidate.id,
            split.split_hash,
            record_hash(validation_records),
            1,
            validation,
            291921,
        )
        results.append(result)
        progress = {
            "epoch": 2,
            "bad_epochs": 1,
            "best_inner": result.identity,
            "patience_best": 1.0,
            "pos_weight": pos_weight(context, split, "inner_train", smoke=False),
            "history": [
                {
                    "epoch": epoch,
                    "training": {
                        "role": "inner_train",
                        "record_hash": record_hash(training_records),
                        "record_count": len(training_records),
                        "loss": 0.3,
                        "elapsed_seconds": 1.0,
                    },
                    "validation": validation,
                }
                for epoch in (1, 2)
            ],
        }
        payload = {
            "schema": "module5_checkpoint_v1",
            "identity": {**identity, "stage": "inner", "selected_inner_configuration": None},
            "progress": progress,
            "model": {},
            "optimizer": {},
            "rng": {},
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        content = buffer.getvalue()
        checksum = hashlib.sha256(content).hexdigest()
        directory = config.output(f"experiments/runs/{run_id}/inner")
        directory.mkdir(parents=True)
        (directory / f"{checksum}.pt").write_bytes(content)
        write_json(directory / "latest.json", {"filename": f"{checksum}.pt", "sha256": checksum})
        ledger.plan(run_id, identity)
        ledger.transition(run_id, "RUNNING")
        ledger.transition(
            run_id,
            "INNER_COMPLETE",
            {
                "inner_result": result.identity,
                "inner_training_seconds": 2.0,
                "parameter_counts": {"trainable": 291921},
            },
        )
    winner = choose_candidate(
        results,
        split_hash=split.split_hash,
        validation_record_hash=results[0].validation_record_hash,
        candidate_ids={candidate.id for candidate in CANDIDATES},
    )
    ledger.transition(winner.experiment_id, "SELECTED", {"selection": winner.identity})
    write_json(
        config.output("experiments/selection/fold1_c2.json"), winner.identity, immutable=True
    )
    return ledger, winner.identity


def test_report_is_read_only_complete_and_explains_selection(
    context: ExperimentContext, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture_panel(context)
    before_hash = context.code_hash

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("report must never construct a model")

    monkeypatch.setattr(trainer, "build_model", forbidden)
    evidence = context.config.output("experiments")
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in evidence.rglob("*") if p.is_file()}
    report = helper.panel_report(context, 1)
    assert report["selection"]["candidate_id"] == "lower_lr"
    assert len(report["compared_experiment_ids"]) == len(report["candidates"]) == 4
    assert {r["first_decisive_field"] for r in report["tie_break_reasoning"]} == {
        "lower_validation_loss"
    }
    assert all(
        r["outer_model_forward_count"] == 0 and r["epochs_completed"] == 2 and r["best_epoch"] == 1
        for r in report["candidates"]
    )
    assert all(before[p] == (p.read_bytes(), p.stat().st_mtime_ns) for p in before)
    assert context.code_hash == before_hash
    path = tmp_path / "report.json"
    helper.publish(path, report)
    old = path.stat().st_mtime_ns
    helper.publish(path, report)
    assert path.stat().st_mtime_ns == old
    with pytest.raises(ValueError, match="Existing report"):
        helper.publish(path, {**report, "outer_model_forward_count": 1})


def test_incomplete_panel_cannot_report(context: ExperimentContext) -> None:
    freeze_protocol(context)
    with pytest.raises(ValueError, match="no production ledger"):
        helper.panel_report(context, 1)


@pytest.mark.parametrize("damage", ["selection", "identity", "checksum", "outer_role"])
def test_report_rejects_tampering(context: ExperimentContext, damage: str) -> None:
    ledger, winner = fixture_panel(context)
    run_id = winner["experiment_id"]
    directory = context.config.output(f"experiments/runs/{run_id}/inner")
    pointer = read_json(directory / "latest.json")
    path = directory / pointer["filename"]
    if damage == "selection":
        write_json(
            context.config.output("experiments/selection/fold1_c2.json"),
            {**winner, "candidate_id": "baseline"},
        )
    elif damage == "identity":
        with ledger.connect() as connection:
            connection.execute("UPDATE runs SET identity='{}' WHERE id=?", (run_id,))
    elif damage == "checksum":
        path.write_bytes(b"corrupt")
    else:
        payload = torch.load(path, weights_only=True)
        payload["progress"]["history"][0]["training"]["role"] = "outer_test"
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        content = buffer.getvalue()
        checksum = hashlib.sha256(content).hexdigest()
        (directory / f"{checksum}.pt").write_bytes(content)
        write_json(directory / "latest.json", {"filename": f"{checksum}.pt", "sha256": checksum})
    with pytest.raises(ValueError):
        helper.panel_report(context, 1)
