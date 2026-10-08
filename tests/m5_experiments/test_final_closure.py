"""Final report regressions use tiny signed-context fixtures; no production inference."""

from __future__ import annotations

import csv
import io
import json
from copy import deepcopy

import pytest

from scripts import module5_final_closure as m
from tests.m5_experiments.test_workflow_outer_robustness import subgroup_results


@pytest.fixture
def views(context):
    output = []
    for result in subgroup_results(context):
        fold = result["outer_fold"]
        for ablation in m.s.g.ORDER:
            item = deepcopy(result)
            item.update(
                ablation=ablation,
                experiment_id=f"fold{fold}_{ablation}",
                provenance="NATIVE",
                candidate="baseline",
                selected_epoch=2,
                threshold=0.5,
                checkpoint_sha256="a" * 64,
                result_sha256="b" * 64,
                record_count=len(result["predictions"]),
                selection_sha256="c" * 64,
                parameter_counts={"trainable": 10},
                inner_training_seconds=1.0,
                final_training_seconds=2.0,
            )
            if (fold, ablation) == (1, "A1"):
                item.update(experiment_id=m.r.RUN, provenance=m.EXCEPTION)
            # Unequal fold metrics test unweighted means and ddof=1, independent of group metrics.
            item["metrics"]["accuracy"] = [0.2, 0.4, 0.9][fold - 1]
            output.append(item)
    return output


def test_frozen_reporting_exception_sample_std_and_groups(context, views):
    before = deepcopy(views)
    with m.s.forbid_scientific_execution():
        cross, robust = m.reports(context, views)
    assert views == before
    metric = cross["ablations"]["C2"]["summary"]["accuracy"]
    assert metric["mean"] == pytest.approx(0.5)
    assert metric["sample_std"] == pytest.approx((0.26 / 2) ** 0.5)
    assert m.EXCEPTION in cross["ablations"]["A1"]["provenance_note"]
    assert cross["ablations"]["A1"]["folds"][0]["provenance"] == m.EXCEPTION
    assert robust["record_count"] == 15
    assert robust["threshold"] == 0.5
    assert not robust["new_inference"] and not robust["retuning"]
    speakers = [
        v for k, v in robust["groups"].items() if k.startswith("pooled_descriptive/speaker")
    ]
    assert len(speakers) == 15
    assert all(v["metrics"]["roc_auc"] is None for v in speakers)
    assert all(v["small_subgroup"] for v in speakers)
    fold_csv, summary_csv = m.tables(cross)
    rows = list(csv.DictReader(io.StringIO(fold_csv.decode())))
    assert len(rows) == 21
    assert sum(v["provenance"] == m.EXCEPTION for v in rows) == 1
    summary = list(csv.DictReader(io.StringIO(summary_csv.decode())))
    assert len(summary) == 49
    assert all(m.EXCEPTION in v["provenance"] for v in summary if v["ablation"] == "A1")


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "hidden", "extra_exception"])
def test_reports_reject_incomplete_or_hidden_exception(context, views, mutation):
    if mutation == "missing":
        views.pop()
    elif mutation == "duplicate":
        views[-1] = deepcopy(views[0])
    elif mutation == "hidden":
        next(v for v in views if v["experiment_id"] == m.r.RUN)["provenance"] = "NATIVE"
    else:
        views[-1]["provenance"] = m.EXCEPTION
    with pytest.raises(ValueError):
        m.reports(context, views)


def test_reporting_view_is_external_and_binds_training_metadata(views):
    cell = views[0]
    result = deepcopy(cell)
    run = {
        "data": {
            k: cell[k]
            for k in (
                "selection",
                "parameter_counts",
                "inner_training_seconds",
                "final_training_seconds",
            )
        }
    }
    before = deepcopy((cell, result, run))
    assert m.reporting_view(cell, result, run)["provenance"] == "NATIVE"
    assert (cell, result, run) == before
    result["final_training_seconds"] += 1
    with pytest.raises(ValueError, match="final_training_seconds"):
        m.reporting_view(cell, result, run)
    result = deepcopy(cell)
    result["metrics"]["accuracy"] += 0.1
    with pytest.raises(ValueError, match="metrics"):
        m.reporting_view(cell, result, run)


def test_reporting_view_preserves_unknown_prior_forwards(views):
    cell = next(v for v in views if v["experiment_id"] == m.r.RUN)
    result = {
        **deepcopy(cell),
        "provenance_flag": m.EXCEPTION,
        "original_prior_forward_count": "UNKNOWN",
    }
    run = {
        "data": {
            k: cell[k]
            for k in (
                "selection",
                "parameter_counts",
                "inner_training_seconds",
                "final_training_seconds",
            )
        }
    }
    assert m.reporting_view(cell, result, run)["provenance"] == m.EXCEPTION
    result["original_prior_forward_count"] = 0
    with pytest.raises(ValueError, match="Exception"):
        m.reporting_view(cell, result, run)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sequence_state", "READY"),
        ("verified_chain_completions", 7),
        ("next_target", {}),
        ("remaining_order", ["C2"]),
        ("previous_completion_sha256", "0" * 64),
        ("model_forwards", 1),
        ("native_writes", 1),
    ],
)
def test_complete_gate(field, value):
    status = {
        "status": "PASS",
        "sequence_state": "COMPLETE",
        "verified_chain_completions": 8,
        "next_target": None,
        "remaining_order": [],
        "previous_completion_sha256": m.FINAL_SHA,
        "model_forwards": 0,
        "native_writes": 0,
    }
    chain = [{"sha256": m.FINAL_SHA}] * 8
    m.require_complete(status, chain)
    status[field] = value
    with pytest.raises(ValueError):
        m.require_complete(status, chain)


@pytest.fixture
def lifecycle():
    runs = {m.r.RUN: {"state": "FINAL_TRAIN_COMPLETE", "claim": m.r.CLAIM}}
    runs.update({str(i): {"state": "OUTER_EVALUATED", "claim": f"{i:032x}"} for i in range(20)})
    runs.update({f"inner{i}": {"state": "INNER_COMPLETE", "claim": None} for i in range(9)})
    events = [[1, m.r.RUN, "", "OUTER_CLAIMED"]]
    for i in range(20):
        events.extend([[2, str(i), "", k] for k in ("OUTER_CLAIMED", "OUTER_EVALUATED")])
    events.extend([[1, m.r.RUN, "", "RUNNING"]] * (173 - len(events)))
    return {
        "native_runs_snapshot": runs,
        "native_events_snapshot": events,
        "native_files_snapshot": {str(i): {} for i in range(168)},
    }


@pytest.mark.parametrize(
    "mutation", ["state", "claim", "duplicate_event", "extra_file", "unfinished"]
)
def test_lifecycle_exact_and_fail_closed(lifecycle, mutation):
    assert m.validate_lifecycle(lifecycle)["unique_native_claims"] == 21
    runs = lifecycle["native_runs_snapshot"]
    if mutation == "state":
        runs["0"]["state"] = "FAILED"
    elif mutation == "claim":
        runs["0"]["claim"] = runs["1"]["claim"]
    elif mutation == "duplicate_event":
        lifecycle["native_events_snapshot"].append([174, "0", "", "OUTER_CLAIMED"])
    elif mutation == "extra_file":
        lifecycle["native_files_snapshot"]["temp"] = {}
    else:
        runs["inner0"]["claim"] = "f" * 32
    with pytest.raises(ValueError):
        m.validate_lifecycle(lifecycle)


@pytest.fixture
def package(tmp_path, context, views, monkeypatch):
    monkeypatch.setattr(m, "OUTPUT", "artifacts/final_closure")
    (tmp_path / "artifacts").mkdir(exist_ok=True)
    for relative in (m.SELF, m.TEST):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic reporting helper")
    cross, robust = m.reports(context, views)
    report = {"status": "PASS", "cells": views, "evidence_files": {}}
    files = m.assemble(tmp_path, report, cross, robust, {"status": "PASS"})
    return tmp_path, files


def test_immutable_package_handoff_and_checksums(package):
    root, files = package
    directory = m.publish(root, files)
    manifest = m.verify_package(directory)
    assert set(manifest["files"]) | {"evidence_manifest.json"} == set(files)
    handoff = json.loads(files["module6_handoff.json"])
    assert handoff["module5_closure_sha256"] == m.r.file_hash(
        directory / "module5_final_closure.json"
    )
    assert len(handoff["primary_checkpoints"]) == 3
    assert "NOT STARTED" in handoff["status"]
    assert "UNKNOWN" in handoff["limitations"][0]
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(ValueError, match="never replace"):
        m.publish(root, files)
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}


@pytest.mark.parametrize("mutation", ["changed", "missing", "extra"])
def test_package_corruption(package, mutation):
    root, files = package
    directory = m.publish(root, files)
    target = directory / "cross_fold_metrics.csv"
    if mutation == "changed":
        target.write_bytes(target.read_bytes() + b"x")
    elif mutation == "missing":
        target.unlink()
    else:
        (directory / "partial").write_bytes(b"x")
    with pytest.raises(ValueError):
        m.verify_package(directory)


def test_changed_evidence_cannot_publish(package):
    root, files = package
    evidence = root / "protected"
    evidence.write_bytes(b"original")
    manifest = json.loads(files["evidence_manifest.json"])
    manifest["input_evidence"] = {"protected": m.r.fingerprint(evidence)}
    files["evidence_manifest.json"] = m.s.canonical_bytes(manifest)
    evidence.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Evidence changed"):
        m.publish(root, files)
    assert not (root / m.OUTPUT).exists()
    assert list((root / "artifacts").glob(".final_closure_pending_*"))
    assert evidence.read_bytes() == b"changed"


def test_no_model_or_ledger_entry_in_reporting():
    import torch

    from vaaniswitch.m5_experiments.ledger import Ledger

    with m.s.forbid_scientific_execution():
        with pytest.raises(RuntimeError, match="forbidden"):
            torch.nn.Linear(1, 1)
        with pytest.raises(RuntimeError, match="forbidden"):
            Ledger(None)


def test_check_gate_rejects_changed_helper(tmp_path):
    path = tmp_path / m.SELF
    path.parent.mkdir(parents=True)
    path.write_text("changed")
    checks = tmp_path / "checks.json"
    checks.write_text(json.dumps({"status": "PASS", "helper_sha256": "old"}))
    with pytest.raises(ValueError, match="Tested helper"):
        m.checks(tmp_path, checks)


def test_cli_save_requires_checks(monkeypatch, capsys):
    monkeypatch.setattr(m, "audit", lambda _: ({"evidence_files": {}}, [], None))
    monkeypatch.setattr(
        m,
        "reports",
        lambda *_: ({"aggregation": "frozen", "ablations": {}}, {"groups": {}, "record_count": 0}),
    )
    assert m.main(["--save"]) == 1
    assert "Test evidence required" in capsys.readouterr().out
