"""Fold-1 exception closure uses retained evidence only, never a scientific model."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from scripts import module5_outer_exception_report as p


@pytest.fixture(autouse=True)
def forbid_scientific_execution(monkeypatch):
    from vaaniswitch.m5_experiments import evaluator
    from vaaniswitch.m5_experiments.ledger import Ledger

    forbidden = Mock(side_effect=AssertionError("No scientific execution or native writes"))
    monkeypatch.setattr(torch.nn.Module, "_call_impl", forbidden)
    monkeypatch.setattr(evaluator, "build_model", forbidden)
    monkeypatch.setattr(evaluator, "ExperimentDataset", forbidden)
    monkeypatch.setattr(Ledger, "__init__", forbidden)
    monkeypatch.setattr(p.r, "build_model", forbidden)
    monkeypatch.setattr(p.r, "ExperimentDataset", forbidden)


def panel():
    cells = []
    for ablation in p.ORDER:
        cell = {
            "ablation": ablation,
            "experiment_id": p.c.run_id(ablation),
            "outer_fold": 1,
            "record_count": 2819,
            "record_hash": p.r.RECORD_HASH,
            "threshold": 0.5,
            "provenance": "NATIVE",
            "native_state": "OUTER_EVALUATED",
            "claim": p.NATIVE_CLAIM.get(ablation),
            "result_sha256": p.NATIVE_SHA.get(ablation),
            "metrics": {"accuracy": 0.5},
        }
        if ablation == "A1":
            cell.update(
                provenance="RECOVERED_PROTOCOL_EXCEPTION",
                native_state="FINAL_TRAIN_COMPLETE",
                claim=p.r.CLAIM,
                native_result_absent=True,
                attempt_id=p.c.ATTEMPT,
                result_sha256=p.c.RESULT_SHA,
                manifest_sha256=p.c.MANIFEST_SHA,
                original_prior_forward_count="UNKNOWN",
                uninterrupted_native_once_only=False,
            )
        cells.append(cell)
    return cells


@pytest.mark.parametrize(
    "ablation,field,value",
    [
        ("A1", "provenance", "NATIVE"),
        ("B1", "provenance", "RECOVERED_PROTOCOL_EXCEPTION"),
        ("A1", "claim", "changed"),
        ("B2", "claim", "changed"),
        ("A1", "result_sha256", "changed"),
        ("A1", "manifest_sha256", "changed"),
        ("C2", "result_sha256", "changed"),
        ("A1", "original_prior_forward_count", 0),
        ("A1", "uninterrupted_native_once_only", True),
        ("A1", "native_result_absent", False),
        ("B3", "record_hash", "changed"),
        ("C1", "record_count", 2873),
        ("A0", "threshold", 0.6),
    ],
)
def test_closure_rejects_relabeling_second_exception_and_changed_bindings(ablation, field, value):
    cells = panel()
    p.validate_cells(cells)
    next(cell for cell in cells if cell["ablation"] == ablation)[field] = value
    with pytest.raises(ValueError):
        p.validate_cells(cells)


def test_panel_requires_exact_seven_cells_in_declared_order():
    cells = panel()
    cells[0], cells[1] = cells[1], cells[0]
    with pytest.raises(ValueError, match="ordered"):
        p.validate_cells(cells)


@pytest.mark.parametrize("damage", ["checksum", "metrics", "order", "label", "hash"])
def test_retained_native_verification_rejects_tampering(tmp_path, monkeypatch, damage):
    records = (
        NS(logical_id="control-row", speaker_id="FC01", condition=NS(value="control")),
        NS(logical_id="dysarthric-row", speaker_id="F03", condition=NS(value="dysarthric")),
    )
    key = p.c.run_id("B1")
    run = {
        "identity": {"experiment_id": key, "ablation": "B1", "outer_fold": 1},
        "state": "OUTER_EVALUATED",
        "claim": p.NATIVE_CLAIM["B1"],
        "data": {"selection": {"epoch": 1}, "final_checkpoint": {}, "final_identity": {}},
    }
    result = {
        "status": "OUTER_EVALUATED",
        "experiment_id": key,
        "identity": run["identity"],
        "selection": run["data"]["selection"],
        "checkpoint": {},
        "ablation": "B1",
        "outer_fold": 1,
        "claim": run["claim"],
        "threshold": 0.5,
        "forward_count": 2,
        "source_load_count": 2,
        "predictions": [
            {
                "logical_id": rec.logical_id,
                "speaker_id": rec.speaker_id,
                "label": i,
                "probability": (0.2, 0.8)[i],
            }
            for i, rec in enumerate(records)
        ],
        "metrics": p.c.native.binary_metrics([0, 1], [0.2, 0.8]),
    }
    path = tmp_path / "runs" / key / "outer_result.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(result))
    run["data"]["outer_result_sha256"] = p.r.file_hash(path)
    events = [
        [1, key, "time", "OUTER_CLAIMED", json.dumps({"claim": run["claim"]})],
        [
            2,
            key,
            "time",
            "OUTER_EVALUATED",
            json.dumps({"outer_result_sha256": p.r.file_hash(path)}),
        ],
    ]
    monkeypatch.setattr(p.c.native, "ROOT", tmp_path)
    assert (
        p.c.native.completed_result(tmp_path, key, run, events, records)["metrics"]["accuracy"] == 1
    )
    if damage == "metrics":
        result["metrics"]["accuracy"] = 0
    elif damage == "order":
        result["predictions"].reverse()
    elif damage == "label":
        result["predictions"][0]["label"] = 1
    elif damage == "hash":
        monkeypatch.setattr(p.c.native, "record_hash", lambda _: "wrong")
    else:
        result["threshold"] = 0.6
    path.write_text(json.dumps(result))
    if damage != "checksum":
        # Rebind checksum to prove semantic checks catch internally checksummed corruption.
        run["data"]["outer_result_sha256"] = p.r.file_hash(path)
        events[-1][-1] = json.dumps({"outer_result_sha256": p.r.file_hash(path)})
    with pytest.raises(ValueError):
        p.c.native.completed_result(tmp_path, key, run, events, records)


def test_read_only_closure_preserves_inputs_and_only_publishes_external_report(
    tmp_path, monkeypatch
):
    sentinel = tmp_path / "native-evidence"
    sentinel.write_bytes(b"immutable")
    before = p.r.fingerprint(sentinel)
    state = {
        "native_files_snapshot": {"native-evidence": before},
        "native_runs_snapshot": {},
        "native_events_snapshot": [],
    }
    acceptance = {"recovery_files_snapshot": {"recovery": "unchanged"}}
    monkeypatch.setattr(p, "prepare", lambda _: (acceptance, NS(), ()))
    monkeypatch.setattr(p, "snapshot", lambda _: deepcopy(state))
    monkeypatch.setattr(p.c, "validate_native_progress", Mock())
    monkeypatch.setattr(p.c, "verify_csv", Mock())
    monkeypatch.setattr(p, "fold1_cells", lambda *_: panel())
    monkeypatch.setattr(p.r, "inventory", lambda _: acceptance["recovery_files_snapshot"])
    monkeypatch.setattr(p.c, "validate_frozen_runtime", Mock())
    report = p.inspect(tmp_path)
    assert report["model_forwards"] == report["production_writes"] == 0
    assert report["disclosure"] == p.DISCLOSURE
    assert not report["cross_fold_aggregation"] and not report["ranking_or_selection"]
    assert list(tmp_path.iterdir()) == [sentinel]
    destination = p.publish(tmp_path, report)
    saved = p.r.fingerprint(destination)
    assert p.publish(tmp_path, report) == destination  # Idempotent; never overwritten.
    assert p.r.fingerprint(destination) == saved and p.r.fingerprint(sentinel) == before
    report["status"] = "changed"
    with pytest.raises(ValueError, match="never overwrite"):
        p.publish(tmp_path, report)
    assert p.r.fingerprint(destination) == saved


@pytest.mark.parametrize("fold", [2, 3])
def test_closure_rejects_outer_activity_in_other_folds(fold):
    from tests.m5_experiments.test_outer_exception_continuation import advanced, base_snapshot

    original = base_snapshot()
    runs, events, files = advanced(original, p.c.ORDER)
    runs[f"fold{fold}_untouched"]["claim"] = "e" * 32
    with pytest.raises(ValueError, match="unfinished"):
        p.c.validate_native_progress(original, runs, events, files, p.c.ORDER)
