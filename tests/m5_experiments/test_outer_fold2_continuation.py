"""Fold-2 handoff: exact closure dependency and synthetic native control-flow proof."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from scripts import module5_outer_fold2_continuation as g


@pytest.fixture(autouse=True)
def no_scientific_model(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("No scientific model construction/forward"))
    monkeypatch.setattr(torch.nn.Module, "_call_impl", forbidden)
    monkeypatch.setattr(torch.nn.Module, "__init__", forbidden)


def baseline():
    runs = {
        g.r.RUN: {"state": "FINAL_TRAIN_COMPLETE", "claim": g.r.CLAIM, "data": {}},
        "fold1_native": {"state": "OUTER_EVALUATED", "claim": "b" * 32, "data": {}},
        "fold3_untouched": {"state": "FINAL_TRAIN_COMPLETE", "claim": None, "data": {}},
    }
    for ablation, epoch in g.EPOCHS.items():
        key = g.run_id(ablation)
        selection = {"experiment_id": key, "candidate_id": "stronger_decay", "epoch": epoch}
        runs[key] = {
            "id": key,
            "identity": {"outer_fold": 2, "ablation": ablation},
            "state": "FINAL_TRAIN_COMPLETE",
            "claim": None,
            "data": {"selection": selection, "final_epoch": epoch, "final_checkpoint": "unchanged"},
        }
    return {
        "native_runs_snapshot": runs,
        "native_events_snapshot": [
            [145, "fold1_native", "2026-10-06T01:00:00+00:00", "OUTER_EVALUATED", "{}"]
        ],
        "native_files_snapshot": {
            "ledger.sqlite3": "original",
            "events.csv": "original",
            "fold1_result.json": "original",
            "final.pt": "original",
        },
    }


def advanced(original):
    value = deepcopy(original)
    key = g.run_id("A0")
    run = value["native_runs_snapshot"][key]
    run.update(state="OUTER_EVALUATED", claim="d" * 32)
    run["data"]["outer_result_sha256"] = "e" * 64
    value["native_events_snapshot"].extend(
        [
            [
                146,
                key,
                "2026-10-06T02:00:00+00:00",
                "OUTER_CLAIMED",
                json.dumps({"claim": "d" * 32}),
            ],
            [
                147,
                key,
                "2026-10-06T02:01:00+00:00",
                "OUTER_EVALUATED",
                json.dumps({"outer_result_sha256": "e" * 64}),
            ],
        ]
    )
    value["native_files_snapshot"].update(
        {
            "ledger.sqlite3": "advanced",
            "events.csv": "advanced",
            f"runs/{key}/outer_result.json": "new result",
        }
    )
    return value


@pytest.mark.parametrize("mode", ["preflight", "verify"])
def test_only_first_fold2_target_enabled(mode):
    g.scope(mode, 2, "A0")
    for fold, ablation in [(1, "A0"), (3, "A0"), (True, "A0"), *((2, a) for a in g.ORDER[1:])]:
        with pytest.raises(ValueError):
            g.scope(mode, fold, ablation)


@pytest.mark.parametrize("completed", [False, True])
def test_exact_native_delta_and_preservation(completed):
    original = baseline()
    g.validate_progress(
        original, advanced(original) if completed else original, completed=completed
    )


@pytest.mark.parametrize(
    "damage",
    [
        "second_claim",
        "a1_claim",
        "fold1",
        "fold3",
        "later_target",
        "prior_target_claim",
        "selection",
        "checkpoint",
        "extra_result",
        "old_event",
        "extra_event",
        "missing_file",
    ],
)
@pytest.mark.parametrize("completed", [False, True])
def test_unauthorized_progress_rejected(damage, completed):
    original = baseline()
    current = advanced(original) if completed else deepcopy(original)
    runs, files = current["native_runs_snapshot"], current["native_files_snapshot"]
    if damage in {"second_claim", "later_target"}:
        runs[g.run_id("A1")]["claim"] = "f" * 32
    elif damage == "a1_claim":
        runs[g.r.RUN]["claim"] = "changed"
    elif damage == "fold1":
        runs["fold1_native"]["data"]["changed"] = True
    elif damage == "fold3":
        runs["fold3_untouched"]["data"]["changed"] = True
    elif damage == "prior_target_claim":
        runs[g.run_id("A0")]["claim"] = "b" * 32  # pending or duplicate completed token
    elif damage == "selection":
        runs[g.run_id("A0")]["data"]["selection"]["epoch"] = 12
    elif damage == "checkpoint":
        files["final.pt"] = "changed"
    elif damage == "extra_result":
        files["runs/later/outer_result.json"] = "unexpected"
    elif damage == "old_event":
        current["native_events_snapshot"][0][-1] = "changed"
    elif damage == "extra_event":
        current["native_events_snapshot"].append([148, "other", "time", "FAILED", "{}"])
    else:
        del files["fold1_result.json"]
    with pytest.raises(ValueError):
        g.validate_progress(original, current, completed=completed)


@pytest.mark.parametrize("missing", [True, False])
def test_closure_required_and_checksum_pinned(tmp_path, missing):
    if not missing:
        path = tmp_path / g.p.REPORT
        path.parent.mkdir(parents=True)
        path.write_text('{"status": "PASS"}')
    with pytest.raises((OSError, ValueError)):
        g.load_closure(tmp_path)


def test_frozen_fold2_epochs_and_candidate():
    runs = baseline()["native_runs_snapshot"]
    bundle = {"selections": {a: deepcopy(runs[g.run_id(a)]["data"]["selection"]) for a in g.ORDER}}
    g.validate_selections(bundle, runs)
    bundle["selections"]["A0"]["epoch"] = 12
    with pytest.raises(ValueError):
        g.validate_selections(bundle, runs)
    bundle["selections"]["A0"] = deepcopy(runs[g.run_id("A0")]["data"]["selection"])
    bundle["selections"]["C2"]["candidate_id"] = "baseline"
    with pytest.raises(ValueError):
        g.validate_selections(bundle, runs)


def test_wrong_fold2_record_hash_or_order_rejected(monkeypatch):
    records = tuple(NS(logical_id=str(i)) for i in range(2873))
    audit = {
        "outer_evaluation_plan": [
            {
                "outer_fold": 2,
                "record_hash": g.RECORD_HASH,
                "logical_ids": [v.logical_id for v in records],
            },
        ]
    }
    monkeypatch.setattr(g.c.native, "record_hash", lambda _: g.RECORD_HASH)
    g.validate_records(records, audit)
    with pytest.raises(ValueError):
        g.validate_records(records[::-1], audit)
    monkeypatch.setattr(g.c.native, "record_hash", lambda _: "changed")
    with pytest.raises(ValueError):
        g.validate_records(records, audit)


@pytest.mark.parametrize("mode", ["preflight", "verify"])
def test_fold2_inspection_is_read_only_and_preserves_recovery(tmp_path, monkeypatch, mode):
    original = baseline()
    current = advanced(original) if mode == "verify" else deepcopy(original)
    sentinel = tmp_path / "protected"
    sentinel.write_bytes(b"unchanged")
    initial = sentinel.read_bytes()
    context = NS(
        split=Mock(return_value=NS(records=Mock(return_value=()))),
        predecessor=NS(source=NS(contract=object())),
    )
    monkeypatch.setattr(g, "load_closure", lambda _: original)
    monkeypatch.setattr(g.p, "prepare", lambda _: ({}, context, ()))
    monkeypatch.setattr(g.p, "snapshot", lambda _: deepcopy(current))
    monkeypatch.setattr(g.c, "verify_csv", Mock())
    exception = Mock()
    monkeypatch.setattr(g.c, "verify_exception", exception)
    monkeypatch.setattr(g, "validate_records", Mock())
    monkeypatch.setattr(g, "validate_selections", Mock())
    monkeypatch.setattr(g.r, "read_json", lambda _: {})
    monkeypatch.setattr(g.p, "final_checkpoint", lambda *_: {"selected_epoch": 11})
    preservation = Mock()
    monkeypatch.setattr(g.p, "assert_preserved", preservation)
    monkeypatch.setattr(g.r, "file_hash", lambda _: g.CLOSURE_SHA)
    monkeypatch.setattr(
        g.c.native,
        "completed_result",
        Mock(
            return_value={
                "record_hash": g.RECORD_HASH,
                "result_sha256": "e" * 64,
                "metrics": {"accuracy": 0.5},
            }
        ),
    )
    output = g.inspect(mode, 2, "A0", tmp_path)
    assert output["model_forwards"] == output["production_writes"] == 0
    assert output["fold1_unchanged"] and output["a1_exception_unchanged"]
    assert output["fold3_outer_untouched"] and output["no_later_fold2_target_ran"]
    assert list(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == initial
    exception.assert_called_once()
    preservation.assert_called_once()
    context.split.assert_called_once_with(2)


@pytest.mark.parametrize("ablation", g.ORDER)
def test_frozen_native_fold2_gate_and_actual_sql_claim_are_target_local(
    tmp_path,
    monkeypatch,
    ablation,
):
    from vaaniswitch.m5_experiments import evaluator
    from vaaniswitch.m5_experiments.ledger import Ledger

    config = NS(output=lambda rel: tmp_path / rel, cpu_threads=2, candidate=lambda _: NS())
    ledger = Ledger(config)  # Real SQL/locks, synthetic temporary namespace only.
    runs = baseline()["native_runs_snapshot"]
    for a in g.ORDER:
        key = g.run_id(a)
        runs[key]["identity"].update(experiment_id=key, split_hash="fold2-only")
        runs[key]["data"].update(
            final_identity={"only": key},
            parameter_counts={},
            inner_training_seconds=0,
            final_training_seconds=0,
        )
    with ledger.connect() as db:
        for key, run in runs.items():
            db.execute(
                "INSERT INTO runs VALUES(?,?,?,?,?)",
                (
                    key,
                    json.dumps(run.get("identity", {})),
                    run["state"],
                    json.dumps(run["data"]),
                    run["claim"],
                ),
            )
    old = {v["id"]: v for v in ledger.all_runs()}
    bundle = {"selections": {a: runs[g.run_id(a)]["data"]["selection"] for a in g.ORDER}}
    selections = deepcopy(bundle)
    split = NS(split_hash="fold2-only")
    context = NS(config=config, split=Mock(return_value=split))
    record = NS(logical_id="fold2-control", speaker_id="FC02")
    dataset = NS(records=[record], loaded=[record.logical_id], guard=Mock())
    batch = NS(provenance=[record])
    model = Mock(return_value=NS(probabilities=NS(tolist=lambda: [0.2])))
    monkeypatch.setattr(evaluator, "require_frozen", Mock())
    original_read = evaluator.read_json
    monkeypatch.setattr(
        evaluator,
        "read_json",
        lambda path: bundle if path.name == "fold2_ablations.json" else original_read(path),
    )
    monkeypatch.setattr(evaluator, "build_model", Mock(return_value=model))
    monkeypatch.setattr(evaluator, "make_optimizer", Mock(return_value=object()))
    monkeypatch.setattr(
        evaluator, "load_checkpoint", Mock(return_value={"progress": {"epoch": g.EPOCHS[ablation]}})
    )

    def fake_dataset(ctx, requested_split, role, requested_ablation, *, permit):
        assert ctx is context and requested_split is split and role == "outer_test"
        assert requested_ablation == ablation
        permit.verify(split, ablation)  # Actual native permit, including SQL claim check.
        return dataset

    monkeypatch.setattr(evaluator, "ExperimentDataset", fake_dataset)
    monkeypatch.setattr(evaluator, "batches", lambda _: [batch])
    monkeypatch.setattr(evaluator, "batch_labels", lambda _: NS(tolist=lambda: [0]))
    result = evaluator.evaluate_outer(context, 2, ablation)
    assert result["outer_fold"] == 2 and model.call_count == 1
    assert result["selection"]["epoch"] == g.EPOCHS[ablation]
    after = {v["id"]: v for v in ledger.all_runs()}
    assert all(after[k] == v for k, v in old.items() if k != g.run_id(ablation))
    assert after[g.r.RUN] == old[g.r.RUN]
    assert bundle == selections
    context.split.assert_called_once_with(2)
    with ledger.connect() as db:
        events = db.execute("SELECT run_id,event FROM events ORDER BY seq").fetchall()
    assert events == [
        (g.run_id(ablation), "OUTER_CLAIMED"),
        (g.run_id(ablation), "OUTER_EVALUATED"),
    ]
    assert evaluator.evaluate_outer(context, 2, ablation) == result and model.call_count == 1
    assert len(list(tmp_path.glob("experiments/runs/*/outer_result.json"))) == 1
