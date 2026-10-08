"""Synthetic regression coverage for the A1-only read-only continuation gate."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from scripts import module5_outer_fold2_a1_continuation as h
from tests.m5_experiments.test_outer_fold2_continuation import advanced, baseline


@pytest.fixture(autouse=True)
def forbid_scientific_execution(monkeypatch):
    from vaaniswitch.m5_experiments import evaluator
    from vaaniswitch.m5_experiments.ledger import Ledger

    forbidden = Mock(side_effect=AssertionError("No scientific execution or native writes"))
    for obj, name in [
        (torch.nn.Module, "__init__"),
        (torch.nn.Module, "_call_impl"),
        (Ledger, "__init__"),
        (evaluator, "evaluate_outer"),
        (h.r, "build_model"),
        (h.r, "ExperimentDataset"),
    ]:
        monkeypatch.setattr(obj, name, forbidden)


def completion():
    value = advanced(baseline())
    value["verification"] = {**h.A0_EXPECTED, "metrics": {"accuracy": 0.5}}
    return value


def a1_completed(original):
    current = deepcopy(original)
    key = h.g.run_id("A1")
    run = current["native_runs_snapshot"][key]
    run.update(state="OUTER_EVALUATED", claim="1" * 32)
    run["data"]["outer_result_sha256"] = "2" * 64
    current["native_events_snapshot"].extend(
        [
            [
                148,
                key,
                "2026-10-06T03:00:00+00:00",
                "OUTER_CLAIMED",
                json.dumps({"claim": "1" * 32}),
            ],
            [
                149,
                key,
                "2026-10-06T03:01:00+00:00",
                "OUTER_EVALUATED",
                json.dumps({"outer_result_sha256": "2" * 64}),
            ],
        ]
    )
    current["native_files_snapshot"].update(
        {
            "ledger.sqlite3": "A1 completed",
            "events.csv": "A1 completed",
            f"runs/{key}/outer_result.json": "A1 result",
        }
    )
    return current


@pytest.mark.parametrize("mode", ["preflight", "verify"])
def test_only_fold2_a1_is_enabled(mode):
    h.scope(mode, 2, "A1")
    for fold, ablation in [
        (1, "A1"),
        (3, "A1"),
        (True, "A1"),
        *((2, a) for a in h.g.ORDER if a != "A1"),
    ]:
        with pytest.raises(ValueError):
            h.scope(mode, fold, ablation)


@pytest.mark.parametrize("completed", [False, True])
def test_exact_a1_two_event_delta_is_accepted(completed):
    original = completion()
    h.validate_progress(
        original, a1_completed(original) if completed else original, completed=completed
    )


@pytest.mark.parametrize(
    "damage",
    [
        "a0_claim",
        "a0_result",
        "fold1",
        "fold1_exception",
        "fold3",
        "second_claim",
        "prior_a1_claim",
        "a1_checkpoint",
        "a1_selection",
        "later_fold2",
        "old_event",
        "extra_event",
        "wrong_event",
        "duplicate_claim",
        "unexpected_result",
        "missing_file",
    ],
)
@pytest.mark.parametrize("completed", [False, True])
def test_all_unapproved_native_changes_fail(damage, completed):
    original = completion()
    current = a1_completed(original) if completed else deepcopy(original)
    runs, files = current["native_runs_snapshot"], current["native_files_snapshot"]
    target = runs[h.g.run_id("A1")]
    if damage == "a0_claim":
        runs[h.g.run_id("A0")]["claim"] = "f" * 32
    elif damage == "a0_result":
        files[f"runs/{h.g.run_id('A0')}/outer_result.json"] = "changed"
    elif damage == "fold1":
        runs["fold1_native"]["data"]["changed"] = True
    elif damage == "fold1_exception":
        runs[h.r.RUN]["claim"] = "f" * 32
    elif damage == "fold3":
        runs["fold3_untouched"]["data"]["changed"] = True
    elif damage in {"second_claim", "later_fold2"}:
        runs[h.g.run_id("B1")]["claim"] = "f" * 32
    elif damage == "prior_a1_claim":
        target.update(state="FINAL_TRAIN_COMPLETE", claim="f" * 32)
    elif damage == "a1_checkpoint":
        target["data"]["final_checkpoint"] = "changed"
    elif damage == "a1_selection":
        target["data"]["selection"]["epoch"] = 3
    elif damage == "old_event":
        current["native_events_snapshot"][0][-1] = "changed"
    elif damage == "extra_event":
        current["native_events_snapshot"].append([150, h.g.run_id("B1"), "time", "FAILED", "{}"])
    elif damage == "wrong_event":
        current["native_events_snapshot"][-1][1] = h.g.run_id("B1")
    elif damage == "duplicate_claim":
        target["claim"] = runs[h.g.run_id("A0")]["claim"]
        if completed:
            current["native_events_snapshot"][-2][-1] = json.dumps({"claim": target["claim"]})
    elif damage == "unexpected_result":
        files[f"runs/{h.g.run_id('B1')}/outer_result.json"] = "unexpected"
    else:
        del files["final.pt"]
    with pytest.raises(ValueError):
        h.validate_progress(original, current, completed=completed)


@pytest.mark.parametrize(
    "field",
    [
        "experiment_id",
        "target_state",
        "target_claim",
        "result_sha256",
        "checkpoint_sha256",
        "candidate",
        "selected_epoch",
        "record_count",
        "record_hash",
        "threshold",
        "provenance",
    ],
)
def test_a0_pins_are_exact(field):
    value = dict(h.A0_EXPECTED)
    h.validate_a0(value)
    value[field] = "changed"
    with pytest.raises(ValueError):
        h.validate_a0(value)


@pytest.mark.parametrize("missing", [False, True])
def test_completion_and_fold1_closure_require_exact_hashes(tmp_path, missing):
    if not missing:
        for relative in (h.COMPLETION, h.p.REPORT):
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"status":"PASS"}')
    with pytest.raises((OSError, ValueError)):
        h.load_completion(tmp_path)
    with pytest.raises((OSError, ValueError)):
        h.g.load_closure(tmp_path)


@pytest.mark.parametrize("ablation", ["A0", "A1"])
@pytest.mark.parametrize("damage", ["hash", "metrics", "order", "labels", "probability"])
def test_retained_predictions_are_independently_recomputed(tmp_path, monkeypatch, ablation, damage):
    records = (
        NS(logical_id="r0", speaker_id="FC02", condition=NS(value="control")),
        NS(logical_id="r1", speaker_id="M01", condition=NS(value="dysarthric")),
    )
    key = h.g.run_id(ablation)
    run = {
        "identity": {"experiment_id": key, "ablation": ablation, "outer_fold": 2},
        "state": "OUTER_EVALUATED",
        "claim": h.A0_EXPECTED["target_claim"],
        "data": {
            "selection": {"epoch": h.g.EPOCHS[ablation]},
            "final_checkpoint": {},
            "final_identity": {},
        },
    }
    result = {
        "status": "OUTER_EVALUATED",
        "experiment_id": key,
        "identity": run["identity"],
        "selection": run["data"]["selection"],
        "checkpoint": {},
        "ablation": ablation,
        "outer_fold": 2,
        "claim": run["claim"],
        "threshold": 0.5,
        "forward_count": 2,
        "source_load_count": 2,
        "predictions": [
            {
                "logical_id": rec.logical_id,
                "speaker_id": rec.speaker_id,
                "label": index,
                "probability": (0.2, 0.8)[index],
            }
            for index, rec in enumerate(records)
        ],
        "metrics": h.c.native.binary_metrics([0, 1], [0.2, 0.8]),
    }
    base = tmp_path / h.r.BASE / "experiments"
    path = base / "runs" / key / "outer_result.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(result))
    run["data"]["outer_result_sha256"] = h.r.file_hash(path)
    events = [
        [1, key, "time", "OUTER_CLAIMED", json.dumps({"claim": run["claim"]})],
        [
            2,
            key,
            "time",
            "OUTER_EVALUATED",
            json.dumps({"outer_result_sha256": h.r.file_hash(path)}),
        ],
    ]
    monkeypatch.setattr(h.c.native, "ROOT", tmp_path)
    synthetic_pins = {
        **h.A0_EXPECTED,
        "record_count": 2,
        "record_hash": h.c.native.record_hash(records),
        "result_sha256": h.r.file_hash(path),
    }
    monkeypatch.setattr(h, "A0_EXPECTED", synthetic_pins)
    monkeypatch.setattr(
        h.p,
        "final_checkpoint",
        lambda *_: {
            "candidate": "stronger_decay",
            "selected_epoch": 11,
            "checkpoint_sha256": h.A0_EXPECTED["checkpoint_sha256"],
        },
    )

    def verify():
        if ablation == "A0":
            return h.verify_a0(
                tmp_path,
                {"native_runs_snapshot": {key: run}, "native_events_snapshot": events},
                records,
            )
        return h.c.native.completed_result(base, key, run, events, records)

    assert verify()["metrics"]["accuracy"] == 1
    if damage == "metrics":
        result["metrics"]["accuracy"] = 0
    elif damage == "order":
        result["predictions"].reverse()
    elif damage == "labels":
        result["predictions"][0]["label"] = 1
    elif damage == "probability":
        result["predictions"][0]["probability"] = float("nan")
    else:
        result["threshold"] = 0.6
    path.write_text(json.dumps(result))
    if damage != "hash":
        # Updated checksums cannot conceal semantic corruption of retained evidence.
        checksum = h.r.file_hash(path)
        run["data"]["outer_result_sha256"] = synthetic_pins["result_sha256"] = checksum
        events[-1][-1] = json.dumps({"outer_result_sha256": checksum})
    with pytest.raises(ValueError):
        verify()


def test_wrong_fold2_record_order_or_hash(monkeypatch):
    records = tuple(NS(logical_id=str(index)) for index in range(2873))
    audit = {
        "outer_evaluation_plan": [
            {
                "outer_fold": 2,
                "record_hash": h.g.RECORD_HASH,
                "logical_ids": [value.logical_id for value in records],
            }
        ]
    }
    monkeypatch.setattr(h.c.native, "record_hash", lambda _: h.g.RECORD_HASH)
    h.g.validate_records(records, audit)
    with pytest.raises(ValueError):
        h.g.validate_records(records[::-1], audit)
    monkeypatch.setattr(h.c.native, "record_hash", lambda _: "changed")
    with pytest.raises(ValueError):
        h.g.validate_records(records, audit)


@pytest.mark.parametrize("mode", ["preflight", "verify"])
@pytest.mark.parametrize("bad_checkpoint", [False, True])
def test_inspection_has_zero_forwards_writes_and_preserves_inputs(
    tmp_path,
    monkeypatch,
    mode,
    bad_checkpoint,
):
    original = completion()
    current = a1_completed(original) if mode == "verify" else deepcopy(original)
    sentinel = tmp_path / "native-and-recovery"
    sentinel.write_bytes(b"unchanged")
    context = NS(
        split=Mock(return_value=NS(records=Mock(return_value=()))),
        predecessor=NS(source=NS(contract=object())),
    )
    monkeypatch.setattr(h, "load_completion", lambda _: original)
    monkeypatch.setattr(h.p, "prepare", lambda _: ({}, context, ()))
    monkeypatch.setattr(h.p, "snapshot", lambda _: deepcopy(current))
    monkeypatch.setattr(h.c, "verify_csv", Mock())
    exception = Mock()
    monkeypatch.setattr(h.c, "verify_exception", exception)
    monkeypatch.setattr(h.g, "validate_records", Mock())
    monkeypatch.setattr(h.g, "validate_selections", Mock())
    monkeypatch.setattr(h.r, "read_json", lambda _: {})
    monkeypatch.setattr(h, "verify_a0", lambda *_: original["verification"])
    monkeypatch.setattr(
        h.p,
        "final_checkpoint",
        lambda *_: {
            "checkpoint_sha256": "changed" if bad_checkpoint else h.A1_CHECKPOINT,
            "candidate": "stronger_decay",
            "selected_epoch": 2,
        },
    )
    preservation = Mock()
    monkeypatch.setattr(h.p, "assert_preserved", preservation)
    monkeypatch.setattr(
        h.r,
        "file_hash",
        lambda path: h.g.CLOSURE_SHA if path == tmp_path / h.p.REPORT else h.COMPLETION_SHA,
    )
    monkeypatch.setattr(
        h.c.native,
        "completed_result",
        Mock(
            return_value={
                "record_count": 2873,
                "record_hash": h.g.RECORD_HASH,
                "result_sha256": "2" * 64,
                "metrics": {"accuracy": 0.5},
            }
        ),
    )
    if bad_checkpoint:
        with pytest.raises(ValueError, match="checkpoint"):
            h.inspect(mode, 2, "A1", tmp_path)
    else:
        output = h.inspect(mode, 2, "A1", tmp_path)
        assert output["model_forwards"] == output["production_writes"] == 0
        assert output["fold1_unchanged"] and output["fold1_a1_exception_unchanged"]
        assert output["fold2_a0_unchanged"] and output["later_fold2_targets_untouched"]
        assert output["fold3_outer_untouched"]
        exception.assert_called_once()
        preservation.assert_called_once()
    assert list(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"unchanged"
