"""Synthetic fail-closed tests for the external Fold2/B2 read-only guard."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from scripts import module5_outer_fold2_b2_continuation as m
from tests.m5_experiments.test_outer_fold2_b1_continuation import a0_completion, b1_completed


@pytest.fixture(autouse=True)
def forbid_execution(monkeypatch):
    from vaaniswitch.m5_experiments import evaluator
    from vaaniswitch.m5_experiments.ledger import Ledger

    forbidden = Mock(side_effect=AssertionError("No scientific execution or native writer"))
    for obj, name in [
        (torch.nn.Module, "__init__"),
        (torch.nn.Module, "_call_impl"),
        (Ledger, "__init__"),
        (evaluator, "evaluate_outer"),
        (m.r, "build_model"),
        (m.r, "ExperimentDataset"),
    ]:
        monkeypatch.setattr(obj, name, forbidden)


def completion():
    value = b1_completed(a0_completion())
    value["verification"] = {**m.B1_EXPECTED, "metrics": {"accuracy": 0.5}}
    return value


def advance(original):
    current = deepcopy(original)
    key = m.g.run_id("B2")
    run = current["native_runs_snapshot"][key]
    run.update(state="OUTER_EVALUATED", claim="5" * 32)
    run["data"]["outer_result_sha256"] = "6" * 64
    current["native_events_snapshot"].extend(
        [
            [
                152,
                key,
                "2026-10-06T08:00:00+00:00",
                "OUTER_CLAIMED",
                json.dumps({"claim": run["claim"]}),
            ],
            [
                153,
                key,
                "2026-10-06T08:01:00+00:00",
                "OUTER_EVALUATED",
                json.dumps({"outer_result_sha256": "6" * 64}),
            ],
        ]
    )
    current["native_files_snapshot"].update(
        {
            "ledger.sqlite3": {"sha256": "after-b2"},
            "events.csv": {"sha256": "after-b2"},
            f"runs/{key}/outer_result.json": {"sha256": "6" * 64},
        }
    )
    return current


@pytest.mark.parametrize("mode", ["preflight", "verify"])
def test_scope_and_cli_exclude_every_other_target(mode):
    m.scope(mode, 2, "B2")
    for fold, ablation in [
        (1, "B2"),
        (3, "B2"),
        (True, "B2"),
        *((2, a) for a in m.g.ORDER if a != "B2"),
    ]:
        with pytest.raises(ValueError):
            m.scope(mode, fold, ablation)
    for invalid in ["train", "outer", "resume", "save"]:
        with pytest.raises(ValueError):
            m.scope(invalid, 2, "B2")
        with pytest.raises(SystemExit):
            m.main([invalid, "--outer-fold", "2", "--ablation", "B2"])


@pytest.mark.parametrize("completed", [False, True])
def test_exact_b2_delta_only(completed):
    original = completion()
    m.validate_progress(original, advance(original) if completed else original, completed=completed)


@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize(
    "damage",
    [
        "a0_claim",
        "a1_result",
        "b1_checkpoint",
        "b1_file",
        "fold1",
        "exception",
        "prior_b2_claim",
        "b2_checkpoint",
        "b2_selection",
        "prior_b2_event",
        "b3",
        "c1",
        "c2",
        "fold3",
        "old_event",
        "extra_event",
        "wrong_event",
        "duplicate_claim",
        "extra_file",
        "missing_file",
        "bad_result_hash",
    ],
)
def test_unapproved_progress_is_rejected(completed, damage):
    original = completion()
    current = advance(original) if completed else deepcopy(original)
    runs, files, events = (
        current["native_runs_snapshot"],
        current["native_files_snapshot"],
        current["native_events_snapshot"],
    )
    target = runs[m.g.run_id("B2")]
    if damage == "a0_claim":
        runs[m.g.run_id("A0")]["claim"] = "f" * 32
    elif damage == "a1_result":
        runs[m.g.run_id("A1")]["data"]["outer_result_sha256"] = "f" * 64
    elif damage == "b1_checkpoint":
        runs[m.g.run_id("B1")]["data"]["final_checkpoint"] = "changed"
    elif damage == "b1_file":
        files[f"runs/{m.g.run_id('B1')}/outer_result.json"]["sha256"] = "changed"
    elif damage == "fold1":
        runs["fold1_native"]["data"]["changed"] = True
    elif damage == "exception":
        runs[m.r.RUN]["claim"] = "f" * 32
    elif damage == "prior_b2_claim":
        target.update(state="FINAL_TRAIN_COMPLETE", claim="f" * 32)
    elif damage == "b2_checkpoint":
        target["data"]["final_checkpoint"] = "changed"
    elif damage == "b2_selection":
        target["data"]["selection"]["epoch"] = 8
    elif damage in {"b3", "c1", "c2"}:
        runs[m.g.run_id(damage.upper())]["claim"] = "f" * 32
    elif damage == "fold3":
        runs["fold3_untouched"]["data"]["changed"] = True
    elif damage == "old_event":
        events[0][-1] = "changed"
    elif damage in {"extra_event", "prior_b2_event"}:
        events.append([154, m.g.run_id("B2"), "2026-10-06T09:00:00+00:00", "OUTER_CLAIMED", "{}"])
    elif damage == "wrong_event":
        events[-1][1] = m.g.run_id("B3")
    elif damage == "duplicate_claim":
        target["claim"] = runs[m.g.run_id("A0")]["claim"]
        if completed:
            events[-2][-1] = json.dumps({"claim": target["claim"]})
    elif damage == "extra_file":
        files["unexpected.tmp"] = {"sha256": "unexpected"}
    elif damage == "missing_file":
        del files["final.pt"]
    else:
        files[f"runs/{m.g.run_id('B2')}/outer_result.json"] = {"sha256": "wrong"}
    with pytest.raises(ValueError):
        m.validate_progress(original, current, completed=completed)


@pytest.mark.parametrize("field", list(m.B1_EXPECTED))
def test_b1_identity_pins(field):
    value = dict(m.B1_EXPECTED)
    m.validate_b1(value)
    value[field] = "changed"
    with pytest.raises(ValueError):
        m.validate_b1(value)


@pytest.mark.parametrize("ablation", ["A0", "A1", "B1", "B2"])
@pytest.mark.parametrize("damage", ["hash", "metrics", "order", "labels", "probability"])
def test_retained_results_recompute_metrics_and_check_signed_records(
    tmp_path,
    monkeypatch,
    ablation,
    damage,
):
    records = (
        NS(logical_id="r0", speaker_id="FC02", condition=NS(value="control")),
        NS(logical_id="r1", speaker_id="M01", condition=NS(value="dysarthric")),
    )
    key = m.g.run_id(ablation)
    run = {
        "identity": {"experiment_id": key, "ablation": ablation, "outer_fold": 2},
        "state": "OUTER_EVALUATED",
        "claim": "a" * 32,
        "data": {
            "selection": {"epoch": m.g.EPOCHS[ablation]},
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
                "label": i,
                "probability": (0.2, 0.8)[i],
            }
            for i, rec in enumerate(records)
        ],
        "metrics": m.c.native.binary_metrics([0, 1], [0.2, 0.8]),
    }
    base = tmp_path / m.r.BASE / "experiments"
    path = base / "runs" / key / "outer_result.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(result))
    checksum = m.r.file_hash(path)
    run["data"]["outer_result_sha256"] = checksum
    events = [
        [1, key, "time", "OUTER_CLAIMED", json.dumps({"claim": run["claim"]})],
        [2, key, "time", "OUTER_EVALUATED", json.dumps({"outer_result_sha256": checksum})],
    ]
    monkeypatch.setattr(m.c.native, "ROOT", tmp_path)
    checkpoint = {
        "checkpoint_sha256": "b" * 64,
        "candidate": "stronger_decay",
        "selected_epoch": m.g.EPOCHS[ablation],
        "selection_sha256": m.k.SELECTION_SHA,
    }
    pins = {
        **m.B1_EXPECTED,
        **checkpoint,
        "experiment_id": key,
        "target_claim": run["claim"],
        "record_count": 2,
        "record_hash": m.c.native.record_hash(records),
        "result_sha256": checksum,
    }
    if ablation != "B2":
        owner, name = {
            "A0": (m.h, "A0_EXPECTED"),
            "A1": (m.k, "A1_EXPECTED"),
            "B1": (m, "B1_EXPECTED"),
        }[ablation]
        monkeypatch.setattr(owner, name, pins)
    monkeypatch.setattr(m.p, "final_checkpoint", lambda *_: checkpoint)

    def verify():
        if ablation == "B2":
            return m.c.native.completed_result(base, key, run, events, records)
        func = {"A0": m.h.verify_a0, "A1": m.k.verify_a1, "B1": m.verify_b1}[ablation]
        return func(
            tmp_path,
            {"native_runs_snapshot": {key: run}, "native_events_snapshot": events},
            records,
        )

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
        # Even refreshing hashes cannot conceal semantic corruption.
        checksum = m.r.file_hash(path)
        run["data"]["outer_result_sha256"] = pins["result_sha256"] = checksum
        events[-1][-1] = json.dumps({"outer_result_sha256": checksum})
    with pytest.raises(ValueError):
        verify()


@pytest.mark.parametrize("mode", ["preflight", "verify"])
@pytest.mark.parametrize("damage", [None, "checkpoint", "selection", "exception", "records"])
def test_orchestration_is_read_only_and_fail_closed(tmp_path, monkeypatch, mode, damage):
    original = completion()
    current = advance(original) if mode == "verify" else deepcopy(original)
    sentinel = tmp_path / "preserved"
    sentinel.write_bytes(b"unchanged")
    context = NS(
        split=Mock(return_value=NS(records=Mock(return_value=()))),
        predecessor=NS(source=NS(contract=object())),
    )
    monkeypatch.setattr(m, "load_completion", lambda _: original)
    monkeypatch.setattr(m.p, "prepare", lambda _: ({}, context, ()))
    monkeypatch.setattr(m.p, "snapshot", lambda _: deepcopy(current))
    monkeypatch.setattr(m.c, "verify_csv", Mock())
    exception = Mock(side_effect=ValueError("exception") if damage == "exception" else None)
    monkeypatch.setattr(m.c, "verify_exception", exception)
    monkeypatch.setattr(
        m.g,
        "validate_records",
        Mock(side_effect=ValueError("records") if damage == "records" else None),
    )
    monkeypatch.setattr(m.g, "validate_selections", Mock())
    monkeypatch.setattr(m.r, "read_json", lambda _: {})
    monkeypatch.setattr(m.r, "file_hash", lambda _: m.k.SELECTION_SHA)
    verifiers = []
    for owner, name, pins in [
        (m.h, "verify_a0", m.h.A0_EXPECTED),
        (m.k, "verify_a1", m.k.A1_EXPECTED),
        (m, "verify_b1", m.B1_EXPECTED),
    ]:
        verifier = Mock(return_value={**pins, "metrics": original["verification"]["metrics"]})
        monkeypatch.setattr(owner, name, verifier)
        verifiers.append(verifier)
    checkpoint = dict(m.B2_EXPECTED)
    if damage in {"checkpoint", "selection"}:
        checkpoint["checkpoint_sha256" if damage == "checkpoint" else "selected_epoch"] = "changed"
    monkeypatch.setattr(m.p, "final_checkpoint", lambda *_: checkpoint)
    preservation = Mock()
    monkeypatch.setattr(m.p, "assert_preserved", preservation)
    monkeypatch.setattr(
        m.c.native,
        "completed_result",
        Mock(
            return_value={
                "record_count": 2873,
                "record_hash": m.g.RECORD_HASH,
                "result_sha256": "6" * 64,
                "metrics": {"accuracy": 0.5},
            }
        ),
    )
    if damage:
        with pytest.raises(ValueError):
            m.inspect(mode, 2, "B2", tmp_path)
    else:
        output = m.inspect(mode, 2, "B2", tmp_path)
        assert output["status"] == "PASS"
        assert output["model_forwards"] == output["production_writes"] == 0
        assert output["fold1_unchanged"] and output["fold1_a1_exception_unchanged"]
        assert output["fold2_a0_unchanged"] and output["fold2_a1_unchanged"]
        assert output["fold2_b1_unchanged"] and output["later_fold2_targets_untouched"]
        assert output["fold3_outer_untouched"]
        assert output["currently_authorized_target"] == "Fold2/B2"
        for verifier in verifiers:
            verifier.assert_called_once()
        exception.assert_called_once()
        preservation.assert_called_once()
    assert list(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"unchanged"


@pytest.mark.parametrize("missing", [False, True])
def test_completion_and_fold1_closure_fail_on_changed_or_missing_evidence(
    tmp_path,
    monkeypatch,
    missing,
):
    original_hash = m.r.file_hash
    monkeypatch.setattr(
        m.r,
        "file_hash",
        lambda path: m.B1_HELPER_SHA if path == tmp_path / m.B1_HELPER else original_hash(path),
    )
    if not missing:
        for relative in (m.COMPLETION, m.p.REPORT):
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"status":"PASS"}')
    with pytest.raises((OSError, ValueError)):
        m.load_completion(tmp_path)
    with pytest.raises((OSError, ValueError)):
        m.g.load_closure(tmp_path)


def test_wrong_fold2_record_hash_and_order(monkeypatch):
    records = tuple(NS(logical_id=str(i)) for i in range(2873))
    audit = {
        "outer_evaluation_plan": [
            {
                "outer_fold": 2,
                "record_hash": m.g.RECORD_HASH,
                "logical_ids": [value.logical_id for value in records],
            },
        ]
    }
    monkeypatch.setattr(m.c.native, "record_hash", lambda _: m.g.RECORD_HASH)
    m.g.validate_records(records, audit)
    with pytest.raises(ValueError):
        m.g.validate_records(records[::-1], audit)
    monkeypatch.setattr(m.c.native, "record_hash", lambda _: "changed")
    with pytest.raises(ValueError):
        m.g.validate_records(records, audit)
