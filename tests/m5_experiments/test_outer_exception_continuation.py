"""Continuation guard tests: synthetic evidence, mocked scientific forwards only."""

from __future__ import annotations

import json
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from scripts import module5_outer_exception_continuation as c

r = c.recovery


@pytest.fixture(autouse=True)
def no_real_model_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Real scientific forward forbidden")

    monkeypatch.setattr(torch.nn.Module, "_call_impl", forbidden)


def rows():
    return (
        NS(logical_id="row-0", speaker_id="FC01", condition=NS(value="control")),
        NS(logical_id="row-1", speaker_id="F03", condition=NS(value="dysarthric")),
    )


def base_snapshot():
    runs = {}
    for a in ("A0", "A1", *c.ORDER):
        identity = {"experiment_id": c.run_id(a), "ablation": a, "outer_fold": 1}
        selection = {"experiment_id": c.run_id(a), "candidate_id": "stronger_dropout", "epoch": 1}
        runs[c.run_id(a)] = {
            "identity": identity,
            "state": "FINAL_TRAIN_COMPLETE",
            "claim": None,
            "data": {
                "selection": selection,
                "final_checkpoint": {"filename": "original.pt"},
                "final_identity": identity,
                "parameter_counts": {},
                "inner_training_seconds": 1.0,
                "final_training_seconds": 1.0,
            },
        }
    runs[c.run_id("A0")]["state"] = "OUTER_EVALUATED"
    runs[c.run_id("A0")]["claim"] = "a" * 32
    runs[c.run_id("A0")]["data"]["outer_result_sha256"] = "b" * 64
    runs[r.RUN]["claim"] = r.CLAIM
    for fold in (2, 3):
        runs[f"fold{fold}_untouched"] = {
            "identity": {"outer_fold": fold},
            "state": "FINAL_TRAIN_COMPLETE",
            "claim": None,
            "data": {"selection": "unchanged"},
        }
    events = [
        [
            133,
            c.run_id("A0"),
            "2026-09-28T20:51:30+00:00",
            "OUTER_CLAIMED",
            json.dumps({"claim": "a" * 32}),
        ],
        [
            134,
            c.run_id("A0"),
            "2026-09-29T04:44:11+00:00",
            "OUTER_EVALUATED",
            json.dumps({"outer_result_sha256": "b" * 64}),
        ],
        [
            135,
            r.RUN,
            "2026-09-29T04:44:19.996521+00:00",
            "OUTER_CLAIMED",
            json.dumps({"claim": r.CLAIM}),
        ],
    ]
    return {
        "native_runs_snapshot": runs,
        "native_events_snapshot": events,
        "native_files_snapshot": {
            "ledger.sqlite3": {"sha256": "original"},
            "events.csv": {"sha256": "original"},
            "protected.pt": {"sha256": "original"},
        },
    }


def advanced(snapshot, completed):
    runs = deepcopy(snapshot["native_runs_snapshot"])
    events = deepcopy(snapshot["native_events_snapshot"])
    files = deepcopy(snapshot["native_files_snapshot"])
    for index, a in enumerate(completed):
        key = c.run_id(a)
        claim = str(index + 1) * 32
        sha = str(index + 1) * 64
        runs[key]["claim"] = claim
        runs[key]["state"] = "OUTER_EVALUATED"
        runs[key]["data"]["outer_result_sha256"] = sha
        events.extend(
            [
                [
                    136 + 2 * index,
                    key,
                    "2026-10-06T00:00:00+00:00",
                    "OUTER_CLAIMED",
                    json.dumps({"claim": claim}),
                ],
                [
                    137 + 2 * index,
                    key,
                    "2026-10-06T00:01:00+00:00",
                    "OUTER_EVALUATED",
                    json.dumps({"outer_result_sha256": sha}),
                ],
            ]
        )
        # Keep chronology monotonic across synthetic successive targets.
        events[-2][2] = f"2026-10-06T00:{index * 2:02d}:00+00:00"
        events[-1][2] = f"2026-10-06T00:{index * 2 + 1:02d}:00+00:00"
        files[f"runs/{key}/outer_result.json"] = {"sha256": sha}
    if completed:
        files["ledger.sqlite3"] = {"sha256": "legitimate advancement"}
        files["events.csv"] = {"sha256": "legitimate advancement"}
    return runs, events, files


@pytest.mark.parametrize("target", c.ORDER)
@pytest.mark.parametrize("mode", ["preflight", "verify"])
def test_only_sequential_remaining_targets_accepted(target, mode):
    completed = c.request_scope(mode, 1, target)
    snapshot = base_snapshot()
    c.validate_native_progress(snapshot, *advanced(snapshot, completed), completed)


@pytest.mark.parametrize(
    "fold,target", [(1, "A0"), (1, "A1"), (1, "unknown"), (2, "B1"), (3, "B1"), (True, "B1")]
)
def test_wrong_targets_folds_rejected(fold, target):
    with pytest.raises(r.ReviewRequired):
        c.request_scope("preflight", fold, target)


@pytest.mark.parametrize(
    "damage",
    [
        "second_claim",
        "wrong_a1_claim",
        "target_claim",
        "native_a1_result",
        "target_result",
        "later_target",
        "fold2",
        "fold3",
        "a0",
        "original_event",
        "checkpoint",
        "selection",
    ],
)
def test_prohibited_native_changes_rejected(damage):
    snapshot = base_snapshot()
    runs, events, files = advanced(snapshot, ())
    if damage in {"second_claim", "target_claim"}:
        runs[c.run_id("B2" if damage == "second_claim" else "B1")]["claim"] = "d" * 32
    elif damage == "wrong_a1_claim":
        runs[r.RUN]["claim"] = "wrong"
    elif damage in {"native_a1_result", "target_result"}:
        key = r.RUN if damage == "native_a1_result" else c.run_id("B1")
        files[f"runs/{key}/outer_result.json"] = {"sha256": "unexpected"}
    elif damage == "later_target":
        runs, events, files = advanced(snapshot, ("B2",))
    elif damage in {"fold2", "fold3"}:
        runs[f"{damage}_untouched"]["data"]["selection"] = "changed"
    elif damage == "a0":
        runs[c.run_id("A0")]["claim"] = "changed"
    elif damage == "original_event":
        events[-1][4] = "{}"
    elif damage == "checkpoint":
        files["protected.pt"]["sha256"] = "changed"
    else:
        runs[c.run_id("B1")]["data"]["selection"]["epoch"] = 2
    with pytest.raises(r.ReviewRequired):
        c.validate_native_progress(snapshot, runs, events, files, ())


@pytest.mark.parametrize("damage", ["extra_event", "repeat_claim", "later", "csv_file_missing"])
def test_post_target_requires_exact_two_event_delta(damage):
    snapshot = base_snapshot()
    runs, events, files = advanced(snapshot, ("B1",))
    if damage == "extra_event":
        events.append([138, c.run_id("B1"), events[-1][2], "FAILED", "{}"])
    elif damage == "repeat_claim":
        events[-1][3] = "OUTER_CLAIMED"
    elif damage == "later":
        runs, events, files = advanced(snapshot, ("B1", "B2"))
    else:
        del files["events.csv"]
    with pytest.raises(r.ReviewRequired):
        c.validate_native_progress(snapshot, runs, events, files, ("B1",))


@pytest.mark.parametrize(
    "field,value",
    [
        ("classification", "NATIVE"),
        ("provenance_flag", "NATIVE"),
        ("record_count", 2818),
        ("record_hash", "changed"),
        ("checkpoint_sha256", "changed"),
        ("threshold", 0.6),
        ("torch", "other"),
        ("original_prior_forward_count", 0),
    ],
)
def test_exact_exception_classification_and_identity(field, value):
    value_dict = {
        **r.TARGET,
        "attempt_id": c.ATTEMPT,
        "provenance_flag": "RECOVERED_PROTOCOL_EXCEPTION",
        "notice": r.NOTICE,
        "recovery_forward_count": 2819,
        "recovery_source_load_count": 2819,
        "recovery_batch_count": 177,
    }
    c.exception_identity(value_dict)
    value_dict[field] = value
    with pytest.raises(r.ReviewRequired):
        c.exception_identity(value_dict)


@pytest.mark.parametrize("missing", [True, False])
def test_missing_or_changed_recovery_rejected(tmp_path, missing):
    directory = tmp_path / r.RECOVERY / c.ATTEMPT
    directory.mkdir(parents=True)
    if not missing:
        (directory / "recovery_result.json").write_bytes(b"changed")
    with pytest.raises((FileNotFoundError, r.ReviewRequired)):
        c.verify_exception(tmp_path, {}, None, (), {}, [])


@pytest.mark.parametrize("damage", ["source", "protocol", "threads"])
def test_frozen_protocol_source_runtime_drift_rejected(monkeypatch, damage):
    context = NS(config=NS(config_hash=r.PROTOCOL, cpu_threads=2), code_hash=r.CODE)
    check = Mock()
    monkeypatch.setattr(c, "require_frozen", check)
    c.validate_frozen_runtime(context)
    if damage == "source":
        context.code_hash = "changed"
    elif damage == "protocol":
        context.config.config_hash = "changed"
    else:
        context.config.cpu_threads = 4
    with pytest.raises(r.ReviewRequired):
        c.validate_frozen_runtime(context)
    assert check.call_count == 2


def test_checkpoint_mismatch_rejected(tmp_path):
    run = base_snapshot()["native_runs_snapshot"][c.run_id("B1")]
    ex = tmp_path / r.BASE / "experiments"
    selection_path = ex / "selection/fold1_ablations.json"
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text(json.dumps({"selections": {"B1": run["data"]["selection"]}}))
    final = ex / "runs" / c.run_id("B1") / "final"
    final.mkdir(parents=True)
    (final / "latest.json").write_text(json.dumps({"filename": "changed.pt", "sha256": "c" * 64}))
    with pytest.raises(r.ReviewRequired, match="pointer"):
        c.validate_final_checkpoint(tmp_path, run)


@pytest.mark.parametrize("mode", ["preflight", "verify"])
def test_inspection_is_read_only_and_preserves_exception(tmp_path, monkeypatch, mode):
    snapshot = base_snapshot()
    completed = () if mode == "preflight" else ("B1",)
    runs, events, files = advanced(snapshot, completed)
    snapshot["recovery_files_snapshot"] = {"immutable": "recovery"}
    ex = tmp_path / r.BASE / "experiments"
    ex.mkdir(parents=True)
    sentinel = ex / "ledger.sqlite3"
    sentinel.write_bytes(b"never modified")
    before = r.fingerprint(sentinel)
    monkeypatch.setattr(c, "load_acceptance", lambda _: snapshot)
    monkeypatch.setattr(c, "frozen_context", lambda *_: (NS(), rows()))
    monkeypatch.setattr(
        r, "inventory", lambda path: files if path == ex else snapshot["recovery_files_snapshot"]
    )
    monkeypatch.setattr(r, "read_production", lambda _: (runs, events))
    monkeypatch.setattr(c, "verify_csv", Mock())
    verify_exception = Mock()
    monkeypatch.setattr(c, "verify_exception", verify_exception)
    monkeypatch.setattr(c, "validate_final_checkpoint", lambda *_: {"checkpoint_sha256": "pinned"})
    monkeypatch.setattr(c, "require_frozen", Mock())
    complete = Mock(return_value={"result_sha256": "s", "metrics": {"accuracy": 0.5}})
    monkeypatch.setattr(c.native, "completed_result", complete)
    for name in ("build_model", "ExperimentDataset", "run", "recovery_model"):
        monkeypatch.setattr(r, name, Mock(side_effect=AssertionError("No scientific execution")))
    report = c.inspect(mode, 1, "B1", root=tmp_path)
    assert report["status"] == "PASS" and report["model_forwards"] == 0
    assert report["production_writes"] == 0 and report["a1_native_claim_unchanged"]
    assert report["fold2_fold3_unchanged"] and report["no_later_target_ran"]
    assert r.fingerprint(sentinel) == before
    assert runs[r.RUN] == snapshot["native_runs_snapshot"][r.RUN]
    verify_exception.assert_called_once()
    assert complete.call_count == (1 if mode == "preflight" else 2)


def test_native_metric_verification_uses_predictions_only(tmp_path, monkeypatch):
    records = rows()
    snapshot = base_snapshot()
    runs, events, _ = advanced(snapshot, ("B1",))
    key, run = c.run_id("B1"), runs[c.run_id("B1")]
    predictions = [
        {
            "logical_id": rec.logical_id,
            "speaker_id": rec.speaker_id,
            "label": i,
            "probability": (0.2, 0.8)[i],
        }
        for i, rec in enumerate(records)
    ]
    result = {
        "status": "OUTER_EVALUATED",
        "experiment_id": key,
        "identity": run["identity"],
        "selection": run["data"]["selection"],
        "checkpoint": run["data"]["final_checkpoint"],
        "ablation": "B1",
        "outer_fold": 1,
        "claim": run["claim"],
        "threshold": 0.5,
        "predictions": predictions,
        "forward_count": 2,
        "source_load_count": 2,
        "metrics": c.native.binary_metrics([0, 1], [0.2, 0.8]),
    }
    path = tmp_path / "runs" / key / "outer_result.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(result))
    checksum = r.file_hash(path)
    run["data"]["outer_result_sha256"] = checksum
    events[-1][4] = json.dumps({"outer_result_sha256": checksum})
    monkeypatch.setattr(c.native, "ROOT", tmp_path)
    own = [e for e in events if e[1] == key]
    assert c.native.completed_result(tmp_path, key, run, own, records)["metrics"]["accuracy"] == 1
    result["metrics"]["accuracy"] = 0
    path.write_text(json.dumps(result))
    run["data"]["outer_result_sha256"] = r.file_hash(path)
    events[-1][4] = json.dumps({"outer_result_sha256": r.file_hash(path)})
    with pytest.raises(ValueError, match="metrics"):
        c.native.completed_result(tmp_path, key, run, own, records)


@pytest.mark.parametrize("ablation", c.ORDER)
def test_frozen_native_evaluator_is_independent_of_a1_claim(tmp_path, monkeypatch, ablation):
    from vaaniswitch.m5_experiments import evaluator

    snapshot = base_snapshot()
    runs = deepcopy(snapshot["native_runs_snapshot"])
    operations, publications = [], {}

    class MemoryLedger:
        def get(self, key):
            return deepcopy(runs[key])

        def lock(self, key):
            operations.append(("lock", key))
            return nullcontext()

        def claim_outer(self, key):
            assert runs[key]["state"] == "FINAL_TRAIN_COMPLETE" and runs[key]["claim"] is None
            runs[key]["claim"] = "e" * 32
            operations.append(("claim", key))
            return "e" * 32

        def transition(self, key, state, data):
            operations.append(("transition", key))
            runs[key]["state"] = state
            runs[key]["data"].update(data)

        def export(self):
            operations.append(("export", "global events CSV"))

    bundle = {
        "selections": {a: runs[c.run_id(a)]["data"]["selection"] for a in ("A0", "A1", *c.ORDER)}
    }
    record = rows()[0]
    batch = NS(provenance=[record])
    dataset = NS(records=[record], loaded=[record.logical_id], guard=Mock())
    model = Mock(return_value=NS(probabilities=NS(tolist=lambda: [0.2])))
    context = NS(
        config=NS(output=lambda rel: tmp_path / rel, cpu_threads=2, candidate=lambda _: NS()),
        split=Mock(return_value=NS()),
    )
    monkeypatch.setattr(evaluator, "require_frozen", Mock())
    monkeypatch.setattr(evaluator, "Ledger", lambda _: MemoryLedger())
    monkeypatch.setattr(
        evaluator, "read_json", lambda path: publications.get(path, bundle)
    )
    monkeypatch.setattr(evaluator, "build_model", Mock(return_value=model))
    monkeypatch.setattr(evaluator, "make_optimizer", Mock(return_value=object()))
    monkeypatch.setattr(evaluator, "load_checkpoint", Mock(return_value={"progress": {"epoch": 1}}))
    monkeypatch.setattr(evaluator, "ExperimentDataset", Mock(return_value=dataset))
    monkeypatch.setattr(evaluator, "batches", lambda _: [batch])
    monkeypatch.setattr(evaluator, "batch_labels", lambda _: NS(tolist=lambda: [0]))
    monkeypatch.setattr(evaluator, "file_hash", lambda _: "f" * 64)

    def publish(path, value, **kwargs):
        assert kwargs == {"immutable": True}
        publications[path] = value

    monkeypatch.setattr(evaluator, "write_json", publish)
    result = evaluator.evaluate_outer(context, 1, ablation)
    assert result["ablation"] == ablation and model.call_count == 1
    assert runs[r.RUN] == snapshot["native_runs_snapshot"][r.RUN]
    assert all(
        runs[key] == old
        for key, old in snapshot["native_runs_snapshot"].items()
        if key != c.run_id(ablation)
    )
    assert all(key == c.run_id(ablation) for op, key in operations if op != "export")
    assert all(c.run_id(ablation) in str(path) for path in publications)
    context.split.assert_called_once_with(1)
    assert evaluator.evaluate_outer(context, 1, ablation) == result
    assert model.call_count == 1  # Native completed reuse makes no extra mocked forward.
