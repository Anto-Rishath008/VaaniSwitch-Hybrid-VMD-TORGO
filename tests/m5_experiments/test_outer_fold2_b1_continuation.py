"""Regression coverage for the Fold2/B1 read-only continuation gate."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from scripts import module5_outer_fold2_b1_continuation as k
from tests.m5_experiments.test_outer_fold2_continuation import advanced, baseline


@pytest.fixture(autouse=True)
def forbid_scientific_execution(monkeypatch):
    from vaaniswitch.m5_experiments import evaluator
    from vaaniswitch.m5_experiments.ledger import Ledger

    forbidden = Mock(side_effect=AssertionError("No scientific execution or native writes"))
    for obj, name in [
        (torch.nn.Module, "_call_impl"),
        (Ledger, "__init__"),
        (evaluator, "evaluate_outer"),
        (k.r, "build_model"),
        (k.r, "ExperimentDataset"),
    ]:
        monkeypatch.setattr(obj, name, forbidden)


def a0_completion():
    value = advanced(baseline())
    value["verification"] = {"metrics": {"accuracy": 0.5}}
    # Real inventory entries are fingerprint dictionaries. Keep synthetic evidence shape aligned.
    value["native_files_snapshot"] = {
        name: fingerprint if isinstance(fingerprint, dict) else {"sha256": str(fingerprint)}
        for name, fingerprint in value["native_files_snapshot"].items()
    }
    return value


def a1_completed(completion):
    current = deepcopy(completion)
    key = k.g.run_id("A1")
    run = current["native_runs_snapshot"][key]
    run.update(state="OUTER_EVALUATED", claim=k.A1_CLAIM)
    run["data"]["outer_result_sha256"] = k.A1_RESULT_SHA
    current["native_events_snapshot"].extend(
        [
            [
                148,
                key,
                "2026-10-06T06:04:08+00:00",
                "OUTER_CLAIMED",
                json.dumps({"claim": k.A1_CLAIM}),
            ],
            [
                149,
                key,
                "2026-10-06T06:07:39+00:00",
                "OUTER_EVALUATED",
                json.dumps({"outer_result_sha256": k.A1_RESULT_SHA}),
            ],
        ]
    )
    current["native_files_snapshot"][f"runs/{key}/outer_result.json"] = {
        "sha256": k.A1_RESULT_SHA
    }
    current["native_files_snapshot"]["ledger.sqlite3"] = {"sha256": "after-a1"}
    current["native_files_snapshot"]["events.csv"] = {"sha256": "after-a1"}
    return current


def b1_completed(completion):
    current = a1_completed(completion)
    key = k.g.run_id("B1")
    run = current["native_runs_snapshot"][key]
    claim, result_sha = "3" * 32, "4" * 64
    run.update(state="OUTER_EVALUATED", claim=claim)
    run["data"]["outer_result_sha256"] = result_sha
    current["native_events_snapshot"].extend(
        [
            [
                150,
                key,
                "2026-10-06T07:00:00+00:00",
                "OUTER_CLAIMED",
                json.dumps({"claim": claim}),
            ],
            [
                151,
                key,
                "2026-10-06T07:01:00+00:00",
                "OUTER_EVALUATED",
                json.dumps({"outer_result_sha256": result_sha}),
            ],
        ]
    )
    current["native_files_snapshot"][f"runs/{key}/outer_result.json"] = {"sha256": result_sha}
    current["native_files_snapshot"]["ledger.sqlite3"] = {"sha256": "after-b1"}
    current["native_files_snapshot"]["events.csv"] = {"sha256": "after-b1"}
    return current


@pytest.mark.parametrize("mode", ["preflight", "verify"])
def test_only_fold2_b1_is_enabled(mode):
    k.scope(mode, 2, "B1")
    invalid_targets = [
        (1, "B1"),
        (3, "B1"),
        (True, "B1"),
        *((2, ablation) for ablation in k.g.ORDER if ablation != "B1"),
    ]
    for fold, ablation in invalid_targets:
        with pytest.raises(ValueError):
            k.scope(mode, fold, ablation)


@pytest.mark.parametrize("completed", [False, True])
def test_exact_a1_prefix_and_optional_b1_delta_are_accepted(completed):
    original = a0_completion()
    current = b1_completed(original) if completed else a1_completed(original)
    k.validate_progress(original, current, b1_completed=completed)


@pytest.mark.parametrize(
    "damage",
    [
        "a1_claim",
        "a1_result",
        "b1_prior_claim",
        "later_fold2",
        "fold3",
        "second_claim",
        "old_event",
        "extra_event",
        "duplicate_claim",
        "a1_file",
        "unexpected_result",
        "missing_file",
    ],
)
@pytest.mark.parametrize("completed", [False, True])
def test_unapproved_native_changes_fail(damage, completed):
    original = a0_completion()
    current = b1_completed(original) if completed else a1_completed(original)
    runs, files, events = (
        current["native_runs_snapshot"],
        current["native_files_snapshot"],
        current["native_events_snapshot"],
    )
    b1 = runs[k.g.run_id("B1")]

    if damage == "a1_claim":
        runs[k.g.run_id("A1")]["claim"] = "f" * 32
    elif damage == "a1_result":
        runs[k.g.run_id("A1")]["data"]["outer_result_sha256"] = "f" * 64
    elif damage == "b1_prior_claim":
        b1.update(state="FINAL_TRAIN_COMPLETE", claim="f" * 32)
    elif damage in {"later_fold2", "second_claim"}:
        runs[k.g.run_id("B2")]["claim"] = "f" * 32
    elif damage == "fold3":
        runs["fold3_untouched"]["data"]["selection"] = "changed"
    elif damage == "old_event":
        events[0][-1] = "changed"
    elif damage == "extra_event":
        events.append([152, k.g.run_id("B2"), "2026-10-06T08:00:00+00:00", "FAILED", "{}"])
    elif damage == "duplicate_claim":
        duplicate = runs[k.g.run_id("A0")]["claim"]
        if completed:
            b1["claim"] = duplicate
            events[-2][-1] = json.dumps({"claim": duplicate})
        else:
            runs[k.g.run_id("B2")]["claim"] = duplicate
    elif damage == "a1_file":
        files[f"runs/{k.g.run_id('A1')}/outer_result.json"]["sha256"] = "changed"
    elif damage == "unexpected_result":
        files[f"runs/{k.g.run_id('B2')}/outer_result.json"] = {"sha256": "unexpected"}
    else:
        protected = next(name for name in files if name not in {"ledger.sqlite3", "events.csv"})
        del files[protected]

    with pytest.raises(ValueError):
        k.validate_progress(original, current, b1_completed=completed)


@pytest.mark.parametrize("field", list(k.A1_EXPECTED))
def test_a1_pins_are_exact(field):
    value = dict(k.A1_EXPECTED)
    k.validate_a1(value)
    value[field] = "changed"
    with pytest.raises(ValueError):
        k.validate_a1(value)


def test_b1_checkpoint_is_pinned_to_uploaded_frozen_final_generation():
    assert k.B1_CHECKPOINT == "0325643987a235655f39da418622a09049f1ee2fc94f0c09421daac0666132d4"


@pytest.mark.parametrize("mode", ["preflight", "verify"])
def test_inspection_is_read_only_and_preserves_all_prior_evidence(tmp_path, monkeypatch, mode):
    completion = a0_completion()
    current = b1_completed(completion) if mode == "verify" else a1_completed(completion)
    sentinel = tmp_path / "native-and-recovery"
    sentinel.write_bytes(b"unchanged")

    context = NS(
        split=Mock(return_value=NS(records=Mock(return_value=()))),
        predecessor=NS(source=NS(contract=object())),
    )
    monkeypatch.setattr(k.h, "load_completion", lambda _: completion)
    monkeypatch.setattr(k.p, "prepare", lambda _: ({}, context, ()))
    monkeypatch.setattr(k.p, "snapshot", lambda _: deepcopy(current))
    monkeypatch.setattr(k.c, "verify_csv", Mock())
    exception = Mock()
    monkeypatch.setattr(k.c, "verify_exception", exception)
    monkeypatch.setattr(k.g, "validate_records", Mock())
    monkeypatch.setattr(k.g, "validate_selections", Mock())
    monkeypatch.setattr(k.r, "read_json", lambda _: {})
    monkeypatch.setattr(
        k.h,
        "verify_a0",
        lambda *_: {
            "metrics": completion["verification"]["metrics"],
            "result_sha256": k.h.A0_EXPECTED["result_sha256"],
        },
    )
    monkeypatch.setattr(k, "verify_a1", lambda *_: dict(k.A1_EXPECTED))
    monkeypatch.setattr(
        k.p,
        "final_checkpoint",
        lambda *_: {
            "checkpoint_sha256": k.B1_CHECKPOINT,
            "candidate": "stronger_decay",
            "selected_epoch": 1,
            "selection_sha256": k.SELECTION_SHA,
        },
    )
    preservation = Mock()
    monkeypatch.setattr(k.p, "assert_preserved", preservation)

    def fake_hash(path):
        text = str(path).replace("\\", "/")
        if text.endswith(k.A1_HELPER):
            return k.A1_HELPER_SHA
        if text.endswith(k.p.REPORT):
            return k.g.CLOSURE_SHA
        if text.endswith(k.h.COMPLETION):
            return k.h.COMPLETION_SHA
        if text.endswith("selection/fold2_ablations.json"):
            return k.SELECTION_SHA
        return "irrelevant"

    monkeypatch.setattr(k.r, "file_hash", fake_hash)
    monkeypatch.setattr(k.c.native, "validate_unclaimed", Mock())
    monkeypatch.setattr(
        k.c.native,
        "completed_result",
        Mock(return_value={
            "record_count": 2873,
            "record_hash": k.g.RECORD_HASH,
            "result_sha256": "4" * 64,
            "metrics": {"accuracy": 0.5},
        }),
    )

    output = k.inspect(mode, 2, "B1", tmp_path)
    assert output["status"] == "PASS"
    assert output["model_forwards"] == output["production_writes"] == 0
    assert output["fold1_unchanged"] and output["fold1_a1_exception_unchanged"]
    assert output["fold2_a0_unchanged"] and output["fold2_a1_unchanged"]
    assert output["later_fold2_targets_untouched"] and output["fold3_outer_untouched"]
    assert output["currently_authorized_target"] == "Fold2/B1"
    assert sentinel.read_bytes() == b"unchanged"
    exception.assert_called_once()
    preservation.assert_called_once()
