"""Synthetic sequential-chain, corruption, no-forward and immutable publication tests."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace as NS

import pytest

from scripts import module5_outer_remaining_sequence as m


@pytest.fixture
def rig(tmp_path, monkeypatch):
    records = {
        fold: tuple(
            NS(
                logical_id=f"f{fold}-r{i}",
                speaker_id=("FC02", "M01")[i],
                condition=NS(value=("control", "dysarthric")[i]),
            )
            for i in range(2)
        )
        for fold in (2, 3)
    }
    monkeypatch.setattr(
        m, "RECORDS", {f: (2, m.c.native.record_hash(v)) for f, v in records.items()}
    )
    monkeypatch.setattr(m.c.native, "ROOT", tmp_path)
    specs = {}
    state = {
        "native_files_snapshot": {
            "ledger.sqlite3": {"sha256": "initial"},
            "events.csv": {"sha256": "initial"},
            "protected.bin": {"sha256": "protected"},
        },
        "native_runs_snapshot": {
            m.r.RUN: {
                "state": "FINAL_TRAIN_COMPLETE",
                "claim": m.r.CLAIM,
                "identity": {"experiment_id": m.r.RUN},
                "data": {},
            }
        },
        "native_events_snapshot": [
            [
                1,
                m.r.RUN,
                "2026-10-07T00:00:00+00:00",
                "OUTER_CLAIMED",
                json.dumps({"claim": m.r.CLAIM}),
            ]
        ],
    }

    def add(fold, ablation, candidate, epoch):
        key = f"fold{fold}_{ablation}_{candidate}_seed42_677ad640faf5"
        count, checksum = m.RECORDS[fold]
        spec = {
            "experiment_id": key,
            "outer_fold": fold,
            "ablation": ablation,
            "candidate": candidate,
            "selected_epoch": epoch,
            "checkpoint_sha256": hashlib.sha256(key.encode()).hexdigest(),
            "selection_sha256": m.old.k.SELECTION_SHA if fold == 2 else "f" * 64,
            "record_count": count,
            "record_hash": checksum,
        }
        specs[key] = spec
        state["native_runs_snapshot"][key] = {
            "identity": {"experiment_id": key, "outer_fold": fold, "ablation": ablation},
            "state": "FINAL_TRAIN_COMPLETE",
            "claim": None,
            "data": {
                "selection": {"epoch": epoch, "candidate_id": candidate},
                "final_checkpoint": {},
                "final_identity": {},
            },
        }
        return spec

    def cp(spec):
        return {
            key: spec[key]
            for key in ("checkpoint_sha256", "candidate", "selected_epoch", "selection_sha256")
        }

    def finish(spec):
        key = spec["experiment_id"]
        run = state["native_runs_snapshot"][key]
        assert run["claim"] is None
        claim = hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()
        result = {
            "status": "OUTER_EVALUATED",
            "experiment_id": key,
            "identity": run["identity"],
            "selection": run["data"]["selection"],
            "checkpoint": {},
            "ablation": spec["ablation"],
            "outer_fold": spec["outer_fold"],
            "claim": claim,
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
                for i, rec in enumerate(records[spec["outer_fold"]])
            ],
            "metrics": m.c.native.binary_metrics([0, 1], [0.2, 0.8]),
        }
        path = tmp_path / m.r.BASE / "experiments/runs" / key / "outer_result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))
        checksum = m.r.file_hash(path)
        run.update(state="OUTER_EVALUATED", claim=claim)
        run["data"]["outer_result_sha256"] = checksum
        events = state["native_events_snapshot"]
        for kind, detail in (
            ("OUTER_CLAIMED", {"claim": claim}),
            ("OUTER_EVALUATED", {"outer_result_sha256": checksum}),
        ):
            seq = events[-1][0] + 1
            when = (datetime(2026, 10, 7, tzinfo=UTC) + timedelta(seconds=seq)).isoformat()
            events.append([seq, key, when, kind, json.dumps(detail)])
        state["native_files_snapshot"][f"runs/{key}/outer_result.json"] = m.r.fingerprint(path)
        for name in ("ledger.sqlite3", "events.csv"):
            state["native_files_snapshot"][name] = {"sha256": str(events[-1][0])}
        return path, result

    prior = []
    for ablation in m.g.ORDER[:-1]:
        spec = add(2, ablation, "stronger_decay", m.g.EPOCHS[ablation])
        finish(spec)
        prior.append(m.verify_cell(tmp_path, state, spec, records[2], cp(spec)))
    plan = [
        add(fold, a, "stronger_decay" if fold == 2 else "baseline", epoch)
        for (fold, a), epoch in zip(m.ORDER, m.EPOCHS, strict=True)
    ]
    anchor = {
        **deepcopy(state),
        "remaining_sequence": plan,
        "protected_helper_sha256": {},
        "verification": {
            **prior[-1],
            "prior_native_results": {v["ablation"]: v for v in prior[:-1]},
        },
    }
    source = tmp_path / m.SELF
    source.parent.mkdir(parents=True)
    source.write_text("# synthetic guard identity\n")
    audit = {
        "outer_evaluation_plan": [
            {
                "outer_fold": fold,
                "record_hash": m.RECORDS[fold][1],
                "logical_ids": [v.logical_id for v in recs],
            }
            for fold, recs in records.items()
        ]
    }
    context = NS(
        split=lambda fold: NS(records=lambda *_: records[fold]),
        predecessor=NS(source=NS(contract=object())),
    )
    monkeypatch.setattr(m, "load_anchor", lambda _: deepcopy(anchor))
    monkeypatch.setattr(m.p, "prepare", lambda _: ({}, context, ()))
    monkeypatch.setattr(m.p, "snapshot", lambda _: deepcopy(state))
    monkeypatch.setattr(
        m.p, "assert_preserved", lambda _, before, *__: m.require(before == state, "Native changed")
    )
    monkeypatch.setattr(
        m.p, "final_checkpoint", lambda _, run: cp(specs[run["identity"]["experiment_id"]])
    )
    monkeypatch.setattr(m.c, "verify_csv", lambda *_a, **_k: None)
    monkeypatch.setattr(m.c, "verify_exception", lambda *_: None)
    real_read = m.r.read_json
    monkeypatch.setattr(
        m.r,
        "read_json",
        lambda path: audit if path == tmp_path / m.c.FINAL_AUDIT else real_read(path),
    )
    return NS(
        root=tmp_path,
        state=state,
        anchor=anchor,
        plan=plan,
        specs=specs,
        records=records,
        audit=audit,
        finish=finish,
        cp=cp,
    )


def test_entire_eight_target_sequence_and_append_only_chain(rig):
    previous_sha = m.ANCHOR_SHA
    for index, spec in enumerate(rig.plan):
        args = (spec["outer_fold"], spec["ablation"])
        before = deepcopy(rig.state)
        ready = m.inspect("preflight", *args, rig.root)
        assert ready["next_target"] == spec
        assert ready["previous_completion_sha256"] == previous_sha
        assert ready["model_forwards"] == ready["native_writes"] == 0
        assert rig.state == before and len(m.read_chain(rig.root)) == index
        with pytest.raises(ValueError, match="No exact completed"):
            m.inspect("verify", *args, rig.root)
        for other in rig.plan:
            if other != spec:
                with pytest.raises(ValueError, match="exact next"):
                    m.inspect("preflight", other["outer_fold"], other["ablation"], rig.root)
        rig.finish(spec)  # Synthetic retained predictions only; no evaluator or model.
        assert m.inspect("status", root=rig.root)["sequence_state"] == "AWAITING_VERIFICATION"
        with pytest.raises(ValueError, match="run verify"):
            m.inspect("preflight", *args, rig.root)
        native_before_verify = deepcopy(rig.state)
        verified = m.inspect("verify", *args, rig.root)
        assert verified["external_completion_written"]
        assert verified["verification"]["metrics"]["accuracy"] == 1.0
        assert rig.state == native_before_verify
        previous_sha = verified["completion_artifact"]["sha256"]
        chain = m.read_chain(rig.root)
        assert len(chain) == index + 1
        assert chain[-1]["sha256"] == previous_sha
        assert (
            chain[-1]["value"]["previous_completion_sha256"] == ready["previous_completion_sha256"]
        )
        for mode in ("preflight", "verify"):
            with pytest.raises(ValueError):
                m.inspect(mode, *args, rig.root)
    status = m.inspect("status", root=rig.root)
    assert status["sequence_state"] == "COMPLETE" and status["next_target"] is None
    assert status["verified_chain_completions"] == 8
    assert status["native_writes"] == status["model_forwards"] == 0


@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize(
    "damage",
    [
        "prior_claim",
        "prior_checkpoint",
        "prior_result",
        "exception",
        "later_fold3",
        "new_run",
        "prior_current_claim",
        "wrong_checkpoint",
        "wrong_selection",
        "old_event",
        "extra_event",
        "missing_event",
        "wrong_event",
        "duplicate_claim",
        "extra_file",
        "missing_file",
        "result_hash",
    ],
)
def test_exact_native_delta_rejects_all_unrequested_changes(rig, completed, damage):
    current = deepcopy(rig.state)
    if completed:
        rig.finish(rig.plan[0])
        current = deepcopy(rig.state)
    runs, events, files = (
        current[name]
        for name in ("native_runs_snapshot", "native_events_snapshot", "native_files_snapshot")
    )
    prior_key = m.g.run_id("C1")
    key = rig.plan[0]["experiment_id"]
    if damage == "prior_claim":
        runs[prior_key]["claim"] = "a" * 32
    elif damage == "prior_checkpoint":
        runs[prior_key]["data"]["final_checkpoint"] = "changed"
    elif damage == "prior_result":
        files[f"runs/{prior_key}/outer_result.json"]["sha256"] = "changed"
    elif damage == "exception":
        runs[m.r.RUN]["claim"] = "a" * 32
    elif damage == "later_fold3":
        runs[rig.plan[1]["experiment_id"]]["claim"] = "a" * 32
    elif damage == "new_run":
        runs["unrequested"] = deepcopy(runs[key])
    elif damage == "prior_current_claim":
        runs[key].update(state="FINAL_TRAIN_COMPLETE", claim="a" * 32)
    elif damage == "wrong_checkpoint":
        runs[key]["data"]["final_checkpoint"] = "changed"
    elif damage == "wrong_selection":
        runs[key]["data"]["selection"]["epoch"] = 99
    elif damage == "old_event":
        events[0][-1] = "changed"
    elif damage == "extra_event":
        events.append([999, key, "2026-10-07T01:00:00+00:00", "FAILED", "{}"])
    elif damage == "missing_event":
        events.pop()
    elif damage == "wrong_event":
        events[-1][1] = rig.plan[1]["experiment_id"]
    elif damage == "duplicate_claim":
        runs[key]["claim"] = runs[prior_key]["claim"]
        if completed:
            events[-2][-1] = json.dumps({"claim": runs[key]["claim"]})
    elif damage == "extra_file":
        files["unexpected.tmp"] = {"sha256": "changed"}
    elif damage == "missing_file":
        del files["protected.bin"]
    else:
        files[f"runs/{key}/outer_result.json"] = {"sha256": "changed"}
    with pytest.raises(ValueError):
        m.validate_delta(rig.anchor, current, rig.plan[0] if completed else None)


@pytest.mark.parametrize(
    "damage",
    ["hash", "order", "speaker", "label", "nan", "inf", "metrics", "threshold", "checkpoint"],
)
@pytest.mark.parametrize("fold", [2, 3])
def test_refreshed_hashes_do_not_hide_semantic_corruption(rig, damage, fold):
    spec = rig.plan[0 if fold == 2 else 1]
    path, result = rig.finish(spec)
    key = spec["experiment_id"]
    if damage in {"hash", "order"}:
        result["predictions"].reverse()
    elif damage == "speaker":
        result["predictions"][0]["speaker_id"] = "wrong"
    elif damage == "label":
        result["predictions"][0]["label"] = 1
    elif damage in {"nan", "inf"}:
        result["predictions"][0]["probability"] = float(damage)
    elif damage == "metrics":
        result["metrics"]["accuracy"] = 0
    elif damage == "threshold":
        result["threshold"] = 0.6
    else:
        result["checkpoint"] = {"changed": True}
    path.write_text(json.dumps(result))
    if damage != "hash":
        checksum = m.r.file_hash(path)
        rig.state["native_runs_snapshot"][key]["data"]["outer_result_sha256"] = checksum
        rig.state["native_events_snapshot"][-1][-1] = json.dumps({"outer_result_sha256": checksum})
        rig.state["native_files_snapshot"][f"runs/{key}/outer_result.json"] = m.r.fingerprint(path)
    with pytest.raises(ValueError):
        m.verify_cell(rig.root, rig.state, spec, rig.records[fold], rig.cp(spec))


@pytest.mark.parametrize("fold", [2, 3])
@pytest.mark.parametrize("damage", ["order", "hash", "count"])
def test_wrong_signed_records(rig, fold, damage, monkeypatch):
    records = rig.records[fold]
    if damage == "order":
        records = records[::-1]
    elif damage == "count":
        records = records[:1]
    else:
        monkeypatch.setattr(m.c.native, "record_hash", lambda _: "wrong")
    with pytest.raises(ValueError):
        m.validate_records(records, fold, rig.audit)


@pytest.mark.parametrize(
    "field", ["checkpoint_sha256", "candidate", "selected_epoch", "selection_sha256"]
)
def test_checkpoint_pins_reject_changes(rig, monkeypatch, field):
    spec = rig.plan[0]
    wrong = rig.cp(spec)
    wrong[field] = "changed"
    monkeypatch.setattr(m.p, "final_checkpoint", lambda *_: wrong)
    with pytest.raises(ValueError, match="checkpoint"):
        m.inspect("preflight", 2, "C2", rig.root)


@pytest.mark.parametrize(
    "damage", ["bytes", "missing", "extra", "gap", "partial", "link", "metrics", "helpers"]
)
def test_completion_tampering_blocks_next_target(rig, damage):
    rig.finish(rig.plan[0])
    m.inspect("verify", 2, "C2", rig.root)
    entry = m.read_chain(rig.root)[0]
    path = rig.root / entry["path"]
    if damage == "bytes":
        path.write_bytes(path.read_bytes() + b" ")
    elif damage == "missing":
        path.unlink()
    elif damage == "extra":
        (path.parent / "extra.json").write_text("{}")
    elif damage == "gap":
        path.parent.rename(path.parent.with_name("01"))
    elif damage == "partial":
        (path.parent.parent / ".pending-abandoned").mkdir()
    else:
        value = entry["value"]
        if damage == "link":
            value["previous_completion_sha256"] = "f" * 64
        elif damage == "metrics":
            value["verification"]["metrics"]["accuracy"] = 0
        else:
            value["protected_helper_sha256"][m.SELF] = "f" * 64
        payload = m.canonical_bytes(value)
        renamed = path.with_name(hashlib.sha256(payload).hexdigest() + ".json")
        path.unlink()
        renamed.write_bytes(payload)
    with pytest.raises((ValueError, OSError)):
        m.inspect("preflight", 3, "A0", rig.root)


def test_no_completed_evidence_overwrite(rig):
    rig.finish(rig.plan[0])
    m.inspect("verify", 2, "C2", rig.root)
    entry = m.read_chain(rig.root)[0]
    before = (rig.root / entry["path"]).read_bytes()
    with pytest.raises(ValueError):
        m.publish_completion(rig.root, entry["value"], [], rig.state)
    assert (rig.root / entry["path"]).read_bytes() == before


def test_publication_failure_preserves_native_and_does_not_advance(rig, monkeypatch):
    rig.finish(rig.plan[0])
    before = deepcopy(rig.state)
    original_write = m.p.write_json

    def broken(*_args, **_kwargs):
        raise OSError("synthetic interrupted external publication")

    monkeypatch.setattr(m.p, "write_json", broken)
    with pytest.raises(OSError):
        m.inspect("verify", 2, "C2", rig.root)
    assert rig.state == before and m.read_chain(rig.root) == []
    monkeypatch.setattr(m.p, "write_json", original_write)
    assert m.inspect("status", root=rig.root)["sequence_state"] == "AWAITING_VERIFICATION"
    assert m.inspect("verify", 2, "C2", rig.root)["verified_chain_completions"] == 1


@pytest.mark.parametrize("boundary", ["prepare", "exception"])
def test_scientific_prerequisite_failure_stops_without_publication(rig, monkeypatch, boundary):
    def fail(*_a, **_k):
        raise ValueError("changed frozen source/protocol/runtime/exception")

    monkeypatch.setattr(
        m.p if boundary == "prepare" else m.c,
        "prepare" if boundary == "prepare" else "verify_exception",
        fail,
    )
    with pytest.raises(ValueError):
        m.inspect("preflight", 2, "C2", rig.root)
    assert m.read_chain(rig.root) == []


@pytest.mark.parametrize("damage", ["code", "protocol", "torch", "threads"])
def test_real_frozen_runtime_validator_rejects_drift(monkeypatch, damage):
    context = NS(config=NS(config_hash=m.r.PROTOCOL, cpu_threads=2), code_hash=m.r.CODE)
    monkeypatch.setattr(m.c, "require_frozen", lambda _: None)
    if damage == "code":
        context.code_hash = "changed"
    elif damage == "protocol":
        context.config.config_hash = "changed"
    elif damage == "threads":
        context.config.cpu_threads = 4
    else:
        monkeypatch.setattr(m.c.torch, "__version__", "changed")
    with pytest.raises(ValueError):
        m.c.validate_frozen_runtime(context)


@pytest.mark.parametrize(
    "operation", ["construct", "forward", "ledger", "train", "evaluate", "dataset"]
)
def test_execution_entrypoints_are_blocked(operation):
    import torch

    from vaaniswitch.m5_experiments import evaluator, trainer
    from vaaniswitch.m5_experiments.ledger import Ledger

    calls = {
        "construct": lambda: torch.nn.Module(),
        "forward": lambda: torch.nn.Module._call_impl(None),
        "ledger": lambda: Ledger(None),
        "train": lambda: trainer.train_stage(),
        "evaluate": lambda: evaluator.evaluate_outer(None, 2, "C2"),
        "dataset": lambda: m.r.ExperimentDataset(),
    }
    with m.forbid_scientific_execution(), pytest.raises(RuntimeError, match="forbidden"):
        calls[operation]()


@pytest.mark.parametrize("missing", [False, True])
def test_anchor_requires_exact_pinned_bytes(tmp_path, missing):
    if not missing:
        path = tmp_path / m.ANCHOR
        path.parent.mkdir(parents=True)
        path.write_text('{"status":"PASS"}')
    with pytest.raises((ValueError, OSError)):
        m.load_anchor(tmp_path)


def test_cli_modes_and_status_scope(rig):
    for mode in ("train", "outer", "resume", "save"):
        with pytest.raises(SystemExit):
            m.main([mode])
    with pytest.raises(ValueError):
        m.inspect("status", 2, "C2", rig.root)
    with pytest.raises(ValueError):
        m.inspect("preflight", root=rig.root)
