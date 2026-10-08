"""Read-only Fold2/A1 continuation after immutable independent Fold2/A0 verification.

Only preflight and verification are supported. Native inference remains manual.
The used A0 helper and all frozen scientific code/evidence remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

if __package__:
    from . import module5_outer_fold2_continuation as g
else:
    import module5_outer_fold2_continuation as g

p, c, r, require = g.p, g.c, g.r, g.require
ROOT = g.ROOT
COMPLETION = f"{r.BASE}/outer_continuation/fold2/a1_handoff/a0_completion.json"
COMPLETION_SHA = "34fecfc92758919c8b1a22a5a59e9f0680b1da1311f609cd1f55af00df21609a"
A1_CHECKPOINT = "6840e0688780b28ce5a103776993224880ed8137ce9122c7ba676b954ed69c67"
A0_EXPECTED = {
    "status": "PASS",
    "experiment_id": "fold2_A0_stronger_decay_seed42_677ad640faf5",
    "provenance": "NATIVE",
    "target_state": "OUTER_EVALUATED",
    "target_claim": "950f7ce40de0443c85a26ed1f74ea13f",
    "checkpoint_sha256": "3dc65c5e1b57860fcd28ddad921ea7f0f4110fefd35781e3f796b5eb5269cc41",
    "candidate": "stronger_decay",
    "selected_epoch": 11,
    "record_count": 2873,
    "record_hash": "e85a74cb8dd33249ac0f27d42521d5dc0cd2d9abd283970c978f278d9fb97353",
    "result_sha256": "745db46496016e77586094ce1939ae3eff25926b476f4fbffaa92f3dc2c278bf",
    "threshold": 0.5,
}


def scope(mode: str, fold: int, ablation: str) -> None:
    require(mode in {"preflight", "verify"}, "Only read-only preflight/verify allowed")
    require(type(fold) is int and fold == 2, "Only Fold 2 is authorized")
    require(ablation == "A1", "Only Fold2/A1 is authorized; A0 and later targets are excluded")


def validate_a0(value: dict[str, Any]) -> None:
    require(
        all(value.get(key) == expected for key, expected in A0_EXPECTED.items()),
        "Pinned Fold2/A0 identity/claim/checkpoint/result/records differ",
    )


def load_completion(root: Path) -> dict[str, Any]:
    path = c.canonical_path(root, COMPLETION)
    require(r.file_hash(path) == COMPLETION_SHA, "A0 completion missing or changed")
    value = r.read_json(path)
    require(
        value["schema"] == "module5_fold2_a0_completion_v1"
        and value["status"] == "PASS"
        and value["protocol_hash"] == r.PROTOCOL
        and value["frozen_code_hash"] == r.CODE
        and value["fold1_closure_sha256"] == g.CLOSURE_SHA
        and value["actual_model_forwards"] == value["native_writes"] == 0
        and value["next_manual_target"] == "Fold2/A1",
        "Invalid A0 completion prerequisite",
    )
    for relative, checksum in value["protected_helper_sha256"].items():
        require(
            r.file_hash(root / relative) == checksum, f"Previously used helper changed: {relative}"
        )
    closure = g.load_closure(root)
    g.validate_progress(closure, value, completed=True)
    validate_a0(value["verification"])
    return value


def validate_progress(
    completion: dict[str, Any], state: dict[str, Any], *, completed: bool
) -> None:
    originals, runs = completion["native_runs_snapshot"], state["native_runs_snapshot"]
    previous_events, events = completion["native_events_snapshot"], state["native_events_snapshot"]
    previous_files, files = completion["native_files_snapshot"], state["native_files_snapshot"]
    require(set(runs) == set(originals), "Native experiment set changed")
    unfinished = [
        key
        for key, value in runs.items()
        if value["claim"] is not None and value["state"] != "OUTER_EVALUATED"
    ]
    require(unfinished == [r.RUN], "Second or changed unfinished claim")
    key = g.run_id("A1")
    for identity, original in originals.items():
        if not completed or identity != key:
            require(runs[identity] == original, f"Preserved/unrequested run changed: {identity}")
    if completed:
        target, original = runs[key], originals[key]
        checksum = target["data"].get("outer_result_sha256")
        require(
            original["state"] == "FINAL_TRAIN_COMPLETE"
            and original["claim"] is None
            and target["state"] == "OUTER_EVALUATED"
            and isinstance(target["claim"], str)
            and bool(re.fullmatch(r"[a-f0-9]{32}", target["claim"]))
            and isinstance(checksum, str)
            and bool(re.fullmatch(r"[a-f0-9]{64}", checksum))
            and target
            == {
                **original,
                "state": "OUTER_EVALUATED",
                "claim": target["claim"],
                "data": {**original["data"], "outer_result_sha256": checksum},
            },
            "Invalid native A1 completion or changed selection/checkpoint",
        )
    require(events[: len(previous_events)] == previous_events, "Earlier native event changed")
    extra = events[len(previous_events) :]
    require(len(extra) == (2 if completed else 0), "Unexpected prior/later/extra outer activity")
    previous_time = datetime.fromisoformat(previous_events[-1][2])
    for index, event in enumerate(extra):
        detail = (
            {"claim": runs[key]["claim"]}
            if index == 0
            else {"outer_result_sha256": runs[key]["data"]["outer_result_sha256"]}
        )
        event_time = datetime.fromisoformat(event[2])
        require(
            event[0] == previous_events[-1][0] + index + 1
            and event[1] == key
            and event[3] == ("OUTER_CLAIMED", "OUTER_EVALUATED")[index]
            and json.loads(event[4]) == detail
            and event_time.utcoffset() is not None
            and event_time >= previous_time,
            "Invalid A1 claim/completion lifecycle",
        )
        previous_time = event_time
    tokens = [v["claim"] for v in runs.values() if v["claim"] is not None]
    require(len(tokens) == len(set(tokens)), "Duplicate claim token")
    mutable = {"ledger.sqlite3", "events.csv"} if completed else set()
    for relative, expected in previous_files.items():
        if relative not in mutable:
            require(
                files.get(relative) == expected, f"Preserved native evidence changed: {relative}"
            )
    additions = {f"runs/{key}/outer_result.json"} if completed else set()
    require(set(files) == set(previous_files) | additions, "Unexpected/missing/partial native file")


def verify_a0(root: Path, state: dict[str, Any], records) -> dict[str, Any]:
    key = g.run_id("A0")
    run = state["native_runs_snapshot"][key]
    events = [e for e in state["native_events_snapshot"] if e[1] == key]
    cell = c.native.completed_result(root / r.BASE / "experiments", key, run, events, records)
    checkpoint = p.final_checkpoint(root, run)
    value = {
        **cell,
        **checkpoint,
        "status": "PASS",
        "provenance": "NATIVE",
        "target_state": run["state"],
        "target_claim": run["claim"],
    }
    validate_a0(value)
    return value


def inspect(mode: str, fold: int, ablation: str, root: Path = ROOT) -> dict[str, Any]:
    scope(mode, fold, ablation)
    root = root.resolve()
    completion = load_completion(root)
    acceptance, context, fold1_records = p.prepare(root)
    before = p.snapshot(root)
    validate_progress(completion, before, completed=mode == "verify")
    runs, events = before["native_runs_snapshot"], before["native_events_snapshot"]
    base = root / r.BASE / "experiments"
    c.verify_csv(base / "events.csv", events, advanced=True)
    c.verify_exception(root, acceptance, context, fold1_records, runs, events)
    records = context.split(2).records(context.predecessor.source.contract, "outer_test")
    g.validate_records(records, r.read_json(root / c.FINAL_AUDIT))
    g.validate_selections(r.read_json(base / "selection/fold2_ablations.json"), runs)
    a0 = verify_a0(root, before, records)
    require(a0["metrics"] == completion["verification"]["metrics"], "A0 recomputed metrics changed")
    key = g.run_id("A1")
    target, own = runs[key], [e for e in events if e[1] == key]
    result_path = base / "runs" / key / "outer_result.json"
    verified = None
    if mode == "preflight":
        c.native.validate_unclaimed(target, own, result_path.exists())
    else:
        verified = c.native.completed_result(base, key, target, own, records)
        require(
            verified["record_count"] == 2873 and verified["record_hash"] == g.RECORD_HASH,
            "A1 result ordered records/hash differ",
        )
    checkpoint = p.final_checkpoint(root, target)
    require(
        checkpoint["checkpoint_sha256"] == A1_CHECKPOINT
        and checkpoint["candidate"] == "stronger_decay"
        and checkpoint["selected_epoch"] == 2,
        "Pinned Fold2/A1 checkpoint/candidate/epoch differs",
    )
    p.assert_preserved(root, before, acceptance, context)
    require(r.file_hash(root / p.REPORT) == g.CLOSURE_SHA, "Fold-1 closure changed during check")
    require(r.file_hash(root / COMPLETION) == COMPLETION_SHA, "A0 completion changed during check")
    output = {
        "status": "PASS",
        "mode": mode,
        "outer_fold": 2,
        "ablation": "A1",
        "experiment_id": key,
        "provenance": "NATIVE",
        "target_state": target["state"],
        "target_claim": target["claim"],
        "protocol_hash": r.PROTOCOL,
        "frozen_code_hash": r.CODE,
        "runtime": {"seed": 42, "torch": "2.14.0+cpu", "cpu_threads": 2, "threshold": 0.5},
        "fold1_closure_sha256": g.CLOSURE_SHA,
        "a0_completion_sha256": COMPLETION_SHA,
        "a0_independent_verification": "PASS",
        "a0_result_sha256": a0["result_sha256"],
        "record_count": 2873,
        "record_hash": g.RECORD_HASH,
        "threshold": 0.5,
        **checkpoint,
        "model_forwards": 0,
        "production_writes": 0,
        "production_unchanged_during_check": True,
        "fold1_unchanged": True,
        "fold1_a1_exception_unchanged": True,
        "fold1_a1_original_prior_forward_count": "UNKNOWN",
        "fold2_a0_unchanged": True,
        "later_fold2_targets_untouched": True,
        "fold3_outer_untouched": True,
        "currently_authorized_target": "Fold2/A1",
        "cross_fold_aggregation": False,
        "execution_plan_changed_by_metrics": False,
        "stop_after_verification": True,
    }
    if verified is not None:
        output.update(result_sha256=verified["result_sha256"], metrics=verified["metrics"])
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "verify"))
    parser.add_argument("--outer-fold", type=int, choices=(2,), required=True)
    parser.add_argument("--ablation", choices=("A1",), required=True)
    args = parser.parse_args(argv)
    try:
        print(
            json.dumps(
                inspect(args.mode, args.outer_fold, args.ablation), indent=2, allow_nan=False
            )
        )
        return 0
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        IndexError,
        StopIteration,
    ) as exc:
        print(json.dumps({"status": "FAIL", "action": "STOP_FOR_REVIEW", "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
