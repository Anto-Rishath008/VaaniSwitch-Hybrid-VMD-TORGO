"""Read-only Fold-2/A0 preflight and verification after the pinned Fold-1 closure.

The future order is A0,A1,B1,B2,B3,C1,C2. Only A0 is enabled by this handoff.
No model inference, native writes, recovery, ranking or aggregation is performed.
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
    from . import module5_outer_exception_report as p
else:
    import module5_outer_exception_report as p

c, r, require = p.c, p.r, p.require
ROOT = p.ROOT
CLOSURE_SHA = "df03075219e2aa584664b3d340d1fe537c60b6c6c588bb2451995e2b0eea362b"
ORDER = p.ORDER
EPOCHS = dict(zip(ORDER, (11, 2, 1, 9, 5, 2, 2), strict=True))
RECORD_HASH = "e85a74cb8dd33249ac0f27d42521d5dc0cd2d9abd283970c978f278d9fb97353"


def run_id(ablation: str) -> str:
    return f"fold2_{ablation}_stronger_decay_seed42_{r.PROTOCOL[:12]}"


def scope(mode: str, fold: int, ablation: str) -> None:
    require(mode in {"preflight", "verify"}, "Only read-only preflight/verify supported")
    require(type(fold) is int and fold == 2, "This handoff is limited to Fold 2")
    require(ablation == "A0", "Only the first Fold-2 target A0 is currently authorized")


def load_closure(root: Path) -> dict[str, Any]:
    path = c.canonical_path(root, p.REPORT)
    require(r.file_hash(path) == CLOSURE_SHA, "Fold-1 closure missing or changed")
    value = r.read_json(path)
    require(
        value["schema"] == "module5_fold1_exception_closure_v1"
        and value["status"] == "PASS"
        and value["outer_fold"] == 1
        and value["protocol_hash"] == r.PROTOCOL
        and value["frozen_code_hash"] == r.CODE
        and value["disclosure"] == p.DISCLOSURE
        and value["model_forwards"] == value["production_writes"] == 0
        and value["fold2_fold3_outer_untouched"] is True,
        "Invalid prerequisite Fold-1 closure",
    )
    p.validate_cells(value["cells"])
    return value


def validate_progress(closure: dict[str, Any], state: dict[str, Any], *, completed: bool) -> None:
    before = closure["native_runs_snapshot"]
    runs = state["native_runs_snapshot"]
    events, previous_events = state["native_events_snapshot"], closure["native_events_snapshot"]
    files, previous_files = state["native_files_snapshot"], closure["native_files_snapshot"]
    require(set(runs) == set(before), "Native experiment set changed")
    pending = [
        key
        for key, value in runs.items()
        if value["claim"] is not None and value["state"] != "OUTER_EVALUATED"
    ]
    require(pending == [r.RUN], "Second or changed unfinished claim; stop for review")
    key = run_id("A0")
    for identity, original in before.items():
        if not completed or identity != key:
            require(runs[identity] == original, f"Unrequested run changed: {identity}")
    if completed:
        target, original = runs[key], before[key]
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
            "Invalid native A0 completion or altered selection/checkpoint",
        )
    require(events[: len(previous_events)] == previous_events, "Original events changed")
    extra = events[len(previous_events) :]
    require(len(extra) == (2 if completed else 0), "Unexpected additional/later native events")
    previous_time = datetime.fromisoformat(previous_events[-1][2])
    for index, event in enumerate(extra):
        expected_type = ("OUTER_CLAIMED", "OUTER_EVALUATED")[index]
        detail = (
            {"claim": runs[key]["claim"]}
            if index == 0
            else {"outer_result_sha256": runs[key]["data"]["outer_result_sha256"]}
        )
        event_time = datetime.fromisoformat(event[2])
        require(
            event[0] == previous_events[-1][0] + index + 1
            and event[1] == key
            and event[3] == expected_type
            and json.loads(event[4]) == detail
            and event_time.utcoffset() is not None
            and event_time >= previous_time,
            "Invalid claim/completion event lifecycle",
        )
        previous_time = event_time
    tokens = [v["claim"] for v in runs.values() if v["claim"] is not None]
    require(len(tokens) == len(set(tokens)), "Duplicate claim token")
    mutable = {"ledger.sqlite3", "events.csv"} if completed else set()
    for relative, expected in previous_files.items():
        if relative not in mutable:
            require(files.get(relative) == expected, f"Preserved evidence changed: {relative}")
    additions = {f"runs/{key}/outer_result.json"} if completed else set()
    require(set(files) == set(previous_files) | additions, "Unexpected/missing/partial native file")


def validate_records(records, audit: dict[str, Any]) -> None:
    plan = next(v for v in audit["outer_evaluation_plan"] if v["outer_fold"] == 2)
    require(
        len(records) == 2873
        and c.native.record_hash(records) == RECORD_HASH
        and plan["record_hash"] == RECORD_HASH
        and [v.logical_id for v in records] == plan["logical_ids"],
        "Fold-2 ordered signed records/hash differ",
    )


def validate_selections(bundle, runs) -> None:
    require(set(bundle["selections"]) == set(ORDER), "Frozen seven-target panel differs")
    for ablation, epoch in EPOCHS.items():
        selection = bundle["selections"][ablation]
        run = runs[run_id(ablation)]
        require(
            selection == run["data"]["selection"]
            and selection["experiment_id"] == run_id(ablation)
            and selection["candidate_id"] == "stronger_decay"
            and selection["epoch"] == run["data"]["final_epoch"] == epoch
            and run["identity"]["outer_fold"] == 2
            and run["identity"]["ablation"] == ablation
            and run["state"] in {"FINAL_TRAIN_COMPLETE", "OUTER_EVALUATED"},
            "Frozen Fold-2 candidate/epoch/identity/readiness differs",
        )


def inspect(mode: str, fold: int, ablation: str, root: Path = ROOT) -> dict[str, Any]:
    scope(mode, fold, ablation)
    root = root.resolve()
    closure = load_closure(root)
    acceptance, context, fold1_records = p.prepare(root)
    before = p.snapshot(root)
    validate_progress(closure, before, completed=mode == "verify")
    runs, events = before["native_runs_snapshot"], before["native_events_snapshot"]
    base = root / r.BASE / "experiments"
    c.verify_csv(base / "events.csv", events, advanced=True)
    # Fold-1 native files and rows are pinned by closure + validate_progress.
    # Reverify the exact immutable exceptional journal/result without any inference.
    c.verify_exception(root, acceptance, context, fold1_records, runs, events)
    records = context.split(2).records(context.predecessor.source.contract, "outer_test")
    validate_records(records, r.read_json(root / c.FINAL_AUDIT))
    bundle = r.read_json(base / "selection/fold2_ablations.json")
    validate_selections(bundle, runs)
    key = run_id(ablation)
    target, own = runs[key], [e for e in events if e[1] == key]
    result_path = base / "runs" / key / "outer_result.json"
    verified = None
    if mode == "preflight":
        c.native.validate_unclaimed(target, own, result_path.exists())
    else:
        verified = c.native.completed_result(base, key, target, own, records)
        require(verified["record_hash"] == RECORD_HASH, "Native Fold-2 result record hash differs")
    checkpoint = p.final_checkpoint(root, target)
    p.assert_preserved(root, before, acceptance, context)
    require(r.file_hash(root / p.REPORT) == CLOSURE_SHA, "Closure changed during inspection")
    output = {
        "status": "PASS",
        "mode": mode,
        "outer_fold": 2,
        "ablation": "A0",
        "experiment_id": key,
        "provenance": "NATIVE",
        "target_state": target["state"],
        "target_claim": target["claim"],
        "protocol_hash": r.PROTOCOL,
        "frozen_code_hash": r.CODE,
        "fold1_closure_sha256": CLOSURE_SHA,
        "record_count": 2873,
        "record_hash": RECORD_HASH,
        "threshold": 0.5,
        **checkpoint,
        "model_forwards": 0,
        "production_writes": 0,
        "production_unchanged_during_check": True,
        "fold1_unchanged": True,
        "a1_exception_unchanged": True,
        "a1_original_prior_forward_count": "UNKNOWN",
        "fold3_outer_untouched": True,
        "no_later_fold2_target_ran": True,
        "future_order": list(ORDER),
        "currently_authorized_target": "Fold2/A0",
        "stop_after_verification": True,
        "cross_fold_aggregation": False,
    }
    if verified is not None:
        output.update(result_sha256=verified["result_sha256"], metrics=verified["metrics"])
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "verify"))
    parser.add_argument("--outer-fold", type=int, choices=(2,), required=True)
    parser.add_argument("--ablation", choices=("A0",), required=True)
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
