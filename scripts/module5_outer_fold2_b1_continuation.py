"""Read-only Fold2/B1 continuation after verified native Fold2/A0 and Fold2/A1.

Only preflight and post-evaluation verification are supported. Native inference remains
manual through the unchanged vaaniswitch-m5 outer command. This helper performs no
model inference, training, tuning, native ledger writes, ranking, or aggregation.
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
    from . import module5_outer_fold2_a1_continuation as h
else:
    import module5_outer_fold2_a1_continuation as h

g, p, c, r, require = h.g, h.p, h.c, h.r, h.require
ROOT = h.ROOT

A1_HELPER = "scripts/module5_outer_fold2_a1_continuation.py"
A1_HELPER_SHA = "f2557e7240100c94f024ea80646cd5a9f48597c45bf15045b1b80160a07fe29d"
SELECTION_SHA = "ff1c4e5a100e503dbc8e95f1772cd92dcf613a50d4f34cb90fd7b5db091cb8d5"
A1_CLAIM = "a7e181ffc4de46c39582f2720c020df6"
A1_RESULT_SHA = "13252f3660e5066c31dbd2ad0b2d0a9e6f85fd664c7a07cecce1f4f8389ff76b"
A1_CHECKPOINT = "6840e0688780b28ce5a103776993224880ed8137ce9122c7ba676b954ed69c67"
B1_CHECKPOINT = "0325643987a235655f39da418622a09049f1ee2fc94f0c09421daac0666132d4"

A1_EXPECTED = {
    "status": "PASS",
    "experiment_id": "fold2_A1_stronger_decay_seed42_677ad640faf5",
    "provenance": "NATIVE",
    "target_state": "OUTER_EVALUATED",
    "target_claim": A1_CLAIM,
    "checkpoint_sha256": A1_CHECKPOINT,
    "candidate": "stronger_decay",
    "selected_epoch": 2,
    "selection_sha256": SELECTION_SHA,
    "record_count": 2873,
    "record_hash": g.RECORD_HASH,
    "result_sha256": A1_RESULT_SHA,
    "threshold": 0.5,
}


def scope(mode: str, fold: int, ablation: str) -> None:
    require(mode in {"preflight", "verify"}, "Only read-only preflight/verify allowed")
    require(type(fold) is int and fold == 2, "Only Fold 2 is authorized")
    require(ablation == "B1", "Only Fold2/B1 is authorized; all other targets are excluded")


def validate_a1(value: dict[str, Any]) -> None:
    require(
        all(value.get(key) == expected for key, expected in A1_EXPECTED.items()),
        "Pinned Fold2/A1 identity/claim/checkpoint/result/records differ",
    )


def _exact_native_completion(
    original: dict[str, Any],
    current: dict[str, Any],
    *,
    claim: str | None = None,
    result_sha: str | None = None,
    label: str,
) -> None:
    expected_claim = claim if claim is not None else current.get("claim")
    expected_result = result_sha if result_sha is not None else current.get("data", {}).get(
        "outer_result_sha256"
    )
    require(
        original["state"] == "FINAL_TRAIN_COMPLETE"
        and original["claim"] is None
        and current["state"] == "OUTER_EVALUATED"
        and isinstance(expected_claim, str)
        and bool(re.fullmatch(r"[a-f0-9]{32}", expected_claim))
        and current["claim"] == expected_claim
        and isinstance(expected_result, str)
        and bool(re.fullmatch(r"[a-f0-9]{64}", expected_result))
        and current.get("data", {}).get("outer_result_sha256") == expected_result
        and current
        == {
            **original,
            "state": "OUTER_EVALUATED",
            "claim": expected_claim,
            "data": {**original["data"], "outer_result_sha256": expected_result},
        },
        f"Invalid native {label} completion or changed training/selection evidence",
    )


def validate_progress(
    completion: dict[str, Any], state: dict[str, Any], *, b1_completed: bool
) -> None:
    """Accept exactly A1 completion plus, optionally, the one B1 completion delta."""

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

    a1_key, b1_key = g.run_id("A1"), g.run_id("B1")

    for identity, original in originals.items():
        current = runs[identity]
        if identity == a1_key:
            _exact_native_completion(
                original,
                current,
                claim=A1_CLAIM,
                result_sha=A1_RESULT_SHA,
                label="A1",
            )
        elif identity == b1_key and b1_completed:
            _exact_native_completion(original, current, label="B1")
        else:
            require(current == original, f"Preserved/unrequested run changed: {identity}")

    require(
        events[: len(previous_events)] == previous_events,
        "Earlier native event prefix changed",
    )
    extra = events[len(previous_events) :]
    require(
        len(extra) == (4 if b1_completed else 2),
        "Unexpected prior/later/extra outer activity",
    )

    expected = [
        (a1_key, "OUTER_CLAIMED", {"claim": A1_CLAIM}),
        (a1_key, "OUTER_EVALUATED", {"outer_result_sha256": A1_RESULT_SHA}),
    ]
    if b1_completed:
        expected.extend(
            [
                (b1_key, "OUTER_CLAIMED", {"claim": runs[b1_key]["claim"]}),
                (
                    b1_key,
                    "OUTER_EVALUATED",
                    {"outer_result_sha256": runs[b1_key]["data"]["outer_result_sha256"]},
                ),
            ]
        )

    previous_time = datetime.fromisoformat(previous_events[-1][2])
    for index, (event, (expected_id, expected_type, expected_detail)) in enumerate(
        zip(extra, expected, strict=True)
    ):
        event_time = datetime.fromisoformat(event[2])
        require(
            event[0] == previous_events[-1][0] + index + 1
            and event[1] == expected_id
            and event[3] == expected_type
            and json.loads(event[4]) == expected_detail
            and event_time.utcoffset() is not None
            and event_time >= previous_time,
            "Invalid A1/B1 claim-completion lifecycle",
        )
        previous_time = event_time

    tokens = [value["claim"] for value in runs.values() if value["claim"] is not None]
    require(len(tokens) == len(set(tokens)), "Duplicate native claim token")

    # A0-completion snapshot predates the A1 result. Ledger/events legitimately advance;
    # all other predecessor files remain byte-identical, and only the exact A1/B1 result
    # files may be added.
    for relative, expected_file in previous_files.items():
        if relative not in {"ledger.sqlite3", "events.csv"}:
            require(
                files.get(relative) == expected_file,
                f"Preserved native evidence changed: {relative}",
            )

    a1_result_rel = f"runs/{a1_key}/outer_result.json"
    require(
        a1_result_rel in files
        and files[a1_result_rel].get("sha256") == A1_RESULT_SHA,
        "Fold2/A1 native result file missing or changed",
    )

    allowed = set(previous_files) | {a1_result_rel}
    if b1_completed:
        b1_result_rel = f"runs/{b1_key}/outer_result.json"
        require(
            b1_result_rel in files
            and files[b1_result_rel].get("sha256") == runs[b1_key]["data"]["outer_result_sha256"],
            "Fold2/B1 native result file missing or changed",
        )
        allowed.add(b1_result_rel)
    require(set(files) == allowed, "Unexpected/missing/partial native file")


def verify_a1(root: Path, state: dict[str, Any], records) -> dict[str, Any]:
    key = g.run_id("A1")
    run = state["native_runs_snapshot"][key]
    events = [event for event in state["native_events_snapshot"] if event[1] == key]
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
    validate_a1(value)
    return value


def inspect(mode: str, fold: int, ablation: str, root: Path = ROOT) -> dict[str, Any]:
    scope(mode, fold, ablation)
    root = root.resolve()

    require(
        r.file_hash(root / A1_HELPER) == A1_HELPER_SHA,
        "Previously used Fold2/A1 helper changed",
    )
    completion = h.load_completion(root)
    acceptance, context, fold1_records = p.prepare(root)
    before = p.snapshot(root)
    validate_progress(completion, before, b1_completed=mode == "verify")

    runs, events = before["native_runs_snapshot"], before["native_events_snapshot"]
    base = root / r.BASE / "experiments"
    c.verify_csv(base / "events.csv", events, advanced=True)
    c.verify_exception(root, acceptance, context, fold1_records, runs, events)

    records = context.split(2).records(context.predecessor.source.contract, "outer_test")
    g.validate_records(records, r.read_json(root / c.FINAL_AUDIT))
    g.validate_selections(r.read_json(base / "selection/fold2_ablations.json"), runs)
    require(
        r.file_hash(base / "selection/fold2_ablations.json") == SELECTION_SHA,
        "Fold2 selection bundle changed",
    )

    a0 = h.verify_a0(root, before, records)
    require(a0["metrics"] == completion["verification"]["metrics"], "A0 recomputed metrics changed")
    a1 = verify_a1(root, before, records)

    key = g.run_id("B1")
    target = runs[key]
    own = [event for event in events if event[1] == key]
    result_path = base / "runs" / key / "outer_result.json"
    verified = None
    if mode == "preflight":
        c.native.validate_unclaimed(target, own, result_path.exists())
    else:
        verified = c.native.completed_result(base, key, target, own, records)
        require(
            verified["record_count"] == 2873 and verified["record_hash"] == g.RECORD_HASH,
            "B1 result ordered records/hash differ",
        )

    checkpoint = p.final_checkpoint(root, target)
    require(
        checkpoint["checkpoint_sha256"] == B1_CHECKPOINT
        and checkpoint["candidate"] == "stronger_decay"
        and checkpoint["selected_epoch"] == 1
        and checkpoint["selection_sha256"] == SELECTION_SHA,
        "Pinned Fold2/B1 checkpoint/candidate/epoch/selection differs",
    )

    p.assert_preserved(root, before, acceptance, context)
    require(r.file_hash(root / p.REPORT) == g.CLOSURE_SHA, "Fold-1 closure changed during check")
    require(
        r.file_hash(root / h.COMPLETION) == h.COMPLETION_SHA,
        "A0 completion changed during check",
    )
    require(r.file_hash(root / A1_HELPER) == A1_HELPER_SHA, "A1 helper changed during check")

    output = {
        "status": "PASS",
        "mode": mode,
        "outer_fold": 2,
        "ablation": "B1",
        "experiment_id": key,
        "provenance": "NATIVE",
        "target_state": target["state"],
        "target_claim": target["claim"],
        "protocol_hash": r.PROTOCOL,
        "frozen_code_hash": r.CODE,
        "runtime": {"seed": 42, "torch": "2.14.0+cpu", "cpu_threads": 2, "threshold": 0.5},
        "fold1_closure_sha256": g.CLOSURE_SHA,
        "a0_completion_sha256": h.COMPLETION_SHA,
        "a0_result_sha256": a0["result_sha256"],
        "a1_independent_verification": "PASS",
        "a1_claim": A1_CLAIM,
        "a1_result_sha256": a1["result_sha256"],
        "a1_checkpoint_sha256": a1["checkpoint_sha256"],
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
        "fold2_a1_unchanged": True,
        "later_fold2_targets_untouched": True,
        "fold3_outer_untouched": True,
        "currently_authorized_target": "Fold2/B1",
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
    parser.add_argument("--ablation", choices=("B1",), required=True)
    args = parser.parse_args(argv)
    try:
        print(
            json.dumps(
                inspect(args.mode, args.outer_fold, args.ablation),
                indent=2,
                allow_nan=False,
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
