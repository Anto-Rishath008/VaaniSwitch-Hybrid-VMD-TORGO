"""Read-only Fold2/B3 guard after immutable independent native B2 verification.

Only preflight and post-evaluation verification are supported. Run native inference
manually with the unchanged CLI. No training, inference, native writes or aggregation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

if __package__:
    from . import module5_outer_fold2_b2_continuation as previous
else:
    import module5_outer_fold2_b2_continuation as previous

h, g, p, c, r, require = (
    previous.h,
    previous.g,
    previous.p,
    previous.c,
    previous.r,
    previous.require,
)
k = previous.k
ROOT = previous.ROOT
COMPLETION = f"{r.BASE}/outer_continuation/fold2/b3_handoff/b2_completion.json"
COMPLETION_SHA = "17eb5b4bd05fe961440bbcb141f9d01419ebf4e1c77a61b43363ef162ee2a64e"
B2_HELPER = "scripts/module5_outer_fold2_b2_continuation.py"
B2_HELPER_SHA = "193c7ac7f3d7a0b0a1250da4bf9b4effd7a2123bbeb4f78b9e78ae32a0fa311d"
B2_EXPECTED = {
    **previous.B1_EXPECTED,
    "experiment_id": "fold2_B2_stronger_decay_seed42_677ad640faf5",
    "target_claim": "50e280d7b8f2491a9966aaeb6263c6df",
    "checkpoint_sha256": previous.B2_CHECKPOINT,
    "selected_epoch": 9,
    "result_sha256": "ec05eb39d0dd1ef7e6a18cb74a5c02b96b6cd5e47900e811f24f9e2a8fffe1b2",
}
B3_CHECKPOINT = "2787a6fb201c79ef2031593f0432438c19e0d9f7f5d5254b01a880b95547bd72"
B3_EXPECTED = {
    "checkpoint_sha256": B3_CHECKPOINT,
    "selected_epoch": 5,
    "candidate": "stronger_decay",
    "selection_sha256": k.SELECTION_SHA,
}


def scope(mode: str, fold: int, ablation: str) -> None:
    require(mode in {"preflight", "verify"}, "Only read-only preflight/verify allowed")
    require(type(fold) is int and fold == 2, "Only Fold 2 is authorized")
    require(ablation == "B3", "Only Fold2/B3 is authorized")


def validate_b2(value: dict[str, Any]) -> None:
    require(
        all(value.get(key) == expected for key, expected in B2_EXPECTED.items()),
        "Pinned Fold2/B2 identity/claim/checkpoint/result/records differ",
    )


def load_completion(root: Path) -> dict[str, Any]:
    require(r.file_hash(root / B2_HELPER) == B2_HELPER_SHA, "Used B2 helper changed")
    path = c.canonical_path(root, COMPLETION)
    require(r.file_hash(path) == COMPLETION_SHA, "B2 completion missing or changed")
    value = r.read_json(path)
    require(
        value["schema"] == "module5_fold2_b2_completion_v1"
        and value["status"] == "PASS"
        and value["protocol_hash"] == r.PROTOCOL
        and value["frozen_code_hash"] == r.CODE
        and value["fold1_closure_sha256"] == g.CLOSURE_SHA
        and value["a0_completion_sha256"] == h.COMPLETION_SHA
        and value["b1_completion_sha256"] == previous.COMPLETION_SHA
        and value["actual_model_forwards"] == value["native_writes"] == 0
        and value["next_manual_target"] == "Fold2/B3"
        and value["b3_checkpoint_discovery"] == B3_EXPECTED,
        "Invalid B2 completion prerequisite",
    )
    for relative, checksum in value["protected_helper_sha256"].items():
        require(r.file_hash(root / relative) == checksum, f"Used helper changed: {relative}")
    # Verify the frozen prefix using the used helper, without asking that helper to
    # accept a later live B3 delta. Its live B2-only scope remains unchanged.
    previous.validate_progress(previous.load_completion(root), value, completed=True)
    validate_b2(value["verification"])
    return value


def validate_progress(
    completion: dict[str, Any], state: dict[str, Any], *, completed: bool
) -> None:
    """Permit no delta, or exactly one B3 claim/completion and its result."""
    originals, runs = completion["native_runs_snapshot"], state["native_runs_snapshot"]
    old_events, events = completion["native_events_snapshot"], state["native_events_snapshot"]
    old_files, files = completion["native_files_snapshot"], state["native_files_snapshot"]
    require(set(runs) == set(originals), "Native experiment set changed")
    unfinished = [
        key
        for key, value in runs.items()
        if value["claim"] is not None and value["state"] != "OUTER_EVALUATED"
    ]
    require(unfinished == [r.RUN], "Second or changed unfinished claim")
    key = g.run_id("B3")
    for identity, original in originals.items():
        if completed and identity == key:
            k._exact_native_completion(original, runs[identity], label="B3")
        else:
            require(runs[identity] == original, f"Preserved/unrequested run changed: {identity}")
    require(events[: len(old_events)] == old_events, "Earlier native event prefix changed")
    extra = events[len(old_events) :]
    require(len(extra) == (2 if completed else 0), "Unexpected prior/later/extra outer activity")
    previous_time = datetime.fromisoformat(old_events[-1][2])
    for index, event in enumerate(extra):
        detail = (
            {"claim": runs[key]["claim"]}
            if index == 0
            else {"outer_result_sha256": runs[key]["data"]["outer_result_sha256"]}
        )
        event_time = datetime.fromisoformat(event[2])
        require(
            event[0] == old_events[-1][0] + index + 1
            and event[1] == key
            and event[3] == ("OUTER_CLAIMED", "OUTER_EVALUATED")[index]
            and json.loads(event[4]) == detail
            and event_time.utcoffset() is not None
            and event_time >= previous_time,
            "Invalid B3 claim-completion lifecycle",
        )
        previous_time = event_time
    tokens = [value["claim"] for value in runs.values() if value["claim"] is not None]
    require(len(tokens) == len(set(tokens)), "Duplicate native claim token")
    mutable = {"ledger.sqlite3", "events.csv"} if completed else set()
    for relative, expected in old_files.items():
        if relative not in mutable:
            require(
                files.get(relative) == expected, f"Preserved native evidence changed: {relative}"
            )
    additions = {f"runs/{key}/outer_result.json"} if completed else set()
    require(set(files) == set(old_files) | additions, "Unexpected/missing/partial native file")
    if completed:
        require(
            files[f"runs/{key}/outer_result.json"]["sha256"]
            == runs[key]["data"]["outer_result_sha256"],
            "B3 result file/ledger checksum differs",
        )


def verify_b2(root: Path, state: dict[str, Any], records) -> dict[str, Any]:
    key = g.run_id("B2")
    run = state["native_runs_snapshot"][key]
    events = [event for event in state["native_events_snapshot"] if event[1] == key]
    cell = c.native.completed_result(root / r.BASE / "experiments", key, run, events, records)
    value = {
        **cell,
        **p.final_checkpoint(root, run),
        "status": "PASS",
        "provenance": "NATIVE",
        "target_state": run["state"],
        "target_claim": run["claim"],
    }
    validate_b2(value)
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
    require(
        r.file_hash(base / "selection/fold2_ablations.json") == k.SELECTION_SHA,
        "Fold2 selection bundle changed",
    )
    # Each verifier checks ordered IDs, signed labels, probabilities and recomputed
    # metrics against the actual retained native result before checking its pins.
    a0 = h.verify_a0(root, before, records)
    a1 = k.verify_a1(root, before, records)
    b1 = previous.verify_b1(root, before, records)
    b2 = verify_b2(root, before, records)
    require(
        b2["metrics"] == completion["verification"]["metrics"],
        "B2 recomputed metrics changed",
    )
    key = g.run_id("B3")
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
            "B3 result ordered records/hash differ",
        )
    checkpoint = p.final_checkpoint(root, target)
    require(checkpoint == B3_EXPECTED, "Pinned B3 checkpoint/candidate/epoch/selection differs")

    p.assert_preserved(root, before, acceptance, context)
    # Recheck the external frozen prerequisites and already-used helpers too.
    require(load_completion(root) == completion, "Completion prerequisite changed during check")
    output = {
        "status": "PASS",
        "mode": mode,
        "outer_fold": 2,
        "ablation": "B3",
        "experiment_id": key,
        "provenance": "NATIVE",
        "target_state": target["state"],
        "target_claim": target["claim"],
        "protocol_hash": r.PROTOCOL,
        "frozen_code_hash": r.CODE,
        "runtime": {"seed": 42, "torch": "2.14.0+cpu", "cpu_threads": 2, "threshold": 0.5},
        "fold1_closure_sha256": g.CLOSURE_SHA,
        "b2_completion_sha256": COMPLETION_SHA,
        "prior_native_results": {
            ablation: {
                "verification": "PASS",
                "claim": cell["target_claim"],
                "result_sha256": cell["result_sha256"],
                "checkpoint_sha256": cell["checkpoint_sha256"],
                "candidate": cell["candidate"],
                "selected_epoch": cell["selected_epoch"],
            }
            for ablation, cell in (("A0", a0), ("A1", a1), ("B1", b1), ("B2", b2))
        },
        "record_count": 2873,
        "record_hash": g.RECORD_HASH,
        "threshold": 0.5,
        **checkpoint,
        "native_run_count": len(runs),
        "native_event_count": len(events),
        "native_file_count": len(before["native_files_snapshot"]),
        "model_forwards": 0,
        "production_writes": 0,
        "production_unchanged_during_check": True,
        "fold1_unchanged": True,
        "fold1_a1_exception_unchanged": True,
        "fold1_a1_provenance": "RECOVERED_PROTOCOL_EXCEPTION",
        "fold1_a1_original_prior_forward_count": "UNKNOWN",
        "fold2_a0_unchanged": True,
        "fold2_a1_unchanged": True,
        "fold2_b1_unchanged": True,
        "fold2_b2_unchanged": True,
        "later_fold2_targets_untouched": True,
        "fold3_outer_untouched": True,
        "frozen_protocol_code_runtime_unchanged": True,
        "currently_authorized_target": "Fold2/B3",
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
    parser.add_argument("--ablation", choices=("B3",), required=True)
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
