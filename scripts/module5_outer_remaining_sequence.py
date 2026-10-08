"""Fixed remaining outer sequence: read-only preflight/status, append-only verify.

No native inference, training, claims or ledger writes. Verify publishes only an
external immutable completion JSON. Run all invocations sequentially.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from contextlib import ExitStack, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

sys.dont_write_bytecode = True

if __package__:
    from . import module5_outer_fold2_c1_continuation as old
else:
    import module5_outer_fold2_c1_continuation as old

h, g, p, c, r, require = old.h, old.g, old.p, old.c, old.r, old.require
ROOT = old.ROOT
BASE = f"{r.BASE}/outer_remaining_sequence"
ANCHOR = f"{BASE}/c1_completion.json"
ANCHOR_SHA = "a132ee00cb7398dc10d3626f90cb8523846f1111650a172b92ffaf1cc347d8d9"
PLAN_SHA = "3cf6ef447c7791df2eec2823e68e36d8ad3b7719b090b6089e10ac8d3499cf48"
CHAIN = f"{BASE}/completions"
SELF = "scripts/module5_outer_remaining_sequence.py"
RUNTIME = {"seed": 42, "torch": "2.14.0+cpu", "cpu_threads": 2, "threshold": 0.5}
ORDER = ((2, "C2"), *((3, a) for a in g.ORDER))
EPOCHS = (2, 5, 5, 9, 6, 10, 5, 2)
RECORDS = {
    2: (2873, g.RECORD_HASH),
    3: (2981, "72eedd08aa36215e3d998e0d19bdf95f655c9cc7686f83f707a5cb053fde266a"),
}
SNAPSHOTS = ("native_files_snapshot", "native_runs_snapshot", "native_events_snapshot")


@contextmanager
def forbid_scientific_execution():
    """Fail before scientific model construction/calls or a native Ledger opens."""
    import torch

    from vaaniswitch.m5_experiments import evaluator, trainer
    from vaaniswitch.m5_experiments.ledger import Ledger

    def forbidden(*args, **kwargs):
        raise RuntimeError("Scientific execution/native Ledger writes are forbidden")

    with ExitStack() as stack:
        for obj, name in (
            (torch.nn.Module, "__init__"),
            (torch.nn.Module, "_call_impl"),
            (Ledger, "__init__"),
            (evaluator, "evaluate_outer"),
            (trainer, "train_stage"),
            (r, "build_model"),
            (r, "ExperimentDataset"),
        ):
            stack.enter_context(patch.object(obj, name, forbidden))
        yield


def snapshot(value: dict[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in SNAPSHOTS}


def canonical_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def load_anchor(root: Path) -> dict[str, Any]:
    require(r.file_hash(root / ANCHOR) == ANCHOR_SHA, "C1 completion anchor changed")
    anchor = r.read_json(root / ANCHOR)
    require(
        anchor["schema"] == "module5_remaining_sequence_anchor_v1"
        and anchor["status"] == "PASS"
        and anchor["protocol_hash"] == r.PROTOCOL
        and anchor["frozen_code_hash"] == r.CODE
        and anchor["runtime"] == RUNTIME
        and anchor["runtime_sha256"] == c.native.digest(RUNTIME)
        and anchor["fold1_closure_sha256"] == g.CLOSURE_SHA
        and anchor["b3_completion_sha256"] == old.COMPLETION_SHA
        and anchor["actual_model_forwards"] == anchor["native_writes"] == 0,
        "Invalid frozen anchor identity/runtime",
    )
    plan = anchor["remaining_sequence"]
    require(
        len(plan) == len(ORDER)
        and [(v["outer_fold"], v["ablation"]) for v in plan] == list(ORDER)
        and [v["selected_epoch"] for v in plan] == list(EPOCHS)
        and [v["candidate"] for v in plan] == ["stronger_decay", *(["baseline"] * 7)]
        and c.native.digest(plan) == anchor["sequence_sha256"] == PLAN_SHA,
        "Frozen remaining sequence changed",
    )
    for relative, checksum in anchor["protected_helper_sha256"].items():
        require(
            r.file_hash(root / relative) == checksum, f"Previously used helper changed: {relative}"
        )
    for relative, expected in anchor["earlier_completion_evidence"].items():
        require(
            r.fingerprint(root / relative) == expected, f"Earlier completion changed: {relative}"
        )
    old.validate_progress(old.load_completion(root), anchor, completed=True)
    value = anchor["verification"]
    require(
        value["experiment_id"] == g.run_id("C1")
        and value["target_claim"] == "ed75e153cfca41a89a1487179a15f4fc"
        and value["result_sha256"]
        == "59b3dac25e0e5afbc717bd028bad82b4bb022158f2f5f4f927418e2d041f13b2"
        and value["checkpoint_sha256"] == old.C1_CHECKPOINT
        and value["candidate"] == "stronger_decay"
        and value["selected_epoch"] == 2,
        "C1 anchor identity differs",
    )
    return anchor


def read_chain(root: Path, *, ignore_stage: str | None = None) -> list[dict[str, Any]]:
    directory = root / CHAIN
    if not directory.exists():
        return []
    require(directory.is_dir() and not directory.is_symlink(), "Invalid completion directory")
    entries = sorted(path for path in directory.iterdir() if path.name != ignore_stage)
    require(len(entries) <= len(ORDER), "Too many completions")
    chain = []
    for index, entry in enumerate(entries):
        require(
            entry.name == f"{index:02d}" and entry.is_dir() and not entry.is_symlink(),
            "Missing/out-of-order/partial completion slot",
        )
        children = list(entry.iterdir())
        require(len(children) == 1, "Missing or extra completion file")
        path = children[0]
        require(
            path.is_file()
            and not path.is_symlink()
            and re.fullmatch(r"[a-f0-9]{64}\.json", path.name) is not None,
            "Unexpected completion filename",
        )
        checksum = r.file_hash(path)
        require(checksum == path.stem, "Previous completion SHA changed")
        value = r.read_json(path)
        require(path.read_bytes() == canonical_bytes(value), "Noncanonical completion")
        chain.append(
            {"path": path.relative_to(root).as_posix(), "sha256": checksum, "value": value}
        )
    return chain


def validate_delta(previous: dict, current: dict, target: dict | None) -> None:
    """Accept no change, or exactly one target's native two-event completion."""
    runs, originals = current["native_runs_snapshot"], previous["native_runs_snapshot"]
    events, old_events = current["native_events_snapshot"], previous["native_events_snapshot"]
    files, old_files = current["native_files_snapshot"], previous["native_files_snapshot"]
    require(set(runs) == set(originals), "Native experiment set changed")
    unfinished = [
        key
        for key, value in runs.items()
        if value["claim"] is not None and value["state"] != "OUTER_EVALUATED"
    ]
    require(unfinished == [r.RUN], "Unexpected unfinished claim or changed Fold1/A1")
    key = target["experiment_id"] if target else None
    for identity, original in originals.items():
        if identity == key:
            old.k._exact_native_completion(original, runs[identity], label=identity)
        else:
            require(runs[identity] == original, f"Prior/later/unrequested run changed: {identity}")
    require(events[: len(old_events)] == old_events, "Earlier event prefix changed")
    extra = events[len(old_events) :]
    require(len(extra) == (2 if target else 0), "Unexpected/missing native events")
    last_time = datetime.fromisoformat(old_events[-1][2])
    for index, event in enumerate(extra):
        detail = (
            {"claim": runs[key]["claim"]}
            if index == 0
            else {"outer_result_sha256": runs[key]["data"]["outer_result_sha256"]}
        )
        when = datetime.fromisoformat(event[2])
        require(
            event[0] == old_events[-1][0] + index + 1
            and event[1] == key
            and event[3] == ("OUTER_CLAIMED", "OUTER_EVALUATED")[index]
            and json.loads(event[4]) == detail
            and when.utcoffset() is not None
            and when >= last_time,
            "Invalid exact claim/completion lifecycle",
        )
        last_time = when
    claims = [run["claim"] for run in runs.values() if run["claim"] is not None]
    require(len(claims) == len(set(claims)), "Duplicate claim token")
    mutable = {"ledger.sqlite3", "events.csv"} if target else set()
    for relative, expected in old_files.items():
        if relative not in mutable:
            require(files.get(relative) == expected, f"Prior native file changed: {relative}")
    additions = {f"runs/{key}/outer_result.json"} if target else set()
    require(set(files) == set(old_files) | additions, "Unexpected/missing/partial native file")
    if target:
        require(
            files[f"runs/{key}/outer_result.json"]["sha256"]
            == runs[key]["data"]["outer_result_sha256"],
            "Result file/ledger checksum differs",
        )


def validate_records(records, fold: int, audit: dict) -> None:
    count, checksum = RECORDS[fold]
    planned = next(v for v in audit["outer_evaluation_plan"] if v["outer_fold"] == fold)
    require(
        len(records) == count
        and c.native.record_hash(records) == checksum
        and planned["record_hash"] == checksum
        and [v.logical_id for v in records] == planned["logical_ids"],
        f"Fold{fold} ordered signed record count/hash differs",
    )


def checkpoint(root: Path, run: dict, spec: dict) -> dict:
    value = p.final_checkpoint(root, run)
    require(
        all(
            value[key] == spec[key]
            for key in ("checkpoint_sha256", "candidate", "selected_epoch", "selection_sha256")
        ),
        "Frozen checkpoint/candidate/epoch/selection changed",
    )
    return value


def verify_cell(root: Path, state: dict, spec: dict, records, saved: dict) -> dict:
    key = spec["experiment_id"]
    run = state["native_runs_snapshot"][key]
    own = [event for event in state["native_events_snapshot"] if event[1] == key]
    value = c.native.completed_result(root / r.BASE / "experiments", key, run, own, records)
    value.update(
        saved,
        status="PASS",
        provenance="NATIVE",
        target_state=run["state"],
        target_claim=run["claim"],
    )
    require(
        value["experiment_id"] == key
        and value["outer_fold"] == spec["outer_fold"]
        and value["ablation"] == spec["ablation"]
        and value["record_count"] == spec["record_count"]
        and value["record_hash"] == spec["record_hash"]
        and value["threshold"] == 0.5,
        "Retained result identity/ordered records differ",
    )
    return value


def verify_prefix(root: Path, anchor: dict, state: dict, records) -> list[dict]:
    values = []
    for ablation in g.ORDER[:-1]:
        key = g.run_id(ablation)
        expected = (
            anchor["verification"]
            if ablation == "C1"
            else anchor["verification"]["prior_native_results"][ablation]
        )
        spec = {
            "experiment_id": key,
            "outer_fold": 2,
            "ablation": ablation,
            "record_count": RECORDS[2][0],
            "record_hash": RECORDS[2][1],
            "selection_sha256": old.k.SELECTION_SHA,
            **expected,
        }
        saved = checkpoint(root, state["native_runs_snapshot"][key], spec)
        value = verify_cell(root, state, spec, records, saved)
        require(
            value["target_claim"] == expected.get("target_claim", expected.get("claim"))
            and value["result_sha256"] == expected["result_sha256"],
            "Pinned completed-prefix claim/result changed",
        )
        if ablation == "C1":
            require(value["metrics"] == expected["metrics"], "C1 recomputed metrics changed")
        values.append(value)
    return values


def completion_document(
    anchor: dict, step: int, previous_sha: str, state: dict, verified: dict, helpers: dict
) -> dict:
    plan = anchor["remaining_sequence"]
    return {
        "schema": "module5_remaining_outer_completion_v1",
        "status": "PASS",
        "step_index": step,
        "target": plan[step],
        "verification": verified,
        "anchor_sha256": ANCHOR_SHA,
        "previous_completion_sha256": previous_sha,
        "sequence_sha256": PLAN_SHA,
        "protocol_hash": r.PROTOCOL,
        "frozen_code_hash": r.CODE,
        "runtime": RUNTIME,
        "runtime_sha256": c.native.digest(RUNTIME),
        "protected_helper_sha256": helpers,
        "actual_model_forwards": 0,
        "native_writes": 0,
        "next_target": plan[step + 1] if step + 1 < len(plan) else None,
        **snapshot(state),
    }


def validate_chain(
    anchor: dict, chain: list, current: dict, verified: dict, helpers: dict
) -> tuple[dict, str]:
    previous, previous_sha = anchor, ANCHOR_SHA
    plan = anchor["remaining_sequence"]
    for step, entry in enumerate(chain):
        value = entry["value"]
        validate_delta(previous, value, plan[step])
        expected = completion_document(
            anchor, step, previous_sha, value, verified[plan[step]["experiment_id"]], helpers
        )
        require(value == expected, "Changed completion identity/snapshot/link/metrics/helpers")
        previous, previous_sha = value, entry["sha256"]
    return previous, previous_sha


def authorize(mode: str, fold: int | None, ablation: str | None, plan: list, step: int) -> None:
    require(mode in {"status", "preflight", "verify"}, "Only status/preflight/verify supported")
    if mode == "status":
        require(fold is None and ablation is None, "Status takes no target")
    else:
        require(step < len(plan), "Sequence complete; no target authorized")
        target = plan[step]
        require(
            type(fold) is int and (fold, ablation) == (target["outer_fold"], target["ablation"]),
            "Only the exact next target is authorized; completed/out-of-order target rejected",
        )


def publish_completion(root: Path, value: dict, chain_before: list, native_before: dict) -> dict:
    """Atomic no-replace directory publication; partial stages fail closed."""
    step = value["step_index"]
    require(step == len(chain_before), "Wrong completion publication slot")
    require(read_chain(root) == chain_before, "Completion chain changed before publication")
    require(p.snapshot(root) == native_before, "Native evidence changed before publication")
    directory = root / CHAIN
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"{step:02d}"
    require(not final.exists(), "Never overwrite a completed verification")
    checksum = hashlib.sha256(canonical_bytes(value)).hexdigest()
    stage = directory / f".pending-{uuid.uuid4().hex}"
    stage.mkdir()
    path = stage / f"{checksum}.json"
    try:
        p.write_json(path, value, immutable=True)
        require(r.file_hash(path) == checksum, "Completion publication checksum differs")
        require(
            read_chain(root, ignore_stage=stage.name) == chain_before, "Concurrent chain change"
        )
        require(p.snapshot(root) == native_before, "Native evidence changed during publication")
        require(not final.exists(), "Completed verification already exists")
        os.rename(stage, final)
    finally:
        # Remove only this invocation's unpublished stage; never remove existing evidence.
        if stage.exists():
            require(stage.resolve().parent == directory.resolve(), "Stage escaped chain directory")
            path.unlink(missing_ok=True)
            stage.rmdir()
    return {"path": (final / path.name).relative_to(root).as_posix(), "sha256": checksum}


def _inspect(mode: str, fold: int | None, ablation: str | None, root: Path) -> dict:
    root = root.resolve()
    anchor = load_anchor(root)
    chain = read_chain(root)
    plan = anchor["remaining_sequence"]
    step = len(chain)
    authorize(mode, fold, ablation, plan, step)
    helpers = {**anchor["protected_helper_sha256"], SELF: r.file_hash(root / SELF)}
    acceptance, context, fold1_records = p.prepare(root)
    before = p.snapshot(root)
    runs, events = before["native_runs_snapshot"], before["native_events_snapshot"]
    c.verify_csv(root / r.BASE / "experiments/events.csv", events, advanced=True)
    c.verify_exception(root, acceptance, context, fold1_records, runs, events)
    audit = r.read_json(root / c.FINAL_AUDIT)
    records = {}
    for outer_fold in (2, 3):
        records[outer_fold] = context.split(outer_fold).records(
            context.predecessor.source.contract, "outer_test"
        )
        validate_records(records[outer_fold], outer_fold, audit)
    prefix = verify_prefix(root, anchor, before, records[2])
    saved = {v["experiment_id"]: checkpoint(root, runs[v["experiment_id"]], v) for v in plan}
    verified = {}
    for spec in plan[:step]:
        key = spec["experiment_id"]
        verified[key] = verify_cell(root, before, spec, records[spec["outer_fold"]], saved[key])
    previous, previous_sha = validate_chain(anchor, chain, before, verified, helpers)
    target = plan[step] if step < len(plan) else None
    pending = target is not None and runs[target["experiment_id"]]["state"] == "OUTER_EVALUATED"
    validate_delta(previous, before, target if pending else None)
    if target:
        key = target["experiment_id"]
        if pending:
            verified[key] = verify_cell(
                root, before, target, records[target["outer_fold"]], saved[key]
            )
        else:
            c.native.validate_unclaimed(
                runs[key],
                [e for e in events if e[1] == key],
                (root / r.BASE / "experiments/runs" / key / "outer_result.json").exists(),
            )
    require(
        mode != "preflight" or not pending, "Native evaluation complete; run verify, never rerun"
    )
    require(mode != "verify" or pending, "No exact completed native target to verify")
    # Recheck native, recovery, protected inputs, prerequisites and chain before any append.
    p.assert_preserved(root, before, acceptance, context)
    require(load_anchor(root) == anchor, "Anchor/prerequisite changed during check")
    require(read_chain(root) == chain, "Completion chain changed during check")
    require(r.file_hash(root / SELF) == helpers[SELF], "Generic guard changed during check")
    publication = None
    if mode == "verify":
        document = completion_document(
            anchor, step, previous_sha, before, verified[target["experiment_id"]], helpers
        )
        publication = publish_completion(root, document, chain, before)
    next_step = step + (1 if publication else 0)
    output = {
        "status": "PASS",
        "mode": mode,
        "sequence_state": (
            "COMPLETE"
            if next_step == len(plan)
            else "AWAITING_VERIFICATION"
            if pending and not publication
            else "READY"
        ),
        "anchor_completion_sha256": ANCHOR_SHA,
        "sequence_sha256": PLAN_SHA,
        "previous_completion_sha256": previous_sha,
        "protocol_hash": r.PROTOCOL,
        "frozen_code_hash": r.CODE,
        "runtime": RUNTIME,
        "fold1_closure_sha256": g.CLOSURE_SHA,
        "verified_chain_completions": next_step,
        "next_target": plan[next_step] if next_step < len(plan) else None,
        "remaining_order": [f"Fold{v['outer_fold']}/{v['ablation']}" for v in plan[next_step:]],
        "completed_fold2_prefix_verified": [v["experiment_id"] for v in prefix],
        "completed_remaining_results_verified": len(verified),
        "native_run_count": len(runs),
        "native_event_count": len(events),
        "native_file_count": len(before["native_files_snapshot"]),
        "model_forwards": 0,
        "native_writes": 0,
        "external_completion_written": publication is not None,
        "production_unchanged_during_check": True,
        "fold1_unchanged": True,
        "fold1_a1_exception_unchanged": True,
        "fold1_a1_provenance": "RECOVERED_PROTOCOL_EXCEPTION",
        "fold1_a1_original_prior_forward_count": "UNKNOWN",
        "completed_fold2_cells_unchanged": True,
        "later_targets_untouched": True,
        "fold3_fully_untouched": all(
            runs[v["experiment_id"]] == anchor["native_runs_snapshot"][v["experiment_id"]]
            for v in plan
            if v["outer_fold"] == 3
        ),
        "frozen_protocol_code_runtime_unchanged": True,
        "cross_fold_aggregation": False,
        "execution_plan_changed_by_metrics": False,
    }
    if target:
        output.update(
            checked_target=target,
            target_state=runs[target["experiment_id"]]["state"],
            target_claim=runs[target["experiment_id"]]["claim"],
        )
    if publication:
        output.update(
            completion_artifact=publication, verification=verified[target["experiment_id"]]
        )
    return output


def inspect(
    mode: str, fold: int | None = None, ablation: str | None = None, root: Path = ROOT
) -> dict:
    with forbid_scientific_execution():
        return _inspect(mode, fold, ablation, root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "verify", "status"))
    parser.add_argument("--outer-fold", type=int, choices=(2, 3))
    parser.add_argument("--ablation", choices=g.ORDER)
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
