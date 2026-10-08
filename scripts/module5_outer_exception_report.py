"""Read-only Fold-1 closure; --save publishes only a new immutable external report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import torch  # noqa: E402

from vaaniswitch.m5_experiments.checkpoint import (  # noqa: E402
    checked_generation,
    validate_progress,
)
from vaaniswitch.m5_experiments.reporting import write_json  # noqa: E402

if __package__:
    from . import module5_outer_exception_continuation as c
else:
    import module5_outer_exception_continuation as c

r = c.recovery
require = r.require
ROOT = c.ROOT
ORDER = ("A0", "A1", "B1", "B2", "B3", "C1", "C2")
REPORT = f"{r.BASE}/outer_exception_report/fold1/fold1_outer_panel.json"
CONTINUATION_SHA = "f69d62c1c283ee9a847ead13bb6586277cbe5ae2276f24a1d943be55d399d054"
NATIVE_SHA = {
    "A0": "a2854e2f44d47e294137e35a3ff7682b221af7a14bdb954455ee9749fa917572",
    "B1": "36f4dbb3e66c5c1ddbd65251e28e3de28dca45d2ec9a78716e209c1e58da9595",
    "B2": "764e02548fc383c4eb0671e54eb32256b40cc35c313d8a952c442adf27e9879a",
    "B3": "55a91123dd614dc10b543d3f2c18dccf6d09b9f41e06d9c2d07142a0c60a0b77",
    "C1": "d1ea1b6ee9a76f3d75407aa9ed9ec4c884b4227c1395fdddb346893fedcba079",
    "C2": "f0cac4ed9f6a8fd730d6b0a373cda3255bbb7cb75df54b3ff4c3a14c80382c86",
}
NATIVE_CLAIM = {
    "A0": "1d8656252af14bd3bd83d59dfb121b76",
    "B1": "4718fd5e346c418081230b3765a52356",
    "B2": "eac59d8646054f898cb14f95932a249f",
    "B3": "29f33238710c437991d42c078979c134",
    "C1": "9b082a22ed954f4cafb318cf930d3f53",
    "C2": "a3b63a1a7f5642da9380267d84a10bf1",
}
DISCLOSURE = (
    "Fold 1/A1 is RECOVERED_PROTOCOL_EXCEPTION and does NOT satisfy uninterrupted "
    "native once-only outer evaluation: its prior native model-forward count "
    "cannot be established and remains UNKNOWN. Its original claim/state and "
    "absent native result are preserved."
)


def prepare(root: Path):
    require(
        r.file_hash(root / "scripts/module5_outer_exception_continuation.py") == CONTINUATION_SHA,
        "Original continuation helper changed",
    )
    acceptance = c.load_acceptance(root)
    context, records = c.frozen_context(root, acceptance)
    return acceptance, context, records


def snapshot(root: Path) -> dict[str, Any]:
    base = root / r.BASE / "experiments"
    files = r.inventory(base)
    runs, events = r.read_production(base / "ledger.sqlite3")
    return {
        "native_files_snapshot": files,
        "native_runs_snapshot": runs,
        "native_events_snapshot": [list(e) for e in events],
    }


def final_checkpoint(root: Path, run: dict[str, Any]) -> dict[str, Any]:
    identity = run["identity"]
    fold, ablation = identity["outer_fold"], identity["ablation"]
    base = root / r.BASE / "experiments"
    selection_path = base / f"selection/fold{fold}_ablations.json"
    selection = r.read_json(selection_path)["selections"][ablation]
    require(
        selection == run["data"]["selection"]
        and selection["experiment_id"] == identity["experiment_id"],
        "Final selection identity differs",
    )
    directory = c.canonical_path(
        root, f"{r.BASE}/experiments/runs/{identity['experiment_id']}/final"
    )
    pointer = r.read_json(directory / "latest.json")
    require(pointer == run["data"]["final_checkpoint"], "Final checkpoint pointer differs")
    path = checked_generation(directory, pointer)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    require(
        saved["schema"] == "module5_checkpoint_v1"
        and saved["identity"] == run["data"]["final_identity"],
        "Final checkpoint identity differs",
    )
    validate_progress(saved["progress"])
    require(
        saved["progress"]["epoch"] == selection["epoch"] == run["data"]["final_epoch"],
        "Final checkpoint epoch differs",
    )
    return {
        "checkpoint_sha256": pointer["sha256"],
        "selected_epoch": selection["epoch"],
        "candidate": selection["candidate_id"],
        "selection_sha256": r.file_hash(selection_path),
    }


def fold1_cells(root, acceptance, context, records, state) -> list[dict[str, Any]]:
    runs, events = state["native_runs_snapshot"], state["native_events_snapshot"]
    c.verify_exception(root, acceptance, context, records, runs, events)
    cells = []
    for ablation in ORDER:
        key = c.run_id(ablation)
        run = runs[key]
        checkpoint = final_checkpoint(root, run)
        if ablation == "A1":
            path = root / r.RECOVERY / c.ATTEMPT / "recovery_result.json"
            result = r.read_json(path)
            cell = {
                "ablation": "A1",
                "experiment_id": key,
                "outer_fold": 1,
                "provenance": "RECOVERED_PROTOCOL_EXCEPTION",
                "result_path": path.relative_to(root).as_posix(),
                "result_sha256": c.RESULT_SHA,
                "manifest_sha256": c.MANIFEST_SHA,
                "attempt_id": c.ATTEMPT,
                "metrics": result["metrics"],
                "record_count": result["record_count"],
                "record_hash": result["record_hash"],
                "threshold": result["threshold"],
                "native_result_absent": True,
                "original_prior_forward_count": "UNKNOWN",
                "uninterrupted_native_once_only": False,
            }
        else:
            require(run["claim"] == NATIVE_CLAIM[ablation], "Pinned native claim changed")
            cell = c.native.completed_result(
                root / r.BASE / "experiments",
                key,
                run,
                [e for e in events if e[1] == key],
                records,
            )
            require(cell["result_sha256"] == NATIVE_SHA[ablation], "Pinned native result changed")
            cell["provenance"] = "NATIVE"
        cells.append({**cell, **checkpoint, "native_state": run["state"], "claim": run["claim"]})
    validate_cells(cells)
    return cells


def validate_cells(cells: list[dict[str, Any]]) -> None:
    require([v["ablation"] for v in cells] == list(ORDER), "Seven ordered cells required")
    for cell in cells:
        ablation = cell["ablation"]
        require(
            cell["experiment_id"] == c.run_id(ablation)
            and cell["outer_fold"] == 1
            and cell["record_count"] == 2819
            and cell["record_hash"] == r.RECORD_HASH
            and cell["threshold"] == 0.5,
            "Fold-1 cell record identity differs",
        )
        if ablation == "A1":
            require(
                cell["provenance"] == "RECOVERED_PROTOCOL_EXCEPTION"
                and cell["native_state"] == "FINAL_TRAIN_COMPLETE"
                and cell["claim"] == r.CLAIM
                and cell["native_result_absent"] is True
                and cell["attempt_id"] == c.ATTEMPT
                and cell["result_sha256"] == c.RESULT_SHA
                and cell["manifest_sha256"] == c.MANIFEST_SHA
                and cell["original_prior_forward_count"] == "UNKNOWN"
                and cell["uninterrupted_native_once_only"] is False,
                "A1 exception cannot be changed or relabeled native",
            )
        else:
            require(
                cell["provenance"] == "NATIVE"
                and cell["native_state"] == "OUTER_EVALUATED"
                and cell["claim"] == NATIVE_CLAIM[ablation]
                and cell["result_sha256"] == NATIVE_SHA[ablation],
                "Second exception or changed native cell",
            )


def assert_preserved(root, before, acceptance, context) -> None:
    require(snapshot(root) == before, "Production changed during read-only inspection")
    require(
        r.inventory(root / r.RECOVERY) == acceptance["recovery_files_snapshot"],
        "Recovered A1 evidence changed",
    )
    c.validate_frozen_runtime(context)


def inspect(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    acceptance, context, records = prepare(root)
    before = snapshot(root)
    c.validate_native_progress(
        acceptance,
        before["native_runs_snapshot"],
        before["native_events_snapshot"],
        before["native_files_snapshot"],
        c.ORDER,
    )
    c.verify_csv(
        root / r.BASE / "experiments/events.csv",
        before["native_events_snapshot"],
        advanced=True,
    )
    cells = fold1_cells(root, acceptance, context, records, before)
    assert_preserved(root, before, acceptance, context)
    return {
        "schema": "module5_fold1_exception_closure_v1",
        "status": "PASS",
        "outer_fold": 1,
        "protocol_hash": r.PROTOCOL,
        "frozen_code_hash": r.CODE,
        "runtime": {"seed": 42, "torch": "2.14.0+cpu", "cpu_threads": 2, "threshold": 0.5},
        "acceptance_sha256": c.ACCEPTANCE_SHA,
        "continuation_helper_sha256": CONTINUATION_SHA,
        "cells": cells,
        "disclosure": DISCLOSURE,
        "native_cells": 6,
        "recovered_protocol_exception_cells": 1,
        "model_forwards": 0,
        "production_writes": 0,
        "production_unchanged_during_check": True,
        "cross_fold_aggregation": False,
        "ranking_or_selection": False,
        "execution_plan_changed_by_metrics": False,
        "fold2_fold3_outer_untouched": True,
        "protected_predecessor_files_verified": 247,
        "recovery_files_snapshot": acceptance["recovery_files_snapshot"],
        **before,
    }


def publish(root: Path, report: dict[str, Any]) -> Path:
    path = c.canonical_path(root, REPORT)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        require(path.read_bytes() == payload.encode(), "Existing closure differs; never overwrite")
    else:
        write_json(path, report, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-fold", type=int, choices=(1,), required=True)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = inspect()
        output = {k: v for k, v in report.items() if not k.endswith("_snapshot")}
        if args.save:
            path = publish(ROOT, report)
            output.update(artifact_path=str(path), artifact_sha256=r.file_hash(path))
        print(json.dumps(output, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError) as exc:
        print(json.dumps({"status": "FAIL", "action": "STOP_FOR_REVIEW", "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
