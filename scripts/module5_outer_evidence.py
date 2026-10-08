"""External read-only outer preflight/result reporting; never evaluates a model.

Frozen evaluator output is preserved. --save adds a separate immutable fold report
with explicit outer record hashes, which the original result schema does not store.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from vaaniswitch.m4_fusion.losses import label_for_condition
from vaaniswitch.m4_fusion.metrics import binary_metrics
from vaaniswitch.m5_experiments.config import ABLATIONS, ExperimentConfig, digest, file_hash
from vaaniswitch.m5_experiments.reporting import read_json, write_json
from vaaniswitch.m5_experiments.source import prepare_experiments, require_frozen
from vaaniswitch.m5_experiments.splits import record_hash

if __package__:
    from .audit_module5_final_retraining import read_state
    from .audit_module5_phase5b import CODE, PROTOCOL, ROOT, fingerprint
    from .report_module5_phase5b import require
else:
    from audit_module5_final_retraining import read_state
    from audit_module5_phase5b import CODE, PROTOCOL, ROOT, fingerprint
    from report_module5_phase5b import require


def validate_unclaimed(run: dict[str, Any], events: list[Any], result_exists: bool) -> None:
    require(
        run["state"] == "FINAL_TRAIN_COMPLETE" and run["claim"] is None,
        "Target is not an unclaimed completed final run; stop for review",
    )
    require(
        not result_exists and not any("OUTER" in e[3] for e in events),
        "Previous outer result/event exists; do not evaluate again",
    )


def validate_predictions(predictions: list[dict[str, Any]], records: tuple[Any, ...]) -> str:
    require(len(predictions) == len(records), "Outer prediction count differs")
    for p, r in zip(predictions, records, strict=True):
        require(
            p["logical_id"] == r.logical_id
            and p["speaker_id"] == r.speaker_id
            and type(p["label"]) is int
            and p["label"] == label_for_condition(r.condition.value),
            "Outer prediction record/speaker/label differs",
        )
        require(
            isinstance(p["probability"], (int, float))
            and not isinstance(p["probability"], bool)
            and math.isfinite(p["probability"])
            and 0 <= p["probability"] <= 1,
            "Invalid prediction probability",
        )
    observed = digest([p["logical_id"] for p in predictions])
    require(observed == record_hash(records), "Outer prediction record hash differs")
    return observed


def completed_result(
    base: Path,
    run_id: str,
    run: dict[str, Any],
    events: list[Any],
    records: tuple[Any, ...],
) -> dict[str, Any]:
    require(run["state"] == "OUTER_EVALUATED" and run["claim"], "Fold result incomplete")
    path = base / "runs" / run_id / "outer_result.json"
    checksum = file_hash(path)
    require(checksum == run["data"]["outer_result_sha256"], "Outer result checksum changed")
    result = read_json(path)
    require(
        result["status"] == "OUTER_EVALUATED"
        and result["experiment_id"] == run_id
        and result["identity"] == run["identity"]
        and result["selection"] == run["data"]["selection"]
        and result["checkpoint"] == run["data"]["final_checkpoint"]
        and result["ablation"] == run["identity"]["ablation"]
        and result["outer_fold"] == run["identity"]["outer_fold"]
        and result["claim"] == run["claim"]
        and result["threshold"] == 0.5,
        "Outer result provenance differs",
    )
    outer_events = [e for e in events if "OUTER" in e[3]]
    require(
        [e[3] for e in outer_events] == ["OUTER_CLAIMED", "OUTER_EVALUATED"],
        "Outer must have exactly one claim and completion",
    )
    require(
        json.loads(outer_events[0][4]) == {"claim": run["claim"]}
        and json.loads(outer_events[1][4]) == {"outer_result_sha256": checksum},
        "Claim/result event details differ",
    )
    observed = validate_predictions(result["predictions"], records)
    require(
        result["forward_count"] == result["source_load_count"] == len(records),
        "Outer forward/source counts differ",
    )
    recomputed = binary_metrics(
        [p["label"] for p in result["predictions"]],
        [p["probability"] for p in result["predictions"]],
    )
    require(recomputed == result["metrics"], "Saved metrics differ from retained predictions")
    return {
        "experiment_id": run_id,
        "ablation": result["ablation"],
        "outer_fold": result["outer_fold"],
        "record_count": len(records),
        "record_hash": observed,
        "threshold": 0.5,
        "metrics": recomputed,
        "checkpoint": result["checkpoint"],
        "final_identity": run["data"]["final_identity"],
        "result_path": str(path.relative_to(ROOT)),
        "result_sha256": checksum,
    }


def inspect(
    audit_path: Path,
    fold: int,
    mode: str,
    ablation: str | None = None,
) -> dict[str, Any]:
    config = ExperimentConfig.from_yaml(
        ROOT / "configs/experiments/module5/protocol.yaml", repo_root=ROOT
    )
    context = prepare_experiments(config)
    require_frozen(context)
    audit = read_json(audit_path)
    require(
        str(torch.__version__) == audit["torch"] == "2.14.0+cpu" and config.cpu_threads == 2,
        "Frozen CPU runtime drift",
    )
    require(
        audit["status"] == "PASS"
        and audit["protocol_hash"] == config.config_hash == PROTOCOL
        and audit["frozen_code_hash"] == context.code_hash == CODE,
        "Audit/frozen identity drift",
    )
    base = config.output("experiments")
    before = {p: fingerprint(base / p) for p in ("ledger.sqlite3", "events.csv")}
    runs, events = read_state(base)
    prior = audit["production_runs_snapshot"]
    require(set(runs) == set(prior), "Production experiment set changed")
    for run_id, run in runs.items():
        require(run["identity"] == prior[run_id]["identity"], "Experiment identity drift")
        require(
            all(run["data"].get(k) == v for k, v in prior[run_id]["data"].items()),
            "Inner/final evidence drift",
        )
        require(
            run["state"] in {"INNER_COMPLETE", "FINAL_TRAIN_COMPLETE", "OUTER_EVALUATED"},
            "Failed/invalidated/pending run exists; stop for review",
        )
        require(
            (prior[run_id]["state"] == "INNER_COMPLETE") == (run["state"] == "INNER_COMPLETE"),
            "Selected/unselected run state drift",
        )
        require(
            (run["state"] == "OUTER_EVALUATED") == (run["claim"] is not None),
            "Unfinished outer claim exists; stop for review",
        )
    # Recheck every selection and all seven final checkpoint files for this fold.
    evidence = audit["production_evidence_before_and_after"]
    expected = {
        p: v
        for p, v in evidence.items()
        if p.startswith("selection/") or (p.startswith(f"runs/fold{fold}_") and "/final/" in p)
    }
    for relative, retained in expected.items():
        require(fingerprint(base / relative) == retained, f"Audited evidence changed: {relative}")
    bundle = read_json(base / f"selection/fold{fold}_ablations.json")
    records = context.split(fold).records(context.predecessor.source.contract, "outer_test")
    plan = next(p for p in audit["outer_evaluation_plan"] if p["outer_fold"] == fold)
    require(
        [r.logical_id for r in records] == plan["logical_ids"]
        and record_hash(records) == plan["record_hash"]
        and len(records) == plan["record_count"],
        "Outer fold metadata/hash differs from audited plan",
    )
    results = []
    targets = [ablation] if ablation else list(ABLATIONS)
    for variant in ABLATIONS:
        selected = bundle["selections"][variant]
        run_id = selected["experiment_id"]
        run = runs[run_id]
        require(
            run["state"] in {"FINAL_TRAIN_COMPLETE", "OUTER_EVALUATED"}
            and run["data"]["selection"] == selected,
            "All seven final runs required",
        )
        own = [e for e in events if e[1] == run_id]
        if run["state"] == "OUTER_EVALUATED":
            result = completed_result(base, run_id, run, own, records)
            if mode == "report":
                results.append(result)
            elif variant in targets:
                raise ValueError("Target already evaluated; use report, do not repeat evaluation")
        else:
            validate_unclaimed(run, own, (base / "runs" / run_id / "outer_result.json").exists())
            require(mode != "report", "All seven immutable outer results required for fold report")
    require_frozen(context)
    require(
        {p: fingerprint(base / p) for p in before} == before, "Ledger/events changed during read"
    )
    return {
        "status": "PASS",
        "mode": mode,
        "outer_fold": fold,
        "target_ablations": targets,
        "protocol_hash": PROTOCOL,
        "frozen_code_hash": CODE,
        "record_count": len(records),
        "record_hash": record_hash(records),
        "threshold": 0.5,
        "closure_audit_sha256": file_hash(audit_path),
        "results": results,
        "model_forwards": 0,
        "aggregation_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "report"))
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--outer-fold", required=True, type=int, choices=(1, 2, 3))
    parser.add_argument("--ablation", choices=ABLATIONS)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    if (args.save or args.mode == "report") and args.ablation:
        parser.error("report/save operate on the entire completed fold")
    if args.save and args.mode != "report":
        parser.error("--save is for report only")
    try:
        report = inspect(args.audit, args.outer_fold, args.mode, args.ablation)
        if args.save:
            path = (
                ROOT
                / "artifacts/module5"
                / PROTOCOL
                / "outer_evidence"
                / (f"fold{args.outer_fold}_outer_report.json")
            )
            if path.exists():
                require(read_json(path) == report, "Existing report differs; preserve for review")
            else:
                write_json(path, report, immutable=True)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
