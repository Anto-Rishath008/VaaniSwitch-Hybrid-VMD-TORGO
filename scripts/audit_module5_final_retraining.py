"""Read-only final-retraining audit; no data loading, training or model forwards."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import time
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from vaaniswitch.m5_experiments.ablation_models import AblationModel
from vaaniswitch.m5_experiments.checkpoint import load_checkpoint
from vaaniswitch.m5_experiments.config import CANDIDATES, ExperimentConfig
from vaaniswitch.m5_experiments.data import pos_weight
from vaaniswitch.m5_experiments.ledger import experiment_id
from vaaniswitch.m5_experiments.reporting import read_json, write_json
from vaaniswitch.m5_experiments.source import ExperimentContext, prepare_experiments, require_frozen
from vaaniswitch.m5_experiments.splits import record_hash
from vaaniswitch.m5_experiments.trainer import build_model, make_optimizer, run_identity

if __package__:
    from .audit_module5_phase5b import CODE, PROTOCOL, ROOT, fingerprint, inventory
    from .audit_module5_phase5c import COMMON, validate_rng
    from .report_module5_phase5b import require
else:
    from audit_module5_phase5b import CODE, PROTOCOL, ROOT, fingerprint, inventory
    from audit_module5_phase5c import COMMON, validate_rng
    from report_module5_phase5b import require

PHASE_C = "phase5c_audit/20260925T040400520005Z/phase5c_audit.json"


def read_state(base: Path) -> tuple[dict[str, Any], list[Any]]:
    with closing(sqlite3.connect((base / "ledger.sqlite3").as_uri() + "?mode=ro", uri=True)) as db:
        runs = {
            i: {"identity": json.loads(j), "state": s, "data": json.loads(d), "claim": c}
            for i, j, s, d, c in db.execute("SELECT id,identity,state,data,claim FROM runs")
        }
        events = list(db.execute("SELECT * FROM events ORDER BY seq"))
    return runs, events


def validate_final_history(
    progress: dict[str, Any],
    selection: dict[str, Any],
    count: int,
    record_digest: str,
    weight: float,
) -> None:
    require(progress["epoch"] == selection["epoch"], "Final epoch differs from frozen selection")
    require(progress["pos_weight"] == weight, "Final weight differs from development-only labels")
    require(
        progress["bad_epochs"] == 0
        and progress["patience_best"] == 0.0
        and progress["best_inner"] is None,
        "Final stage contains selection/early-stopping state",
    )
    require(
        [h["epoch"] for h in progress["history"]] == list(range(1, selection["epoch"] + 1)),
        "Final history is incomplete or has repeated epochs",
    )
    for h in progress["history"]:
        require(set(h) == {"epoch", "training"}, "Final stage contains validation/extra fields")
        t = h["training"]
        require(
            set(t) == {"loss", "record_count", "record_hash", "role", "elapsed_seconds"},
            "Final training contains validation metrics/extra fields",
        )
        require(
            t["role"] == "development"
            and t["record_count"] == count
            and t["record_hash"] == record_digest,
            "Final role/record coverage differs",
        )
        require(
            math.isfinite(t["loss"])
            and t["loss"] >= 0
            and math.isfinite(t["elapsed_seconds"])
            and t["elapsed_seconds"] >= 0,
            "Invalid final loss/time",
        )


def audit_final(
    context: ExperimentContext,
    run_id: str,
    run: dict[str, Any],
    selection: dict[str, Any],
    backbone: dict[str, torch.Tensor],
) -> dict[str, Any]:
    config = context.config
    fold, ablation = run["identity"]["outer_fold"], run["identity"]["ablation"]
    split, candidate = context.split(fold), config.candidate(COMMON[fold])
    expected = run_identity(context, split, ablation, candidate, run_id, smoke=False)
    require(run["identity"] == expected, f"Run identity drift: {run_id}")
    require(
        run["data"]["selection"] == run["data"]["inner_result"] == selection,
        "Final selection differs from retained inner evidence",
    )
    directory = config.output(f"experiments/runs/{run_id}/final")
    pointer = read_json(directory / "latest.json")
    require(pointer == run["data"]["final_checkpoint"], "Final pointer/ledger drift")
    require(
        {p.name for p in directory.iterdir()} == {"latest.json", pointer["filename"]},
        "Final must contain exactly one checkpoint generation and pointer",
    )
    identity = {
        **expected,
        "stage": "final",
        "selection_status": "SELECTED",
        "selected_inner_configuration": selection,
        "all_development_speakers": list(split.speakers("development")),
        "outer_test_speakers": list(split.speakers("outer_test")),
    }
    require(run["data"]["final_identity"] == identity, "Final identity drift")
    model = build_model(context, ablation, candidate, load_pretrained=False)
    optimizer = make_optimizer(model, candidate)
    saved = load_checkpoint(directory, model, optimizer, identity, pointer=pointer)
    validate_rng(saved["rng"])
    counts = model.parameter_counts()
    require(counts == run["data"]["parameter_counts"], "Final architecture/counts differ")
    state = saved["model"]
    prefix = "branches.mel.backbone."
    frozen = {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
    require(
        set(frozen) == (set(backbone) if ablation in {"A0", "A1", "C1", "C2"} else set()),
        "Final backbone inventory differs",
    )
    require(all(torch.equal(v, backbone[k]) for k, v in frozen.items()), "Frozen ResNet drift")
    records = split.records(context.predecessor.source.contract, "development")
    weight = pos_weight(context, split, "development", smoke=False)
    require(len(split.speakers("development")) == 10, "Final requires ten development speakers")
    p = saved["progress"]
    validate_final_history(p, selection, len(records), record_hash(records), weight)
    require(
        run["data"]["final_epoch"] == selection["epoch"]
        and run["data"]["final_pos_weight"] == weight,
        "Final ledger epoch/weight drift",
    )
    seconds = sum(h["training"]["elapsed_seconds"] for h in p["history"])
    require(seconds == run["data"]["final_training_seconds"], "Final timing drift")
    steps = math.ceil(len(records) / 16) * selection["epoch"]
    require(
        all(v["step"].item() == steps for v in saved["optimizer"]["state"].values()),
        "Final AdamW steps inconsistent with exact epoch count",
    )
    return {
        "status": "PASS",
        "experiment_id": run_id,
        "outer_fold": fold,
        "ablation": ablation,
        "candidate": candidate.id,
        "epoch": p["epoch"],
        "development_record_count": len(records),
        "development_record_hash": record_hash(records),
        "development_pos_weight": weight,
        "development_speakers": list(split.speakers("development")),
        "checkpoint": pointer,
        "checkpoint_bytes": (directory / pointer["filename"]).stat().st_size,
        "parameter_counts": counts,
        "model_tensors": len(state),
        "frozen_backbone_tensors": len(frozen),
        "optimizer_steps": steps,
        "rng_round_trip": "PASS",
        "training_seconds": seconds,
    }


def audit(transcript: Path | None = None) -> dict[str, Any]:
    config = ExperimentConfig.from_yaml(
        ROOT / "configs/experiments/module5/protocol.yaml", repo_root=ROOT
    )
    context = prepare_experiments(config)
    require_frozen(context)
    require(
        config.config_hash == PROTOCOL and context.code_hash == CODE, "Frozen code/protocol drift"
    )
    require(str(torch.__version__) == "2.14.0+cpu" and config.cpu_threads == 2, "CPU runtime drift")
    torch.set_num_threads(2)
    base = config.output("experiments")
    before, protocol = inventory(base), fingerprint(config.output("protocol.json"))
    prior_paths = [
        config.output("phase5a_audit/20260922T155051Z/phase5a_audit.json"),
        config.output("phase5b_audit/20260923T180221073595Z/phase5b_audit.json"),
        config.output(PHASE_C),
    ]
    for path in prior_paths:
        require(read_json(path)["status"] == "PASS", f"Predecessor phase audit failed: {path}")
    old = read_json(prior_paths[-1])
    require(old["protocol_hash"] == PROTOCOL and old["frozen_code_hash"] == CODE, "Prior identity")
    retained = {
        k: v
        for k, v in old["production_evidence_before_and_after"].items()
        if k.startswith(("runs/", "selection/", "reports/"))
    }
    require(all(before.get(k) == v for k, v in retained.items()), "Retained inner evidence changed")
    protected = read_json(config.output("phase5a_audit/20260922T155051Z/protected_inputs.json"))[
        "protected_files"
    ]
    for relative, expected in protected.items():
        require(
            fingerprint(ROOT / relative) == expected, f"Protected predecessor changed: {relative}"
        )
    runs, events = read_state(base)
    chosen = {
        s["experiment_id"]: s for b in old["selected_results"] for s in b["selections"].values()
    }
    expected_ids = {experiment_id(config, f, "C2", c.id) for f in (1, 2, 3) for c in CANDIDATES}
    expected_ids.update(chosen)
    require(
        set(runs) == expected_ids and len(runs) == 30, "Expected 30 exact production identities"
    )
    require(
        Counter(r["state"] for r in runs.values())
        == {"FINAL_TRAIN_COMPLETE": 21, "INNER_COMPLETE": 9},
        "Unexpected ledger states",
    )
    require(all(r["claim"] is None for r in runs.values()), "Outer claim exists")
    require(len(events) == 132 and [e[0] for e in events] == list(range(1, 133)), "Event inventory")
    with (base / "events.csv").open(encoding="utf-8", newline="") as stream:
        csv_events = list(csv.reader(stream))
    require(
        csv_events[0] == ["sequence", "experiment_id", "time", "event", "detail"]
        and csv_events[1:] == [[str(v) for v in e] for e in events],
        "Event CSV mismatch",
    )
    for run_id, run in runs.items():
        own = [e for e in events if e[1] == run_id]
        expected_events = ["PLANNED", "RUNNING", "INNER_COMPLETE"]
        if run_id in chosen:
            expected_events += ["SELECTED", "FINAL_TRAIN_COMPLETE"]
        require([e[3] for e in own] == expected_events, f"Unexpected run lifecycle: {run_id}")
        require(json.loads(own[0][4]) == run["identity"], "Planned identity mismatch")
        merged: dict[str, Any] = {}
        for e in own[1:]:
            merged.update(json.loads(e[4]) or {})
        require(merged == run["data"], "Ledger/event data mismatch")
        directory = base / "runs" / run_id
        require(
            {p.name for p in directory.iterdir()}
            == ({"inner", "final"} if run_id in chosen else {"inner"}),
            "Unexpected run artifacts",
        )
        require(
            {p.relative_to(base).as_posix() for p in (directory / "inner").iterdir()}
            == {k for k in retained if k.startswith(f"runs/{run_id}/inner/")},
            "Unexpected retained inner files",
        )
    require({p.name for p in (base / "runs").iterdir()} == expected_ids, "Unexpected run namespace")
    require(
        {p.name for p in base.iterdir()}
        == {"ledger.sqlite3", "events.csv", "locks", "runs", "reports", "selection"},
        "Unexpected production output, outer or robustness evidence",
    )
    for folder in ("reports", "selection"):
        require(
            {p.relative_to(base).as_posix() for p in (base / folder).iterdir()}
            == {k for k in retained if k.startswith(folder + "/")},
            "Unexpected reporting output",
        )
    require(not list(config.output("reports").glob("*")), "Aggregation/robustness output exists")
    for bundle in old["selected_results"]:
        require(
            read_json(base / f"selection/fold{bundle['outer_fold']}_ablations.json") == bundle,
            "Immutable Phase 5C bundle drift",
        )
    signed = torch.load(
        ROOT / "artifacts/checkpoints/module4/fold1_seed42_6fc69cf0ad55/best.pt",
        map_location="cpu",
        weights_only=True,
    )["model_state"]
    prefix = "encoders.mel_encoder.backbone."
    backbone = {k.removeprefix(prefix): v for k, v in signed.items() if k.startswith(prefix)}
    require(len(backbone) == 120, "Signed ResNet inventory")
    with patch.object(
        AblationModel, "forward", side_effect=AssertionError("Audit forbids forwards")
    ):
        checkpoints = [
            audit_final(context, i, runs[i], s, backbone) for i, s in sorted(chosen.items())
        ]
    outer_plan = []
    for fold in (1, 2, 3):
        split = context.split(fold)
        records = split.records(context.predecessor.source.contract, "outer_test")
        outer_plan.append(
            {
                "outer_fold": fold,
                "speakers": list(split.speakers("outer_test")),
                "record_count": len(records),
                "record_hash": record_hash(records),
                "logical_ids": [r.logical_id for r in records],
            }
        )
    pasted_evidence = None
    if transcript:
        text = transcript.read_text(encoding="utf-8-sig")
        pasted, _ = json.JSONDecoder().raw_decode(text[text.index("{") :])
        require(
            {r["id"]: {k: v for k, v in r.items() if k != "id"} for r in pasted["runs"]} == runs,
            "Pasted ledger differs from retained evidence",
        )
        pasted_evidence = {"path": str(transcript), **fingerprint(transcript), "matches": True}
    require_frozen(context)
    require(
        inventory(base) == before and fingerprint(config.output("protocol.json")) == protocol,
        "Audit changed production evidence or protocol",
    )
    return {
        "status": "PASS",
        "final_retraining": "COMPLETE / PASSED",
        "next": "OUTER EVALUATION READY FOR MANUAL EXECUTION",
        "protocol_hash": PROTOCOL,
        "frozen_code_hash": CODE,
        "torch": str(torch.__version__),
        "production_runs": 30,
        "states": dict(Counter(r["state"] for r in runs.values())),
        "events": len(events),
        "final_checkpoints": checkpoints,
        "final_checkpoint_count": 21,
        "per_fold_final_count": {
            str(f): sum(c["outer_fold"] == f for c in checkpoints) for f in (1, 2, 3)
        },
        "final_epoch_total": sum(c["epoch"] for c in checkpoints),
        "inner_files_unchanged": len(retained),
        "protected_files_unchanged": len(protected),
        "protected_scope": "Fresh SHA-256/size/mtime for 247 predecessor files; retained accepted "
        "raw/A0/Module1/Module2 full-tree evidence, not a repeated corpus audit.",
        "failed_invalidated": 0,
        "outer_claims": 0,
        "outer_events": 0,
        "outer_results": 0,
        "robustness_started": False,
        "outer_model_forward_count": 0,
        "audit_model_forwards": 0,
        "initialization_resume_evidence": "Frozen train_stage starts fresh when final/latest.json "
        "is absent; otherwise loads final only. Exact final optimizer steps/identities/history and "
        "focused mocked regressions support this. "
        "No separately logged initialization/resume trace.",
        "outer_forward_evidence": "Frozen development-only code, exact development histories and "
        "no claims/events/results; derived, not an independent historical counter.",
        "outer_evaluation_plan": outer_plan,
        "production_runs_snapshot": runs,
        "production_evidence_before_and_after": before,
        "frozen_protocol_fingerprint": protocol,
        "prior_audits": {str(p.relative_to(ROOT)): fingerprint(p) for p in prior_paths},
        "pasted_ledger": pasted_evidence,
        "module5_signed_off": False,
        "module6_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-transcript", type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    path = (
        ROOT
        / "artifacts/module5"
        / PROTOCOL
        / "final_retraining_audit"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
        / "final_retraining_audit.json"
    )
    try:
        report = audit(args.ledger_transcript)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, AssertionError) as exc:
        report = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    report["elapsed_seconds"] = time.perf_counter() - start
    write_json(path, report, immutable=True)
    print(
        json.dumps(
            {
                "status": report["status"],
                "artifact": path.as_posix(),
                "elapsed_seconds": report["elapsed_seconds"],
                "error": report.get("error"),
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
