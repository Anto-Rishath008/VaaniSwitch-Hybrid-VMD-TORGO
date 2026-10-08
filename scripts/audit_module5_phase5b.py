"""Read-only Phase 5B closure audit; no dataset loads, training or model forwards.

Only the timestamped audit artifact is written. scripts/ is outside the frozen
code hash (src/vaaniswitch/m[012345]*/*.py). Run from the project root.
"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from vaaniswitch.m4_fusion.losses import label_for_condition
from vaaniswitch.m5_experiments.ablation_models import AblationModel
from vaaniswitch.m5_experiments.checkpoint import load_checkpoint
from vaaniswitch.m5_experiments.config import CANDIDATES, ExperimentConfig, file_hash
from vaaniswitch.m5_experiments.ledger import experiment_id
from vaaniswitch.m5_experiments.reporting import read_json, write_json
from vaaniswitch.m5_experiments.source import prepare_experiments, require_frozen
from vaaniswitch.m5_experiments.trainer import build_model, make_optimizer

if __package__:
    from .report_module5_phase5b import panel_report, read_runs, require
else:
    from report_module5_phase5b import panel_report, read_runs, require

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9"
CODE = "7696cad010914ec37f494d9f1a11a2e1761e46f353e61d97a07399f746a34ff8"


def fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": file_hash(path)}


def inventory(path: Path) -> dict[str, Any]:
    return {
        p.relative_to(path).as_posix(): fingerprint(p)
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }


def validate_history(progress: dict[str, Any], positives: int, negatives: int) -> None:
    """Cross-check stored confusion-derived metrics and the frozen patience rule."""
    bad, best = 0, 0.0
    for epoch in progress["history"]:
        require(bad < 5, "Epoch exists after early stopping")
        metrics = epoch["validation"]
        tn, fp, fn, tp = (metrics[k] for k in ("TN", "FP", "FN", "TP"))
        require(
            all(type(n) is int and n >= 0 for n in (tn, fp, fn, tp)), "Invalid confusion counts"
        )
        require(
            tn + fp == negatives and tp + fn == positives,
            "Confusion counts differ from signed validation labels",
        )
        require(metrics["confusion_matrix"] == [[tn, fp], [fn, tp]], "Confusion matrix differs")

        def ratio(a: int, b: int) -> float:
            return a / b if b else 0.0

        expected = {
            "accuracy": (tn + tp) / (positives + negatives),
            "sensitivity": tp / positives,
            "specificity": tn / negatives,
            "precision": ratio(tp, tp + fp),
            "balanced_accuracy": (tp / positives + tn / negatives) / 2,
            "macro_f1": (ratio(2 * tp, 2 * tp + fp + fn) + ratio(2 * tn, 2 * tn + fp + fn)) / 2,
        }
        for key, value in expected.items():
            require(
                math.isclose(metrics[key], value, rel_tol=1e-12, abs_tol=1e-12),
                f"Confusion-derived metric differs: {key}",
            )
        require(math.isfinite(metrics["loss"]) and metrics["loss"] >= 0, "Invalid loss")
        require(0 <= metrics["roc_auc"] <= 1, "Invalid ROC-AUC")
        significant = epoch["epoch"] == 1 or metrics["macro_f1"] > best + 1e-4
        bad = 0 if significant else bad + 1
        if significant:
            best = metrics["macro_f1"]
    require(
        progress["bad_epochs"] == bad and progress["patience_best"] == best,
        "Early stopping state differs",
    )
    require(bad == 5 or progress["epoch"] == 30, "Candidate stopped before frozen stopping rule")


def audit() -> dict[str, Any]:
    config = ExperimentConfig.from_yaml(
        ROOT / "configs/experiments/module5/protocol.yaml", repo_root=ROOT
    )
    context = prepare_experiments(config)
    require_frozen(context)
    require(config.config_hash == PROTOCOL and context.code_hash == CODE, "Frozen hash drift")
    torch.set_num_threads(config.cpu_threads)
    base = config.output("experiments")
    before = inventory(base)
    protocol_before = fingerprint(config.output("protocol.json"))
    protected_path = config.output("phase5a_audit/20260922T155051Z/protected_inputs.json")
    protected = read_json(protected_path)["protected_files"]
    for relative, expected in protected.items():
        require(fingerprint(ROOT / relative) == expected, f"Protected file changed: {relative}")
    runs = read_runs(base / "ledger.sqlite3")
    expected_ids = {experiment_id(config, f, "C2", c.id) for f in (1, 2, 3) for c in CANDIDATES}
    require(set(runs) == expected_ids, "Production run set differs from exactly 12 C2 candidates")
    require({p.name for p in (base / "runs").iterdir()} == expected_ids, "Run directories differ")
    require(not any(r["claim"] is not None for r in runs.values()), "Outer claim exists")
    states = Counter(r["state"] for r in runs.values())
    require(states == {"SELECTED": 3, "INNER_COMPLETE": 9}, "Unexpected candidate states")
    with sqlite3.connect((base / "ledger.sqlite3").as_uri() + "?mode=ro", uri=True) as db:
        events = list(db.execute("SELECT * FROM events ORDER BY seq"))
    with (base / "events.csv").open(newline="", encoding="utf-8") as stream:
        csv_events = list(csv.reader(stream))
    require(
        csv_events[0] == ["sequence", "experiment_id", "time", "event", "detail"],
        "Event CSV schema differs",
    )
    require(
        csv_events[1:] == [[str(v) for v in e] for e in events], "Event CSV differs from ledger"
    )
    require(len(events) == 39, "Unexpected event count")
    for run_id, run in runs.items():
        own = [e for e in events if e[1] == run_id]
        expected = ["PLANNED", "RUNNING", "INNER_COMPLETE"]
        if run["state"] == "SELECTED":
            expected.append("SELECTED")
        require([e[3] for e in own] == expected, f"Unexpected lifecycle: {run_id}")
        require(json.loads(own[0][4]) == run["identity"], "Planned identity differs")
        merged: dict[str, Any] = {}
        for e in own[1:]:
            merged.update(json.loads(e[4]) or {})
        require(merged == run["data"], "Ledger data differs from event history")
    require(
        set(p.name for p in base.iterdir())
        == {"ledger.sqlite3", "events.csv", "locks", "runs", "reports", "selection"},
        "Unexpected production evidence (final/outer/partial)",
    )
    for directory, names in (
        ("selection", {f"fold{f}_c2.json" for f in (1, 2, 3)}),
        ("reports", {f"fold{f}_phase5b.json" for f in (1, 2, 3)}),
    ):
        require(
            {p.name for p in (base / directory).iterdir()} == names,
            f"Unexpected {directory} evidence",
        )
    m4 = ROOT / "artifacts/checkpoints/module4/fold1_seed42_6fc69cf0ad55/best.pt"
    signed = torch.load(m4, map_location="cpu", weights_only=True)["model_state"]
    prefix = "encoders.mel_encoder.backbone."
    backbone = {k.removeprefix(prefix): v for k, v in signed.items() if k.startswith(prefix)}
    require(len(backbone) == 120, "Signed frozen backbone inventory differs")
    reports, checkpoints = [], []
    # An accidental forward in any audit dependency fails immediately.
    with patch.object(
        AblationModel, "forward", side_effect=AssertionError("Audit forbids forward")
    ):
        for fold in (1, 2, 3):
            report = panel_report(context, fold)
            require(
                report == read_json(base / f"reports/fold{fold}_phase5b.json"),
                f"Reporter disagreement for fold {fold}",
            )
            require(
                sum(c["ledger_status"] == "SELECTED" for c in report["candidates"]) == 1,
                "Exactly one winner required per fold",
            )
            split = context.split(fold)
            records = split.records(context.predecessor.source.contract, "inner_validation")
            positives = sum(label_for_condition(r.condition.value) for r in records)
            for candidate_report in report["candidates"]:
                run_id = candidate_report["experiment_id"]
                candidate = config.candidate(candidate_report["candidate_id"])
                directory = base / f"runs/{run_id}/inner"
                pointer = read_json(directory / "latest.json")
                require(
                    {p.name for p in directory.parent.iterdir()} == {"inner"},
                    "Final/outer/partial stage exists",
                )
                require(
                    {p.name for p in directory.iterdir()} == {"latest.json", pointer["filename"]},
                    "Unexpected checkpoint generations/partials",
                )
                identity = {
                    **runs[run_id]["identity"],
                    "stage": "inner",
                    "selection_status": "RUNNING",
                    "selected_inner_configuration": None,
                    "all_development_speakers": list(split.speakers("development")),
                    "outer_test_speakers": list(split.speakers("outer_test")),
                }
                model = build_model(context, "C2", candidate, load_pretrained=False)
                optimizer = make_optimizer(model, candidate)
                payload = load_checkpoint(directory, model, optimizer, identity)
                counts = model.parameter_counts()
                require(counts == runs[run_id]["data"]["parameter_counts"], "Parameter count drift")
                require(counts["trainable"] == 291921, "C2 architecture drift")
                state = payload["model"]
                for name, value in backbone.items():
                    require(
                        torch.equal(state["branches.mel.backbone." + name], value),
                        f"Frozen backbone drift: {run_id}/{name}",
                    )
                progress = payload["progress"]
                validate_history(progress, positives, len(records) - positives)
                train_count = progress["history"][0]["training"]["record_count"]
                steps = math.ceil(train_count / 16) * progress["epoch"]
                require(
                    all(s["step"].item() == steps for s in payload["optimizer"]["state"].values()),
                    "AdamW step count differs from complete inner-training epochs",
                )
                checkpoints.append(
                    {
                        "experiment_id": run_id,
                        "sha256": pointer["sha256"],
                        "checkpoint_bytes": (directory / pointer["filename"]).stat().st_size,
                        "model_tensors": len(state),
                        "matching_frozen_backbone_tensors": len(backbone),
                        "parameter_counts": counts,
                        "optimizer_steps": steps,
                        "epochs_completed": progress["epoch"],
                        "best_epoch": progress["best_inner"]["epoch"],
                        "pos_weight": progress["pos_weight"],
                        "status": "PASS",
                    }
                )
                del payload, optimizer, model
            reports.append(report)
    require_frozen(context)
    require(inventory(base) == before, "Audit changed production evidence")
    require(
        fingerprint(config.output("protocol.json")) == protocol_before, "Audit changed protocol"
    )
    return {
        "status": "PASS",
        "phase5b": "COMPLETE / PASSED",
        "phase5c": "NOT STARTED / READY FOR MANUAL EXECUTION",
        "protocol_hash": PROTOCOL,
        "frozen_code_hash": CODE,
        "production_runs": len(runs),
        "states": dict(states),
        "events": len(events),
        "events_csv_matches_ledger": True,
        "outer_claims": 0,
        "outer_events": 0,
        "failed_invalidated_runs": 0,
        "final_retrains": 0,
        "outer_results": 0,
        "outer_model_forward_count": 0,
        "outer_forward_evidence": "Derived from frozen code, exact epoch role/record hashes, "
        "and absence of claims/events/results; no independent historical forward counter.",
        "audit_model_forwards": 0,
        "reporter_agreement": True,
        "checkpoint_validation": checkpoints,
        "panels": reports,
        "protected_files_unchanged": len(protected),
        "protected_scope": "247 retained predecessor files SHA-256/size/mtime; "
        "Phase 5A raw/cache tree inventories not rerun.",
        "production_evidence_before_and_after": before,
        "frozen_protocol_fingerprint": protocol_before,
        "module5_signed_off": False,
        "module6_started": False,
    }


def main() -> int:
    start = time.perf_counter()
    path = (
        ROOT
        / "artifacts/module5"
        / PROTOCOL
        / "phase5b_audit"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
        / "phase5b_audit.json"
    )
    try:
        report = audit()
    except (OSError, ValueError, KeyError, RuntimeError, AssertionError) as exc:
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
