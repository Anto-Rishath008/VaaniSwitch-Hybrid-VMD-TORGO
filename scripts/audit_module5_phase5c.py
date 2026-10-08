"""Read-only Phase 5C evidence audit and descriptive inner-validation report.

No source data loads, forwards or training. Only a new timestamped audit is written.
This external script is outside src/vaaniswitch/m[012345]*/*.py (frozen code scope).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sqlite3
import statistics
import time
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

from vaaniswitch.m4_fusion.losses import label_for_condition
from vaaniswitch.m5_experiments.ablation_models import AblationModel
from vaaniswitch.m5_experiments.checkpoint import load_checkpoint
from vaaniswitch.m5_experiments.config import ABLATIONS, CANDIDATES, ExperimentConfig
from vaaniswitch.m5_experiments.data import pos_weight
from vaaniswitch.m5_experiments.ledger import experiment_id
from vaaniswitch.m5_experiments.reporting import read_json, write_json
from vaaniswitch.m5_experiments.search import InnerResult, metric_key
from vaaniswitch.m5_experiments.source import ExperimentContext, prepare_experiments, require_frozen
from vaaniswitch.m5_experiments.splits import record_hash
from vaaniswitch.m5_experiments.trainer import build_model, make_optimizer, run_identity

if __package__:
    from .audit_module5_phase5b import (
        CODE,
        PROTOCOL,
        ROOT,
        fingerprint,
        inventory,
        validate_history,
    )
    from .report_module5_phase5b import panel_report, read_runs, require
else:
    from audit_module5_phase5b import CODE, PROTOCOL, ROOT, fingerprint, inventory, validate_history
    from report_module5_phase5b import panel_report, read_runs, require

COMMON = {1: "stronger_dropout", 2: "stronger_decay", 3: "baseline"}
PAIRS = (("A1", "A0"), ("B3", "B1"), ("B3", "B2"), ("C1", "A1"), ("C2", "C1"))
METRICS = ("macro_f1", "balanced_accuracy", "roc_auc")


def validate_rng(state: dict[str, Any]) -> None:
    """Exercise restoration on isolated CPU generators, preserving process RNG state."""
    require(set(state) == {"python", "numpy", "torch"}, "RNG fields differ")
    py = random.Random()
    py.setstate(state["python"])
    require(py.getstate() == state["python"], "Python RNG round trip differs")
    n = state["numpy"]
    require(isinstance(n, list) and len(n) == 5 and n[0] == "MT19937", "NumPy RNG schema")
    require(
        len(n[1]) == 624 and all(type(v) is int and 0 <= v < 2**32 for v in n[1]),
        "NumPy RNG keys invalid",
    )
    require(
        type(n[2]) is int and 0 <= n[2] <= 624 and n[3] in (0, 1) and math.isfinite(n[4]),
        "NumPy RNG position/cache invalid",
    )
    ng = np.random.RandomState()
    ng.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    restored = ng.get_state()
    require(
        restored[0] == n[0] and restored[1].tolist() == n[1] and list(restored[2:]) == n[2:],
        "NumPy RNG round trip differs",
    )
    t = state["torch"]
    generator = torch.Generator(device="cpu")
    require(
        isinstance(t, torch.Tensor)
        and t.dtype == torch.uint8
        and t.device.type == "cpu"
        and t.shape == generator.get_state().shape,
        "CPU torch RNG schema",
    )
    generator.set_state(t)
    require(torch.equal(generator.get_state(), t), "Torch RNG round trip differs")


def validate_bundle(
    context: ExperimentContext,
    fold: int,
    bundle: dict[str, Any],
    runs: dict[str, Any],
    c2: dict[str, Any],
) -> None:
    require(
        set(bundle) == {"outer_fold", "config_hash", "common_candidate", "selections"},
        "Selection bundle schema",
    )
    require(
        bundle["outer_fold"] == fold
        and bundle["config_hash"] == PROTOCOL
        and bundle["common_candidate"] == COMMON[fold],
        "Common candidate/protocol drift",
    )
    require(set(bundle["selections"]) == set(ABLATIONS), "Seven matched ablations required")
    require(bundle["selections"]["C2"] == c2, "C2 was not reused exactly")
    for ablation, selected in bundle["selections"].items():
        run_id = experiment_id(context.config, fold, ablation, COMMON[fold])
        require(
            selected["experiment_id"] == run_id and selected["candidate_id"] == COMMON[fold],
            "Independent ablation settings or experiment identity",
        )
        run = runs[run_id]
        require(
            run["state"] == "SELECTED"
            and run["data"]["selection"] == selected
            and run["data"]["inner_result"] == selected,
            "Selection differs from ledger",
        )


def summaries(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    require([b["outer_fold"] for b in bundles] == [1, 2, 3], "Exactly ordered three folds required")
    differences = {
        f"{left}-{right}": {
            metric: [
                b["selections"][left]["metrics"][metric] - b["selections"][right]["metrics"][metric]
                for b in bundles
            ]
            for metric in METRICS
        }
        for left, right in PAIRS
    }
    stats = {}
    for ablation in ABLATIONS:
        stats[ablation] = {}
        for metric in METRICS:
            values = [b["selections"][ablation]["metrics"][metric] for b in bundles]
            stats[ablation][metric] = {
                "values": values,
                "mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values),
                "fold_count": 3,
                "ddof": 1,
            }
    return {
        "scope": "descriptive INNER validation only; not outer generalization or significance",
        "differences": differences,
        "summary": stats,
        "macro_f1_fold_winners": [
            max(ABLATIONS, key=lambda a: b["selections"][a]["metrics"]["macro_f1"]) for b in bundles
        ],
    }


def audit_inner(
    context: ExperimentContext,
    run_id: str,
    run: dict[str, Any],
    backbone: dict[str, torch.Tensor],
) -> dict[str, Any]:
    original = run["identity"]
    fold, ablation = original["outer_fold"], original["ablation"]
    split = context.split(fold)
    candidate = context.config.candidate(original["candidate"]["id"])
    expected = run_identity(context, split, ablation, candidate, run_id, smoke=False)
    require(original == expected, f"Experiment identity drift: {run_id}")
    directory = context.config.output(f"experiments/runs/{run_id}/inner")
    pointer = read_json(directory / "latest.json")
    require(
        {p.name for p in directory.parent.iterdir()} == {"inner"},
        f"Final/outer/partial stage exists: {run_id}",
    )
    require(
        {p.name for p in directory.iterdir()} == {"latest.json", pointer["filename"]},
        f"Unexpected checkpoint generations/partials: {run_id}",
    )
    identity = {
        **expected,
        "stage": "inner",
        "selection_status": "RUNNING",
        "selected_inner_configuration": None,
        "all_development_speakers": list(split.speakers("development")),
        "outer_test_speakers": list(split.speakers("outer_test")),
    }
    model = build_model(context, ablation, candidate, load_pretrained=False)
    optimizer = make_optimizer(model, candidate)
    payload = load_checkpoint(directory, model, optimizer, identity)
    validate_rng(payload["rng"])
    counts = model.parameter_counts()
    require(counts == run["data"]["parameter_counts"], "Ablation parameter count drift")
    state = payload["model"]
    prefix = "branches.mel.backbone."
    actual_backbone = {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
    has_mel = ablation in {"A0", "A1", "C1", "C2"}
    require(set(actual_backbone) == (set(backbone) if has_mel else set()), "Backbone keys differ")
    for name, tensor in actual_backbone.items():
        require(torch.equal(tensor, backbone[name]), f"Frozen backbone drift: {run_id}/{name}")
    p = payload["progress"]
    require(
        p["pos_weight"] == pos_weight(context, split, "inner_train", smoke=False),
        "pos_weight differs from inner-training-only labels",
    )
    contract = context.predecessor.source.contract
    role_records = {
        role: split.records(contract, role) for role in ("inner_train", "inner_validation")
    }
    val = role_records["inner_validation"]
    positives = sum(label_for_condition(r.condition.value) for r in val)
    validate_history(p, positives, len(val) - positives)
    for epoch in p["history"]:
        require(set(epoch) == {"epoch", "training", "validation"}, "Inner epoch fields differ")
        for key, role in (("training", "inner_train"), ("validation", "inner_validation")):
            evidence = epoch[key]
            require(
                evidence["role"] == role
                and evidence["record_count"] == len(role_records[role])
                and evidence["record_hash"] == record_hash(role_records[role]),
                "Epoch record/speaker role drift",
            )
            require(
                math.isfinite(evidence["loss"])
                and evidence["loss"] >= 0
                and math.isfinite(evidence["elapsed_seconds"])
                and evidence["elapsed_seconds"] >= 0,
                "Nonfinite/negative stage loss/time",
            )
        require(epoch["validation"]["threshold"] == 0.5, "Threshold drift")
    result = InnerResult(**p["best_inner"])
    require(result.identity == run["data"]["inner_result"], "Checkpoint/ledger best differs")
    best = max(p["history"], key=lambda h: metric_key(h["validation"], counts["trainable"]))
    require(
        result.epoch == best["epoch"]
        and result.metrics == best["validation"]
        and result.trainable_parameters == counts["trainable"]
        and result.role == "inner_validation"
        and result.experiment_id == run_id
        and result.candidate_id == candidate.id
        and result.split_hash == split.split_hash
        and result.validation_record_hash == record_hash(val),
        "Selected epoch evidence drift",
    )
    seconds = sum(h["training"]["elapsed_seconds"] for h in p["history"])
    require(seconds == run["data"]["inner_training_seconds"], "Training wall-time evidence drift")
    steps = math.ceil(len(role_records["inner_train"]) / 16) * p["epoch"]
    require(
        all(s["step"].item() == steps for s in payload["optimizer"]["state"].values()),
        "AdamW step count differs from complete training epochs",
    )
    return {
        "status": "PASS",
        "experiment_id": run_id,
        "outer_fold": fold,
        "ablation": ablation,
        "candidate": candidate.id,
        "sha256": pointer["sha256"],
        "checkpoint_bytes": (directory / pointer["filename"]).stat().st_size,
        "parameter_counts": counts,
        "model_tensors": len(state),
        "matching_frozen_backbone_tensors": len(actual_backbone),
        "optimizer_steps": steps,
        "rng_round_trip": "PASS",
        "epochs_completed": p["epoch"],
        "selected_epoch": result.epoch,
        "pos_weight": p["pos_weight"],
        "training_seconds": seconds,
    }


def audit(transcript: Path | None = None) -> dict[str, Any]:
    config = ExperimentConfig.from_yaml(
        ROOT / "configs/experiments/module5/protocol.yaml", repo_root=ROOT
    )
    context = prepare_experiments(config)
    require_frozen(context)
    require(config.config_hash == PROTOCOL and context.code_hash == CODE, "Frozen hash drift")
    require(
        str(torch.__version__) == "2.14.0+cpu" and config.cpu_threads == 2,
        "Frozen CPU runtime drift",
    )
    torch.set_num_threads(config.cpu_threads)
    base = config.output("experiments")
    before = inventory(base)
    protocol_before = fingerprint(config.output("protocol.json"))
    old_path = config.output("phase5b_audit/20260923T180221073595Z/phase5b_audit.json")
    old = read_json(old_path)
    require(old["status"] == "PASS" and old["frozen_code_hash"] == CODE, "Phase 5B audit identity")
    unchanged = {
        k: v
        for k, v in old["production_evidence_before_and_after"].items()
        if k.startswith(("runs/", "selection/", "reports/"))
    }
    require(all(before.get(k) == v for k, v in unchanged.items()), "Phase 5B C2 evidence changed")
    protected_path = config.output("phase5a_audit/20260922T155051Z/protected_inputs.json")
    protected = read_json(protected_path)["protected_files"]
    for relative, expected in protected.items():
        require(fingerprint(ROOT / relative) == expected, f"Protected file changed: {relative}")
    runs = read_runs(base / "ledger.sqlite3")
    expected_ids = {experiment_id(config, f, "C2", c.id) for f in (1, 2, 3) for c in CANDIDATES}
    expected_ids.update(
        experiment_id(config, f, a, COMMON[f]) for f in (1, 2, 3) for a in ABLATIONS if a != "C2"
    )
    require(set(runs) == expected_ids and len(runs) == 30, "Production run set differs")
    require({p.name for p in (base / "runs").iterdir()} == expected_ids, "Run directories differ")
    states = Counter(r["state"] for r in runs.values())
    require(states == {"SELECTED": 21, "INNER_COMPLETE": 9}, "Production states differ")
    require(all(r["claim"] is None for r in runs.values()), "Outer claim exists")
    with closing(sqlite3.connect((base / "ledger.sqlite3").as_uri() + "?mode=ro", uri=True)) as db:
        events = list(db.execute("SELECT * FROM events ORDER BY seq"))
    with (base / "events.csv").open(newline="", encoding="utf-8") as stream:
        exported = list(csv.reader(stream))
    require(
        exported[0] == ["sequence", "experiment_id", "time", "event", "detail"]
        and exported[1:] == [[str(v) for v in e] for e in events],
        "Event CSV/ledger mismatch",
    )
    require(
        len(events) == 111 and [e[0] for e in events] == list(range(1, 112)),
        "Unexpected event inventory",
    )
    for run_id, run in runs.items():
        own = [e for e in events if e[1] == run_id]
        sequence = ["PLANNED", "RUNNING", "INNER_COMPLETE"]
        if run["state"] == "SELECTED":
            sequence.append("SELECTED")
        require([e[3] for e in own] == sequence, f"Unexpected lifecycle: {run_id}")
        require(json.loads(own[0][4]) == run["identity"], "Planned identity differs")
        merged: dict[str, Any] = {}
        for e in own[1:]:
            merged.update(json.loads(e[4]) or {})
        require(merged == run["data"], "Event history/ledger data differ")
    require(
        {p.name for p in base.iterdir()}
        == {"ledger.sqlite3", "events.csv", "locks", "runs", "reports", "selection"},
        "Unexpected final/outer/partial production evidence",
    )
    require(
        {p.name for p in (base / "selection").iterdir()}
        == {f"fold{f}_{s}.json" for f in (1, 2, 3) for s in ("c2", "ablations")},
        "Selection file inventory differs",
    )
    require(
        {p.name for p in (base / "reports").iterdir()}
        == {f"fold{f}_phase5b.json" for f in (1, 2, 3)},
        "Unexpected report inventory",
    )
    bundles, final_plan = [], []
    for fold in (1, 2, 3):
        panel = panel_report(context, fold)
        require(panel == read_json(base / f"reports/fold{fold}_phase5b.json"), "C2 report changed")
        bundle = read_json(base / f"selection/fold{fold}_ablations.json")
        validate_bundle(context, fold, bundle, runs, panel["selection"])
        bundles.append(bundle)
        split = context.split(fold)
        development = split.records(context.predecessor.source.contract, "development")
        require(len(split.speakers("development")) == 10, "Ten development speakers required")
        final_plan.append(
            {
                "outer_fold": fold,
                "common_candidate": COMMON[fold],
                "development_speakers": list(split.speakers("development")),
                "development_record_count": len(development),
                "development_record_hash": record_hash(development),
                "development_only_pos_weight": pos_weight(
                    context, split, "development", smoke=False
                ),
                "epochs": {a: bundle["selections"][a]["epoch"] for a in ABLATIONS},
            }
        )
    signed = torch.load(
        ROOT / "artifacts/checkpoints/module4/fold1_seed42_6fc69cf0ad55/best.pt",
        map_location="cpu",
        weights_only=True,
    )["model_state"]
    prefix = "encoders.mel_encoder.backbone."
    backbone = {k.removeprefix(prefix): v for k, v in signed.items() if k.startswith(prefix)}
    require(len(backbone) == 120, "Signed backbone inventory")
    with patch.object(
        AblationModel, "forward", side_effect=AssertionError("Audit forbids forward")
    ):
        checkpoints = [audit_inner(context, i, r, backbone) for i, r in sorted(runs.items())]
    transcript_evidence = None
    if transcript:
        text = transcript.read_text(encoding="utf-8-sig")
        pasted, _ = json.JSONDecoder().raw_decode(text[text.index("{") :])
        require(
            {r["id"]: {k: v for k, v in r.items() if k != "id"} for r in pasted["runs"]} == runs,
            "Pasted ledger differs from retained ledger",
        )
        transcript_evidence = {"path": str(transcript), **fingerprint(transcript), "matches": True}
    require_frozen(context)
    require(inventory(base) == before, "Production evidence changed during audit")
    require(
        fingerprint(config.output("protocol.json")) == protocol_before, "Frozen protocol changed"
    )
    return {
        "status": "PASS",
        "phase5c": "COMPLETE / PASSED",
        "next_phase": "FINAL RETRAINING READY FOR MANUAL EXECUTION",
        "protocol_hash": PROTOCOL,
        "frozen_code_hash": CODE,
        "torch": str(torch.__version__),
        "production_runs": 30,
        "states": dict(states),
        "events": len(events),
        "c2_reused": 3,
        "new_matched_inner_runs": 18,
        "phase5b_files_unchanged": len(unchanged),
        "selected_results": bundles,
        "checkpoint_validation": checkpoints,
        "descriptive_inner_results": summaries(bundles),
        "final_retraining_plan": final_plan,
        "protected_files_unchanged": len(protected),
        "protected_scope": "Fresh SHA-256/size/mtime checks on 247 predecessor files; "
        "prior accepted raw/A0/Module 1/2 full-tree audits not rerun.",
        "failed_invalidated_runs": 0,
        "final_training_count": 0,
        "outer_claims": 0,
        "outer_events": 0,
        "outer_results": 0,
        "outer_model_forward_count": 0,
        "outer_forward_evidence": "Frozen guards and exact inner role/record histories; "
        "no outer claims/events/results. Derived evidence, not an independent runtime counter.",
        "audit_model_forwards": 0,
        "production_evidence_before_and_after": before,
        "frozen_protocol_fingerprint": protocol_before,
        "pasted_ledger": transcript_evidence,
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
        / "phase5c_audit"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
        / "phase5c_audit.json"
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
