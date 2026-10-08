"""External final reports from verified retained predictions; no scientific execution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import uuid
from collections import Counter
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree

sys.dont_write_bytecode = True

from vaaniswitch.m5_experiments.aggregation import METRICS, aggregate  # noqa: E402
from vaaniswitch.m5_experiments.robustness import robustness  # noqa: E402

if __package__:
    from . import module5_outer_remaining_sequence as s
else:
    import module5_outer_remaining_sequence as s

r, p, c, require = s.r, s.p, s.c, s.require
ROOT = s.ROOT
OUTPUT = f"{r.BASE}/final_closure"
FINAL_SHA = "acdafedcab014c98f61e747558daadb0805ce32c233a1c342fbe093972be98a7"
SELF = "scripts/module5_final_closure.py"
TEST = "tests/m5_experiments/test_final_closure.py"
EXCEPTION = "RECOVERED_PROTOCOL_EXCEPTION"
SOURCES = {
    "aggregation": "src/vaaniswitch/m5_experiments/aggregation.py",
    "robustness": "src/vaaniswitch/m5_experiments/robustness.py",
    "protocol_identity": "src/vaaniswitch/m5_experiments/config.py",
    "robustness_tests": "tests/m5_experiments/test_workflow_outer_robustness.py",
}
LIMITATIONS = [
    p.DISCLOSURE,
    "Only 15 speakers and three folds; descriptive variability is not significance.",
    "Module4 development exposed Fold2/3 speakers; these are not pristine external validation.",
    "One seed (42), fixed threshold 0.5, no calibration or threshold optimization.",
    "Common C2-selected settings control ablations; simpler models were not separately tuned.",
    "Capacity differs across ablations; nonlinear complementarity is descriptive, not causal.",
    "Small and single-class subgroups; null AUC retained with its reason.",
    "RMS normalization may suppress amplitude cues; 235 CORE Mel center crops.",
    "NumPy VMD lacks independent-package cross-validation; 48 valid records hit its ceiling.",
    "No clinical diagnostic or state-of-the-art claim; no guaranteed accuracy.",
    "Full raw/cache corpora not rescanned; signed prior audits and 247 files checked.",
]


def require_complete(status: dict, chain: list) -> None:
    require(
        status["status"] == "PASS"
        and status["sequence_state"] == "COMPLETE"
        and status["verified_chain_completions"] == len(chain) == 8
        and status["next_target"] is None
        and status["remaining_order"] == []
        and status["previous_completion_sha256"] == chain[-1]["sha256"] == FINAL_SHA
        and status["model_forwards"] == status["native_writes"] == 0,
        "Complete pinned eight-entry chain required",
    )


def validate_lifecycle(state: dict) -> dict:
    runs, events = state["native_runs_snapshot"], state["native_events_snapshot"]
    counts = Counter(v["state"] for v in runs.values())
    require(
        counts == {"OUTER_EVALUATED": 20, "FINAL_TRAIN_COMPLETE": 1, "INNER_COMPLETE": 9},
        "Unexpected final ledger states",
    )
    claims = [v["claim"] for v in runs.values() if v["claim"] is not None]
    require(len(claims) == len(set(claims)) == 21, "Claim tokens must be unique")
    for key, run in runs.items():
        own = [v[3] for v in events if v[1] == key and "OUTER" in v[3]]
        expected = (
            ["OUTER_CLAIMED"]
            if key == r.RUN
            else ["OUTER_CLAIMED", "OUTER_EVALUATED"]
            if run["state"] == "OUTER_EVALUATED"
            else []
        )
        require(own == expected, f"Unexpected outer lifecycle: {key}")
        if key == r.RUN:
            require(
                run["claim"] == r.CLAIM and run["state"] == "FINAL_TRAIN_COMPLETE",
                "Original exceptional claim/state changed",
            )
        elif run["state"] == "INNER_COMPLETE":
            require(run["claim"] is None, "Unselected candidate has a claim")
    require(all(e[1] in runs for e in events), "Event references unknown run")
    require(len(events) == 173, "Unexpected native event count")
    require(len(state["native_files_snapshot"]) == 168, "Unexpected native file count")
    return {
        "run_count": len(runs),
        "states": dict(counts),
        "unique_native_claims": len(claims),
        "outer_claim_events": 21,
        "outer_completion_events": 20,
        "expected_unfinished_claim": r.RUN,
        "unexpected_unfinished_claims": 0,
        "native_file_count": 168,
        "native_event_count": 173,
    }


def reporting_view(cell: dict, result: dict, run: dict) -> dict:
    """Adapt the exceptional schema in memory; never create a native A1 result."""
    require(result["metrics"] == cell["metrics"], "Reporting metrics differ from audit")
    require(
        result["experiment_id"] == cell["experiment_id"]
        and result["outer_fold"] == cell["outer_fold"]
        and result["ablation"] == cell["ablation"]
        and result["threshold"] == 0.5,
        "Reporting identity differs",
    )
    if cell["provenance"] == EXCEPTION:
        require(
            cell["experiment_id"] == r.RUN
            and result["provenance_flag"] == EXCEPTION
            and result["original_prior_forward_count"] == "UNKNOWN",
            "Exception disclosure missing",
        )
    else:
        require(cell["provenance"] == "NATIVE", "Unknown result provenance")
        for name in ("parameter_counts", "inner_training_seconds", "final_training_seconds"):
            require(result[name] == run["data"][name], f"Native {name} differs")
    return {
        **deepcopy(cell),
        "selection": deepcopy(run["data"]["selection"]),
        "parameter_counts": deepcopy(run["data"]["parameter_counts"]),
        "inner_training_seconds": run["data"]["inner_training_seconds"],
        "final_training_seconds": run["data"]["final_training_seconds"],
        "predictions": deepcopy(result["predictions"]),
    }


def audit(root: Path) -> tuple[dict, list, object]:
    status = s.inspect("status", root=root)
    chain = s.read_chain(root)
    require_complete(status, chain)
    acceptance, context, records1 = p.prepare(root)
    before = p.snapshot(root)
    lifecycle = validate_lifecycle(before)
    fold1 = p.fold1_cells(root, acceptance, context, records1, before)
    require(
        r.file_hash(root / p.REPORT) == s.g.CLOSURE_SHA
        and fold1 == r.read_json(root / p.REPORT)["cells"],
        "Pinned Fold1 closure changed",
    )
    anchor = s.load_anchor(root)
    records = {1: records1}
    for fold in (2, 3):
        records[fold] = context.split(fold).records(
            context.predecessor.source.contract, "outer_test"
        )
    cells = fold1 + s.verify_prefix(root, anchor, before, records[2])
    for spec in anchor["remaining_sequence"]:
        key = spec["experiment_id"]
        saved = s.checkpoint(root, before["native_runs_snapshot"][key], spec)
        cells.append(s.verify_cell(root, before, spec, records[spec["outer_fold"]], saved))
    require(len(cells) == 21, "All 21 cells required")
    views = []
    for cell in cells:
        result = r.read_json(root / cell["result_path"])
        require(r.file_hash(root / cell["result_path"]) == cell["result_sha256"], "Result drift")
        c.native.validate_predictions(result["predictions"], records[cell["outer_fold"]])
        require(
            c.native.binary_metrics(
                [v["label"] for v in result["predictions"]],
                [v["probability"] for v in result["predictions"]],
            )
            == cell["metrics"],
            "Prediction metric recomputation differs",
        )
        cell["speakers"] = sorted({v.speaker_id for v in records[cell["outer_fold"]]})
        cell["claim"] = before["native_runs_snapshot"][cell["experiment_id"]]["claim"]
        views.append(
            reporting_view(cell, result, before["native_runs_snapshot"][cell["experiment_id"]])
        )
    p.assert_preserved(root, before, acceptance, context)
    require(s.read_chain(root) == chain, "Chain changed during audit")
    protected = r.read_json(root / "artifacts/module5/implementation/protected_inputs.json")
    evidence = {
        f"{r.BASE}/experiments/{name}": value
        for name, value in before["native_files_snapshot"].items()
    }
    evidence.update(
        {
            f"{r.RECOVERY}/{name}": value
            for name, value in acceptance["recovery_files_snapshot"].items()
        }
    )
    evidence.update(protected["protected_files"])
    evidence.update(anchor["earlier_completion_evidence"])
    paths = [
        p.REPORT,
        s.ANCHOR,
        c.FINAL_AUDIT,
        f"{r.BASE}/protocol.json",
        *anchor["protected_helper_sha256"],
        s.SELF,
        *SOURCES.values(),
        *(entry["path"] for entry in chain),
    ]
    evidence.update({name: r.fingerprint(root / name) for name in paths})
    report = {
        "schema": "module5_final_outer_audit_v1",
        "status": "PASS",
        "protocol_hash": r.PROTOCOL,
        "frozen_code_hash": r.CODE,
        "runtime": s.RUNTIME,
        "sequence_status": status,
        "completion_chain": [{k: e[k] for k in ("path", "sha256")} for e in chain],
        "cells": cells,
        "native_cells": 20,
        "recovered_protocol_exception_cells": 1,
        "exception_disclosure": p.DISCLOSURE,
        "model_forwards_during_closure": 0,
        "training_during_closure": 0,
        "native_scientific_writes_during_closure": 0,
        "protected_predecessor_files_verified": len(protected["protected_files"]),
        "recovery_files_verified": len(acceptance["recovery_files_snapshot"]),
        "raw_cache_audit_scope": LIMITATIONS[-1],
        "lifecycle": lifecycle,
        "evidence_files": evidence,
    }
    return report, views, context


def reports(context, views: list) -> tuple[dict, dict]:
    keys = {(v["outer_fold"], v["ablation"]) for v in views}
    require(
        len(views) == len(keys) == 21 and keys == {(f, a) for f in (1, 2, 3) for a in s.g.ORDER},
        "Exactly 21 outer cells required",
    )
    exceptions = [v for v in views if v["provenance"] != "NATIVE"]
    require(
        len(exceptions) == 1
        and exceptions[0]["experiment_id"] == r.RUN
        and exceptions[0]["provenance"] == EXCEPTION,
        "Exactly the authorized Fold1/A1 exception required",
    )
    before = deepcopy(views)
    cross = aggregate(views)
    cross["exception_disclosure"] = p.DISCLOSURE
    cross["ablations"]["A1"]["provenance_note"] = f"Fold1/A1 = {EXCEPTION}"
    cross["sources"] = {k: SOURCES[k] for k in ("aggregation", "protocol_identity")}
    robust = robustness(context, views)
    robust.update(
        {
            "status": "COMPLETE / PASSED",
            "sources": {k: SOURCES[k] for k in ("robustness", "robustness_tests")},
            "documentation_section": (
                "docs/modules/5_accuracy_robustness/MODULE.md#reporting-and-limitations"
            ),
            "dimensions": [
                "gender",
                "primary_microphone",
                "prompt_type",
                "duration",
                "mel_handling",
                "speaker",
            ],
            "threshold": 0.5,
            "new_inference": False,
            "retuning": False,
        }
    )
    require(views == before, "Frozen reporting mutated retained prediction views")
    return cross, robust


def csv_bytes(rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def tables(cross: dict) -> tuple[bytes, bytes]:
    rows, summaries = [], []
    for ablation, item in cross["ablations"].items():
        for fold in item["folds"]:
            rows.append(
                {
                    "ablation": ablation,
                    "outer_fold": fold["outer_fold"],
                    "provenance": fold["provenance"],
                    "experiment_id": fold["experiment_id"],
                    "candidate": fold["candidate"],
                    "selected_epoch": fold["selected_epoch"],
                    "record_count": fold["record_count"],
                    "result_sha256": fold["result_sha256"],
                    "checkpoint_sha256": fold["checkpoint_sha256"],
                    "trainable_parameters": fold["parameter_counts"]["trainable"],
                    "inner_training_seconds": fold["inner_training_seconds"],
                    "final_training_seconds": fold["final_training_seconds"],
                    **{k: fold["metrics"][k] for k in (*METRICS, "TN", "FP", "FN", "TP")},
                }
            )
        note = item.get("provenance_note", "Three NATIVE folds")
        for metric, values in item["summary"].items():
            summaries.append({"ablation": ablation, "metric": metric, "provenance": note, **values})
    return csv_bytes(rows), csv_bytes(summaries)


def verify_evidence(root: Path, evidence: dict) -> None:
    for name, expected in evidence.items():
        require(
            r.fingerprint(c.canonical_path(root, name)) == expected,
            f"Evidence changed: {name}",
        )


def checks(root: Path, path: Path) -> dict:
    value = r.read_json(path)
    require(value["status"] == "PASS", "Closure checks must pass before sign-off")
    require(value["helper_sha256"] == r.file_hash(root / SELF), "Tested helper changed")
    require(value["test_sha256"] == r.file_hash(root / TEST), "Closure tests changed")
    for name in ("focused", "module5", "ruff", "help"):
        item = value["checks"][name]
        require(item["exit_code"] == 0, f"Failed check: {name}")
        for relative, expected in item["files"].items():
            require(
                r.fingerprint(root / relative) == expected, f"Check evidence changed: {relative}"
            )
        if name in {"focused", "module5"}:
            suites = ElementTree.parse(root / item["junit"]).getroot()
            totals = Counter()
            for suite in suites.iter("testsuite"):
                for field in ("tests", "failures", "errors", "skipped"):
                    totals[field] += int(suite.get(field, 0))
            require(
                totals["tests"] > 0
                and not any(totals[k] for k in ("failures", "errors", "skipped")),
                f"Incomplete or failed {name} suite",
            )
    return value


def assemble(root: Path, report: dict, cross: dict, robust: dict, tested: dict) -> dict:
    require(report["status"] == "PASS", "Audit must pass")
    require(robust["status"] == "COMPLETE / PASSED", "Frozen robustness incomplete")
    fold_csv, summary_csv = tables(cross)
    files = {
        "final_outer_audit.json": s.canonical_bytes(report),
        "cross_fold_metrics.json": s.canonical_bytes(cross),
        "cross_fold_metrics.csv": fold_csv,
        "ablation_summary.csv": summary_csv,
        "robustness_summary.json": s.canonical_bytes(robust),
        "validation_checks.json": s.canonical_bytes(tested),
    }
    primary = []
    for cell in report["cells"]:
        if cell["ablation"] == "C2":
            primary.append(
                {
                    k: cell[k]
                    for k in (
                        "outer_fold",
                        "experiment_id",
                        "candidate",
                        "selected_epoch",
                        "checkpoint_sha256",
                        "selection_sha256",
                    )
                }
            )
            primary[-1]["checkpoint_path"] = (
                f"{r.BASE}/experiments/runs/{cell['experiment_id']}/final/"
                f"{cell['checkpoint_sha256']}.pt"
            )
    closure = {
        "schema": "module5_final_closure_v1",
        "status": "COMPLETE / PASSED / SIGNED OFF",
        "signoff_scope": "Exception-aware completion; not uninterrupted native compliance",
        "protocol_hash": r.PROTOCOL,
        "frozen_code_hash": r.CODE,
        "runtime": s.RUNTIME,
        "acceptance": {
            "all_21_cells_accounted_for": True,
            "exception_provenance_preserved": True,
            "retained_metrics_reproducible": True,
            "frozen_hashes_intact": True,
            "completion_chain_intact": True,
            "frozen_aggregation_complete": True,
            "frozen_robustness_complete": True,
            "integrity_blockers": [],
            "no_scientific_rerun_during_closure": True,
            "focused_and_module5_tests_pass": True,
        },
        "artifacts": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        "limitations": LIMITATIONS,
        "primary_model_policy": (
            "Retain the three pre-outer C2 development-trained checkpoints at their frozen "
            "inner-selected epochs. No single deployment checkpoint or ensemble is selected "
            "from outer performance; Module6 ASR/TTS does not require a classifier refit."
        ),
        "primary_checkpoints": sorted(primary, key=lambda v: v["outer_fold"]),
        "module6": "NOT STARTED; handoff prepared only",
        "model_forwards_during_closure": 0,
        "native_scientific_writes_during_closure": 0,
    }
    files["module5_final_closure.json"] = s.canonical_bytes(closure)
    handoff = {
        "status": "PREPARED ONLY / MODULE 6 NOT STARTED",
        "protocol_hash": r.PROTOCOL,
        "frozen_code_hash": r.CODE,
        "module5_closure_sha256": hashlib.sha256(files["module5_final_closure.json"]).hexdigest(),
        "cross_fold_metrics_sha256": closure["artifacts"]["cross_fold_metrics.json"],
        "primary_model_policy": closure["primary_model_policy"],
        "primary_checkpoints": closure["primary_checkpoints"],
        "limitations": LIMITATIONS,
        "objective": (
            "Frozen English ASR -> preserved raw transcript -> auxiliary WER/CER against "
            "authorized references -> explicit optional correction -> standard English TTS. "
            "No voice cloning or ASR research training."
        ),
        "sources": [
            "PROJECT_SCOPE_LOCK.md",
            "FINAL_WORKFLOW.md",
            "docs/modules/6_english_asr_tts/MODULE.md",
        ],
        "next_step": (
            "In a separately authorized Module6 session, read its AGENTS/MODULE/TEST_PLAN; "
            "implement and test separate frozen ASR and standard TTS wrappers. "
            "Module5 artifacts are immutable inputs. No Module6 execution is authorized here."
        ),
    }
    files["module6_handoff.json"] = s.canonical_bytes(handoff)
    files["evidence_manifest.json"] = s.canonical_bytes(
        {
            "schema": "module5_final_evidence_manifest_v1",
            "files": {
                name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                for name, data in files.items()
            },
            "reporting_helper": {
                SELF: r.fingerprint(root / SELF),
                TEST: r.fingerprint(root / TEST),
            },
            "input_evidence": report["evidence_files"],
            "source_policy": "Frozen functions called directly on verified reporting views",
            "exception_disclosure": p.DISCLOSURE,
        }
    )
    return files


def verify_package(directory: Path) -> dict:
    manifest = r.read_json(directory / "evidence_manifest.json")
    require(
        {p.name for p in directory.iterdir()}
        == set(manifest["files"]) | {"evidence_manifest.json"},
        "Unexpected/missing package files",
    )
    for name, expected in manifest["files"].items():
        require(Path(name).name == name, "Invalid package filename")
        path = directory / name
        require(path.is_file() and not path.is_symlink(), "Invalid package file")
        require(
            {"sha256": r.file_hash(path), "bytes": path.stat().st_size} == expected,
            f"Package checksum differs: {name}",
        )
    return manifest


def publish(root: Path, files: dict) -> Path:
    directory = c.canonical_path(root, OUTPUT)
    require(not directory.exists(), "Final closure exists; never replace retained reports")
    stage = directory.with_name(f".final_closure_pending_{uuid.uuid4().hex}")
    stage.mkdir()
    # A failure preserves its staging evidence; never deletes or replaces retained reports.
    for name, data in files.items():
        require(Path(name).name == name, "Invalid package filename")
        with (stage / name).open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    manifest = verify_package(stage)
    verify_evidence(root, manifest["input_evidence"])
    require(not directory.exists(), "Concurrent closure publication; stop")
    os.rename(stage, directory)
    return directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--checks", type=Path)
    parser.add_argument("--verify-package", action="store_true")
    args = parser.parse_args(argv)
    try:
        require(not (args.save and args.verify_package), "Choose save or verify")
        with s.forbid_scientific_execution():
            if args.verify_package:
                value = verify_package(ROOT / OUTPUT)
                verify_evidence(ROOT, value["input_evidence"])
                verify_evidence(ROOT, value["reporting_helper"])
                print(
                    json.dumps(
                        {
                            "status": "PASS",
                            "package": OUTPUT,
                            "manifest_sha256": r.file_hash(
                                ROOT / OUTPUT / "evidence_manifest.json"
                            ),
                        }
                    )
                )
                return 0
            report, views, context = audit(ROOT)
            cross, robust = reports(context, views)
            verify_evidence(ROOT, report["evidence_files"])
            output = {
                "status": "PASS",
                "native_cells": 20,
                "exceptional_cells": 1,
                "aggregation": cross["aggregation"],
                "robustness_groups": len(robust["groups"]),
                "robustness_records": robust["record_count"],
                "model_forwards": 0,
                "native_scientific_writes": 0,
                "summary": {a: v["summary"] for a, v in cross["ablations"].items()},
            }
            if args.save:
                require(args.checks is not None, "Test evidence required for sign-off")
                files = assemble(ROOT, report, cross, robust, checks(ROOT, args.checks))
                directory = publish(ROOT, files)
                output.update(
                    module5="COMPLETE / PASSED / SIGNED OFF",
                    module6="NOT STARTED",
                    artifacts={name: r.fingerprint(directory / name) for name in files},
                )
            print(json.dumps(output, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError) as exc:
        print(json.dumps({"status": "FAIL", "action": "STOP_FOR_REVIEW", "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
