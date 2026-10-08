"""Read-only continuation gate for one pinned Fold 1/A1 recovery exception.

This tool never evaluates a model, claims a run, writes a ledger, or aggregates.
Native inference remains the separate, unchanged vaaniswitch-m5 outer command.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import torch  # noqa: E402

from vaaniswitch.m5_experiments.checkpoint import validate_progress  # noqa: E402
from vaaniswitch.m5_experiments.config import ExperimentConfig, canonical_path  # noqa: E402
from vaaniswitch.m5_experiments.source import prepare_experiments, require_frozen  # noqa: E402

if __package__:
    from . import module5_outer_evidence as native
    from . import module5_recover_outer_exception as recovery
else:
    import module5_outer_evidence as native
    import module5_recover_outer_exception as recovery

ROOT = Path(__file__).resolve().parents[1]
ORDER = ("B1", "B2", "B3", "C1", "C2")
ATTEMPT = "a1_20261005T110007Z_af06fd40be7c"
ACCEPTANCE = f"{recovery.BASE}/outer_continuation/fold1/acceptance.json"
ACCEPTANCE_SHA = "25ff8e11510e760b8a7844d04542f485fcd4b26e5caec97923d2df84130bb120"
RESULT_SHA = "86279a525cec7960fb1e395db86364571c990229d10ba4a491b91bf7511c1659"
MANIFEST_SHA = "03a51736746620d07319364ec5ffbf321f10d1a72caf793345948b7fdcfc1ba6"
FINAL_AUDIT = (
    f"{recovery.BASE}/final_retraining_audit/20260928T195038869875Z/final_retraining_audit.json"
)
require = recovery.require


def run_id(ablation: str) -> str:
    return f"fold1_{ablation}_stronger_dropout_seed42_{recovery.PROTOCOL[:12]}"


def request_scope(mode: str, fold: int, target: str) -> tuple[str, ...]:
    require(mode in {"preflight", "verify"}, "Read-only preflight/verify modes only")
    require(type(fold) is int and fold == 1, "Only Fold 1 is authorized")
    require(target in ORDER, "Only B1/B2/B3/C1/C2; A0/A1 must never be rerun")
    count = ORDER.index(target) + (mode == "verify")
    return ORDER[:count]


def load_acceptance(root: Path) -> dict[str, Any]:
    path = canonical_path(root, ACCEPTANCE)
    require(recovery.file_hash(path) == ACCEPTANCE_SHA, "Continuation acceptance changed")
    value = recovery.read_json(path)
    proof = value["recovery_verifier"]
    require(
        value["schema"] == "module5_exact_a1_exception_acceptance_v1"
        and value["protocol_hash"] == recovery.PROTOCOL
        and value["frozen_code_hash"] == recovery.CODE
        and value["future_order"] == list(ORDER)
        and value["recovery_attempt_id"] == ATTEMPT
        and proof["status"] == "PASS"
        and proof["production_unchanged"] is True
        and proof["verifier_model_forwards"] == 0
        and proof["result_sha256"] == RESULT_SHA
        and proof["manifest_sha256"] == MANIFEST_SHA
        and proof["record_count"] == 2819
        and proof["record_hash"] == recovery.RECORD_HASH,
        "Invalid recovery verifier acceptance",
    )
    for relative, checksum in value["protected_script_hashes"].items():
        require(
            recovery.file_hash(canonical_path(root, relative)) == checksum,
            f"Protected helper changed: {relative}",
        )
    request = canonical_path(
        root, f"{recovery.BASE}/outer_continuation/fold1/authorization_request.txt"
    )
    require(recovery.file_hash(request) == value["request_sha256"], "Authorization changed")
    return value


def frozen_context(root: Path, acceptance: dict[str, Any]):
    for name, checksum in recovery.FORENSIC_HASHES.items():
        path = canonical_path(root, f"{recovery.FORENSICS}/{name}")
        require(recovery.file_hash(path) == checksum, f"Changed forensic evidence: {name}")
    forensic = recovery.read_json(root / recovery.FORENSICS / "forensic_audit.json")
    require(
        forensic["status"] == "FORENSIC_AUDIT_COMPLETE_RECOVERY_BLOCKED"
        and forensic["zero_a1_forwards_proven"] is False
        and acceptance["native_files_snapshot"] == forensic["production_evidence_before_and_after"],
        "Forensic state or production snapshot differs",
    )
    for relative, expected in forensic["frozen_source_fingerprints_before_and_after"].items():
        require(
            recovery.fingerprint(canonical_path(root, relative)) == expected,
            f"Frozen source changed: {relative}",
        )
    require(
        recovery.fingerprint(root / FINAL_AUDIT) == acceptance["final_retraining_audit"],
        "Final-retraining audit changed",
    )
    audit = recovery.read_json(root / FINAL_AUDIT)
    require(audit["status"] == "PASS", "Final-retraining audit not PASS")
    binding = acceptance["recovery_authorization_binding"]
    for name, item in binding["inputs"].items():
        # These two files legitimately advance only through the separately checked native prefix.
        if name in {"original_ledger", "original_events_csv"}:
            relative = "ledger.sqlite3" if name == "original_ledger" else "events.csv"
            require(
                item["path"] == f"{recovery.BASE}/experiments/{relative}"
                and item["sha256"] == acceptance["native_files_snapshot"][relative]["sha256"],
                "Original production hash binding differs",
            )
        else:
            require(
                recovery.file_hash(canonical_path(root, item["path"])) == item["sha256"],
                f"Immutable recovery input changed: {name}",
            )
    policy = recovery.read_json(root / recovery.RECOVERY / "authorization_policy.json")
    protected_path = root / "artifacts/module5/implementation/protected_inputs.json"
    require(
        recovery.file_hash(protected_path) == policy["protected_reference_sha256"],
        "Protected reference changed",
    )
    for relative, expected in recovery.read_json(protected_path)["protected_files"].items():
        require(
            recovery.fingerprint(canonical_path(root, relative)) == expected,
            f"Predecessor changed: {relative}",
        )
    config = ExperimentConfig.from_yaml(
        root / "configs/experiments/module5/protocol.yaml", repo_root=root
    )
    context = prepare_experiments(config)
    validate_frozen_runtime(context)
    records = context.split(1).records(context.predecessor.source.contract, "outer_test")
    recovery.validate_records(records)
    plan = next(p for p in audit["outer_evaluation_plan"] if p["outer_fold"] == 1)
    require(
        [r.logical_id for r in records] == plan["logical_ids"]
        and plan["record_hash"] == recovery.RECORD_HASH,
        "Audited record order differs",
    )
    return context, records


def validate_frozen_runtime(context: Any) -> None:
    require_frozen(context)
    require(
        context.config.config_hash == recovery.PROTOCOL and context.code_hash == recovery.CODE,
        "Frozen protocol/source drift",
    )
    require(
        str(torch.__version__) == "2.14.0+cpu" and context.config.cpu_threads == 2,
        "Frozen CPU runtime drift",
    )


def validate_native_progress(
    acceptance: dict[str, Any],
    runs: dict[str, Any],
    events: list[Any],
    files: dict[str, Any],
    completed: tuple[str, ...],
) -> None:
    """Allow only the exact completed prefix; every other run/file stays immutable."""
    baseline = acceptance["native_runs_snapshot"]
    original_events = acceptance["native_events_snapshot"]
    original_files = acceptance["native_files_snapshot"]
    require(set(runs) == set(baseline), "Production experiment set changed")
    unfinished = [
        key
        for key, run in runs.items()
        if run["claim"] is not None and run["state"] != "OUTER_EVALUATED"
    ]
    require(unfinished == [recovery.RUN], "A second/changed unfinished claim exists")
    completed_ids = [run_id(a) for a in completed]
    for key, original in baseline.items():
        current = runs[key]
        if key not in completed_ids:
            require(current == original, f"Unrequested/later run or A1/A0 changed: {key}")
        else:
            require(
                original["state"] == "FINAL_TRAIN_COMPLETE" and original["claim"] is None,
                "Original target was not unclaimed",
            )
            require(
                current["state"] == "OUTER_EVALUATED"
                and isinstance(current["claim"], str)
                and bool(re.fullmatch(r"[a-f0-9]{32}", current["claim"])),
                "Expected native completion/unique claim missing",
            )
            require(current["identity"] == original["identity"], "Native target identity changed")
            checksum = current["data"].get("outer_result_sha256")
            require(
                isinstance(checksum, str) and bool(re.fullmatch(r"[a-f0-9]{64}", checksum)),
                "Missing native result checksum",
            )
            require(
                current["data"] == {**original["data"], "outer_result_sha256": checksum},
                "Native target training/selection data changed",
            )
    rows = [list(e) for e in events]
    require(rows[: len(original_events)] == original_events, "Original ledger event prefix changed")
    extra = rows[len(original_events) :]
    require(len(extra) == 2 * len(completed_ids), "Unexpected pending/later/extra native events")
    previous_time = datetime.fromisoformat(original_events[-1][2])
    for index, event in enumerate(extra):
        expected_id = completed_ids[index // 2]
        expected_type = "OUTER_CLAIMED" if index % 2 == 0 else "OUTER_EVALUATED"
        require(
            event[0] == original_events[-1][0] + index + 1
            and event[1] == expected_id
            and event[3] == expected_type,
            "Native events violate sequential target order",
        )
        timestamp = datetime.fromisoformat(event[2])
        require(
            timestamp.utcoffset() is not None and timestamp >= previous_time,
            "Invalid native event time",
        )
        previous_time = timestamp
        detail = (
            {"claim": runs[expected_id]["claim"]}
            if index % 2 == 0
            else {"outer_result_sha256": runs[expected_id]["data"]["outer_result_sha256"]}
        )
        require(json.loads(event[4]) == detail, "Native claim/result event detail differs")
    tokens = [run["claim"] for run in runs.values() if run["claim"] is not None]
    require(len(tokens) == len(set(tokens)), "Duplicate native claim token")
    mutable = {"ledger.sqlite3", "events.csv"} if completed else set()
    for relative, expected in original_files.items():
        if relative not in mutable:
            require(files.get(relative) == expected, f"Immutable native file changed: {relative}")
    allowed_new = {f"runs/{key}/outer_result.json" for key in completed_ids}
    require(
        set(files) == set(original_files) | allowed_new,
        "Unexpected/partial/later native file, or missing evidence",
    )


def verify_csv(path: Path, events: list[Any], *, advanced: bool) -> None:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    expected = events if advanced else events[:-1]  # Original export ended before A1 claim 135.
    require(
        rows[0] == ["sequence", "experiment_id", "time", "event", "detail"]
        and rows[1:] == [[str(v) for v in row] for row in expected],
        "Native event CSV disagrees with its allowed ledger prefix",
    )


def exception_identity(result: dict[str, Any]) -> None:
    require(
        all(result.get(k) == v for k, v in recovery.TARGET.items())
        and result["attempt_id"] == ATTEMPT
        and result["provenance_flag"] == "RECOVERED_PROTOCOL_EXCEPTION"
        and result["classification"] == "PROTOCOL_EXCEPTION"
        and result["notice"] == recovery.NOTICE
        and result["recovery_forward_count"] == result["recovery_source_load_count"] == 2819
        and result["recovery_batch_count"] == 177,
        "Wrong exceptional recovery classification/identity/counts",
    )


def verify_exception(
    root: Path,
    acceptance: dict[str, Any],
    context: Any,
    records: tuple[Any, ...],
    runs: dict[str, Any],
    events: list[Any],
) -> None:
    directory = canonical_path(root, f"{recovery.RECOVERY}/{ATTEMPT}")
    require(
        recovery.file_hash(directory / "recovery_result.json") == RESULT_SHA,
        "Changed/missing A1 recovery result",
    )
    require(
        recovery.file_hash(directory / "recovery_manifest.json") == MANIFEST_SHA,
        "Changed/missing A1 recovery manifest",
    )
    require(
        recovery.inventory(root / recovery.RECOVERY) == acceptance["recovery_files_snapshot"],
        "An A1 recovery artifact changed",
    )
    recovery.validate_target(
        runs[recovery.RUN],
        events,
        (root / recovery.BASE / f"experiments/runs/{recovery.RUN}/outer_result.json").exists(),
    )
    prepared = recovery.Prepared(
        root,
        context,
        runs[recovery.RUN],
        records,
        acceptance["recovery_authorization_binding"],
        acceptance["native_files_snapshot"],
        recovery.read_json(root / recovery.RECOVERY / "authorization_policy.json"),
    )
    auth = recovery.authorization(prepared, directory)
    progress = recovery.inspect_journal(directory, records)
    require(
        (directory / "predictions.jsonl").read_bytes()
        == b"".join(recovery.encoded(row) for row in progress.rows),
        "Recovered prediction stream differs",
    )
    result = recovery.read_json(directory / "recovery_result.json")
    exception_identity(result)
    # Reuse the original pure verifier computation against the ORIGINAL production snapshot.
    # The original strict verifier is unchanged; native continuation deltas are checked separately.
    require(
        result == recovery.final_result(prepared, directory, auth, progress),
        "Recovered result/metrics/provenance differs",
    )
    manifest = recovery.read_json(directory / "recovery_manifest.json")
    require(
        manifest
        == {
            "status": recovery.TARGET["status"],
            "classification": "PROTOCOL_EXCEPTION",
            "attempt_id": ATTEMPT,
            "notice": recovery.NOTICE,
            "artifact_hashes": {**result["artifact_hashes"], "recovery_result.json": RESULT_SHA},
            "production_fingerprints_hash": recovery.digest(acceptance["native_files_snapshot"]),
        },
        "Recovery manifest/artifact hashes differ",
    )


def validate_final_checkpoint(root: Path, run: dict[str, Any]) -> dict[str, Any]:
    identity = run["identity"]
    bundle = recovery.read_json(root / recovery.BASE / "experiments/selection/fold1_ablations.json")
    selection = bundle["selections"][identity["ablation"]]
    require(
        selection == run["data"]["selection"]
        and selection["experiment_id"] == identity["experiment_id"],
        "Target selection differs",
    )
    directory = canonical_path(
        root, f"{recovery.BASE}/experiments/runs/{identity['experiment_id']}/final"
    )
    pointer = recovery.read_json(directory / "latest.json")
    require(
        pointer == run["data"]["final_checkpoint"]
        and bool(re.fullmatch(r"[a-f0-9]{64}", pointer["sha256"]))
        and pointer["filename"] == pointer["sha256"] + ".pt",
        "Final checkpoint pointer differs",
    )
    path = canonical_path(root, (directory / pointer["filename"]).relative_to(root).as_posix())
    require(recovery.file_hash(path) == pointer["sha256"], "Final checkpoint checksum differs")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    require(
        saved["schema"] == "module5_checkpoint_v1"
        and saved["identity"] == run["data"]["final_identity"],
        "Final checkpoint identity differs",
    )
    validate_progress(saved["progress"])
    require(
        saved["progress"]["epoch"] == selection["epoch"] == run["data"]["final_epoch"],
        "Final selected epoch differs",
    )
    return {
        "checkpoint_sha256": pointer["sha256"],
        "selected_epoch": selection["epoch"],
        "candidate": selection["candidate_id"],
        "selection_sha256": recovery.file_hash(
            root / recovery.BASE / "experiments/selection/fold1_ablations.json"
        ),
    }


def inspect(mode: str, fold: int, target: str, root: Path = ROOT) -> dict[str, Any]:
    completed = request_scope(mode, fold, target)
    root = root.resolve()
    acceptance = load_acceptance(root)
    context, records = frozen_context(root, acceptance)
    ex = canonical_path(root, f"{recovery.BASE}/experiments")
    before = recovery.inventory(ex)
    runs, events = recovery.read_production(ex / "ledger.sqlite3")
    validate_native_progress(acceptance, runs, events, before, completed)
    verify_csv(ex / "events.csv", events, advanced=bool(completed))
    verify_exception(root, acceptance, context, records, runs, events)
    # A0 and every preceding native result must independently verify before the next target.
    verified = {}
    for ablation in ("A0", *completed):
        key = run_id(ablation)
        verified[ablation] = native.completed_result(
            ex,
            key,
            runs[key],
            [e for e in events if e[1] == key],
            records,
        )
    key = run_id(target)
    target_run = runs[key]
    own = [e for e in events if e[1] == key]
    if mode == "preflight":
        native.validate_unclaimed(target_run, own, (ex / f"runs/{key}/outer_result.json").exists())
    checkpoint = validate_final_checkpoint(root, target_run)
    require_frozen(context)
    require(recovery.inventory(ex) == before, "Production changed during read-only inspection")
    require(
        recovery.inventory(root / recovery.RECOVERY) == acceptance["recovery_files_snapshot"],
        "A1 recovery changed during inspection",
    )
    require(
        recovery.read_production(ex / "ledger.sqlite3") == (runs, events),
        "Ledger changed during inspection",
    )
    result = {
        "status": "PASS",
        "mode": mode,
        "outer_fold": fold,
        "ablation": target,
        "experiment_id": key,
        "target_provenance": "NATIVE",
        "target_state": target_run["state"],
        "target_claim": target_run["claim"],
        "protocol_hash": recovery.PROTOCOL,
        "frozen_code_hash": recovery.CODE,
        "record_count": 2819,
        "record_hash": recovery.RECORD_HASH,
        "threshold": 0.5,
        **checkpoint,
        "model_forwards": 0,
        "production_writes": 0,
        "production_unchanged_during_check": True,
        "aggregation_performed": False,
        "verified_native_prefix": ["A0", *completed],
        "a1_native_claim_unchanged": True,
        "a1_native_result_absent": True,
        "a1_provenance": "RECOVERED_PROTOCOL_EXCEPTION",
        "a1_original_prior_forward_count": "UNKNOWN",
        "a1_recovery_result_sha256": RESULT_SHA,
        "a1_recovery_manifest_sha256": MANIFEST_SHA,
        "no_later_target_ran": True,
        "fold2_fold3_unchanged": True,
        "stop_after_target_verification": True,
    }
    if mode == "verify":
        result["result_sha256"] = verified[target]["result_sha256"]
        result["metrics"] = verified[target]["metrics"]  # Single target, no ranking/aggregation.
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "verify"))
    parser.add_argument("--outer-fold", type=int, choices=(1,), required=True)
    parser.add_argument("--ablation", choices=ORDER, required=True)
    args = parser.parse_args(argv)
    try:
        print(
            json.dumps(
                inspect(args.mode, args.outer_fold, args.ablation), indent=2, allow_nan=False
            )
        )
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, IndexError) as exc:
        print(json.dumps({"status": "FAIL", "action": "STOP_FOR_REVIEW", "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
