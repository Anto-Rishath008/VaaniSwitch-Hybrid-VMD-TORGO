"""Read-only Phase 5A cache/predecessor audit; never regenerate Mel or run a model."""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from audit_module5_protected_inputs import collect

from vaaniswitch.m5_experiments.config import SIGNED_HASHES, ExperimentConfig, digest, file_hash
from vaaniswitch.m5_experiments.no_vmd import NoVMDCache
from vaaniswitch.m5_experiments.reporting import read_json, write_json
from vaaniswitch.m5_experiments.source import prepare_experiments, require_frozen

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_HASH = "677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9"
CODE_HASH = "7696cad010914ec37f494d9f1a11a2e1761e46f353e61d97a07399f746a34ff8"
A0_HASH = "9b57afd31f1b5d7d7c03c4d82828f3479bf0e115a0a35d3a262c0d190337f8d4"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def inspect_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"runs": 0, "states": {}, "outer_claims": 0, "outer_events": 0}
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        rows = list(connection.execute("SELECT id,state,claim FROM runs"))
        events = list(connection.execute("SELECT event FROM events"))
    finally:
        connection.close()
    states = {state: sum(row[1] == state for row in rows) for state in {r[1] for r in rows}}
    return {
        "runs": len(rows),
        "states": states,
        "outer_claims": sum(row[2] is not None for row in rows),
        "outer_events": sum("OUTER" in event[0] for event in events),
    }


def main() -> int:
    start = time.perf_counter()
    config = ExperimentConfig.from_yaml(
        ROOT / "configs/experiments/module5/protocol.yaml", repo_root=ROOT
    )
    context = prepare_experiments(config)
    require_frozen(context)
    protocol_path = config.output("protocol.json")
    protocol = read_json(protocol_path)
    require(
        config.config_hash
        == protocol["config_hash"]
        == digest(protocol["protocol"])
        == PROTOCOL_HASH,
        "Protocol hash mismatch",
    )
    require(context.code_hash == protocol["code_hash"] == CODE_HASH, "Frozen code drift")
    require(protocol["protocol"]["signed_hashes"] == SIGNED_HASHES, "Predecessor hash mismatch")
    cache = NoVMDCache(context)
    require(cache.config_hash == A0_HASH, "A0 hash mismatch")
    namespace = ROOT / "data/processed/module5_ablation/no_vmd_mel" / A0_HASH
    records = sorted(
        context.predecessor.source.contract.records, key=lambda record: record.logical_id
    )
    require(len(records) == 8673, "Expected 8673 signed CORE records")
    retained = {}
    for stage, expected in (
        (
            "build",
            {"selected": 8673, "computed": 8667, "cache_hit": 6, "validated": 8673, "failed": 0},
        ),
        (
            "validate",
            {"selected": 8673, "computed": 0, "cache_hit": 0, "validated": 8673, "failed": 0},
        ),
    ):
        path = config.output(f"a0/full_{stage}.json")
        report = read_json(path)
        require(all(report[k] == v for k, v in expected.items()), f"{stage} counts mismatch")
        require(
            report["status"] == "PASS"
            and report["scope"] == f"a0_full_{stage}"
            and report["outer_model_forwards"] == 0
            and not report["failures"],
            f"{stage} failed",
        )
        require(
            report["a0_config_hash"] == A0_HASH
            and report["experiment_config_hash"] == PROTOCOL_HASH
            and Path(report["namespace"]).resolve() == namespace.resolve(),
            f"{stage} identity mismatch",
        )
        failures_path = config.output(f"a0/full_{stage}_failures.csv")
        with failures_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            require(
                reader.fieldnames == ["logical_id", "error"] and not list(reader),
                "Production failure CSV",
            )
        retained[stage] = {
            "report": report,
            "report_sha256": file_hash(path),
            "failure_csv_sha256": file_hash(failures_path),
            "failure_rows": 0,
        }

    expected_files = {
        f"{r.logical_id}/{name}" for r in records for name in ("mel.npy", "metadata.json")
    }
    actual_files, tree_before = set(), {}
    for directory, directories, filenames in os.walk(namespace, followlinks=False):
        parent = Path(directory)
        for name in directories:
            path = parent / name
            require(
                path.resolve() == path and not name.startswith("."),
                f"Partial/redirected directory: {path}",
            )
        for name in filenames:
            path = parent / name
            require(path.resolve() == path, f"Redirected artifact: {path}")
            relative = path.relative_to(namespace).as_posix()
            actual_files.add(relative)
            stat = path.stat()
            tree_before[relative] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            require(stat.st_size > 0, f"Zero-byte artifact: {relative}")
    require(actual_files == expected_files, "Unexpected/missing/partial A0 artifacts")
    print(
        "A0 inventory matches 8,673 logical entries; checking arrays and signed primaries.",
        flush=True,
    )
    bindings = []
    minimum, maximum = float("inf"), float("-inf")
    for index, record in enumerate(records, 1):
        directory = cache.directory(record.logical_id)
        require(directory == namespace / record.logical_id, "A0 namespace mismatch")
        values = cache.load(record, check_source=True)
        metadata_path = directory / "metadata.json"
        metadata = read_json(metadata_path)
        require(
            metadata["binding"] == cache.binding(record), "Exact signed-record metadata mismatch"
        )
        require(
            values.shape == (224, 224) and values.dtype == np.float32 and np.isfinite(values).all(),
            "Mel shape/dtype/finiteness",
        )
        minimum, maximum = min(minimum, float(values.min())), max(maximum, float(values.max()))
        bindings.append(
            {
                "logical_id": record.logical_id,
                "primary_audio_path": record.primary_audio_path,
                "primary_audio_sha256": record.primary_audio_sha256,
                "manifest_fingerprint": SIGNED_HASHES["manifest_fingerprint"],
                "a0_config_hash": A0_HASH,
                "experiment_protocol_hash": PROTOCOL_HASH,
                "protocol_file_sha256": file_hash(protocol_path),
                "metadata_sha256": file_hash(metadata_path),
                "mel_sha256": metadata["mel_sha256"],
            }
        )
        if index % 1000 == 0:
            print(f"Read-only A0 audit: {index}/{len(records)}", flush=True)
    for relative, before in tree_before.items():
        stat = (namespace / relative).stat()
        require(
            before == {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
            "A0 changed during audit",
        )

    print(
        "A0 checks passed; comparing protected predecessors with the preserved baseline.",
        flush=True,
    )
    protected = collect(ROOT)
    baseline = read_json(ROOT / ".tmp/module5_before.json")
    require(protected == baseline, "Protected predecessor inventory/content differs")
    m2_root = ROOT / "data/processed/representations" / SIGNED_HASHES["representation_config_hash"]
    m2_logical = sum(1 for _ in m2_root.rglob("metadata.json"))
    require(m2_logical == 8673, "Module 2 logical count differs")
    ledger = inspect_ledger(config.output("experiments/ledger.sqlite3"))
    require(
        ledger["runs"] == 0 and ledger["outer_claims"] == ledger["outer_events"] == 0,
        "Production experiment activity needs review before declaring Phase 5B not started",
    )
    require(
        not list(config.output("experiments").glob("runs/*/outer*")), "Unexpected outer results"
    )
    require_frozen(context)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = config.output(f"phase5a_audit/{stamp}")
    output.mkdir(parents=True, exist_ok=False)
    bindings_path = output / "a0_protocol_bindings.csv"
    with bindings_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(bindings[0]))
        writer.writeheader()
        writer.writerows(bindings)
    write_json(
        output / "protected_inputs.json",
        {
            "status": "PASS",
            **protected,
            "module2_logical_directories": m2_logical,
            "changes": [],
            "verification": "trees: path/size/mtime; protected files: content SHA256/size/mtime",
        },
        immutable=True,
    )
    report = {
        "status": "PASS",
        "phase5a": "COMPLETE / PASSED",
        "phase5b": "NOT STARTED / READY FOR MANUAL EXECUTION",
        "module5": "MANUAL EXPERIMENTS IN PROGRESS / NOT SIGNED OFF",
        "module6": "NOT STARTED",
        "protocol_hash": PROTOCOL_HASH,
        "frozen_code_hash": CODE_HASH,
        "a0_config_hash": A0_HASH,
        "frozen": True,
        "frozen_verification": (
            "Existing protocol.json exactly equals require_frozen current document"
        ),
        "frozen_schema_note": (
            "Original protocol has no frozen boolean; file presence plus exact equality is the lock"
        ),
        "namespace": namespace.relative_to(ROOT).as_posix(),
        "signed_hashes": SIGNED_HASHES,
        "retained_reports": retained,
        "splits": protocol["splits"],
        "inventory": {
            "logical_directories": len(records),
            "mel_npy": len(records),
            "metadata_json": len(records),
            "files": len(actual_files),
            "total_bytes": sum(v["bytes"] for v in tree_before.values()),
            "unexpected_partial_files": 0,
            "zero_byte_required_files": 0,
            "inventory_sha256": digest(tree_before),
        },
        "mel_integrity": {
            "validated": len(records),
            "shape": [224, 224],
            "dtype": "float32",
            "all_finite": True,
            "minimum": minimum,
            "maximum": maximum,
            "transforms_run": 0,
        },
        "provenance": {
            "exact_signed_record_bindings": len(bindings),
            "primary_waveform_sha256_checked": len(bindings),
            "entry_protocol_hash_field": False,
            "protocol_binding": "Transitive through frozen protocol, retained full-run reports, "
            "and this checksummed per-entry audit binding table; existing cache metadata unchanged",
            "bindings_file": bindings_path.relative_to(ROOT).as_posix(),
            "bindings_sha256": file_hash(bindings_path),
        },
        "M5-CODE-001": "CLOSED; exact signed-record regression separately rerun",
        "protected": {
            "status": "PASS",
            "file_count": len(protected["protected_files"]),
            "trees": protected["trees"],
            "module2_logical_directories": m2_logical,
            "changed_files": 0,
            "verification": "trees: path/size/mtime; 247 files: SHA256/size/mtime",
        },
        "production_ledger": ledger,
        "outer_model_forwards": 0,
        "audit_model_forwards": 0,
        "elapsed_seconds": time.perf_counter() - start,
    }
    write_json(output / "phase5a_audit.json", report, immutable=True)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(output / "phase5a_audit.json"),
                "inventory": report["inventory"],
                "protected": report["protected"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
