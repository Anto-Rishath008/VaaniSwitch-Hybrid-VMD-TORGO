"""Verify signed trees and Module 0-3 artifacts before/after Module 4 work."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from vaaniswitch.m0_data.reporting import snapshot_raw_tree
from vaaniswitch.m4_fusion.reporting import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
TREES = {
    "raw_torgo": "data/raw/torgo/extracted",
    "module1_production_cache": "data/processed/vmd/"
    "bfd7aec36d99a88cfacc72df85e124d3f4b99b76ed47d71c3ced9a153fa92685",
    "module2_representation_cache": "data/processed/representations/"
    "d8a36c6ab8c48b15e61ecd63194ffa30ede4e7e7e7ab0a30c197002652f9d65d",
}


def collect() -> dict[str, Any]:
    trees = {name: snapshot_raw_tree(ROOT / relative) for name, relative in TREES.items()}
    paths = list((ROOT / "src/vaaniswitch/m3_encoders").rglob("*.py"))
    paths += list((ROOT / "tests/m3_encoders").rglob("*.py"))
    paths += list((ROOT / "docs/modules/3_parallel_token_encoders").glob("*"))
    paths += [
        ROOT / p
        for p in (
            "configs/models/encoders.yaml",
            "configs/representations/default.yaml",
            "data/manifests/torgo_manifest.jsonl",
            "data/splits/speaker_folds.json",
        )
    ]
    paths += list((ROOT / "artifacts/metrics").glob("module[0123]*"))
    paths += list((ROOT / "artifacts/logs").glob("module[0123]*"))
    files = {
        p.relative_to(ROOT).as_posix(): {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "size": p.stat().st_size,
            "mtime_ns": p.stat().st_mtime_ns,
        }
        for p in paths
        if p.is_file()
    }
    return {"trees": trees, "protected_files": files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["before", "after"], required=True)
    args = parser.parse_args()
    reference = ROOT / ".tmp/module4_before.json"
    if args.phase == "before" and reference.exists():
        raise ValueError(
            "before snapshot already exists; preserve the original implementation baseline"
        )
    current = collect()
    signed = read_json(ROOT / "artifacts/metrics/module3_full_signoff_audit.json")[
        "protected_inputs"
    ]
    for name, tree in current["trees"].items():
        expected = signed[name]
        if (
            tree["file_count"] != expected["file_count"]
            or tree["total_bytes"] != expected["total_bytes"]
            or tree["metadata_sha256"] != expected["path_size_mtime_sha256"]
        ):
            raise ValueError(f"protected tree differs from Module 3 signature: {name}")
    if args.phase == "before":
        write_json(reference, current)
        print("PASS: pre-implementation snapshot captured")
        return 0
    before = read_json(reference)
    changed = [
        key
        for key in set(before["protected_files"]) | set(current["protected_files"])
        if before["protected_files"].get(key) != current["protected_files"].get(key)
    ]
    trees_equal = before["trees"] == current["trees"]
    result = {
        "status": "PASS" if trees_equal and not changed else "FAIL",
        "verification": "trees use path/size/mtime SHA-256; "
        "signed files use content SHA-256, size, mtime",
        "before": before["trees"],
        "after": current["trees"],
        "trees_unchanged": trees_equal,
        "module2_logical_directories": sum(
            1 for _ in (ROOT / TREES["module2_representation_cache"]).rglob("metadata.json")
        ),
        "protected_file_count": len(current["protected_files"]),
        "protected_file_changes": sorted(changed),
        "protected_files": current["protected_files"],
    }
    write_json(ROOT / "artifacts/metrics/module4_protected_inputs_audit.json", result)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "protected_files"}, indent=2
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
