"""Capture and compare signed Modules 0-4 without writing to their namespaces."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from vaaniswitch.m0_data.reporting import snapshot_raw_tree
from vaaniswitch.m4_fusion.reporting import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def collect(root: Path) -> dict[str, Any]:
    signed = read_json(root / "artifacts/metrics/module4_full_signoff_audit.json")
    hashes = signed["hashes"]
    representation_hash = hashes["representation_config_hash"]
    trees = {
        "raw_torgo": "data/raw/torgo/extracted",
        "module1_production_cache": f"data/processed/vmd/{hashes['module1_config_hash']}",
        "module2_representation_cache": f"data/processed/representations/{representation_hash}",
    }
    current = {name: snapshot_raw_tree(root / path) for name, path in trees.items()}
    if current != signed["protected_inputs"]["recorded_trees"]:
        raise ValueError("protected trees differ from signed Module 4 inventories")
    paths: set[Path] = set()
    for category in ("src/vaaniswitch", "tests", "docs/modules"):
        for directory in (root / category).iterdir():
            if directory.is_dir() and directory.name.startswith(
                tuple(f"m{i}_" if category != "docs/modules" else f"{i}_" for i in range(5))
            ):
                paths.update(p for p in directory.rglob("*") if "__pycache__" not in p.parts)
    for category in ("metrics", "logs", "plots", "checkpoints"):
        for path in (root / "artifacts" / category).glob("module[01234]*"):
            paths.update(path.rglob("*") if path.is_dir() else [path])
    for category in (
        "configs/data",
        "configs/vmd",
        "configs/representations",
        "configs/models",
        "data/manifests",
        "data/splits",
    ):
        paths.update((root / category).rglob("*"))
    files = {
        p.relative_to(root).as_posix(): {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "size": p.stat().st_size,
            "mtime_ns": p.stat().st_mtime_ns,
        }
        for p in sorted(paths)
        if p.is_file()
    }
    return {"trees": current, "protected_files": files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["before", "after"], required=True)
    args = parser.parse_args()
    before_path = ROOT / ".tmp/module5_before.json"
    current = collect(ROOT)
    if args.phase == "before":
        if before_path.exists():
            raise ValueError("original Module 5 baseline already exists; never replace it")
        write_json(before_path, current)
        print(f"PASS: captured three signed trees and {len(current['protected_files'])} files")
        return 0
    before = read_json(before_path)
    changed = sorted(
        k
        for k in set(before["protected_files"]) | set(current["protected_files"])
        if before["protected_files"].get(k) != current["protected_files"].get(k)
    )
    passed = not changed and before["trees"] == current["trees"]
    result = {
        "status": "PASS" if passed else "FAIL",
        "trees": current["trees"],
        "module2_logical_directories": sum(
            1
            for _ in (
                ROOT
                / "data/processed/representations"
                / "d8a36c6ab8c48b15e61ecd63194ffa30ede4e7e7e7ab0a30c197002652f9d65d"
            ).rglob("metadata.json")
        ),
        "protected_file_count": len(current["protected_files"]),
        "changes": changed,
        "verification": "trees: path/size/mtime; protected files: content SHA256/size/mtime",
        "protected_files": current["protected_files"],
    }
    write_json(ROOT / "artifacts/module5/implementation/protected_inputs.json", result)
    print(f"{result['status']}: {len(current['protected_files'])} files; changes={changed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
