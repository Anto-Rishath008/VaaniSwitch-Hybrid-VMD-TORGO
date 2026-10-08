"""Read-only Phase-0/hash comparison; never imports or executes Modules 0-5."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def mapping_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def inspect(root: Path) -> dict[str, Any]:
    phase0_path = root / "docs/modules/6_english_asr_tts/PHASE0_EVIDENCE.json"
    phase0 = json.loads(phase0_path.read_text())
    before = json.loads((root / "artifacts/module6/setup/preservation_before.json").read_text())
    assert sha(phase0_path) == before["phase0_evidence_sha256"], "Phase-0 evidence drift"
    protected = phase0["upstream_verification"]["protected_sha256"]
    changed = [p for p, h in protected.items() if not (root / p).is_file() or sha(root / p) != h]
    source = {
        p.relative_to(root).as_posix(): sha(p)
        for p in sorted((root / "src/vaaniswitch").glob("m[012345]*/*.py"))
    }
    code = mapping_digest(source)
    env = {
        p.relative_to(root).as_posix(): sha(p) for p in (root / ".venv").rglob("*") if p.is_file()
    }
    env_changed = sorted(
        p
        for p in env.keys() | before["venv_files"].keys()
        if env.get(p) != before["venv_files"].get(p)
    )
    ledger = {
        p.relative_to(root).as_posix(): sha(p)
        for p in (root / "artifacts/module5").rglob("*")
        if p.is_file() and any(x in p.name for x in (".sqlite", ".db"))
    }
    ledger_changed = sorted(
        p
        for p in ledger.keys() | before["module5_ledger_files"].keys()
        if ledger.get(p) != before["module5_ledger_files"].get(p)
    )
    audio_expected = phase0["panel_audio_verification"]["primary_audio_hashes"]
    audio_checked = {}
    with (root / "data/manifests/torgo_manifest.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["logical_id"] in audio_expected:
                audio_checked[row["logical_id"]] = sha(root / row["primary_audio_path"])
    audio_changed = [
        key for key, value in audio_expected.items() if audio_checked.get(key) != value
    ]
    result = {
        "phase0_protected_files_compared": len(protected),
        "changed_protected_files": changed,
        "frozen_scientific_code_sha256": code,
        "scientific_source_files": len(source),
        "scientific_code_matches": code == phase0["upstream_verification"]["frozen_code_hash"],
        "scientific_venv_files_compared": len(env),
        "changed_venv_files": env_changed,
        "ledger_files_compared": len(ledger),
        "changed_ledger_files": ledger_changed,
        "primary_wavs_compared": len(audio_checked),
        "changed_primary_wavs": audio_changed,
        "protocol_identity": phase0["upstream_verification"]["protocol_hash"],
        "protocol_check": "Frozen protocol file and defining scientific code byte-identical",
        "phase0_evidence_sha256": sha(phase0_path),
        "checkpoint_checks": [
            {**c, "current_sha256": sha(root / c["checkpoint_path"])}
            for c in phase0["upstream_verification"]["primary_checkpoints"]
        ],
        "fold1_a1": "RECOVERED_PROTOCOL_EXCEPTION",
        "original_native_forward_count": "UNKNOWN",
        "scope": "Targeted preservation only; no Module 5 execution or full raw-tree audit",
    }
    result["status"] = (
        "PASS"
        if not (changed or env_changed or ledger_changed or audio_changed)
        and result["scientific_code_matches"]
        else "FAIL"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(inspect(args.root.resolve()), indent=2))
