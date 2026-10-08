"""Read-only runtime metadata and source-to-audio projection; no M0-M5 imports."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .artifacts import digest, sha256, verify_checksum
from .config import RECORD_ID


def resolve_smoke_audio(root: Path) -> dict[str, Any]:
    phase0 = json.loads((root / "docs/modules/6_english_asr_tts/PHASE0_EVIDENCE.json").read_text())
    expected = phase0["panel_audio_verification"]["primary_audio_hashes"][RECORD_ID]
    manifest = root / "data/manifests/torgo_manifest.jsonl"
    verify_checksum(
        manifest,
        phase0["upstream_verification"]["protected_sha256"]["data/manifests/torgo_manifest.jsonl"],
    )
    matches = []
    with manifest.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["logical_id"] == RECORD_ID:
                # Project only audio bindings; no transcript/prompt fields cross this boundary.
                matches.append(
                    {
                        k: row[k]
                        for k in (
                            "logical_id",
                            "primary_audio_path",
                            "primary_audio_sha256",
                            "frame_count",
                        )
                    }
                )
    if len(matches) != 1 or matches[0]["primary_audio_sha256"] != expected:
        raise ValueError("development primary audio binding mismatch")
    job = matches[0]
    source = (root / job["primary_audio_path"]).resolve()
    if not source.is_relative_to((root / "data/raw/torgo").resolve()):
        raise ValueError("source outside read-only TORGO")
    verify_checksum(source, expected)
    return job


def environment_record(root: Path) -> dict[str, Any]:
    packages = dict(
        sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
    )
    env = Path(sys.prefix).resolve()
    if env != (root / ".venv-m6").resolve() or sys.prefix == sys.base_prefix:
        raise RuntimeError("Module 6 must use the isolated .venv-m6 interpreter")
    if "include-system-site-packages = false" not in (env / "pyvenv.cfg").read_text():
        raise RuntimeError("system site packages must be isolated")
    cmd = (
        "$os=Get-CimInstance Win32_OperatingSystem;"
        "$cpu=Get-CimInstance Win32_Processor;"
        "$mem=Get-CimInstance Win32_ComputerSystem;"
        "@{os=$os.Caption;os_version=$os.Version;cpu=$cpu.Name;"
        "ram_bytes=$mem.TotalPhysicalMemory}|ConvertTo-Json -Compress"
    )
    hardware = json.loads(
        subprocess.check_output(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd], text=True, timeout=30
        )
    )
    hardware["nvidia_smi"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv"],
        text=True,
        timeout=15,
    ).strip()
    record = {
        "python": platform.python_version(),
        "python_full": sys.version,
        "executable": sys.executable,
        "environment_path": str(env),
        "base_prefix": sys.base_prefix,
        "platform": platform.platform(),
        "environment_cfg_sha256": sha256(env / "pyvenv.cfg"),
        "python_executable_sha256": sha256(Path(sys.executable)),
        "packages": packages,
        "hardware": hardware,
        "lock_sha256": sha256(root / "configs/module6/environment/requirements-win-py312.lock"),
    }
    record["identity_sha256"] = digest(record)
    return record
