"""Module 6 output boundary and immutable, finite JSON records."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def output_path(root: Path, path: Path | str) -> Path:
    """Allow only ordinary paths below the literal artifacts/module6 tree."""
    root = root.resolve()
    candidate = Path(path)
    if ".." in candidate.parts:
        raise ValueError("parent traversal is prohibited")
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("output outside project") from exc
    if relative.parts[:2] != ("artifacts", "module6") or len(relative.parts) < 3:
        raise ValueError("writes must be inside artifacts/module6")
    cursor = root
    for part in relative.parts:
        cursor /= part
        # Runtime is Python 3.12; root mypy defaults to 3.11 stubs without is_junction.
        if cursor.is_symlink() or cursor.is_junction():  # type: ignore[attr-defined, unused-ignore]
            raise ValueError("symlink/junction output prohibited")
        if cursor.is_file() and cursor.stat().st_nlink > 1:
            raise ValueError("hardlinked output prohibited")
    if not candidate.resolve().is_relative_to(root / "artifacts/module6"):
        raise ValueError("resolved output outside Module 6")
    return candidate


def write_bytes(root: Path, path: Path | str, payload: bytes) -> Path:
    target = output_path(root, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return target


def write_json(root: Path, path: Path | str, value: Any) -> Path:
    return write_bytes(root, path, canonical_bytes(value) + b"\n")


def complete_json(root: Path, path: Path | str, value: Any) -> Path:
    """Publish a complete record without replacing an earlier completion."""
    target = output_path(root, path)
    temporary = write_json(root, target.with_suffix(".pending.json"), value)
    os.link(temporary, target)  # atomic exclusive publication on NTFS
    temporary.unlink()
    return target


def verify_checksum(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")
    return actual
