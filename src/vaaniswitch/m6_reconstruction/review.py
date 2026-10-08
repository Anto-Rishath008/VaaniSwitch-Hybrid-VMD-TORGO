"""Optional explicit human correction contract; no automatic correction."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifacts import sha256, write_bytes, write_json


def create_human_correction(
    root: Path,
    record_dir: Path,
    corrected_text: str,
    *,
    actor: str,
    reason: str,
    reference_visible: bool,
) -> dict[str, Any]:
    raw = record_dir / "raw_asr.txt"
    if not raw.is_file():
        raise ValueError("sealed raw transcript required before correction")
    if reference_visible:
        raise ValueError("independent corrected-text evaluation requires reference_visible=false")
    parent_sha = sha256(raw)
    revision = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    target = record_dir / "corrections" / revision
    corrected = write_bytes(root, target / "corrected.txt", corrected_text.encode("utf-8"))
    metadata = {
        "revision": revision,
        "parent_raw_sha256": parent_sha,
        "corrected_sha256": sha256(corrected),
        "actor": actor,
        "source": "explicit_human_edit",
        "reason": reason,
        "reference_visible": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(root, target / "correction.json", metadata)
    if sha256(raw) != parent_sha:
        raise RuntimeError("raw transcript changed during correction")
    return metadata
