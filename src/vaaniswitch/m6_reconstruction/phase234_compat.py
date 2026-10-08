"""Compatibility layer: retain interrupted-Codex Phase-2 contracts unchanged."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import verify_checksum
from .metrics import score as project_score
from .normalization import NORMALIZATION as NORMALIZER_SPEC
from .normalization import NORMALIZATION_SHA as NORMALIZER_SHA256
from .references import REFERENCE_LIMITATION, read_reference
from .schemas import AudioJob

__all__ = ["NORMALIZER_SHA256", "NORMALIZER_SPEC"]
_FOLD = {
    "F03": "1",
    "FC01": "1",
    "M04": "1",
    "MC03": "1",
    "MC04": "1",
    "F04": "2",
    "FC02": "2",
    "M01": "2",
    "M02": "2",
    "MC02": "2",
    "F01": "3",
    "FC03": "3",
    "M03": "3",
    "M05": "3",
    "MC01": "3",
}


def inherited_group(speaker_id: str) -> str:
    try:
        return _FOLD[speaker_id]
    except KeyError as exc:
        raise ValueError(f"speaker is outside frozen three-fold mapping: {speaker_id}") from exc


@dataclass(frozen=True)
class ReferenceAdapter:
    logical_id: str
    eligible: bool
    source_text: str | None
    scoring_text: str | None
    exclusion_reason: str | None
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(frozen=True)
class MetricAdapter:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def load_manifest(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "data/manifests/torgo_manifest.jsonl"
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            logical_id = str(row["logical_id"])
            if logical_id in rows:
                raise ValueError(f"duplicate manifest logical_id: {logical_id}")
            rows[logical_id] = row
    if len(rows) != 9494:
        raise ValueError(f"manifest record-count drift: expected 9494, got {len(rows)}")
    return rows


def signed_core_ids(root: Path, manifest: dict[str, dict[str, Any]]) -> set[str]:
    """Read the retained signed selection; do not derive membership from mutable flags."""
    path = root / "artifacts/metrics/module2_representation_summary.json"
    verify_checksum(path, "03b11f02aa17b7e72cd9a406d09092f36584f782b2551b9f099a0a45acc653d9")
    ids = json.loads(path.read_bytes())["selection"]["logical_ids"]
    if len(ids) != 8673 or len(set(ids)) != 8673 or not set(ids).issubset(manifest):
        raise ValueError("signed CORE selection identity/count drift")
    return set(ids)


def build_reference_inventory(
    root: Path, manifest: dict[str, dict[str, Any]]
) -> dict[str, ReferenceAdapter]:
    result: dict[str, ReferenceAdapter] = {}
    core_count = 0
    image_count = 0
    eligible_count = 0
    core_ids = signed_core_ids(root, manifest)
    for logical_id, row in manifest.items():
        core = logical_id in core_ids
        core_count += int(core)
        payload = read_reference(root, row, core_member=core)
        eligible = bool(payload["eligible"])
        eligible_count += int(eligible)
        image_count += int(core and payload.get("exclusion_reason") == "IMAGE_STIMULUS_REFERENCE")
        result[logical_id] = ReferenceAdapter(
            logical_id=logical_id,
            eligible=eligible,
            source_text=payload.get("original_decoded"),
            scoring_text=payload.get("scoring_reference"),
            exclusion_reason=payload.get("exclusion_reason"),
            payload=payload,
        )
    if core_count != 8673:
        raise ValueError(f"signed CORE drift: expected 8673, got {core_count}")
    if image_count != 111:
        raise ValueError(f"CORE image-reference drift: expected 111, got {image_count}")
    if eligible_count != 8562:
        raise ValueError(f"M6 reference eligibility drift: expected 8562, got {eligible_count}")
    return result


def audio_job(row: dict[str, Any]) -> dict[str, Any]:
    return AudioJob(
        logical_id=row["logical_id"],
        primary_audio_path=row["primary_audio_path"],
        primary_audio_sha256=row["primary_audio_sha256"],
        frame_count=int(row["frame_count"]),
    ).to_dict()


def score(
    logical_id: str,
    reference: str | None,
    hypothesis: str,
    *,
    exclusion_reason: str | None = None,
) -> MetricAdapter:
    reference_payload: dict[str, Any] = {
        "original_decoded": reference,
        "scoring_reference": reference,
        "eligible": reference is not None,
        "exclusion_reason": exclusion_reason if reference is None else None,
        "limitation": REFERENCE_LIMITATION,
    }
    raw = project_score(reference_payload, hypothesis, asr_status="COMPLETE")
    word = raw["wer"]
    char = raw["cer"]
    payload = {
        "logical_id": logical_id,
        "eligible": bool(raw["scored"]),
        "scored": bool(raw["scored"]),
        "exclusion_reason": raw["exclusion_reason"],
        "reference_original": raw["reference_original"],
        "hypothesis_original": raw["hypothesis_original"],
        "reference_normalized": raw["normalized_reference"],
        "hypothesis_normalized": raw["normalized_hypothesis"],
        "wer": word["ratio"] if word else None,
        "cer": char["ratio"] if char else None,
        "word_counts": word,
        "char_counts": char,
        "metric_config_sha256": raw["metric_config_sha256"],
        "normalization_sha256": raw["normalization_sha256"],
        "reference_limitation": raw["reference_limitation"],
    }
    return MetricAdapter(payload)


def _micro(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row["metric"] for row in rows if row["metric"]["eligible"]]
    if not valid:
        return {"records": len(rows), "scored_records": 0, "wer": None, "cer": None}
    ws = wd = wi = wr = 0
    cs = cd = ci = cr = 0
    for metric in valid:
        wc = metric["word_counts"]
        cc = metric["char_counts"]
        ws += int(wc["substitutions"])
        wd += int(wc["deletions"])
        wi += int(wc["insertions"])
        wr += int(wc["reference_length"])
        cs += int(cc["substitutions"])
        cd += int(cc["deletions"])
        ci += int(cc["insertions"])
        cr += int(cc["reference_length"])
    return {
        "records": len(rows),
        "scored_records": len(valid),
        "wer": (ws + wd + wi) / wr if wr else None,
        "cer": (cs + cd + ci) / cr if cr else None,
        "word_counts": {
            "substitutions": ws,
            "deletions": wd,
            "insertions": wi,
            "reference_length": wr,
        },
        "char_counts": {
            "substitutions": cs,
            "deletions": cd,
            "insertions": ci,
            "reference_length": cr,
        },
    }


def aggregate(rows: Iterable[dict[str, Any]], group_fields: tuple[str, ...] = ()) -> dict[str, Any]:
    values = list(rows)
    if not group_fields:
        return _micro(values)
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in values:
        key = tuple(str(row["metadata"].get(field, "")) for field in group_fields)
        groups.setdefault(key, []).append(row)
    return {
        "fields": list(group_fields),
        "groups": {"|".join(key): _micro(group) for key, group in sorted(groups.items())},
    }


def sample_mean_sd(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "sample_sd": None}
    mean = sum(values) / len(values)
    if len(values) < 2:
        return {"n": len(values), "mean": mean, "sample_sd": None}
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return {"n": len(values), "mean": mean, "sample_sd": math.sqrt(variance)}
