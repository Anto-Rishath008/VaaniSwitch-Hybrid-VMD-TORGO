"""Evaluator-only prompt-based references; never imported by recognition workers."""

from __future__ import annotations

import base64
import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any

from .schemas import REFERENCE, split_id

REFERENCE_RULES = "m6_prompt_reference_v1"
REFERENCE_LIMITATION = "TORGO prompt-based reference; not guaranteed verbatim audio transcription"
SENSE = re.compile(r"([A-Za-z]+(?:['-][A-Za-z]+)*)\s+\[as in [^\[\]\r\n]+\]")


def derive_reference(text: str | None, *, core_member: bool) -> dict[str, Any]:
    value = text.strip() if text is not None else ""
    result: dict[str, Any] = {
        "state": REFERENCE,
        "original_decoded": text,
        "scoring_reference": None,
        "eligible": False,
        "eligibility_state": "EXCLUDED",
        "exclusion_reason": None,
        "rule_id": REFERENCE_RULES,
        "limitation": REFERENCE_LIMITATION,
    }
    reason = None
    if not core_member:
        reason = "NOT_IN_SIGNED_CORE"
    elif text is None:
        reason = "MISSING_REFERENCE"
    elif not value:
        reason = "BLANK_REFERENCE"
    elif value.casefold() == "xxx":
        reason = "PLACEHOLDER_REFERENCE"
    elif re.search(r"\.(?:jpg|jpeg|png|gif|bmp|tiff?|webp)(?:\s|$)", value, re.IGNORECASE):
        reason = "IMAGE_STIMULUS_REFERENCE"
    elif any(unicodedata.category(c).startswith("C") and c not in "\t\r\n" for c in value):
        reason = "UNSUPPORTED_CONTROL_CHARACTER"
    elif "[" in value or "]" in value:
        match = SENSE.fullmatch(value)
        if match is None:
            reason = "UNSUPPORTED_BRACKET_PATTERN"
        else:
            result.update(scoring_reference=match.group(1), rule_id="single_word_sense_v1")
    else:
        result["scoring_reference"] = value
    if reason:
        result["exclusion_reason"] = reason
        if reason in {"UNSUPPORTED_CONTROL_CHARACTER", "UNSUPPORTED_BRACKET_PATTERN"}:
            result["eligibility_state"] = "REVIEW_REQUIRED"
    else:
        result.update(eligible=True, eligibility_state="ELIGIBLE")
    return result


def read_reference(root: Path, row: dict[str, Any], *, core_member: bool) -> dict[str, Any]:
    speaker, session, prompt = split_id(row["logical_id"])
    if (row["speaker_id"], row["session_id"], row["prompt_id"]) != (speaker, session, prompt):
        raise ValueError("reference composite identity mismatch")
    relative = Path(row["prompt_path"])
    expected = Path("data/raw/torgo/extracted") / row["speaker_group"] / speaker / session
    if relative.as_posix() != (expected / "prompts" / f"{prompt}.txt").as_posix():
        raise ValueError("reference path does not match composite identity")
    path = (root / relative).resolve()
    if not path.is_relative_to((root / "data/raw/torgo/extracted").resolve()):
        raise ValueError("reference outside TORGO")
    payload = path.read_bytes() if path.is_file() else None
    encoding = None
    decoded = None
    if payload is not None:
        for codec in ("utf-8-sig", "cp1252"):
            try:
                decoded = payload.decode(codec)
                encoding = codec
                break
            except UnicodeDecodeError:
                continue
        if decoded is None:
            result = derive_reference(None, core_member=core_member)
            result.update(
                eligibility_state="REVIEW_REQUIRED", exclusion_reason="UNDECODABLE_REFERENCE"
            )
        else:
            if decoded.strip() != row["transcript"]:
                raise ValueError("prompt bytes no longer match the signed manifest text")
            result = derive_reference(decoded, core_member=core_member)
    else:
        result = derive_reference(None, core_member=core_member)
    result.update(
        logical_id=row["logical_id"],
        prompt_path=relative.as_posix(),
        encoding=encoding,
        raw_bytes_base64=base64.b64encode(payload).decode("ascii") if payload is not None else None,
        raw_prompt_sha256=hashlib.sha256(payload).hexdigest() if payload is not None else None,
    )
    return result
