"""Explicit JiWER 4 counts and aggregation, separate from recognition."""

from __future__ import annotations

import importlib.metadata
import statistics
from dataclasses import asdict
from typing import Any

import jiwer

from .artifacts import digest
from .normalization import NORMALIZATION_SHA, normalize

METRIC_CONFIG = {
    "id": "m6_metrics_v1",
    "backend": "jiwer",
    "version": "4.0.0",
    "normalizer_sha256": NORMALIZATION_SHA,
    "wer": "word S+D+I / reference words",
    "cer": "character S+D+I / reference characters; spaces removed",
    "unavailable_reference": "null",
    "failures": "count separately; never empty hypothesis",
    "aggregation": "sum per-record counts; sample SD ddof=1; no clamping",
}
METRIC_SHA = digest(METRIC_CONFIG)


def _counts(result: Any, reference_length: int, hypothesis_length: int) -> dict[str, Any]:
    errors = result.substitutions + result.deletions + result.insertions
    return {
        "hits": result.hits,
        "substitutions": result.substitutions,
        "deletions": result.deletions,
        "insertions": result.insertions,
        "reference_length": reference_length,
        "hypothesis_length": hypothesis_length,
        "errors": errors,
        "ratio": errors / reference_length,
        "percentage": 100.0 * errors / reference_length,
        "alignment": [[asdict(chunk) for chunk in alignment] for alignment in result.alignments],
    }


def score(
    reference: dict[str, Any], raw_hypothesis: str | None, *, asr_status: str = "COMPLETE"
) -> dict[str, Any]:
    if importlib.metadata.version("jiwer") != "4.0.0":
        raise ValueError("JiWER version drift")
    ref = reference.get("scoring_reference")
    norm_ref = normalize(ref) if isinstance(ref, str) else None
    norm_hyp = normalize(raw_hypothesis) if isinstance(raw_hypothesis, str) else None
    result: dict[str, Any] = {
        "metric_config_sha256": METRIC_SHA,
        "normalization_sha256": NORMALIZATION_SHA,
        "reference_original": reference.get("original_decoded"),
        "scoring_reference": ref,
        "hypothesis_original": raw_hypothesis,
        "normalized_reference": norm_ref,
        "normalized_hypothesis": norm_hyp,
        "reference_eligible": reference["eligible"],
        "reference_limitation": reference["limitation"],
        "asr_status": asr_status,
        "wer": None,
        "cer": None,
        "scored": False,
        "exclusion_reason": None,
    }
    if asr_status != "COMPLETE" or raw_hypothesis is None:
        result["exclusion_reason"] = "ASR_FAILURE"
    elif not reference["eligible"]:
        result["exclusion_reason"] = reference["exclusion_reason"]
    elif not norm_ref:
        result["exclusion_reason"] = "EMPTY_NORMALIZED_REFERENCE"
    else:
        assert norm_hyp is not None
        words = jiwer.ReduceToListOfListOfWords()
        chars = jiwer.ReduceToListOfListOfChars()
        word = jiwer.process_words(
            norm_ref, norm_hyp, reference_transform=words, hypothesis_transform=words
        )
        char_ref, char_hyp = norm_ref.replace(" ", ""), norm_hyp.replace(" ", "")
        char = jiwer.process_characters(
            char_ref, char_hyp, reference_transform=chars, hypothesis_transform=chars
        )
        result.update(
            scored=True,
            wer=_counts(word, len(norm_ref.split()), len(norm_hyp.split())),
            cer=_counts(char, len(char_ref), len(char_hyp)),
        )
    return result


def _micro(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row["metrics"] for row in rows if row["metrics"]["scored"]]
    output: dict[str, Any] = {
        "records": len(rows),
        "scored_records": len(valid),
        "asr_failures": sum(r["metrics"]["exclusion_reason"] == "ASR_FAILURE" for r in rows),
        "unscored_records": len(rows) - len(valid),
    }
    for name in ("wer", "cer"):
        totals = {
            key: sum(m[name][key] for m in valid)
            for key in (
                "hits",
                "substitutions",
                "deletions",
                "insertions",
                "errors",
                "reference_length",
                "hypothesis_length",
            )
        }
        denominator = totals["reference_length"]
        output[name] = {
            **totals,
            "ratio": totals["errors"] / denominator if denominator else None,
            "percentage": 100.0 * totals["errors"] / denominator if denominator else None,
        }
    return output


def aggregate(rows: list[dict[str, Any]], *, scope: str) -> dict[str, Any]:
    ids = [row["logical_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate logical records in metrics")
    result: dict[str, Any] = {
        "scope": scope,
        "metric_config_sha256": METRIC_SHA,
        "corpus": _micro(rows),
        "per_record": rows,
    }
    for key in ("speaker_id", "condition", "prompt_type", "inherited_group"):
        groups = sorted({str(row[key]) for row in rows})
        result[key] = {
            group: _micro([row for row in rows if str(row[key]) == group]) for group in groups
        }
    three = result["inherited_group"]
    summary: dict[str, Any] = {
        "groups": len(three),
        "ddof": 1,
        "weighting": "equal weight per inherited group",
        "wer": None,
        "cer": None,
    }
    if set(three) == {"1", "2", "3"}:
        for metric in ("wer", "cer"):
            values = [three[str(group)][metric]["ratio"] for group in (1, 2, 3)]
            if all(value is not None for value in values):
                summary[metric] = {
                    "mean": statistics.mean(values),
                    "sample_sd": statistics.stdev(values),
                }
    result["three_group_mean_sample_sd"] = summary
    result["completion_status"] = "PARTIAL" if result["corpus"]["asr_failures"] else "COMPLETE"
    return result
