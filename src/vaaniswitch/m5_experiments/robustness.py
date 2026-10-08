"""Descriptive C2 subgroup metrics from immutable predictions, never model selection."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from vaaniswitch.m4_fusion.metrics import binary_metrics
from vaaniswitch.m5_experiments.source import ExperimentContext


def robustness(context: ExperimentContext, results: list[dict[str, Any]]) -> dict[str, Any]:
    c2 = [r for r in results if r["ablation"] == "C2"]
    if sorted(r["outer_fold"] for r in c2) != [1, 2, 3]:
        raise ValueError("robustness requires all three C2 outer results")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    contract = context.predecessor.source.contract
    for result in c2:
        split = context.split(result["outer_fold"])
        expected = {r.logical_id for r in split.records(contract, "outer_test")}
        actual = [p["logical_id"] for p in result["predictions"]]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("robustness requires complete speaker-bound outer predictions")
        for prediction in result["predictions"]:
            record = contract.record_by_id[prediction["logical_id"]]
            if record.logical_id in seen:
                raise ValueError("duplicate outer logical record")
            seen.add(record.logical_id)
            assert record.duration_seconds is not None and record.primary_microphone is not None
            duration = record.duration_seconds
            values = {
                "gender": record.gender.value,
                "primary_microphone": record.primary_microphone.value,
                "prompt_type": record.prompt_type.value,
                "duration": "<=8s" if duration <= 8 else ">8s",
                "mel_handling": "cropped"
                if duration > 8
                else ("padded" if duration < 8 else "exact"),
                "speaker": record.speaker_id,
            }
            for dimension, value in values.items():
                for scope in (f"fold{result['outer_fold']}", "pooled_descriptive"):
                    groups[f"{scope}/{dimension}/{value}"].append(prediction)
    summaries = {}
    for key, predictions in sorted(groups.items()):
        labels = [p["label"] for p in predictions]
        summaries[key] = {
            "sample_count": len(predictions),
            "speaker_count": len({p["speaker_id"] for p in predictions}),
            "control_count": labels.count(0),
            "dysarthric_count": labels.count(1),
            "small_subgroup": len(predictions) < 30,
            "metrics": binary_metrics(labels, [p["probability"] for p in predictions]),
        }
    return {
        "scope": "descriptive C2 outer predictions only; no selection feedback",
        "record_count": len(seen),
        "groups": summaries,
        "single_class_auc": "null with reason; no small subgroup suppression",
    }
