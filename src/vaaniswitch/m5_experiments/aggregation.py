"""Unweighted fold means and sample standard deviations; no significance claims."""

from __future__ import annotations

import math
import statistics
from typing import Any

from vaaniswitch.m4_fusion.metrics import binary_metrics
from vaaniswitch.m5_experiments.config import ABLATIONS, ExperimentConfig, file_hash
from vaaniswitch.m5_experiments.ledger import Ledger
from vaaniswitch.m5_experiments.reporting import read_json

METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "precision",
    "sensitivity",
    "specificity",
    "roc_auc",
)


def load_outer_results(config: ExperimentConfig) -> list[dict[str, Any]]:
    results = []
    for run in Ledger(config).all_runs():
        if run["state"] == "OUTER_EVALUATED":
            path = config.output(f"experiments/runs/{run['id']}/outer_result.json")
            if file_hash(path) != run["data"]["outer_result_sha256"]:
                raise ValueError("immutable outer evidence checksum changed")
            result = read_json(path)
            if (
                result["identity"] != run["identity"]
                or result["selection"] != run["data"]["selection"]
            ):
                raise ValueError("outer result provenance mismatch")
            predictions = result["predictions"]
            recomputed = binary_metrics(
                [p["label"] for p in predictions], [p["probability"] for p in predictions]
            )
            if recomputed != result["metrics"]:
                raise ValueError("outer result metrics disagree with predictions")
            results.append(result)
    return results


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [(r["ablation"], r["outer_fold"]) for r in results]
    if len(keys) != len(set(keys)) or set(keys) != {(a, f) for a in ABLATIONS for f in (1, 2, 3)}:
        raise ValueError("aggregation requires exactly seven ablations across three outer folds")
    report: dict[str, Any] = {
        "aggregation": "unweighted fold mean +/- sample std (ddof=1)",
        "inference": "descriptive fold-wise comparison; three folds establish no significance",
        "ablations": {},
    }
    for ablation in ABLATIONS:
        folds = sorted(
            (r for r in results if r["ablation"] == ablation), key=lambda r: r["outer_fold"]
        )
        summary = {}
        for metric in METRICS:
            values = [r["metrics"][metric] for r in folds]
            valid = [v for v in values if v is not None]
            if any(not math.isfinite(v) for v in valid):
                raise ValueError("non-finite fold metric")
            summary[metric] = {
                "fold_values": values,
                "available_folds": len(valid),
                "mean": statistics.mean(valid) if len(valid) == 3 else None,
                "sample_std": statistics.stdev(valid) if len(valid) == 3 else None,
            }
        matrix = [
            [sum(r["metrics"]["confusion_matrix"][i][j] for r in folds) for j in range(2)]
            for i in range(2)
        ]
        report["ablations"][ablation] = {
            "summary": summary,
            "folds": [
                {k: v for k, v in r.items() if k not in {"predictions", "identity"}} for r in folds
            ],
            "pooled_confusion_matrix_secondary": matrix,
        }
    c2 = report["ablations"]["C2"]["summary"]["macro_f1"]["fold_values"]
    report["c2_macro_f1_differences"] = {}
    for ablation in ABLATIONS[:-1]:
        baseline = report["ablations"][ablation]["summary"]["macro_f1"]["fold_values"]
        differences = [a - b for a, b in zip(c2, baseline, strict=True)]
        report["c2_macro_f1_differences"][ablation] = {
            "fold_differences": differences,
            "positive_folds": sum(d > 0 for d in differences),
            "mean_difference": statistics.mean(differences),
        }
    return report
