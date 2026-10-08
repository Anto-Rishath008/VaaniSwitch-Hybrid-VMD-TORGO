"""Inner-only lexicographic model selection and independent macro-F1 patience tracking."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from vaaniswitch.m5_experiments.config import digest


@dataclass(frozen=True)
class InnerResult:
    experiment_id: str
    candidate_id: str
    split_hash: str
    validation_record_hash: str
    epoch: int
    metrics: dict[str, Any]
    trainable_parameters: int
    role: str = "inner_validation"

    @property
    def identity(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def selection_hash(self) -> str:
        return digest(self.identity)


def metric_key(metrics: dict[str, Any], trainable_parameters: int) -> tuple[float, ...]:
    values = [metrics.get(k) for k in ("macro_f1", "balanced_accuracy", "roc_auc", "loss")]
    if any(
        not isinstance(v, (float, int)) or isinstance(v, bool) or not math.isfinite(v)
        for i, v in enumerate(values)
        if i != 2 or v is not None
    ):
        raise ValueError("selection metrics must be finite; only AUC may be unavailable")
    if any(not 0 <= metrics[k] <= 1 for k in ("macro_f1", "balanced_accuracy")):
        raise ValueError("invalid selection score")
    auc = metrics["roc_auc"]
    if auc is not None and not 0 <= auc <= 1:
        raise ValueError("invalid AUC")
    return (
        float(metrics["macro_f1"]),
        float(metrics["balanced_accuracy"]),
        float(auc) if auc is not None else -1.0,
        -float(metrics["loss"]),
        -float(trainable_parameters),
    )


def choose_candidate(
    results: list[InnerResult],
    *,
    split_hash: str,
    validation_record_hash: str,
    candidate_ids: set[str],
) -> InnerResult:
    if (
        not results
        or {r.candidate_id for r in results} != candidate_ids
        or len(results) != len(candidate_ids)
    ):
        raise ValueError("all predefined candidates must complete exactly once")
    for result in results:
        if (
            result.role != "inner_validation"
            or result.split_hash != split_hash
            or result.validation_record_hash != validation_record_hash
            or type(result.epoch) is not int
            or not 1 <= result.epoch <= 30
        ):
            raise ValueError("selection accepts matching inner-validation evidence only")
    # Stable ID is a deterministic last fallback, never a scientific tie-breaker.
    return max(
        sorted(results, key=lambda r: r.candidate_id, reverse=True),
        key=lambda r: metric_key(r.metrics, r.trainable_parameters),
    )
