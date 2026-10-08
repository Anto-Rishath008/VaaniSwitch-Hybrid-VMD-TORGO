"""Binary validation metrics with explicit undefined cases and tie-aware rank AUC."""

from collections.abc import Sequence
from typing import Any

import numpy as np


def binary_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    if threshold != 0.5:
        raise ValueError("Module 4 threshold is fixed at 0.5")
    if not labels or len(labels) != len(probabilities) or any(x not in (0, 1) for x in labels):
        raise ValueError("metrics require equal nonempty sequences and binary labels")
    scores = np.asarray(probabilities, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("probabilities must be finite values in [0,1]")
    positive = np.asarray(labels) == 1
    predicted = scores >= threshold
    tp, tn = int((positive & predicted).sum()), int((~positive & ~predicted).sum())
    fp, fn = int((~positive & predicted).sum()), int((positive & ~predicted).sum())
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1_positive = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f1_negative = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    recalls = [value for value in (sensitivity, specificity) if value is not None]
    auc = None
    if positive.any() and (~positive).any():
        order = np.argsort(scores, kind="stable")
        ranks = np.empty(len(scores), dtype=np.float64)
        start = 0
        while start < len(scores):
            stop = start + 1
            while stop < len(scores) and scores[order[stop]] == scores[order[start]]:
                stop += 1
            ranks[order[start:stop]] = (start + 1 + stop) / 2
            start = stop
        n_positive = int(positive.sum())
        auc = float(
            (ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
            / (n_positive * (len(scores) - n_positive))
        )
    return {
        "record_count": len(labels),
        "threshold": threshold,
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": sum(recalls) / len(recalls),
        "macro_f1": (f1_positive + f1_negative) / 2,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "roc_auc": auc,
        "roc_auc_unavailable_reason": "only one true class present" if auc is None else None,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "undefined_policy": "absent-class recall null; balanced accuracy over present classes; "
        "macro-F1 over fixed labels [0,1], zero division gives 0",
    }
