"""Speaker policies; preserve each original record's mass when adding variants."""
from collections import Counter
import numpy as np
from .common import require


def weights(partition, policy='B0'):
    partition.check(fitting=True)
    counts = Counter(r.speaker_id for r in partition.records)
    require(policy in ('B0', 'B1', 'B2'), 'Unknown weighting policy')
    if policy in ('B0', 'B1'):
        value = np.array([1 / counts[r.speaker_id] for r in partition.records], dtype=np.float64)
    else:
        median = float(np.median(list(counts.values())))
        value = np.array([np.clip(np.sqrt(median / counts[r.speaker_id]), .5, 2.)
                          for r in partition.records], dtype=np.float64)
    value /= value.mean()
    require(np.isfinite(value).all() and (value > 0).all(), 'Invalid training weights')
    return value


def expanded_weights(partition, original_weights, parent_ids):
    """Total mass stays N ORIGINAL records, preserving LR C under augmentation."""
    partition.check(fitting=True)
    ids = [r.logical_id for r in partition.records]
    require(len(original_weights) == len(ids) and np.isfinite(original_weights).all()
            and (original_weights > 0).all(), 'Invalid base weights')
    require(set(parent_ids) == set(ids), 'Variants lost/invented original records')
    counts = Counter(parent_ids)
    lookup = dict(zip(ids, original_weights, strict=True))
    result = np.array([lookup[i] / counts[i] for i in parent_ids], dtype=np.float64)
    require(np.isclose(result.sum(), original_weights.sum()), 'Augmented objective mass changed')
    return result


def weight_report(records, value):
    require(len(records) == len(value), 'Weight report row mismatch')
    speakers, classes = {}, {'0': 0., '1': 0.}
    for r, w in zip(records, value, strict=True):
        speakers[r.speaker_id] = speakers.get(r.speaker_id, 0.) + float(w)
        classes[str(r.label)] += float(w)
    return {'speakers': speakers, 'classes': classes, 'total': float(value.sum()),
            'min': float(value.min()), 'max': float(value.max()),
            'max_to_min': float(value.max()/value.min()), 'additional_class_weight': None}
