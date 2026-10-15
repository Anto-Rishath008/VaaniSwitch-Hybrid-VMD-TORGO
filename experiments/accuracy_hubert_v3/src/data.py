"""Inherited immutable CORE manifest plus speaker-grouped development partitions."""
from dataclasses import dataclass
import numpy as np
from .common import digest, require
from experiments.accuracy_hubert_v2.src.data import Corpus, Record, guard_records, safe_original


def separated(*groups):
    for attribute in ('speaker_id', 'logical_id', 'primary_audio_path', 'primary_audio_sha256'):
        sets = [{getattr(r, attribute) for r in group} for group in groups]
        require(all(not sets[i] & sets[j] for i in range(len(sets)) for j in range(i)),
                f'Cross-partition leakage: {attribute}')


@dataclass(frozen=True)
class Partition:
    records: tuple
    features: np.ndarray
    allowed: tuple
    forbidden: tuple
    role: str
    evidence_kind: str = 'REAL_DEVELOPMENT'

    def check(self, fitting=False):
        guard_records(self.records, self.allowed, self.forbidden)
        require(self.role in {'train', 'validation', 'development', 'calibration', 'assessment'},
                'Outer partition forbidden in model selection')
        if fitting:
            require(self.role in {'train', 'development'}, 'Fitting requires training role')
        require(self.features.ndim == 2 and len(self.features) == len(self.records)
                and np.isfinite(self.features).all(), 'Feature shape/finiteness')

    @property
    def labels(self):
        self.check()
        return np.asarray([r.label for r in self.records], dtype=np.float32)

    @property
    def identity(self):
        return {'ids': [r.logical_id for r in self.records], 'allowed': list(self.allowed),
                'forbidden': list(self.forbidden), 'role': self.role, 'evidence_kind': self.evidence_kind}


def development_cv(corpus, split):
    labels = {r.speaker_id: r.label for r in corpus.select(split.development)}
    buckets = [[], [], []]
    for label in (0, 1):
        speakers = sorted((s for s in split.development if labels[s] == label),
                          key=lambda s: (digest(['v3-cv', 42, split.fold, label, s]), s))
        require(len(speakers) >= 3, 'Insufficient development speakers for grouped CV')
        for i, s in enumerate(speakers):
            buckets[i % 3].append(s)
    pairs = []
    for val in buckets:
        train = tuple(sorted(set(split.development) - set(val)))
        val = tuple(sorted(val))
        separated(corpus.select(train), corpus.select(val), corpus.select(split.test))
        require({labels[s] for s in train} == {labels[s] for s in val} == {0, 1}, 'CV class missing')
        pairs.append((train, val))
    require(sorted(s for _, val in pairs for s in val) == sorted(split.development), 'CV coverage')
    return pairs


def partition(corpus, speakers, split, role, cache, layer, records=None):
    selected = tuple(records) if records is not None else corpus.select(speakers)
    guard_records(selected, speakers, split.test)
    values = np.stack([cache.load(r)[(6, 9, 12).index(layer)] for r in selected])
    result = Partition(selected, values, tuple(speakers), split.test, role,
                       getattr(corpus, 'evidence_kind', 'REAL_DEVELOPMENT'))
    result.check()
    return result
