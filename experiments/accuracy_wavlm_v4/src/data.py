"""Verified inherited CORE identities and V4 role/segmentation contracts."""
from dataclasses import dataclass
from math import gcd
import numpy as np
from .common import digest, require, sha
from experiments.accuracy_hubert_v2.src.data import Corpus, Record, safe_original, guard_records, training_weights
from experiments.accuracy_hubert_v3.src.data import development_cv, separated


@dataclass(frozen=True)
class Partition:
    records: tuple
    allowed: tuple
    forbidden: tuple
    role: str
    context: int

    def check(self, fitting=False):
        require(self.role in {'train', 'validation', 'development'}, 'Outer data forbidden in development')
        require(not set(self.allowed) & set(self.forbidden), 'Allowed/forbidden speakers overlap')
        guard_records(self.records, self.allowed, self.forbidden)
        for attr in ('primary_audio_path', 'primary_audio_sha256'):
            require(len({getattr(r, attr) for r in self.records}) == len(self.records), 'Duplicate microphone/audio identity')
        if fitting:
            require(self.role in {'train', 'development'}, 'Fit requires training role')
        return self

    @property
    def identity(self):
        self.check()
        return {'records': [vars(r) for r in self.records], 'allowed': self.allowed,
            'forbidden': self.forbidden, 'role': self.role, 'outer_context': self.context}

    @property
    def labels(self):
        self.check()
        return np.array([r.label for r in self.records], dtype=np.float32)

    def weights(self):
        self.check(fitting=True)
        return training_weights(self.records, self.allowed, self.forbidden)


def partition(corpus, split, speakers, role, records=None):
    return Partition(tuple(records) if records is not None else corpus.select(speakers),
        tuple(speakers), split.test, role, split.fold).check()


def segment(values, rate, seconds):
    from scipy.signal import resample_poly
    values = np.asarray(values, dtype=np.float32)
    require(values.ndim in (1, 2) and len(values) > 0 and rate > 0, 'Empty/invalid audio')
    require(np.isfinite(values).all(), 'Non-finite source waveform')
    if values.ndim == 2:
        values = values.mean(axis=1, dtype=np.float32)
    if rate != 16000:
        divisor = gcd(int(rate), 16000)
        values = resample_poly(values, 16000//divisor, int(rate)//divisor).astype(np.float32)
    require(len(values) >= 400, 'Audio shorter than WavLM receptive field (400 samples); stop without discarding record')
    maximum = int(seconds*16000)
    require(maximum >= 400, 'Segment too short')
    start = max(0, (len(values)-maximum)//2)
    stop = min(len(values), start+maximum)
    result = values[start:stop].copy()
    require(np.isfinite(result).all(), 'Non-finite resampled waveform')
    return result, {'source_samples_16k': len(values), 'start_sample_16k': start,
        'end_sample_16k': stop, 'valid_samples': len(result), 'right_padding_samples': 0,
        'source_rate': rate, 'target_rate': 16000, 'crop': 'center', 'maximum_seconds': seconds}


def audio(record, seconds):
    import soundfile as sf
    path = safe_original(record.primary_audio_path)
    require(sha(path) == record.primary_audio_sha256, 'Source checksum: '+record.logical_id)
    values, rate = sf.read(path, dtype='float32', always_2d=True)
    require(rate == record.sample_rate and len(values) == record.frame_count, 'Frozen audio header differs')
    return segment(values, rate, seconds)


def audit(corpus):
    result = {'corpus': corpus.audit(), 'development_cv': {}}
    for split in corpus.splits:
        pairs = development_cv(corpus, split)
        result['development_cv'][str(split.fold)] = [{'train': list(t), 'validation': list(v)} for t, v in pairs]
        for t, v in pairs:
            partition(corpus, split, t, 'train')
            partition(corpus, split, v, 'validation')
            separated(corpus.select(t), corpus.select(v), corpus.select(split.test))
    return result
