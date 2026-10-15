"""Training-only augmented feature cache with pinned V2 extraction semantics."""
import gc
import io
import time
import uuid
import numpy as np
import torch
from .common import ROOT, atomic_bytes, digest, read, require, sha, source_identity, versions, write_json, within_v3
from .augmentation import augment, parameters, choose_records, VERSION
from experiments.accuracy_hubert_v2.src.features import Extractor, audio


class AugmentedCache:
    def __init__(self, root, model_lock, features_config, sources=None):
        self.root = within_v3(root)
        self.identity = {'model_lock': model_lock, 'preprocessing': features_config,
                         'augmentation_version': VERSION, 'source_code': sources or source_identity(),
                         'versions': versions(), 'waveform_order': 'V2 decode-center-crop; channel transform; V2 processor'}

    def expected(self, record):
        return {'record_id': record.logical_id, 'speaker': record.speaker_id, 'label': record.label,
                'audio_path': record.primary_audio_path, 'audio_sha256': record.primary_audio_sha256,
                'augmentation': parameters(record), 'extraction': self.identity}

    def location(self, record):
        return self.root/digest(self.expected(record))

    def load(self, record):
        path = self.location(record)/'complete.json'
        if not path.exists():
            return None
        doc = read(path)
        require(doc['metadata_sha256'] == digest({k: v for k, v in doc.items() if k != 'metadata_sha256'}),
                'Augmented metadata checksum mismatch')
        require(doc['identity'] == self.expected(record) and doc['status'] == 'COMPLETE', 'Augmented identity mismatch')
        require('/' not in doc['filename'] and '\\' not in doc['filename'] and ':' not in doc['filename']
                and doc['filename'] not in ('.', '..'), 'Unsafe augmented payload path')
        payload = path.parent/doc['filename']
        require(sha(payload) == doc['sha256'], 'Augmented feature checksum mismatch')
        values = np.load(payload, allow_pickle=False)
        require(values.dtype == np.float32 and values.shape == (3, 1536) and np.isfinite(values).all(),
                'Augmented feature shape/dtype/finiteness')
        require(doc['layers'] == [6, 9, 12] and doc['layer_hashes'] == [digest(v.tolist()) for v in values],
                'Augmented layer binding mismatch')
        return values

    def save(self, record, values, segment, wave_sha, diagnostics):
        require(self.load(record) is None, 'Never overwrite completed augmentation')
        require(values.dtype == np.float32 and values.shape == (3, 1536) and np.isfinite(values).all(),
                'Invalid augmented feature output')
        directory = self.location(record)
        filename = 'features-'+uuid.uuid4().hex+'.npy'
        buffer = io.BytesIO()
        np.save(buffer, values, allow_pickle=False)
        atomic_bytes(directory/filename, buffer.getvalue())
        metadata = {'status': 'COMPLETE', 'identity': self.expected(record),
                   'filename': filename, 'sha256': sha(directory/filename), 'segment': segment,
                   'waveform_sha256': wave_sha, 'diagnostics': diagnostics, 'layers': [6, 9, 12],
                   'layer_hashes': [digest(v.tolist()) for v in values]}
        write_json(directory/'complete.json', {**metadata, 'metadata_sha256': digest(metadata)})


def bounded_infer(extractor, waves, cap=2):
    """Exact-length batches, halve only on OOM; never skip a waveform."""
    require(waves and len({len(w) for w in waves}) == 1, 'Exact-length batch required')
    output, offset = [], 0
    batch = min(cap, len(waves))
    while offset < len(waves):
        try:
            values = extractor.infer(waves[offset:offset+batch])
        except torch.cuda.OutOfMemoryError:
            if batch == 1:
                raise
            batch = max(1, batch//2)
            gc.collect()
            torch.cuda.empty_cache()
            continue
        require(values.shape == (min(batch, len(waves)-offset), 3, 1536) and values.dtype == np.float32,
                'Extraction output shape/dtype differs from batch; no retry')
        require(np.isfinite(values).all(), 'Non-finite extraction; no retry')
        output.extend(values)
        offset += len(values)
    return np.stack(output)


class AugmentProvider:
    def __init__(self, cache, checkpoint, cfg):
        self.cache, self.checkpoint, self.cfg = cache, checkpoint, cfg
        self.extractor = None
        self.generated = 0
        self.hits = 0
        self.extraction_seconds = 0.
        self.access = []

    def get(self, partition, layer, budget=16):
        partition.check(fitting=True)
        records = choose_records(partition, budget)
        self.access.append({'partition': partition.identity, 'augmented_ids': [r.logical_id for r in records]})
        values = []
        for r in records:
            value = self.cache.load(r)
            if value is None:
                if self.extractor is None:
                    self.extractor = Extractor(self.cfg, self.checkpoint)
                wave, segment = audio(r)  # verifies source SHA before read
                changed, diagnostics = augment(wave, parameters(r))
                import hashlib
                started = time.perf_counter()
                value = bounded_infer(self.extractor, [changed], cap=1)[0]
                torch.cuda.synchronize()
                self.extraction_seconds += time.perf_counter()-started
                self.cache.save(r, value, segment, hashlib.sha256(changed.tobytes()).hexdigest(), diagnostics)
                self.generated += 1
            else:
                self.hits += 1
            values.append(value[(6, 9, 12).index(layer)])
        return records, np.stack(values)

    def close(self):
        if self.extractor is not None:
            self.extractor.close()
            self.extractor = None
