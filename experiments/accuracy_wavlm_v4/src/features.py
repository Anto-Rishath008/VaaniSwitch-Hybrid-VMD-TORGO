"""V4-only immutable frozen embeddings; role checks precede audio access."""
import io
import numpy as np
import torch
from .common import ROOT, atomic, digest, inside, publish, read, require, sha, versions
from .data import audio
from .model import inputs, load_backbone, pool, feature_lengths
from .numerics import finite


class Cache:
    def __init__(self, root, identity):
        self.identity = identity
        self.root = inside(root)/digest(identity)

    def expected(self, record):
        return {'record': vars(record), 'extraction': self.identity}

    def path(self, record):
        parts = record.logical_id.split('/')
        require(len(parts) == 3 and all(p not in ('', '.', '..') for p in parts), 'Unsafe record ID')
        return inside(self.root/record.logical_id)

    def load(self, record):
        root = self.path(record)
        if not (root/'complete.json').exists():
            return None
        doc = read(root/'complete.json')
        require(doc['identity'] == self.expected(record), 'Stale cache identity')
        require(sha(root/'features.npy') == doc['sha256'], 'Corrupt embedding')
        values = np.load(root/'features.npy', allow_pickle=False)
        require(values.dtype == np.float32 and values.shape == (2, 1536), 'Embedding dimensions/dtype')
        return finite(values, 'cached_features')

    def save(self, record, values, segment):
        require(values.dtype == np.float32 and values.shape == (2, 1536), 'Embedding save shape')
        finite(values, 'features_to_save')
        old = self.load(record)
        if old is not None:
            require(np.array_equal(old, values), 'Cannot overwrite completed embedding')
            return
        root = self.path(record)
        stream = io.BytesIO()
        np.save(stream, values, allow_pickle=False)
        # A payload without a completion marker is never reused. Preserve it on interruption.
        if (root/'features.npy').exists():
            require((root/'features.npy').read_bytes() == stream.getvalue(), 'Incomplete cache payload differs; requires review')
        else:
            atomic(root/'features.npy', stream.getvalue())
        publish(root/'complete.json', {'identity': self.expected(record), 'sha256': sha(root/'features.npy'),
            'segment': segment, 'layers': [9, 12], 'pooling': 'mean_plus_population_std'})


class Provider:
    def __init__(self, directory, lock, cfg, root=None):
        self.directory, self.cfg = directory, cfg
        self.cache = Cache(root or ROOT/'cache/features', {'model': lock, 'versions': versions(),
            'source': {n: sha(ROOT/'src'/n) for n in ('features.py', 'data.py', 'model.py', 'numerics.py')},
            'seconds': cfg['frozen_seconds'], 'layers': cfg['layers'], 'precision': 'fp32',
            'crop': 'center', 'padding': 'none_exact_length_batch_one', 'normalize': 'official_processor_do_normalize_false'})
        self.backbone = self.processor = None
        self.generated = 0

    def extract(self, record):
        cached = self.cache.load(record)
        if cached is not None:
            return cached
        if self.backbone is None:
            self.backbone, self.processor, self.loading = load_backbone(self.directory)
            self.backbone.cuda().eval()
        waveform, crop = audio(record, self.cfg['frozen_seconds'])
        values, mask = inputs(self.processor, waveform, 'cuda')
        with torch.inference_mode():
            result = self.backbone(values, attention_mask=mask, output_hidden_states=True)
            lengths = feature_lengths(mask.sum(1), self.backbone.config.conv_kernel, self.backbone.config.conv_stride)
            out = np.stack([pool(result.hidden_states[i], lengths, 'mean_std')[0].cpu().numpy()
                for i in self.cfg['layers']]).astype(np.float32)
        self.cache.save(record, out, crop)
        self.generated += 1
        return out

    def get(self, partition, candidate):
        partition.check()
        values = np.stack([self.extract(r)[self.cfg['layers'].index(candidate['layer'])] for r in partition.records])
        return values[:, :768] if candidate['pooling'] == 'mean' else values

    def close(self):
        import gc
        self.backbone = self.processor = None
        gc.collect()
        torch.cuda.empty_cache()
