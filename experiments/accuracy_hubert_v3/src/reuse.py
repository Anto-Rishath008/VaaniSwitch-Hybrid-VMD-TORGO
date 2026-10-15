"""Verify sealed V2 lineage without calling any V2 writer or launcher."""
import numpy as np
from .common import ROOT, V1, V2, digest, read, require, sha, publish
from experiments.accuracy_hubert_v2.src.reuse import acquire_readonly, reference_cache
from experiments.accuracy_hubert_v2.src.features import validate_cache


def verify_lineage():
    lineage = read(ROOT/'V2_LINEAGE.json')
    for relative, checksum in lineage['v2_artifacts'].items():
        require(sha(V2/relative) == checksum, f'V2 frozen artifact changed: {relative}')
    frozen = read(V2/'logs/frozen_run.json')
    for name, checksum in frozen['source_files'].items():
        require(sha(V2/name) == checksum, f'V2 frozen source changed: {name}')
    for directory in sorted((V2/'results/outer').iterdir()):
        seal = read(directory/'seal.json')
        require(seal['status'] == 'SEALED' and seal['identity'] == read(directory/'claim.json')['identity']
                and seal['result_sha256'] == sha(directory/'outer_result.json'), 'V2 outer seal invalid')
        require(seal['identity']['freeze_sha256'] == sha(V2/'results/FROZEN_DECISIONS.json'),
                'V2 decision binding changed')
        checkpoint = seal['identity']['checkpoint']
        require(sha(checkpoint['path']) == checkpoint['sha256'], 'V2 selected checkpoint changed')
    require(len(list((V2/'results/outer').iterdir())) == 9, 'Expected nine V2 results')
    for path, checksum in read(V2/'logs/stages/reports.json')['artifacts'].items():
        require(sha(path) == checksum, 'V2 report artifact changed')
    directory, lock = acquire_readonly(frozen['config'])
    return frozen, directory, lock


class VerifiedMemoryCache:
    """Snapshot of checksum-validated frozen entries, no writable cache API."""
    def __init__(self, source, records):
        self.source, self.identity = source, source.identity
        self.values = {}
        self.record_identities = {}
        for r in records:
            values = source.load(r)
            values.flags.writeable = False
            self.values[r.logical_id] = values
            self.record_identities[r.logical_id] = r

    def load(self, record):
        require(self.record_identities.get(record.logical_id) == record, 'Original cache record identity mismatch')
        return self.values[record.logical_id]

    def save(self, *args, **kwargs):
        raise RuntimeError('Original feature cache is read-only')


def prepare_reuse(corpus, output):
    frozen, directory, lock = verify_lineage()
    cache = reference_cache(frozen['config'], lock)
    expected_ids = {r.logical_id for r in corpus.records}
    actual_ids = {p.parent.relative_to(cache.root).as_posix() for p in cache.root.rglob('complete.json')}
    require(expected_ids == actual_ids, 'Original feature inventory changed')
    validation = validate_cache(corpus.records, cache)
    require(validation == read(V2/'logs/feature_validation.json') == read(V1/'logs/feature_validation.json'),
            'Original feature provenance/checksums changed')
    for r in corpus.records:
        n = r.frame_count
        start = max(0, (n-128000)//2)
        expected = {'source_samples_16k': n, 'start_sample_16k': start,
                    'end_sample_16k': min(n, start+128000), 'cropped': n > 128000,
                    'valid_samples': min(n, 128000), 'right_padding_samples': 0,
                    'source_rate': 16000, 'target_rate': 16000}
        require(r.sample_rate == 16000 and read(cache.location(r)/'complete.json')['segment'] == expected,
                'Original crop preprocessing differs')
    report = {'status': 'PASS', 'count': len(corpus.records), 'new_original_extractions': 0,
              'validation_sha256': digest(validation), 'cache_identity': cache.identity,
              'cache_root': str(cache.root), 'cropped': validation['cropped_recordings'],
              'access': 'read-only; all payload/per-layer hashes checked; no V2 writer called'}
    publish(output/'feature_reuse.json', report)
    return VerifiedMemoryCache(cache, corpus.records), directory, frozen['config'], lock
