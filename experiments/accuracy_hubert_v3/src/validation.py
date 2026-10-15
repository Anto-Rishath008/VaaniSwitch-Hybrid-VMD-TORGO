"""Bounded actual-GPU validation; never calls develop or evaluate_frozen."""
import random
import time
import uuid
import numpy as np
import torch
from .common import ROOT, digest, publish, require, sha, source_identity, write_json
from .data import partition, separated, safe_original
from .augmentation import augment, parameters
from .features import AugmentedCache, AugmentProvider, Extractor, audio
from .selection import candidate
from .training import fit, predict, load_checkpoint, rng_state, restore_rng
from experiments.accuracy_hubert_v2.src.training import Partition as V2Partition, logistic_fit


def bounded_validation(corpus, cache, checkpoint, inherited_cfg, lock, cfg, env):
    started = time.perf_counter()
    namespace = ROOT/'validation'/('cuda-'+uuid.uuid4().hex)
    namespace.mkdir(parents=True)
    sources = source_identity()
    identity = digest({'namespace': str(namespace), 'sources': sources, 'bounded_validation_only': True})
    split = corpus.splits[0]
    training = []
    for speaker in split.train:
        available = corpus.select([speaker])
        near = min(available, key=lambda r: (abs(r.duration_seconds-1), r.logical_id))
        longest = max(available, key=lambda r: (r.duration_seconds, r.logical_id))
        require(near != longest, 'Smoke needs two distinct training records per speaker')
        training.extend([near, longest])
    validation = []
    for speaker in split.validation:
        available = sorted(corpus.select([speaker]), key=lambda r: (abs(r.duration_seconds-1), r.logical_id))
        validation.extend(available[:2])
    train = partition(corpus, split.train, split, 'train', cache, 9, training)
    val = partition(corpus, split.validation, split, 'validation', cache, 9, validation)
    separated(train.records, val.records, corpus.select(split.test))
    before = {r.logical_id: sha(safe_original(r.primary_audio_path))
              for r in training+validation}
    acache = AugmentedCache(namespace/'augmented', lock, inherited_cfg['features'])
    provider = AugmentProvider(acache, checkpoint, inherited_cfg)
    torch.cuda.reset_peak_memory_stats()
    try:
        provider.extractor = Extractor(inherited_cfg, checkpoint)
        longest = max(training, key=lambda r: r.duration_seconds)
        wave, _ = audio(longest)
        augmented_wave, _ = augment(wave, parameters(longest))
        benchmark = provider.extractor.benchmark([augmented_wave])
        augmented = provider.get(train, 9, budget=1)
        generated = provider.generated
        again = provider.get(train, 9, budget=1)
        require(generated == provider.generated == 8 and np.array_equal(augmented[1], again[1]), 'Augmented cache reuse failed')
        require(np.isfinite(augmented[1]).all(), 'Non-finite actual augmented features')
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
    finally:
        provider.close()
    require(peak_reserved <= env['gpu_total_bytes']*.8, 'Measured smoke exceeds GPU memory cap')
    baseline = fit(train, val, candidate(9, .1), cfg, namespace/'baseline', identity)
    payload = load_checkpoint(baseline['checkpoint'], baseline['identity'])
    # Execute unchanged V2 LR code on the identical SMALL development panel, writing only V3.
    old_train = V2Partition(train.records, train.features, train.allowed, train.forbidden, 'inner_train')
    old_val = V2Partition(val.records, val.features, val.allowed, val.forbidden, 'inner_validation')
    old = logistic_fit(old_train, old_val, .1, namespace/'v2_recipe_parity', identity, 9, inherited_cfg)
    old_payload = load_checkpoint(old['checkpoint'], identity)
    require(np.array_equal(payload['model'].coef_, old_payload['model'].coef_)
            and np.array_equal(payload['model'].intercept_, old_payload['model'].intercept_), 'V2 LR recipe parity failure')
    for spec in (candidate(9, .1, weighting='B2'), candidate(9, .1, normalization='N1'),
                 candidate(9, .1, normalization='N2'), candidate(9, .1, augmentation=True)):
        fit(train, val, spec, cfg, namespace/'panels'/spec['id'], identity,
            augmented=augmented if spec['augmentation'] else None)
    expected = predict(payload, val.features)
    recovered = {}
    for failure in ('after_start', 'after_checkpoint'):
        directory = namespace/failure
        try:
            fit(train, val, candidate(9, .1), cfg, directory, identity, interrupt=failure)
        except InterruptedError:
            pass
        else:
            raise AssertionError('Expected controlled interruption')
        result = fit(train, val, candidate(9, .1), cfg, directory, identity)
        state = load_checkpoint(result['checkpoint'], result['identity'])
        require(np.array_equal(expected, predict(state, val.features)), 'Interrupted fit recovery differs')
        require(fit(train, val, candidate(9, .1), cfg, directory, identity) == result, 'Completed fit not reused')
        recovered[failure] = 'PASS: exact prediction equality and completed-fit reuse'
    saved = rng_state()
    def draws():
        return random.random(), np.random.rand(), torch.rand(3), torch.rand(3, device='cuda').cpu()
    first = draws()
    restore_rng(saved)
    second = draws()
    require(first[0] == second[0] and first[1] == second[1] and torch.equal(first[2], second[2])
            and torch.equal(first[3], second[3]), 'RNG restoration differs')
    require(before == {r.logical_id: sha(safe_original(r.primary_audio_path)) for r in training+validation},
            'Actual development audio changed')
    require(source_identity() == sources, 'Source changed during smoke')
    report = {'status': 'PASS', 'namespace': str(namespace), 'kind': 'BOUNDED_REAL_DEVELOPMENT_VALIDATION',
              'source_files': sources, 'environment': env, 'train_speakers': list(split.train),
              'validation_speakers': list(split.validation), 'forbidden_outer_speakers': list(split.test),
              'training_records': [r.logical_id for r in training], 'validation_records': [r.logical_id for r in validation],
              'train_count': len(training), 'validation_count': len(validation), 'real_outer_forwards': 0,
              'new_original_features': 0, 'augmented_features': generated, 'augmentation_shape': [generated, 3, 1536],
              'augmentation_profiles': sorted({parameters(r)['profile'] for r in augmented[0]}),
              'extraction_seconds': provider.extraction_seconds, 'benchmark': benchmark,
              'peak_allocated_bytes': peak_allocated, 'peak_reserved_bytes': peak_reserved,
              'V2_LR_coefficient_parity': 'EXACT', 'recovery': recovered, 'rng_restoration': 'PASS',
              'finite_numerical_state': 'PASS', 'source_audio_sha_immutability': 'PASS',
              'variant_access': provider.access, 'seconds': time.perf_counter()-started,
              'full_training_executed': False, 'v3_accuracy': None,
              'mlp': 'Not implemented; no development-only justification',
              'note': 'Small panel metrics inside checkpoints are diagnostics, not final V3 accuracy'}
    publish(namespace/'REPORT.json', report)
    write_json(ROOT/'validation/latest_cuda.json', {'path': str(namespace/'REPORT.json'),
               'sha256': sha(namespace/'REPORT.json')}, replace=True)
    return report
