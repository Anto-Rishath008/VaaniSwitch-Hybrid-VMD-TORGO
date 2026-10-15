"""Bounded real development audio/model CUDA checks, never a full study."""
import copy
import time
import uuid
import numpy as np
import torch
from .common import ROOT, digest, load_checkpoint, publish, require, sha, sources, write
from .data import audio, partition, separated, safe_original
from .features import Provider
from .model import inputs, load_backbone
from .numerics import finite
from .selection import candidates
from .training import (fit_lr, fit_neural, memory, predict_lr, predict_neural,
    rng_state, restore_rng)


def bounded(corpus, directory, model_lock, cfg, env):
    root = ROOT/'validation'/('cuda-'+uuid.uuid4().hex)
    root.mkdir(parents=True)
    started = time.perf_counter()
    identity = digest({'root': str(root), 'sources': sources(), 'model': model_lock, 'bounded': True})
    split = corpus.splits[0]
    speakers = []
    for label in (0, 1):
        available = [s for s in split.train if corpus.select([s])[0].label == label]
        speakers.extend(sorted(available, key=lambda s: digest(['v4-smoke', s]))[:2])
    training = []
    for speaker in speakers:
        records = corpus.select([speaker])
        short = min(records, key=lambda r: (abs(r.duration_seconds-1), r.logical_id))
        longest = max(records, key=lambda r: (r.duration_seconds, r.logical_id))
        require(short != longest, 'Smoke panel needs distinct recording lengths')
        training.extend((short, longest))
    validation = [min(corpus.select([s]), key=lambda r: (abs(r.duration_seconds-1), r.logical_id)) for s in split.validation]
    train = partition(corpus, split, speakers, 'train', training)
    val = partition(corpus, split, split.validation, 'validation', validation)
    separated(train.records, val.records, corpus.select(split.test))
    audio_before = {r.logical_id: sha(safe_original(r.primary_audio_path)) for r in training+validation}
    for r in corpus.records:
        require(round(r.frame_count*16000/r.sample_rate) >= 400, 'CORE has sub-receptive-field recording')
    provider = Provider(directory, model_lock, cfg, root/'features')
    torch.cuda.reset_peak_memory_stats()
    frozen, neural = candidates(cfg)
    candidate = frozen[0]
    x, vx = provider.get(train, candidate), provider.get(val, candidate)
    require(provider.generated == 10, 'Expected exactly ten bounded extraction records')
    require(np.array_equal(provider.get(train, candidate), x) and provider.generated == 10, 'Cache reuse differs')
    frozen_memory = memory()
    loading_info = provider.loading
    frozen_parameters = sum(p.numel() for p in provider.backbone.parameters())
    require(all(not p.requires_grad and p.grad is None for p in provider.backbone.parameters()), 'Frozen encoder gradient')
    provider.close()
    baseline = fit_lr(train, val, x, vx, candidate, cfg, root/'frozen_baseline', identity)
    baseline_payload = load_checkpoint(baseline['checkpoint'], baseline['identity'])
    expected = predict_lr(baseline_payload, vx)
    lr_recovery = {}
    for stop in ('after_start', 'after_checkpoint'):
        destination = root/('lr_'+stop)
        try:
            fit_lr(train, val, x, vx, candidate, cfg, destination, identity, interrupt=stop)
        except InterruptedError:
            pass
        else:
            raise AssertionError('Missing controlled LR interruption')
        fit = fit_lr(train, val, x, vx, candidate, cfg, destination, identity)
        actual = predict_lr(load_checkpoint(fit['checkpoint'], fit['identity']), vx)
        require(np.array_equal(expected, actual), 'LR recovery prediction drift')
        lr_recovery[stop] = 'PASS exact predictions'
    bounded_cfg = {**cfg, 'max_epochs': 1, 'accumulation_steps': 4, 'checkpoint_every_updates': 1}
    fits = {}
    for spec in neural:
        print('CUDA bounded '+spec['id'], flush=True)
        fits[spec['id']] = fit_neural(train, val, spec, bounded_cfg, directory, root/spec['id'], identity)
    spec = neural[0]
    interrupted_root = root/'interrupted_neural'
    try:
        fit_neural(train, val, spec, bounded_cfg, directory, interrupted_root, identity, interrupt_after_step=1)
    except InterruptedError:
        pass
    else:
        raise AssertionError('Missing deliberate neural interruption')
    recovered = fit_neural(train, val, spec, bounded_cfg, directory, interrupted_root, identity)
    direct = fits[spec['id']]
    a = load_checkpoint(direct['checkpoint'], direct['identity'])
    b = load_checkpoint(recovered['checkpoint'], recovered['identity'])
    require(all(torch.equal(a['model'][k], b['model'][k]) for k in a['model']), 'Resume parameter drift')
    def equal_tree(a, b):
        if isinstance(a, torch.Tensor):
            return torch.equal(a, b)
        if isinstance(a, np.ndarray):
            return np.array_equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
        if isinstance(a, (tuple, list)):
            return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
        return a == b
    require(equal_tree(a['optimizer'], b['optimizer']) and equal_tree(a['scheduler'], b['scheduler'])
        and equal_tree(a['rng'], b['rng']), 'Optimizer/scheduler/RNG resume differs')
    require(direct['validation_predictions'] == recovered['validation_predictions'], 'Resumed predictions differ')
    reloaded = predict_neural(direct, directory, spec, bounded_cfg, val.records)
    require(np.array_equal(reloaded, [r['probability'] for r in direct['validation_predictions']]), 'Checkpoint prediction reload differs')
    require(fit_neural(train, val, spec, bounded_cfg, directory, root/spec['id'], identity) == direct, 'Completed neural fit was not reused')
    import random
    saved_rng = rng_state()
    def draws():
        return random.random(), np.random.rand(), torch.rand(3), torch.rand(3, device='cuda').cpu()
    first = draws()
    restore_rng(saved_rng)
    require(equal_tree(first, draws()), 'RNG roundtrip failed')
    require(audio_before == {r.logical_id: sha(safe_original(r.primary_audio_path)) for r in training+validation}, 'Development audio changed')
    max_allocated = max([frozen_memory['peak_allocated']]+[r['memory']['peak_allocated'] for r in fits.values()]+[recovered['memory']['peak_allocated']])
    max_reserved = max([frozen_memory['peak_reserved']]+[r['memory']['peak_reserved'] for r in fits.values()]+[recovered['memory']['peak_reserved']])
    require(max_reserved <= env['total_vram']*.8, 'GPU allocation exceeds declared allowance')
    report = {'status': 'PASS', 'evidence': 'BOUNDED_DEVELOPMENT_INFRASTRUCTURE_ONLY',
        'sources': sources(), 'environment': env, 'model_lock': model_lock, 'loading_info': loading_info,
        'records': {'train': [r.logical_id for r in train.records], 'validation': [r.logical_id for r in val.records],
            'forbidden_outer_speakers': list(split.test)}, 'audio_sha256': audio_before,
        'frozen_features': {'train_shape': list(x.shape), 'validation_shape': list(vx.shape),
            'generated_records': provider.generated, 'encoder_parameters': frozen_parameters, 'trainable_encoder_parameters': 0,
            'memory': frozen_memory, 'cache_reuse': 'PASS'},
        'neural_fits': fits, 'lr_recovery': lr_recovery,
        'neural_recovery': 'PASS exact trainable parameters, optimizer, scheduler, RNG and predictions; completed fit reuse',
        'resume_fit': recovered, 'rng_roundtrip': 'PASS Python NumPy Torch CPU CUDA',
        'peak_allocated': max_allocated, 'peak_reserved': max_reserved,
        'bounded_config': bounded_cfg, 'wall_seconds': time.perf_counter()-started,
        'full_development_executed': False, 'real_outer_inference_executed': False, 'v4_accuracy': None}
    publish(root/'REPORT.json', report)
    write(ROOT/'validation/latest_cuda.json', {'path': str(root/'REPORT.json'), 'sha256': sha(root/'REPORT.json')}, replace=True)
    print({'status': 'PASS', 'report': str(root/'REPORT.json'), 'peak_allocated_GiB': max_allocated/1024**3,
        'peak_reserved_GiB': max_reserved/1024**3, 'seconds': report['wall_seconds']}, flush=True)
    return report
