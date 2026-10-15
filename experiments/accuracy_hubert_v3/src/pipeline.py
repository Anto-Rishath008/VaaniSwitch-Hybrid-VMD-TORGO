"""Explicit prepare/validate/develop/authorized-outer entry points; no implicit full run."""
import os
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.dont_write_bytecode = True
import argparse
from pathlib import Path
import time
import traceback
import uuid
import numpy as np
from .common import (ROOT, TITLE, ProcessLock, config, digest, environment, progress,
                     publish, read, require, sha, source_identity, verify_preservation, write_json)
from .data import Corpus, development_cv
from .reuse import prepare_reuse, verify_lineage
from .features import AugmentedCache, AugmentProvider
from .selection import select_fold
from .training import load_checkpoint, predict, prediction_rows, metrics
from .reporting import development_report, final_report, historical_baselines
from experiments.accuracy_hubert_v2.src.training import seal_outer
from experiments.accuracy_hubert_v2.src.pipeline import Tee


def freeze_identity(cfg, env):
    return {'source_files': source_identity(), 'config': cfg, 'versions': env['versions'],
            'gpu': env['gpu'], 'study': TITLE, 'lineage_sha256': sha(ROOT/'V2_LINEAGE.json')}


def assert_frozen(frozen):
    require(source_identity() == frozen['identity']['source_files'], 'Frozen V3 source identity mismatch')
    require(config() == frozen['identity']['config'], 'Frozen V3 configuration mismatch')
    require(len(frozen['selections']) == 3
            and [s['outer_fold'] for s in frozen['selections']] == [1, 2, 3], 'All three decisions must freeze first')
    for selection in frozen['selections']:
        for result in selection['final'].values():
            reference = result['checkpoint']
            require(sha(reference['path']) == reference['sha256'], 'Frozen final checkpoint changed')


def evaluate_frozen(corpus, cache, frozen, output, authorize_outer=False):
    require(authorize_outer is True, 'Explicit real V3 outer authorization required')
    assert_frozen(frozen)
    freeze_path = output/'FROZEN_DECISIONS.json'
    require(read(freeze_path) == frozen and read(output/'FROZEN_DECISIONS.seal.json')['sha256'] == sha(freeze_path),
            'Frozen decision file/seal mismatch')
    results = []
    for split, selection in zip(corpus.splits, frozen['selections'], strict=True):
        reference_rows = None
        for name in ('reference', 'selected'):
            assert_frozen(frozen)
            result = selection['final'][name]
            identity = {'frozen_decisions': digest(frozen), 'checkpoint': result['checkpoint'],
                        'fold': split.fold, 'model': name, 'threshold': .5}
            def infer():
                nonlocal reference_rows
                # First outer fitting-independent feature use/forward occurs only inside durable claim.
                records = corpus.select(split.test)
                same = name == 'selected' and result['checkpoint'] == selection['final']['reference']['checkpoint']
                started = time.perf_counter()
                if same:
                    require(reference_rows is not None, 'Reference must be sealed before deduplicated selected panel')
                    rows = reference_rows
                else:
                    payload = load_checkpoint(result['checkpoint'], result['identity'])
                    x = np.stack([cache.load(r)[(6, 9, 12).index(result['candidate']['layer'])] for r in records])
                    rows = prediction_rows(records, predict(payload, x),
                                           getattr(corpus, 'evidence_kind', 'REAL_SEALED_OUTER'))
                return {'model': name, 'outer_fold': split.fold, 'identity': identity,
                        'candidate': result['candidate'], 'predictions': rows,
                        'metrics': metrics([r['label'] for r in rows], [r['probability'] for r in rows]),
                        'parameter_count': result['parameter_count'], 'training_seconds': result['training_seconds'],
                        'training_time_scope': 'final fit only; search totals in final report runtime_evidence',
                        'evaluation_seconds': time.perf_counter()-started,
                        'deduplicated_from_reference': same, 'study': TITLE}
            value = seal_outer(output/'outer'/f'fold{split.fold}_{name}', identity, infer)
            if name == 'reference':
                reference_rows = value['predictions']
            results.append(value)
    return results


def develop(corpus, cache, checkpoint, inherited_cfg, lock, cfg, env):
    identity = freeze_identity(cfg, env)
    publish(ROOT/'logs/frozen_run.json', identity)
    run_id = digest(identity)
    freeze_path = ROOT/'results/FROZEN_DECISIONS.json'
    if freeze_path.exists():
        frozen = read(freeze_path)
        require(frozen['identity'] == identity, 'Frozen run identity differs on resume')
        assert_frozen(frozen)
        seal_path = ROOT/'results/FROZEN_DECISIONS.seal.json'
        if not seal_path.exists():
            require(not list((ROOT/'results/outer').rglob('claim.json')),
                    'Missing decision seal with consumed outer claims requires review')
            publish(seal_path, {'sha256': sha(freeze_path), 'run_identity': run_id})
        require(read(seal_path)['sha256'] == sha(freeze_path), 'Frozen decision seal mismatch on resume')
        development_report(frozen['selections'], ROOT/'results')
        return frozen
    augmented = AugmentedCache(ROOT/'cache/augmented', lock, inherited_cfg['features'])
    provider = AugmentProvider(augmented, checkpoint, inherited_cfg)
    selections = []
    attempt_started = time.perf_counter()
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        for split in corpus.splits:
            progress(f'Development only: outer-fold context {split.fold}; no outer inference')
            require(source_identity() == identity['source_files'], 'Sources changed during development')
            selections.append(select_fold(corpus, split, cache, provider, cfg,
                                          ROOT/'checkpoints'/f'fold{split.fold}', run_id))
            if cfg['calibration']:
                from .calibration import diagnostic
                diagnostic(corpus, split, cache, cfg, ROOT/'experiments/calibration'/f'fold{split.fold}', run_id)
    finally:
        provider.close()
        write_json(ROOT/'logs'/('augmentation-access-'+uuid.uuid4().hex+'.json'),
                   {'run_identity': run_id, 'partition_receipts': provider.access, 'generated': provider.generated,
                    'cache_hits': provider.hits, 'extraction_seconds': provider.extraction_seconds,
                    'development_attempt_seconds': time.perf_counter()-attempt_started,
                    'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                    'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0})
    receipts = [read(p) for p in sorted((ROOT/'logs').glob('augmentation-access-*.json'))]
    receipts = [r for r in receipts if r['run_identity'] == run_id]
    runtime = {'development_attempt_seconds_sum': sum(r['development_attempt_seconds'] for r in receipts),
               'augmented_extraction_seconds_sum': sum(r['extraction_seconds'] for r in receipts),
               'new_augmented_entries_sum': sum(r['generated'] for r in receipts),
               'gpu_peak_allocated_bytes': max(r['gpu_peak_allocated_bytes'] for r in receipts),
               'gpu_peak_reserved_bytes': max(r['gpu_peak_reserved_bytes'] for r in receipts),
               'fit_seconds_successful_checkpoints': sum(read(p)['training_seconds']
                                                         for p in (ROOT/'checkpoints').rglob('complete.json')),
               'timing_scope': 'Development attempts including checkpoint/selection overhead; excludes initial integrity verification. Failed attempts count when receipts survive; hard-killed interval may be missing.'}
    frozen = {'identity': identity, 'threshold': .5, 'selections': selections, 'runtime_evidence': runtime,
              'outer_authorization': 'separate explicit switch required'}
    assert_frozen(frozen)
    publish(ROOT/'results/FROZEN_DECISIONS.json', frozen)
    publish(ROOT/'results/FROZEN_DECISIONS.seal.json',
            {'sha256': sha(ROOT/'results/FROZEN_DECISIONS.json'), 'run_identity': run_id})
    development_report(selections, ROOT/'results')
    return frozen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'validate', 'develop', 'outer'))
    parser.add_argument('--authorize-outer', action='store_true')
    args = parser.parse_args(argv)
    require(args.mode != 'outer' or args.authorize_outer, 'Explicit --authorize-outer required')
    require(not args.authorize_outer or args.mode in ('develop', 'outer'), 'Authorization mode mismatch')
    (ROOT/'logs').mkdir(exist_ok=True)
    log = ROOT/'logs'/(args.mode+'-'+uuid.uuid4().hex+'.log')
    with ProcessLock(ROOT/'logs/process.lock'), log.open('x', encoding='utf-8') as stream:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(old_out, stream), Tee(old_err, stream)
        try:
            cfg = config()
            env = environment(cfg)
            corpus = Corpus()
            progress('CUDA/environment and frozen dataset PASS')
            if args.mode in ('develop', 'outer'):
                progress('Verifying complete protected inventories before production work')
                verify_preservation()
            cache, checkpoint, inherited_cfg, lock = prepare_reuse(corpus, ROOT/'validation')
            progress('All 8673 original features and V2 sealed lineage PASS; no re-extraction')
            audit = {'dataset': corpus.audit(), 'cv': {str(s.fold): development_cv(corpus, s) for s in corpus.splits},
                     'source_files': source_identity(), 'config': cfg, 'environment': env}
            write_json(ROOT/'validation'/('audit-'+uuid.uuid4().hex+'.json'), audit)
            if args.mode == 'prepare':
                progress('Preparation complete; no classifier training or outer inference')
            elif args.mode == 'validate':
                from .validation import bounded_validation
                report = bounded_validation(corpus, cache, checkpoint, inherited_cfg, lock, cfg, env)
                progress('Bounded development CUDA validation PASS: '+report['namespace'])
            else:
                if args.mode == 'develop':
                    frozen = develop(corpus, cache, checkpoint, inherited_cfg, lock, cfg, env)
                else:
                    require((ROOT/'results/FROZEN_DECISIONS.json').exists(), 'Complete development and freeze first')
                    frozen = read(ROOT/'results/FROZEN_DECISIONS.json')
                    require(frozen['identity'] == freeze_identity(cfg, env), 'Frozen environment/config/source changed')
                if args.authorize_outer:
                    verify_lineage()
                    results = evaluate_frozen(corpus, cache, frozen, ROOT/'results', authorize_outer=True)
                    final_report(corpus, results, frozen['selections'], frozen, ROOT/'results', ROOT/'figures', historical_baselines())
                    progress('Explicitly authorized exploratory outer comparison sealed')
                else:
                    progress('DEVELOPMENT COMPLETE; decisions frozen. Outer inference remains unauthorized and unexecuted.')
            if args.mode in ('develop', 'outer'):
                progress('Verifying complete protected inventories after production work')
                verify_preservation()
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            sys.stdout, sys.stderr = old_out, old_err


if __name__ == '__main__':
    main()
