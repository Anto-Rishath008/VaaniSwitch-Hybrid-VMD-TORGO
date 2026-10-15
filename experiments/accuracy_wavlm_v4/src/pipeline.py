"""Safe default: bounded validation. Full development and outer require distinct flags."""
import os
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
sys.dont_write_bytecode = True
import argparse
import shutil
import time
import traceback
import uuid
from .common import ROOT, ProcessLock, config, digest, environment, event, publish, read, require, sha, sources
from .data import Corpus, audit
from .features import Provider
from .lineage import prepare as prepare_lineage, verify as verify_lineage
from .model_files import acquire


def require_authorization(mode, authorize_development=False, authorize_outer=False):
    if mode == 'Develop':
        require(authorize_development is True, 'Full development requires separate user approval and -AuthorizeDevelopment')
        require(not authorize_outer, 'Outer must be a separate invocation after development review')
    elif mode == 'Outer':
        require(authorize_outer is True, 'Real outer inference requires separate explicit -AuthorizeOuter')
        require(not authorize_development, 'Do not combine development and outer authorization')
    else:
        require(not authorize_development and not authorize_outer, 'Authorization flags do not belong on bounded modes')


def freeze_run(cfg, env, model_lock):
    return {'sources': sources(), 'versions': env['versions'], 'config': cfg, 'model': model_lock,
        'title': 'V4 POST-HOC EXPLORATORY', 'lineage': sha(ROOT/'V3_LINEAGE.json')}


def develop(corpus, cfg, env, directory, model_lock, authorize_development=False):
    require(authorize_development is True, 'Development not authorized')
    from .selection import select_fold
    from .reporting import development, historical_development_report
    from .evaluation import assert_frozen
    smoke_ref = read(ROOT/'validation/latest_cuda.json')
    require(sha(smoke_ref['path']) == smoke_ref['sha256'], 'Smoke report checksum changed')
    smoke = read(smoke_ref['path'])
    require(smoke['status'] == 'PASS' and smoke['sources'] == sources()
        and smoke['environment']['versions'] == env['versions'] and smoke['model_lock'] == model_lock,
        'Current implementation requires matching bounded CUDA validation')
    require(shutil.disk_usage(ROOT).free >= 85*1024**3, 'Need 85 GiB free planning allowance for retained development checkpoints; do not delete evidence')
    identity = freeze_run(cfg, env, model_lock)
    publish(ROOT/'logs/frozen_run.json', identity)
    run_id = digest(identity)
    path = ROOT/'results/FROZEN_DECISIONS.json'
    if path.exists():
        frozen = read(path)
        assert_frozen(frozen, cfg, model_lock)
        require(not list((ROOT/'results/outer').glob('*/claim.json')) or (ROOT/'results/FROZEN_DECISIONS.seal.json').exists(), 'Missing freeze seal after outer claim')
        publish(ROOT/'results/FROZEN_DECISIONS.seal.json', {'sha256': sha(path)})
        development(frozen['selections'], ROOT/'results')
        historical_development_report(frozen['selections'], ROOT/'results')
        return frozen
    provider = Provider(directory, model_lock, cfg)
    started = time.perf_counter()
    try:
        selections = []
        for split in corpus.splits:
            value = select_fold(corpus, split, provider, directory, cfg,
                ROOT/'experiments'/f'context{split.fold}', run_id)
            publish(ROOT/'experiments'/f'context{split.fold}'/'selection.json', value)
            selections.append(value)
        frozen = {**identity, 'selections': selections, 'development_wall_seconds': time.perf_counter()-started}
        publish(path, frozen)
        publish(ROOT/'results/FROZEN_DECISIONS.seal.json', {'sha256': sha(path)})
        development(selections, ROOT/'results')
        historical_development_report(selections, ROOT/'results')
        return frozen
    finally:
        provider.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['Prepare', 'Validate', 'Develop', 'Outer', 'Preserve'], default='Validate')
    parser.add_argument('--authorize-development', action='store_true')
    parser.add_argument('--authorize-outer', action='store_true')
    args = parser.parse_args(argv)
    require_authorization(args.mode, args.authorize_development, args.authorize_outer)
    with ProcessLock(ROOT/'logs/process.lock'):
        try:
            if args.mode == 'Preserve':
                from .preservation import run
                run('after')
                return
            require((ROOT/'validation/protected_before.json').exists(), 'Missing original preservation baseline')
            if args.mode == 'Prepare':
                prepare_lineage()
            else:
                verify_lineage()
            cfg, env = config(), environment()
            directory, model_lock = acquire(download=args.mode == 'Prepare')
            corpus = Corpus()
            publish(ROOT/'validation/data_audit.json', audit(corpus))
            if args.mode == 'Prepare':
                publish(ROOT/'validation/environment.json', env)
            elif args.mode == 'Validate':
                from .validation import bounded
                bounded(corpus, directory, model_lock, cfg, env)
            elif args.mode == 'Develop':
                develop(corpus, cfg, env, directory, model_lock, args.authorize_development)
            elif args.mode == 'Outer':
                from .evaluation import evaluate
                from .reporting import final, historical
                provider = Provider(directory, model_lock, cfg)
                try:
                    frozen = read(ROOT/'results/FROZEN_DECISIONS.json')
                    results = evaluate(corpus, provider, directory, frozen, cfg, model_lock, args.authorize_outer)
                    final(corpus, results, frozen, ROOT/'results', ROOT/'figures', historical())
                finally:
                    provider.close()
            print('Completed '+args.mode+'; no automatic transition to another mode.', flush=True)
        except BaseException as exc:
            event(ROOT/'logs', 'PIPELINE_FAILED', mode=args.mode, error=repr(exc), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
