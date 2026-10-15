"""Explicitly authorized once-only outer inference after every development decision freezes."""
import time
from .common import ROOT, digest, load_checkpoint, publish, read, require, sha, sources, versions, write
from .training import metrics, prediction_rows, predict_lr, predict_neural


def assert_frozen(frozen, cfg, model_lock):
    require(frozen['sources'] == sources() and frozen['versions'] == versions(), 'Frozen source/dependency identity changed')
    require(frozen['config'] == cfg and frozen['model'] == model_lock, 'Frozen model/configuration changed')
    require([s['outer_context'] for s in frozen['selections']] == [1, 2, 3], 'All contexts must freeze before outer inference')
    for selection in frozen['selections']:
        for result in selection['final'].values():
            load_checkpoint(result['checkpoint'], result['identity'])
            if result['candidate']['family'] == 'B':
                load_checkpoint(result['best_checkpoint'], result['identity'])


def seal(root, identity, infer):
    claim, result_path, seal_path = root/'claim.json', root/'result.json', root/'seal.json'
    if seal_path.exists():
        require(read(claim)['identity'] == identity and read(seal_path)['sha256'] == sha(result_path), 'Outer seal identity/checksum')
        result = read(result_path)
        require(result['identity'] == identity, 'Outer result identity')
        return result
    require(not claim.exists(), 'Outer claim consumed without seal; STOP for explicit review, no automatic retry')
    # Unlike idempotent result publication, the inference claim must be exclusively created.
    write(claim, {'identity': identity, 'status': 'CONSUMED_BEFORE_FORWARD'})
    result = {'identity': identity, **infer()}
    publish(result_path, result)
    publish(seal_path, {'sha256': sha(result_path), 'identity': identity})
    return result


def evaluate(corpus, provider, directory, frozen, cfg, model_lock, authorize_outer=False):
    require(authorize_outer is True, 'Separate explicit outer authorization required')
    assert_frozen(frozen, cfg, model_lock)
    path = ROOT/'results/FROZEN_DECISIONS.json'
    require(read(path) == frozen and read(ROOT/'results/FROZEN_DECISIONS.seal.json')['sha256'] == sha(path), 'Decision seal changed')
    results = []
    for split, selection in zip(corpus.splits, frozen['selections'], strict=True):
        for name in ('frozen_reference', 'selected'):
            fit = selection['final'][name]
            identity = {'decisions': digest(frozen), 'fold': split.fold, 'name': name, 'fit': fit['checkpoint']}
            def infer():
                records = corpus.select(split.test)
                candidate = fit['candidate']
                started = time.perf_counter()
                if candidate['family'] == 'A':
                    import numpy as np
                    x = np.stack([provider.extract(r)[cfg['layers'].index(candidate['layer'])] for r in records])
                    if candidate['pooling'] == 'mean':
                        x = x[:, :768]
                    probabilities = predict_lr(load_checkpoint(fit['checkpoint'], fit['identity']), x)
                else:
                    provider.close()
                    probabilities = predict_neural(fit, directory, candidate, cfg, records)
                rows = prediction_rows(records, probabilities, 'POST_HOC_EXPLORATORY_OUTER')
                return {'name': name, 'fold': split.fold, 'candidate': candidate,
                    'predictions': rows, 'metrics': metrics([r.label for r in records], probabilities),
                    'evaluation_seconds': time.perf_counter()-started}
            results.append(seal(ROOT/'results/outer'/f'fold{split.fold}_{name}', identity, infer))
    return results
