"""Fixed development-only comparison; no historical result imports."""
import numpy as np
from .common import digest, publish, require
from .data import development_cv, partition
from .training import fit_lr, fit_neural


def candidates(cfg):
    frozen = [{'id': f'A_L{layer}_{pool}_C{c:g}', 'family': 'A', 'layer': layer,
        'pooling': pool, 'C': c, 'complexity': 0} for layer in cfg['layers']
        for pool in cfg['pooling'] for c in cfg['C']]
    neural = [{'id': f'B_last{depth}_{pool}', 'family': 'B', 'depth': depth,
        'pooling': pool, 'complexity': depth+(pool == 'attention')} for depth, pool in
        [(2, 'mean'), (4, 'mean'), (2, 'attention')]]
    return frozen, neural


def aggregate(candidate, fits):
    require(len(fits) == 3, 'Three development folds required')
    fields = ('macro_f1', 'balanced_accuracy', 'accuracy', 'roc_auc', 'sensitivity', 'specificity', 'log_loss')
    mean, sd = {}, {}
    for field in fields:
        values = [f['validation_metrics'][field] for f in fits]
        require(all(v is not None for v in values), 'Undefined two-class selection metric')
        mean[field] = float(np.mean(values))
        sd[field] = float(np.std(values, ddof=1))
    return {'candidate': candidate, 'mean': mean, 'sample_std': sd, 'fits': fits,
        'evidence_kind': 'DEVELOPMENT_GROUPED_CV'}


def selection_key(value):
    require(value['evidence_kind'] == 'DEVELOPMENT_GROUPED_CV', 'Outer evidence cannot select candidates')
    m, c = value['mean'], value['candidate']
    return (-m['macro_f1'], -m['balanced_accuracy'], -m['accuracy'], -m['roc_auc'],
        m['log_loss'], c['complexity'], c['id'])


def select_fold(corpus, split, provider, directory, cfg, root, run_id):
    pairs = development_cv(corpus, split)
    frozen, neural = candidates(cfg)
    comparisons = []
    for candidate in frozen+neural:
        fits = []
        for index, (t, v) in enumerate(pairs):
            train, val = partition(corpus, split, t, 'train'), partition(corpus, split, v, 'validation')
            fit_root = root/'cv'/candidate['id']/f'split{index+1}'
            if candidate['family'] == 'A':
                x, vx = provider.get(train, candidate), provider.get(val, candidate)
                fit = fit_lr(train, val, x, vx, candidate, cfg, fit_root, run_id)
            else:
                provider.close()
                fit = fit_neural(train, val, candidate, cfg, directory, fit_root, run_id)
            fits.append(fit)
        comparison = aggregate(candidate, fits)
        publish(root/'cv'/candidate['id']/'summary.json', comparison)
        comparisons.append(comparison)
    reference = min(comparisons[:len(frozen)], key=selection_key)
    winner = min(comparisons, key=selection_key)
    dev = partition(corpus, split, split.development, 'development')
    final = {}
    for name, choice in [('frozen_reference', reference), ('selected', winner)]:
        if name == 'selected' and choice['candidate'] == reference['candidate']:
            final[name] = final['frozen_reference']
            continue
        candidate = choice['candidate']
        if candidate['family'] == 'A':
            final[name] = fit_lr(dev, None, provider.get(dev, candidate), None, candidate, cfg,
                root/'final'/name, run_id)
        else:
            # Epoch budget is fixed solely by inner folds; final refit never early-stops on outer data.
            epoch_count = int(np.median([f['best_epoch'] for f in choice['fits']]))
            provider.close()
            final[name] = fit_neural(dev, None, candidate, cfg, directory, root/'final'/name,
                run_id, epochs=epoch_count)
    return {'outer_context': split.fold, 'comparisons': comparisons, 'reference': reference,
        'selected': winner, 'final': final, 'threshold': .5,
        'split_identity': split.identity, 'cv_partitions': [[list(t), list(v)] for t, v in pairs]}
