"""Predeclared small staged grid; candidate functions receive development roles only."""
import numpy as np
from .common import digest, publish, require
from .data import development_cv, partition
from .training import fit, selection_key


def candidate(layer, C, weighting='B0', normalization='N0', augmentation=False, prefix='ref'):
    return {'id': f'{prefix}_L{layer:02d}_C{C:g}_{weighting}_{normalization}_A{int(augmentation)}',
            'layer': layer, 'C': C, 'weighting': weighting, 'normalization': normalization,
            'augmentation': augmentation,
            'complexity': int(weighting != 'B0') + int(normalization != 'N0') + int(augmentation)}


def aggregate_cv(candidate_spec, fits):
    require(len(fits) == 3, 'Expected three grouped CV results')
    fields = ('macro_f1', 'balanced_accuracy', 'accuracy', 'roc_auc', 'log_loss', 'brier', 'ece_10_equal_width')
    summary, variation = {}, {}
    for field in fields:
        values = [r['validation_metrics'][field] for r in fits]
        summary[field] = float(np.mean(values)) if all(v is not None for v in values) else None
        variation[field] = float(np.std(values, ddof=1)) if summary[field] is not None else None
    return {'candidate': candidate_spec, 'metrics': summary, 'sample_std': variation, 'fits': fits,
            'selection_unit': 'equal-weight mean across three speaker-disjoint development folds'}


def qualifies(result, reference, cfg):
    gains = [a['validation_metrics']['macro_f1']-b['validation_metrics']['macro_f1']
             for a, b in zip(result['fits'], reference['fits'], strict=True)]
    return (result['metrics']['macro_f1']-reference['metrics']['macro_f1'] >= cfg['combination_min_macro_f1_gain']
            and sum(x > 0 for x in gains) >= cfg['combination_min_improved_splits']
            and result['metrics']['balanced_accuracy'] >= reference['metrics']['balanced_accuracy'])


def combine(reference, ablations, cfg):
    qualifying = [a for a in ablations if qualifies(a, reference, cfg)]
    weight = [a for a in qualifying if a['candidate']['weighting'] != 'B0']
    norms = [a for a in qualifying if a['candidate']['normalization'] != 'N0']
    augmentations = [a for a in qualifying if a['candidate']['augmentation']]
    if sum(bool(v) for v in (weight, norms, augmentations)) < 2:
        return None
    anchor = reference['candidate']
    return candidate(anchor['layer'], anchor['C'], 'B2' if weight else 'B0',
                     min(norms, key=selection_key)['candidate']['normalization'] if norms else 'N0',
                     bool(augmentations), 'combined')


def select_fold(corpus, split, cache, provider, cfg, output, run_identity):
    """All mutations are under output. Historical outer predictions are never read here."""
    output.mkdir(parents=True, exist_ok=True)
    memo = {}
    def panel(speakers, role, layer):
        key = (tuple(speakers), role, layer)
        if key not in memo:
            memo[key] = partition(corpus, speakers, split, role, cache, layer)
        return memo[key]

    grid = [candidate(layer, C) for layer in cfg['layers'] for C in cfg['C']]
    reproduction = []
    for spec in grid:
        r = fit(panel(split.train, 'train', spec['layer']), panel(split.validation, 'validation', spec['layer']),
                spec, cfg, output/'reproduction'/spec['id'], run_identity)
        reproduction.append(r)
    publish(output/'reproduction.json', {'status': 'COMPLETE_DEVELOPMENT_ONLY',
            'purpose': 'Faithful V2 E1 original inner split and nine-candidate recipe; no outer score imported',
            'split': split.identity, 'fits': reproduction})
    pairs = development_cv(corpus, split)

    def cv(spec):
        fits = []
        for index, (train_speakers, val_speakers) in enumerate(pairs):
            train = panel(train_speakers, 'train', spec['layer'])
            validation = panel(val_speakers, 'validation', spec['layer'])
            augmented = provider.get(train, spec['layer'], cfg['augmentation_per_speaker']) if spec['augmentation'] else None
            fits.append(fit(train, validation, spec, cfg, output/'cv'/spec['id']/f'split{index}',
                            run_identity, augmented=augmented))
        result = aggregate_cv(spec, fits)
        publish(output/'cv'/spec['id']/'summary.json', result)
        return result

    references = [cv(spec) for spec in grid]
    reference = min(references, key=selection_key)
    anchor = reference['candidate']
    mods = [candidate(anchor['layer'], anchor['C'], weighting='B2', prefix='weight'),
            candidate(anchor['layer'], anchor['C'], normalization='N1', prefix='robust'),
            candidate(anchor['layer'], anchor['C'], normalization='N2', prefix='l2'),
            candidate(anchor['layer'], anchor['C'], augmentation=True, prefix='channel')]
    ablations = [cv(spec) for spec in mods]
    combined = combine(reference, ablations, cfg)
    comparisons = [reference] + ablations + ([cv(combined)] if combined else [])
    winner = min(comparisons, key=selection_key)
    final = {}
    for name, choice in (('reference', reference), ('selected', winner)):
        if name == 'selected' and choice['candidate'] == reference['candidate']:
            final[name] = final['reference']
            continue
        spec = choice['candidate']
        dev = panel(split.development, 'development', spec['layer'])
        aug = provider.get(dev, spec['layer'], cfg['augmentation_per_speaker']) if spec['augmentation'] else None
        final[name] = fit(dev, None, spec, cfg, output/'final'/name, run_identity, augmented=aug)
    selection = {'outer_fold': split.fold, 'split': split.identity, 'cv_splits': pairs,
                 'reference': reference, 'winner': winner, 'references': references,
                 'ablations': ablations, 'combined': comparisons[-1] if combined else None,
                 'B1': 'Identical to B0; no duplicate fit; additional class balancing = none',
                 'final': final, 'run_identity': run_identity,
                 'candidate_fit_budget_max': 51, 'final_fit_budget_max': 2,
                 'reproduction_fits': reproduction}
    # JSON canonicalization makes list/tuple representation stable across resumes.
    import json
    selection = json.loads(json.dumps(selection))
    publish(output/'selection.json', selection)
    return selection
