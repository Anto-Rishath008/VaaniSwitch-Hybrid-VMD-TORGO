"""Observed-only development and future sealed exploratory reporting."""
import csv
import io
import os
import numpy as np
from .common import ROOT, STUDY, atomic, digest, inside, publish, read, require, sha
from .training import metrics
from experiments.accuracy_hubert_v3.src.reporting import speaker_summary

FIELDS = ('accuracy', 'macro_f1', 'balanced_accuracy', 'sensitivity', 'specificity', 'roc_auc')


def bytes_publish(path, value):
    if path.exists():
        require(path.read_bytes() == value, 'Published report differs: '+str(path))
    else:
        atomic(path, value)


def table(path, rows):
    require(bool(rows), 'Empty report table')
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=sorted(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    bytes_publish(path, buffer.getvalue().encode())


def development(selections, output):
    rows = [{'outer_context': s['outer_context'], 'candidate': c['candidate']['id'],
        'evidence': 'DEVELOPMENT_GROUPED_CV', **c['mean'],
        **{k+'_sample_std': v for k, v in c['sample_std'].items()}}
        for s in selections for c in s['comparisons']]
    table(output/'ablation_results.csv', rows)
    publish(output/'selected_configurations.json', {str(s['outer_context']): {
        'selected': s['selected']['candidate'], 'frozen_reference': s['reference']['candidate'],
        'final_fit_epochs': {n: r.get('completed_epochs') for n, r in s['final'].items()}}
        for s in selections})
    text = '# V4 development evidence\n\nGrouped speaker CV used for selection; no outer accuracy claim.\n\n'
    text += '\n'.join(f"Context {s['outer_context']}: {s['selected']['candidate']['id']}; mean development macro-F1 {s['selected']['mean']['macro_f1']:.6f}." for s in selections)
    bytes_publish(output/'DEVELOPMENT_REPORT.md', (text+'\n').encode())


def compare_hubert_development(selections, previous):
    """Report after V4 selection freezes; never feed inherited scores back into selection."""
    require(len(selections) == len(previous) == 3, 'Three contexts for development reference')
    rows = []
    for current, old in zip(selections, previous, strict=True):
        require(current['outer_context'] == old['outer_fold'], 'Historical development context mismatch')
        require(current['cv_partitions'] == old['cv_splits'], 'Historical development speaker partitions differ')
        reference = old['reference']
        rows.append({'outer_context': current['outer_context'],
            'wavlm_frozen_candidate': current['reference']['candidate'],
            'wavlm_frozen_development': current['reference']['mean'],
            'hubert_frozen_candidate': reference['candidate'],
            'hubert_frozen_development': reference['metrics'],
            'macro_f1_difference': current['reference']['mean']['macro_f1']-reference['metrics']['macro_f1'],
            'caveat': 'Historical V3 best frozen-HuBERT reference on identical grouped CV definition; differing grids and previous development selection. Descriptive only; not independent or a selection input.'})
    return rows


def historical_development_report(selections, output):
    path = STUDY/'experiments/accuracy_hubert_v3/results/FROZEN_DECISIONS.json'
    before = read(ROOT/'validation/protected_before.json')['study']
    require(sha(path) == before[path.relative_to(STUDY).as_posix()]['sha256'], 'V3 frozen development evidence changed')
    previous = read(path)
    rows = compare_hubert_development(selections, previous['selections'])
    publish(output/'hubert_development_comparison.json', {'rows': rows, 'v3_sha256': sha(path),
        'timing': 'Read only after all V4 selections and final fits are frozen',
        'cv_provenance': 'Unchanged V3 development_cv imported and source-pinned by V3_LINEAGE.json'})


def historical():
    v2 = STUDY/'experiments/accuracy_hubert_v2/results/final_results.json'
    v3 = STUDY/'experiments/accuracy_hubert_v3/results/final_results.json'
    # Files are also covered by the protected content baseline; reports never feed selection.
    baseline = read(ROOT/'validation/protected_before.json')['study']
    for p in (v2, v3):
        require(sha(p) == baseline[p.relative_to(STUDY).as_posix()]['sha256'], 'Historical report changed')
    old2, old3 = read(v2), read(v3)
    return {'Original B2': old2['baselines']['B2'], 'Original C2': old2['baselines']['C2'],
        'V2 E1': old2['models']['E1']['fold_aggregation'],
        'V3 Selected': old3['models']['selected']['fold_aggregation']}


def final(corpus, results, frozen, output, figure_dir, baselines, synthetic=False):
    output, figure_dir = inside(output), inside(figure_dir)
    if synthetic:
        require(output.is_relative_to(ROOT/'validation'), 'Synthetic reports only in validation')
    require(len(results) == 6, 'Need three folds and both reference/selected results')
    identity = digest({'results': results, 'frozen': frozen, 'baselines': baselines, 'synthetic': synthetic})
    seal = output/'REPORT_SEAL.json'
    if seal.exists():
        spec = read(seal)
        require(spec['identity'] == identity, 'Report identity changed')
        for p, checksum in spec['files'].items():
            require(sha(inside(p)) == checksum, 'Report checksum changed')
        return read(output/'final_results.json')
    summary = {'title': 'SYNTHETIC TEST FIXTURE' if synthetic else 'V4 POST-HOC EXPLORATORY',
        'independent_confirmation': False, 'threshold': .5, 'baselines': baselines, 'models': {},
        'frozen_decisions_digest': digest(frozen), 'limitations': [
            '15 speakers and previously exposed outer results; no untouched external test set.',
            'No clinical diagnostic validity, significance, SOTA or guaranteed 90 percent claim.',
            'Equal-fold mean and sample SD; speaker aggregation is descriptive.',
            'Development search, early stopping, overlapping development contexts and one seed limit inference.']}
    folds, records, speakers = [], [], []
    for name in ('frozen_reference', 'selected'):
        cells = sorted([r for r in results if r['name'] == name], key=lambda r: r['fold'])
        require([r['fold'] for r in cells] == [1, 2, 3], 'Outer fold coverage')
        all_rows = []
        for split, cell in zip(corpus.splits, cells, strict=True):
            expected = corpus.select(split.test)
            rows = cell['predictions']
            require(len(rows) == len(expected) and {r['logical_id'] for r in rows} == {r.logical_id for r in expected}, 'Recording coverage')
            require(all(corpus.by_id[r['logical_id']].speaker_id == r['speaker_id']
                and corpus.by_id[r['logical_id']].label == r['label']
                and r['prediction'] == int(r['probability'] >= .5) for r in rows), 'Prediction identity/threshold')
            require(cell['metrics'] == metrics([r['label'] for r in rows], [r['probability'] for r in rows]), 'Metric disagreement')
            fit = frozen['selections'][split.fold-1]['final'][name]
            folds.append({'model': name, 'fold': split.fold, **{k: cell['metrics'][k] for k in FIELDS},
                'candidate': cell['candidate']['id'], 'training_seconds': fit['training_seconds'],
                'trainable_parameters': fit['parameter_count'], 'evaluation_seconds': cell['evaluation_seconds']})
            all_rows.extend(rows)
        require(len(all_rows) == len(corpus.records) == len({r['logical_id'] for r in all_rows}), 'Duplicate/missing overall recording')
        agg = {k: {'mean': float(np.mean([r['metrics'][k] for r in cells])),
            'sample_std': float(np.std([r['metrics'][k] for r in cells], ddof=1)),
            'fold_values': [r['metrics'][k] for r in cells]} for k in FIELDS}
        sp, sm = speaker_summary(all_rows)
        summary['models'][name] = {'fold_aggregation': agg, 'speaker_descriptive': sm, 'folds': cells}
        records.extend({**r, 'model': name} for r in all_rows)
        speakers.extend({**r, 'model': name} for r in sp)
    publish(output/'final_results.json', summary)
    table(output/'fold_results.csv', folds)
    table(output/'record_predictions.csv', records)
    table(output/'speaker_predictions.csv', speakers)
    development(frozen['selections'], output)
    lines = ['# '+summary['title'], '', 'Recording endpoint; equal-fold mean and sample SD. Positive class: dysarthric.',
        'Reused TORGO folds: POST-HOC EXPLORATORY. Independent external validation has not occurred.', '',
        '| Model | Accuracy mean +/- SD | Macro-F1 mean +/- SD | Speaker accuracy |', '|---|---:|---:|---:|']
    for name, value in summary['models'].items():
        a, f = value['fold_aggregation']['accuracy'], value['fold_aggregation']['macro_f1']
        lines.append(f"| {name} | {a['mean']:.4f} +/- {a['sample_std']:.4f} | {f['mean']:.4f} +/- {f['sample_std']:.4f} | {value['speaker_descriptive']['metrics']['accuracy']:.4f} |")
    lines += ['', 'Historical comparison (same endpoint, exploratory):']
    lines += [f"- {n}: accuracy {v['accuracy']['mean']:.4f} +/- {v['accuracy']['sample_std']:.4f}." for n, v in baselines.items()]
    lines += ['', *summary['limitations']]
    bytes_publish(output/'FINAL_EXPERIMENT_REPORT.md', ('\n'.join(lines)+'\n').encode())
    figures(summary, frozen, speakers, figure_dir)
    files = list(output.glob('*.csv')) + [output/n for n in ('final_results.json', 'FINAL_EXPERIMENT_REPORT.md',
        'DEVELOPMENT_REPORT.md', 'selected_configurations.json')] + list(figure_dir.glob('*.png'))
    publish(seal, {'identity': identity, 'files': {str(p): sha(p) for p in files}})
    return summary


def figures(summary, frozen, speakers, output):
    os.environ['MPLCONFIGDIR'] = str(ROOT/'cache/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def finish(fig, name):
        fig.suptitle(summary['title'])
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=140, bbox_inches='tight')
        plt.close(fig)
        bytes_publish(output/name, buffer.getvalue())
    comparisons = [(s['outer_context'], c) for s in frozen['selections'] for c in s['comparisons']]
    fig, ax = plt.subplots(figsize=(10, 4))
    for context, c in comparisons:
        if c['candidate']['family'] == 'B':
            for i, fit in enumerate(c['fits']):
                curve = fit['curve']
                ax.plot([e['epoch'] for e in curve], [e['train_weighted_loss'] for e in curve], alpha=.5,
                    label=f"F{context}/{c['candidate']['id']}/CV{i+1}")
    ax.set(xlabel='Epoch', ylabel='Observed weighted training BCE', title='All neural development fits; LR has no logged iterative curve')
    finish(fig, 'training_curves.png')
    labels = list(summary['baselines'])+list(summary['models'])
    values = list(summary['baselines'].values())+[v['fold_aggregation'] for v in summary['models'].values()]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, [100*v['accuracy']['mean'] for v in values], yerr=[100*v['accuracy']['sample_std'] for v in values], capsize=3)
    ax.set(ylabel='Recording accuracy (%)', title='Fold mean and sample SD; historical comparisons')
    ax.tick_params(axis='x', rotation=20)
    finish(fig, 'accuracy_comparison.png')
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (name, value) in zip(axes, summary['models'].items()):
        matrix = np.sum([r['metrics']['confusion_matrix'] for r in value['folds']], axis=0)
        ax.imshow(matrix, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha='center', va='center')
        ax.set(title=name, xlabel='Predicted', ylabel='True', xticks=[0, 1], yticks=[0, 1])
    finish(fig, 'confusion_matrix.png')
    fig, ax = plt.subplots(figsize=(12, 4))
    ids = sorted({r['speaker_id'] for r in speakers})
    for i, name in enumerate(summary['models']):
        by_id = {r['speaker_id']: r for r in speakers if r['model'] == name}
        ax.bar(np.arange(len(ids))+(i-.5)*.36, [100*by_id[s]['recording_accuracy'] for s in ids], width=.36, label=name)
    ax.set(xticks=np.arange(len(ids)), xticklabels=ids, ylabel='Per-speaker recording accuracy (%)')
    ax.legend()
    finish(fig, 'speaker_accuracy_analysis.png')
    fig, ax = plt.subplots(figsize=(10, 4))
    memory_rows = [(f"F{ctx}/{c['candidate']['id']}/CV{i+1}", f['memory'])
        for ctx, c in comparisons if c['candidate']['family'] == 'B' for i, f in enumerate(c['fits'])]
    ax.bar([r[0] for r in memory_rows], [r[1]['peak_reserved']/1024**3 for r in memory_rows])
    ax.set(ylabel='Peak reserved VRAM (GiB)', title='Neural development fits')
    ax.tick_params(axis='x', rotation=90, labelsize=6)
    finish(fig, 'gpu_memory_analysis.png')
