"""Observed metrics only; developmental evidence never masquerades as outer results."""
import csv
import io
import json
import os
import statistics
import numpy as np
from .common import ROOT, V2, atomic_bytes, digest, publish, read, require, sha
from .training import metrics

FIELDS = ('accuracy', 'balanced_accuracy', 'macro_f1', 'precision', 'sensitivity', 'specificity',
          'roc_auc', 'brier', 'log_loss', 'ece_10_equal_width')


def byte_publish(path, value):
    if path.exists():
        require(path.read_bytes() == value, f'Refusing changed report: {path}')
    else:
        atomic_bytes(path, value)


def csv_publish(path, rows):
    require(rows, 'Empty report table')
    stream = io.StringIO(newline='')
    # A frozen JSON roundtrip sorts object keys. Column order must not depend on
    # the in-memory dict order or completed development cannot resume byte-identically.
    writer = csv.DictWriter(stream, fieldnames=sorted(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    byte_publish(path, stream.getvalue().encode())


def speaker_summary(rows):
    out = []
    for speaker in sorted({r['speaker_id'] for r in rows}):
        group = [r for r in rows if r['speaker_id'] == speaker]
        require(len({r['label'] for r in group}) == 1, 'Speaker label mismatch')
        p = float(np.mean([r['probability'] for r in group]))
        m = metrics([r['label'] for r in group], [r['probability'] for r in group])
        out.append({'speaker_id': speaker, 'label': group[0]['label'], 'probability': p,
                    'prediction': int(p >= .5), 'recording_count': len(group),
                    'recording_accuracy': m['accuracy'], 'sensitivity': m['sensitivity'],
                    'specificity': m['specificity']})
    m = metrics([r['label'] for r in out], [r['probability'] for r in out])
    return out, {'metrics': m, 'recording_accuracy_sample_std_between_speakers':
                 float(np.std([r['recording_accuracy'] for r in out], ddof=1)) if len(out) > 1 else None,
                 'incorrect_speakers': [r['speaker_id'] for r in out if r['prediction'] != r['label']],
                 'unit': 'descriptive speaker mean probability >=0.5'}


def channel_summary(rows):
    return {mic: metrics([r['label'] for r in rows if r['microphone'] == mic],
                         [r['probability'] for r in rows if r['microphone'] == mic])
            for mic in sorted({r['microphone'] for r in rows})}


def ablation_rows(selections):
    rows = []
    for selection in selections:
        entries = [selection['reference']] + selection['ablations']
        if selection['combined']:
            entries.append(selection['combined'])
        for r in entries:
            rows.append({'outer_fold_context': selection['outer_fold'], 'candidate': r['candidate']['id'],
                         'evidence': 'development grouped CV only', **r['metrics']})
    return rows


def development_report(selections, output):
    rows = ablation_rows(selections)
    csv_publish(output/'ablation_results.csv', rows)
    publish(output/'selected_configurations.json',
            {str(s['outer_fold']): {'reference': s['reference']['candidate'], 'selected': s['winner']['candidate']}
             for s in selections})
    lines = ['# V3 development-only report', '',
             'Grouped CV and selection evidence; these are not outer-test results or independent estimates.', '',
             '| Outer-fold context | Recipe | CV macro-F1 | CV balanced accuracy |', '|---|---|---:|---:|']
    for row in rows:
        lines.append(f"| {row['outer_fold_context']} | {row['candidate']} | {row['macro_f1']:.6f} | {row['balanced_accuracy']:.6f} |")
    lines += ['', 'This document contains development selection evidence only.',
              'Real outer status is recorded separately in FROZEN_DECISIONS.json, outer seals and FINAL_EXPERIMENT_REPORT.md.',
              'Development metrics are not final V3 outer accuracy. All decisions must freeze before any outer evaluation.']
    byte_publish(output/'DEVELOPMENT_REPORT.md', ('\n'.join(lines)+'\n').encode())


def historical_baselines():
    doc = read(V2/'results/final_results.json')
    out = {f'Original {m}': doc['baselines'][m] for m in ('B2', 'C2')}
    out.update({f'V2 {m}': doc['models'][m]['fold_aggregation'] for m in ('E1', 'E2', 'E3')})
    return out


def final_report(corpus, results, selections, frozen, output, figure_dir, baselines, data_kind='REAL_SEALED_OUTER'):
    require(data_kind in ('REAL_SEALED_OUTER', 'SYNTHETIC_TEST_FIXTURE'), 'Report evidence kind')
    report_identity = digest({'results': results, 'frozen': frozen, 'kind': data_kind, 'baselines': baselines})
    seal = output/'report_seal.json'
    if seal.exists():
        prior = read(seal)
        require(prior['identity'] == report_identity, 'Report identity changed')
        for path, checksum in prior['artifacts'].items():
            require(sha(path) == checksum, 'Sealed report changed')
        return read(output/'final_results.json')
    models = sorted({r['model'] for r in results})
    require(models == ['reference', 'selected'], 'Missing reference/selected panel')
    summary = {'title': 'V3 POST-HOC EXPLORATORY STUDY', 'data_kind': data_kind,
               'frozen': frozen, 'baselines': baselines, 'models': {},
               'runtime_evidence': frozen.get('runtime_evidence', {}),
               'limitations': ['Original/V1/V2 outer results informed this direction; not an independent benchmark',
                               '15 speakers, overlapping development sets, single fixed seed, selection optimism',
                               'Channel differences are observational and confounded; not causal',
                               'Calibration diagnostics are separate and do not alter primary outer probabilities',
                               'No clinical, significance, SOTA, or guaranteed improvement claim']}
    predictions, fold_rows, speaker_rows = [], [], []
    for model in models:
        cells = sorted([r for r in results if r['model'] == model], key=lambda r: r['outer_fold'])
        require([r['outer_fold'] for r in cells] == [1, 2, 3], 'Missing/duplicate outer fold')
        rows = []
        for cell in cells:
            current = [{**p, 'model': model, 'outer_fold': cell['outer_fold']} for p in cell['predictions']]
            expected = corpus.select(corpus.splits[cell['outer_fold']-1].test)
            require({p['logical_id'] for p in current} == {r.logical_id for r in expected}
                    and len(current) == len(expected), 'Outer prediction coverage differs')
            require(all(corpus.by_id[p['logical_id']].speaker_id == p['speaker_id']
                        and corpus.by_id[p['logical_id']].label == p['label']
                        and p['prediction'] == int(p['probability'] >= .5) for p in current), 'Prediction binding differs')
            require(cell['metrics'] == metrics([p['label'] for p in current], [p['probability'] for p in current]),
                    'Stored metrics disagree with retained predictions')
            rows += current
            fold_rows.append({'model': model, 'outer_fold': cell['outer_fold'],
                              **{k: cell['metrics'][k] for k in FIELDS},
                              'confusion_matrix': json.dumps(cell['metrics']['confusion_matrix']),
                              'training_seconds': cell['training_seconds'], 'parameter_count': cell['parameter_count'],
                              'candidate': cell['candidate']['id']})
        require(len(rows) == len(corpus.records) and len({p['logical_id'] for p in rows}) == len(rows),
                'Duplicate/missing overall predictions')
        aggregation = {}
        for key in FIELDS:
            values = [r['metrics'][key] for r in cells]
            aggregation[key] = {'fold_values': values,
                                'mean': statistics.mean(values) if all(v is not None for v in values) else None,
                                'sample_std': statistics.stdev(values) if all(v is not None for v in values) else None}
        speakers, sm = speaker_summary(rows)
        require(len(speakers) == len(corpus.mapping), 'Speaker coverage lost')
        speaker_rows += [{**s, 'model': model} for s in speakers]
        predictions += rows
        summary['models'][model] = {'fold_aggregation': aggregation, 'speaker_descriptive': sm,
                                    'microphones': channel_summary(rows), 'folds': cells,
                                    'channel_by_speaker_session': {
                                        key: channel_summary([r for r in rows if '/'.join(r['logical_id'].split('/')[:2]) == key])
                                        for key in sorted({'/'.join(r['logical_id'].split('/')[:2]) for r in rows})},
                                    'comparison_pp': {name: {k: 100*(aggregation[k]['mean']-b[k]['mean'])
                                                          for k in ('accuracy', 'macro_f1')}
                                                      for name, b in baselines.items()}}
    publish(output/'final_results.json', summary)
    csv_publish(output/'fold_results.csv', fold_rows)
    csv_publish(output/'record_predictions.csv', predictions)
    csv_publish(output/'speaker_predictions.csv', speaker_rows)
    development_report(selections, output)
    lines = ['# V3 POST-HOC EXPLORATORY STUDY', '', f'Evidence: {data_kind}', '',
             'Same recording-level endpoint; arithmetic outer-fold mean and sample SD (ddof=1).',
             'Positive class: dysarthric. Fixed threshold: 0.5. No independent confirmation.', '',
             '| Model | Recording accuracy mean ± SD | Macro-F1 mean ± SD | Descriptive speaker accuracy |',
             '|---|---:|---:|---:|']
    for model, value in summary['models'].items():
        a, f = value['fold_aggregation']['accuracy'], value['fold_aggregation']['macro_f1']
        sp = value['speaker_descriptive']['metrics']['accuracy']
        lines.append(f"| {model} | {100*a['mean']:.2f}% ± {100*a['sample_std']:.2f}% | {100*f['mean']:.2f}% ± {100*f['sample_std']:.2f}% | {100*sp:.2f}% |")
    lines += ['', 'Historical comparisons (same recording endpoint, no independent significance claim):', '',
              '| Historical model | Accuracy mean ± SD | Macro-F1 mean ± SD |', '|---|---:|---:|']
    for name, b in baselines.items():
        a, f = b['accuracy'], b['macro_f1']
        lines.append(f"| {name} | {100*a['mean']:.2f}% ± {100*a['sample_std']:.2f}% | {100*f['mean']:.2f}% ± {100*f['sample_std']:.2f}% |")
    lines += ['', 'Full per-fold, per-speaker, class rates, calibration/reliability bins, microphone diagnostics,',
              'between-speaker variation, runtimes, trainable parameter counts and all predictions are retained',
              'in JSON/CSV. Missing class-specific speaker rates are null, not invented zeros.',
              'ablation_results.csv contains development-only evidence; selected_configurations.json freezes choices.',
              'training_curves.png explains that no iterative loss histories are available for sklearn LBFGS.', '',
              *summary['limitations']]
    byte_publish(output/'FINAL_EXPERIMENT_REPORT.md', ('\n'.join(lines)+'\n').encode())
    figures(summary, speaker_rows, data_kind, figure_dir)
    names = ('final_results.json', 'fold_results.csv', 'record_predictions.csv', 'speaker_predictions.csv',
             'ablation_results.csv', 'selected_configurations.json', 'FINAL_EXPERIMENT_REPORT.md', 'DEVELOPMENT_REPORT.md')
    artifacts = [output/n for n in names] + list(figure_dir.glob('*.png'))
    publish(seal, {'identity': report_identity, 'artifacts': {str(p): sha(p) for p in artifacts}})
    return summary


def figures(summary, speakers, kind, output):
    os.environ['MPLCONFIGDIR'] = str(ROOT/'cache/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def finish(fig, name):
        fig.suptitle('SYNTHETIC TEST FIXTURE' if kind == 'SYNTHETIC_TEST_FIXTURE' else 'Post-hoc exploratory V3')
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=140, bbox_inches='tight')
        plt.close(fig)
        byte_publish(output/name, buffer.getvalue())
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis('off')
    ax.text(.5, .5, 'Training curves: not applicable\nOnly sklearn logistic regression is fitted.\nNo per-iteration loss history is exposed or fabricated.\nSee checkpoints for solver iteration counts.',
            ha='center', va='center')
    finish(fig, 'training_curves.png')
    labels = list(summary['baselines']) + list(summary['models'])
    stats = list(summary['baselines'].values()) + [m['fold_aggregation'] for m in summary['models'].values()]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, [100*s['accuracy']['mean'] for s in stats],
           yerr=[100*s['accuracy']['sample_std'] for s in stats], capsize=3)
    ax.set(ylabel='Recording accuracy (%)', title='Fold mean ± sample SD; historical comparisons')
    ax.tick_params(axis='x', rotation=20)
    finish(fig, 'accuracy_comparison.png')
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, (model, value) in zip(axes, summary['models'].items(), strict=True):
        matrix = np.sum([r['metrics']['confusion_matrix'] for r in value['folds']], axis=0)
        ax.imshow(matrix, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha='center', va='center')
        ax.set(title=model+' pooled recordings', xlabel='Predicted', ylabel='True', xticks=[0, 1], yticks=[0, 1])
    finish(fig, 'confusion_matrix.png')
    fig, ax = plt.subplots(figsize=(11, 4))
    ids = sorted({r['speaker_id'] for r in speakers})
    for i, model in enumerate(summary['models']):
        by_id = {r['speaker_id']: r for r in speakers if r['model'] == model}
        ax.bar(np.arange(len(ids))+(i-.5)*.35, [100*by_id[s]['recording_accuracy'] for s in ids], width=.35, label=model)
    ax.set(xticks=np.arange(len(ids)), xticklabels=ids, ylabel='Recording accuracy within speaker (%)')
    ax.legend()
    finish(fig, 'speaker_accuracy_analysis.png')
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (model, value) in enumerate(summary['models'].items()):
        channels = value['microphones']
        ax.bar(np.arange(len(channels))+(i-.5)*.35, [100*v['accuracy'] for v in channels.values()], width=.35, label=model)
    ax.set(xticks=np.arange(len(channels)), xticklabels=list(channels), ylabel='Recording accuracy (%)',
           title='Observational microphone groups; confounded, not causal')
    ax.legend()
    finish(fig, 'microphone_analysis.png')
