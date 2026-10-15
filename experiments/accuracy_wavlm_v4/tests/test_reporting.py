import copy
import unittest
from types import SimpleNamespace
from experiments.accuracy_wavlm_v4.src.common import ROOT, read
from experiments.accuracy_wavlm_v4.src.reporting import final, compare_hubert_development
from experiments.accuracy_wavlm_v4.src.training import metrics, prediction_rows
from test_v4 import output, record


def fixture():
    records = tuple(record(f's{fold}{label}', label) for fold in (1, 2, 3) for label in (0, 1))
    corpus = SimpleNamespace(records=records, by_id={r.logical_id: r for r in records},
        splits=[SimpleNamespace(fold=f, test=tuple(r.speaker_id for r in records[(f-1)*2:f*2])) for f in (1, 2, 3)],
        select=lambda speakers: tuple(r for r in records if r.speaker_id in speakers))
    results, selections = [], []
    candidate = {'id': 'synthetic_A', 'family': 'A'}
    metric = metrics([0, 1], [.2, .8])
    fields = {k: metric[k] for k in ('macro_f1', 'balanced_accuracy', 'accuracy', 'roc_auc', 'sensitivity', 'specificity', 'log_loss')}
    comparison = {'candidate': candidate, 'mean': fields, 'sample_std': {k: 0. for k in fields}, 'fits': []}
    for split in corpus.splits:
        selection = {'outer_context': split.fold, 'comparisons': [comparison], 'selected': comparison,
            'reference': comparison, 'final': {}}
        for name in ('frozen_reference', 'selected'):
            results.append({'name': name, 'fold': split.fold, 'candidate': candidate,
                'predictions': prediction_rows(corpus.select(split.test), [.2, .8], 'SYNTHETIC_TEST_FIXTURE'),
                'metrics': metric, 'evaluation_seconds': 0.})
            selection['final'][name] = {'training_seconds': 0., 'parameter_count': 4}
        selections.append(selection)
    return corpus, results, {'selections': selections}


class Reporting(unittest.TestCase):
    def test_development_reference_preserves_unfavorable_difference(self):
        current = [{'outer_context': f, 'cv_partitions': [['train', 'validation']], 'reference': {'candidate': {'id':'wavlm'}, 'mean': {'macro_f1': .4}}} for f in (1,2,3)]
        previous = [{'outer_fold': f, 'cv_splits': [['train', 'validation']], 'reference': {'candidate': {'id':'hubert'}, 'metrics': {'macro_f1': .6}}} for f in (1,2,3)]
        rows = compare_hubert_development(current, previous)
        self.assertTrue(all(abs(r['macro_f1_difference']+.2) < 1e-8 for r in rows))
        previous[0]['cv_splits'] = []
        with self.assertRaises(ValueError):
            compare_hubert_development(current, previous)

    def test_observed_fixture_report_and_sealed_reuse(self):
        corpus, results, frozen = fixture()
        root = output('synthetic-report')
        report = final(corpus, results, frozen, root/'results', root/'figures', {}, synthetic=True)
        self.assertIn('SYNTHETIC', report['title'])
        self.assertEqual(report['models']['selected']['fold_aggregation']['accuracy']['sample_std'], 0.)
        self.assertEqual(len(list((root/'figures').glob('*.png'))), 5)
        self.assertEqual(final(corpus, results, frozen, root/'results', root/'figures', {}, synthetic=True), report)

    def test_prediction_identity_error_rejected(self):
        corpus, results, frozen = fixture()
        results[0]['predictions'][0]['label'] = 1
        root = output('bad-report')
        with self.assertRaises(ValueError):
            final(corpus, results, frozen, root/'results', root/'figures', {}, synthetic=True)

    def test_missing_record_rejected(self):
        corpus, results, frozen = fixture()
        results[0]['predictions'].pop()
        root = output('missing-report')
        with self.assertRaises(ValueError):
            final(corpus, results, frozen, root/'results', root/'figures', {}, synthetic=True)

    def test_synthetic_cannot_publish_production_result(self):
        corpus, results, frozen = fixture()
        with self.assertRaises(ValueError):
            final(corpus, results, frozen, ROOT/'results', ROOT/'figures', {}, synthetic=True)
