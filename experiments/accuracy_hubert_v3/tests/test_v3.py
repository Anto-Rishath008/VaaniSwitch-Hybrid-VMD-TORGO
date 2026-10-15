"""Synthetic fixtures only, except explicitly named read-only live contract tests."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from experiments.accuracy_hubert_v3.src import common, pipeline, training, reporting, selection
from experiments.accuracy_hubert_v3.src.data import Record, Partition, Corpus, development_cv, separated
from experiments.accuracy_hubert_v3.src.weighting import weights, expanded_weights, weight_report
from experiments.accuracy_hubert_v3.src.normalization import fit_normalizer, transform, l2_rows, weighted_quantiles
from experiments.accuracy_hubert_v3.src.augmentation import parameters, augment, choose_records
from experiments.accuracy_hubert_v3.src.features import AugmentedCache, bounded_infer
from experiments.accuracy_hubert_v3.src.reuse import VerifiedMemoryCache
from experiments.accuracy_hubert_v3.src.calibration import calibration_roles, diagnostic
from experiments.accuracy_hubert_v2.src.training import seal_outer, rng_state, restore_rng


def record(speaker, label, index):
    return Record(f'{speaker}/session/{index:03d}', speaker, 'dysarthric' if label else 'control',
                  f'data/raw/{speaker}_{index}.wav', common.digest([speaker, index]), 1., 16000, 16000,
                  'head' if index % 2 else 'array')


def panel(role='train', counts=(2, 4, 3, 5), width=12, seed=10):
    records = tuple(record(f'S{s}', s//2, i) for s, n in enumerate(counts) for i in range(n))
    x = np.random.default_rng(seed).normal(size=(len(records), width)).astype(np.float32)
    x[:, 0] += [r.label for r in records]
    return Partition(records, x, tuple(f'S{i}' for i in range(4)), ('OUTER',), role, 'SYNTHETIC_TEST_FIXTURE')


class SyntheticCorpus:
    def __init__(self):
        self.evidence_kind = 'SYNTHETIC_TEST_FIXTURE'
        self.mapping = {f'S{s:02d}': 1+s%3 for s in range(15)}
        self.records = tuple(record(f'S{s:02d}', int(s >= 7), i) for s in range(15) for i in range(2))
        self.by_id = {r.logical_id: r for r in self.records}
        self.splits = []
        for fold in (1, 2, 3):
            test = tuple(s for s, f in self.mapping.items() if f == fold)
            dev = tuple(sorted(set(self.mapping)-set(test)))
            labels = {r.speaker_id: r.label for r in self.records}
            val = tuple(next(s for s in dev if labels[s] == c) for c in (0, 1))
            train = tuple(sorted(set(dev)-set(val)))
            self.splits.append(SimpleNamespace(fold=fold, test=test, development=dev, train=train, validation=val,
                                              identity={'outer_fold': fold, 'inner_train': list(train),
                                                        'inner_validation': list(val), 'outer_test': list(test)}))
    def select(self, speakers):
        return tuple(r for r in self.records if r.speaker_id in speakers)


class SyntheticCache:
    def load(self, r):
        rng = np.random.default_rng(int(common.digest(r.logical_id)[:8], 16))
        x = rng.normal(size=(3, 12)).astype(np.float32)
        x[:, 0] += 1.5*r.label
        return x


class SyntheticProvider:
    def __init__(self, *args):
        self.access, self.generated, self.hits, self.extraction_seconds = [], 0, 0, 0.
    def get(self, partition, layer, budget=16):
        partition.check(fitting=True)
        self.access.append(partition.identity)
        return partition.records, partition.features+.01
    def close(self):
        pass


class V3Tests(unittest.TestCase):
    def setUp(self):
        (common.ROOT/'validation').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-test-', dir=common.ROOT/'validation')
        self.root = Path(self.temp.name)
        self.cfg = common.config()
    def tearDown(self):
        self.temp.cleanup()

    def test_live_original_manifest_and_split_contract(self):
        c = Corpus()
        self.assertEqual(len(c.records), 8673)
        self.assertEqual(len(c.mapping), 15)
        self.assertEqual([len(c.select(s.test)) for s in c.splits], [2819, 2873, 2981])

    def test_live_pinned_v2_sources_match_lineage(self):
        lineage = common.read(common.ROOT/'V2_LINEAGE.json')
        frozen = common.read(common.V2/'logs/frozen_run.json')
        self.assertEqual(frozen['config']['features']['revision'], 'dba3bb02fda4248b6e082697eee756de8fe8aa8a')
        for name, checksum in frozen['source_files'].items():
            self.assertEqual(common.sha(common.V2/name), checksum)
            self.assertEqual(lineage['v2_artifacts'][name], checksum)

    def test_grouped_cv_coverage_and_disjoint_roles(self):
        c = SyntheticCorpus()
        for split in c.splits:
            pairs = development_cv(c, split)
            self.assertEqual(sorted(s for _, v in pairs for s in v), sorted(split.development))
            for t, v in pairs:
                separated(c.select(t), c.select(v), c.select(split.test))
                self.assertEqual({r.label for r in c.select(t)}, {0, 1})
                self.assertEqual({r.label for r in c.select(v)}, {0, 1})

    def test_hash_leakage_rejected_even_if_different_speakers(self):
        from dataclasses import replace
        a, b = record('a', 0, 1), record('b', 1, 1)
        with self.assertRaisesRegex(ValueError, 'primary_audio_sha256'):
            separated((a,), (replace(b, primary_audio_sha256=a.primary_audio_sha256),))

    def test_outer_partition_rejected(self):
        p = panel('outer')
        with self.assertRaisesRegex(ValueError, 'Outer'):
            p.check()

    def test_validation_cannot_fit_weights_scaler_or_augment(self):
        p = panel('validation')
        for fn in (lambda: weights(p), lambda: fit_normalizer(p, np.ones(len(p.records)), 'N0'),
                   lambda: choose_records(p)):
            with self.assertRaises(ValueError): fn()

    def test_forbidden_speaker_rejected_before_fit(self):
        p = panel()
        p = Partition(p.records, p.features, p.allowed, ('S0',), 'train')
        with self.assertRaises(ValueError): weights(p)

    def test_B0_B1_equivalence_and_equal_speaker_mass(self):
        p = panel()
        a, b = weights(p, 'B0'), weights(p, 'B1')
        np.testing.assert_array_equal(a, b)
        report = weight_report(p.records, a)
        np.testing.assert_allclose(list(report['speakers'].values()), len(p.records)/4)

    def test_B0_matches_v2_weighting_exactly(self):
        from experiments.accuracy_hubert_v2.src.data import training_weights
        p = panel()
        np.testing.assert_array_equal(weights(p), training_weights(p.records, p.allowed, p.forbidden))

    def test_class_masses_follow_speaker_counts_without_double_balance(self):
        p = panel()
        records = tuple(r for r in p.records if r.speaker_id != 'S3')
        p = Partition(records, p.features[:len(records)], p.allowed, p.forbidden, p.role)
        report = weight_report(p.records, weights(p))
        self.assertAlmostEqual(report['classes']['0']/report['classes']['1'], 2.)
        self.assertIsNone(report['additional_class_weight'])

    def test_B2_is_bounded_and_distinct(self):
        p = panel(counts=(1, 20, 2, 30))
        w = weights(p, 'B2')
        self.assertLessEqual(w.max()/w.min(), 4.+1e-12)
        self.assertAlmostEqual(w.mean(), 1.)
        self.assertFalse(np.allclose(w, weights(p)))

    def test_augmented_weights_preserve_every_parent_speaker_class_and_C_mass(self):
        p = panel()
        base = weights(p)
        ids = [r.logical_id for r in p.records]
        augmented_ids = ids + ids[:3]
        w = expanded_weights(p, base, augmented_ids)
        for i, parent in enumerate(ids):
            self.assertAlmostEqual(w[np.array(augmented_ids) == parent].sum(), base[i])
        self.assertAlmostEqual(w.sum(), len(ids))
        self.assertNotAlmostEqual(w.mean(), 1.)

    def test_variant_foreign_parent_rejected(self):
        p = panel()
        with self.assertRaises(ValueError): expanded_weights(p, weights(p), ['outer/record/id'])

    def test_normalizers_train_only_and_validation_outlier_has_no_effect(self):
        p = panel()
        for method in ('N0', 'N1', 'N2'):
            spec = fit_normalizer(p, weights(p), method)
            before = transform(p.features, spec)
            transform(np.full((2, p.features.shape[1]), 1e8, np.float32), spec)
            np.testing.assert_array_equal(before, transform(p.features, spec))
            self.assertEqual(spec['fit_identity']['ids'], [r.logical_id for r in p.records])

    def test_N0_matches_v2_scaler_transform(self):
        from experiments.accuracy_hubert_v2.src.training import fit_scaler, Partition as OldPartition
        p = panel()
        old = OldPartition(p.records, p.features, p.allowed, p.forbidden, 'inner_train')
        scaler, _ = fit_scaler(old)
        np.testing.assert_array_equal(transform(p.features, fit_normalizer(p, weights(p), 'N0')), scaler.transform(p.features))

    def test_robust_zero_iqr_and_constant_dimensions(self):
        p = panel()
        p.features[:, 1] = 7.
        spec = fit_normalizer(p, weights(p), 'N1')
        self.assertGreaterEqual(spec['zero_iqr_dimensions'], 1)
        np.testing.assert_array_equal(transform(p.features, spec)[:, 1], 0)

    def test_l2_zero_vector_and_scale_invariance(self):
        x = np.array([[0., 0.], [3., 4.]], np.float32)
        np.testing.assert_array_equal(l2_rows(x)[0], [0, 0])
        np.testing.assert_allclose(l2_rows(x), l2_rows(x*8), atol=1e-7)

    def test_weighted_quantile_definition(self):
        x = np.array([[0.], [10.]])
        np.testing.assert_allclose(weighted_quantiles(x, np.ones(2))[:, 0], [0, 5, 10])

    def test_nonfinite_features_scaler_and_weights_rejected(self):
        p = panel()
        for location in ('input', 'scaler', 'weights'):
            q = copy.deepcopy(p)
            w = weights(q)
            spec = fit_normalizer(q, w, 'N0')
            if location == 'input': q.features[0, 0] = np.nan
            elif location == 'scaler': spec['scaler'].scale_[0] = np.nan
            else: w[0] = np.nan
            with self.assertRaises(ValueError):
                if location == 'weights': fit_normalizer(q, w, 'N0')
                else: transform(q.features, spec)

    def test_augmentation_deterministic_and_input_immutable(self):
        r = record('s', 0, 1)
        x = np.random.default_rng(1).normal(0, .1, 16000).astype(np.float32)
        before = x.copy()
        first, stats = augment(x, parameters(r))
        second, _ = augment(x, parameters(r))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(x, before)
        self.assertFalse(np.array_equal(x, first))
        self.assertEqual(first.shape, x.shape)
        self.assertAlmostEqual(20*np.log10(stats['pre_noise_rms']/stats['noise_rms']), parameters(r)['snr_db'], places=10)

    def test_silence_and_peak_safety(self):
        p = parameters(record('s', 0, 2))
        silent, stats = augment(np.zeros(16000, np.float32), p)
        self.assertTrue(stats['silent_input'])
        np.testing.assert_array_equal(silent, 0)
        loud, _ = augment(np.ones(16000, np.float32)*2, p)
        self.assertLessEqual(float(np.abs(loud).max()), .990001)

    def test_channel_bandwidth_attenuates_high_frequency(self):
        p = parameters(record('s', 0, 3))
        p.update(profile='band_noise', lowpass_hz=4000., highpass_hz=60., snr_db=35., gain_db=0.)
        t = np.arange(32000)/16000
        x = (.1*np.sin(2*np.pi*1000*t)+.1*np.sin(2*np.pi*7000*t)).astype(np.float32)
        y, _ = augment(x, p)
        low = abs(np.dot(y[2000:], np.exp(-2j*np.pi*1000*t[2000:])))
        high = abs(np.dot(y[2000:], np.exp(-2j*np.pi*7000*t[2000:])))
        self.assertLess(high/low, .2)

    def test_augmentation_bounds_and_seed_identity(self):
        r = record('s', 0, 1)
        p = parameters(r)
        self.assertNotEqual(p['seed'], parameters(record('s', 0, 2))['seed'])
        p['snr_db'] = 5.
        with self.assertRaises(ValueError): augment(np.ones(1000, np.float32), p)

    def test_budget_and_record_membership(self):
        p = panel(counts=(30, 30, 30, 30))
        self.assertEqual(len(choose_records(p)), 64)
        self.assertEqual(choose_records(p), choose_records(p))
        with self.assertRaises(ValueError): choose_records(p, 17)

    def cache(self, sources=None):
        return AugmentedCache(self.root/'aug', {'revision': 'SYNTHETIC'}, {'sample_rate': 16000}, sources or {'test': 'synthetic'})

    def test_augmented_cache_roundtrip_checksum_and_corruption(self):
        c, r = self.cache(), record('s', 0, 1)
        x = np.zeros((3, 1536), np.float32)
        c.save(r, x, {}, 'synthetic', {})
        np.testing.assert_array_equal(c.load(r), x)
        doc = common.read(c.location(r)/'complete.json')
        (c.location(r)/doc['filename']).write_bytes(b'corrupt synthetic payload')
        with self.assertRaisesRegex(ValueError, 'checksum'): c.load(r)

    def test_augmented_metadata_tampering_is_rejected(self):
        c, r = self.cache(), record('s', 0, 1)
        c.save(r, np.zeros((3, 1536), np.float32), {'crop': 'SYNTHETIC'}, 'synthetic', {})
        path = c.location(r)/'complete.json'
        doc = common.read(path)
        doc['segment']['crop'] = 'changed'
        common.write_json(path, doc, replace=True)
        with self.assertRaisesRegex(ValueError, 'metadata checksum'): c.load(r)

    def test_cache_identity_includes_source_audio_seed_and_encoder(self):
        from dataclasses import replace
        r = record('s', 0, 1)
        c = self.cache()
        original = c.location(r)
        self.assertNotEqual(original, self.cache({'test': 'changed-source'}).location(r))
        self.assertNotEqual(original, c.location(replace(r, primary_audio_sha256='other')))
        altered = AugmentedCache(self.root/'aug', {'revision': 'DIFFERENT'}, {'sample_rate': 16000}, {'test': 'synthetic'})
        self.assertNotEqual(original, altered.location(r))

    def test_cache_bad_shape_nonfinite_and_overwrite_rejected(self):
        c, r = self.cache(), record('s', 0, 1)
        for x in (np.zeros((3, 10), np.float32), np.full((3, 1536), np.nan, np.float32)):
            with self.assertRaises(ValueError): c.save(r, x, {}, '', {})
        c.save(r, np.zeros((3, 1536), np.float32), {}, '', {})
        with self.assertRaises(ValueError): c.save(r, np.zeros((3, 1536), np.float32), {}, '', {})

    def test_output_cannot_escape_V3(self):
        with self.assertRaises(ValueError): common.within_v3(common.V2/'cache/forbidden')

    def test_readonly_original_cache_forbids_save_and_rebinding(self):
        r = record('s', 0, 1)
        source = SimpleNamespace(identity={}, load=lambda _: np.zeros((3, 1536), np.float32))
        c = VerifiedMemoryCache(source, (r,))
        with self.assertRaises(RuntimeError): c.save(r, None)
        with self.assertRaises(ValueError): c.load(record('s', 0, 2))
        with self.assertRaises(ValueError): c.load(r)[0, 0] = 1

    def test_OOM_retries_bounded_and_no_sample_skipped(self):
        sizes = []
        class Fake:
            def infer(self, waves):
                sizes.append(len(waves))
                if len(waves) > 1: raise torch.cuda.OutOfMemoryError('synthetic OOM')
                return np.zeros((len(waves), 3, 1536), np.float32)
        result = bounded_infer(Fake(), [np.zeros(800)]*4, cap=4)
        self.assertEqual(sizes, [4, 2, 1, 1, 1, 1])
        self.assertEqual(result.shape, (4, 3, 1536))

    def test_OOM_at_one_and_numerical_failure_stop(self):
        class Bad:
            def infer(self, waves): raise torch.cuda.OutOfMemoryError('synthetic')
        with self.assertRaises(torch.cuda.OutOfMemoryError): bounded_infer(Bad(), [np.zeros(800)])
        class NonFinite:
            def infer(self, waves): return np.full((1, 3, 1536), np.nan)
        with self.assertRaises(ValueError): bounded_infer(NonFinite(), [np.zeros(800)])

    def test_exact_length_batches_required(self):
        with self.assertRaises(ValueError): bounded_infer(None, [np.zeros(800), np.zeros(801)])

    def test_empty_extractor_output_stops_instead_of_looping(self):
        class Empty:
            def infer(self, waves): return np.zeros((0, 3, 1536), np.float32)
        with self.assertRaisesRegex(ValueError, 'shape/dtype'): bounded_infer(Empty(), [np.zeros(800)])

    def test_checkpoint_roundtrip_interruption_and_resume(self):
        p = panel()
        spec = selection.candidate(9, .1)
        baseline = training.fit(p, None, spec, self.cfg, self.root/'base', 'synthetic')
        expected = training.predict(training.load_checkpoint(baseline['checkpoint'], baseline['identity']), p.features)
        for point in ('after_start', 'after_checkpoint'):
            directory = self.root/point
            with self.assertRaises(InterruptedError): training.fit(p, None, spec, self.cfg, directory, 'synthetic', interrupt=point)
            r = training.fit(p, None, spec, self.cfg, directory, 'synthetic')
            actual = training.predict(training.load_checkpoint(r['checkpoint'], r['identity']), p.features)
            np.testing.assert_array_equal(expected, actual)
            with patch.object(training.LogisticRegression, 'fit', side_effect=AssertionError('Repeated fit')):
                self.assertEqual(training.fit(p, None, spec, self.cfg, directory, 'synthetic'), r)

    def test_checkpoint_wrong_identity_and_corrupt_payload_stop(self):
        p = panel()
        r = training.fit(p, None, selection.candidate(9, .1), self.cfg, self.root/'fit', 'synthetic')
        with self.assertRaises(ValueError): training.load_checkpoint(r['checkpoint'], 'wrong')
        Path(r['checkpoint']['path']).write_bytes(b'corrupt synthetic checkpoint')
        with self.assertRaises(ValueError): training.load_checkpoint(r['checkpoint'], r['identity'])

    def test_completed_fit_metrics_tampering_rejected_on_resume(self):
        p = panel()
        directory = self.root/'fit'
        spec = selection.candidate(9, .1)
        r = training.fit(p, None, spec, self.cfg, directory, 'synthetic')
        r['parameter_count'] = 999
        common.write_json(directory/'complete.json', r, replace=True)
        with self.assertRaisesRegex(ValueError, 'metrics/metadata'):
            training.fit(p, None, spec, self.cfg, directory, 'synthetic')

    def test_classifier_rejects_augmented_foreign_speaker(self):
        p = panel()
        with self.assertRaisesRegex(ValueError, 'leakage'):
            training.fit(p, None, selection.candidate(9, .1, augmentation=True), self.cfg, self.root/'fit', 'synthetic',
                         augmented=((record('OUTER', 1, 1),), np.zeros((1, 12), np.float32)))

    def test_fit_rejects_train_validation_overlap(self):
        p = panel()
        v = Partition(p.records, p.features, p.allowed, p.forbidden, 'validation')
        with self.assertRaisesRegex(ValueError, 'leakage'):
            training.fit(p, v, selection.candidate(9, .1), self.cfg, self.root/'fit', 'synthetic')

    def test_finite_LR_coefficients_and_convergence_are_enforced(self):
        p = panel()
        r = training.fit(p, None, selection.candidate(9, .1), self.cfg, self.root/'fit', 'synthetic')
        payload = training.load_checkpoint(r['checkpoint'], r['identity'])
        payload['model'].coef_[0, 0] = np.nan
        with self.assertRaises(ValueError): training.predict(payload, p.features)
        cfg = {**self.cfg, 'logistic_max_iter': 1}
        from sklearn.exceptions import ConvergenceWarning
        with self.assertRaises(ConvergenceWarning):
            training.fit(p, None, selection.candidate(9, .1), cfg, self.root/'nonconverge', 'synthetic')

    def test_RNG_python_numpy_torch_roundtrip(self):
        saved = rng_state()
        first = (random.random(), np.random.rand(), torch.rand(2))
        restore_rng(saved)
        second = (random.random(), np.random.rand(), torch.rand(2))
        self.assertEqual(first[:2], second[:2])
        self.assertTrue(torch.equal(first[2], second[2]))

    def test_outer_requires_authorization_before_any_access(self):
        with self.assertRaisesRegex(ValueError, 'authorization'):
            pipeline.evaluate_frozen(None, None, None, self.root)

    def test_once_only_outer_claim_and_seal(self):
        fn = lambda: {'kind': 'SYNTHETIC_TEST_FIXTURE', 'score': .25}
        self.assertEqual(seal_outer(self.root/'outer', {'x': 1}, fn), fn())
        self.assertEqual(seal_outer(self.root/'outer', {'x': 1}, lambda: self.fail('repeat')), fn())
        with self.assertRaises(ValueError): seal_outer(self.root/'outer', {'x': 2}, fn)

    def test_ambiguous_outer_claim_never_retries(self):
        def crash(): raise InterruptedError('synthetic inference interruption')
        with self.assertRaises(InterruptedError): seal_outer(self.root/'outer', {'x': 1}, crash)
        with self.assertRaisesRegex(ValueError, 'AMBIGUOUS'):
            seal_outer(self.root/'outer', {'x': 1}, lambda: self.fail('retry'))

    def test_frozen_config_or_missing_decisions_rejected(self):
        with self.assertRaisesRegex(ValueError, 'identity'):
            pipeline.assert_frozen({'identity': {'source_files': {}}, 'selections': []})
        with self.assertRaisesRegex(ValueError, 'three'):
            pipeline.assert_frozen({'identity': {'source_files': common.source_identity(), 'config': self.cfg}, 'selections': []})

    def test_metric_hand_calculation_and_undefined_speaker_rates(self):
        m = training.metrics([0, 0, 1, 1], [.1, .7, .4, .9])
        self.assertEqual(m['confusion_matrix'], [[1, 1], [1, 1]])
        self.assertEqual(m['accuracy'], .5)
        self.assertEqual(m['macro_f1'], .5)
        self.assertEqual(m['roc_auc'], .75)
        one = training.metrics([0, 0], [.1, .8])
        self.assertIsNone(one['sensitivity'])
        self.assertIsNone(one['roc_auc'])
        self.assertEqual(one['specificity'], .5)

    def test_speaker_average_uses_all_predictions_fixed_half_threshold(self):
        rows = [{'speaker_id': 'S', 'label': 1, 'probability': p, 'microphone': 'head'} for p in (.2, .8)]
        speakers, _ = reporting.speaker_summary(rows)
        self.assertEqual(speakers[0]['probability'], .5)
        self.assertEqual(speakers[0]['prediction'], 1)
        self.assertIsNone(speakers[0]['specificity'])

    def test_tie_rules_prefer_declared_simplicity(self):
        def result(spec):
            return {'candidate': spec, 'metrics': {'macro_f1': .5, 'balanced_accuracy': .5, 'roc_auc': .5, 'log_loss': .7}}
        a = result(selection.candidate(9, .1))
        b = result(selection.candidate(9, .1, augmentation=True))
        self.assertLess(training.selection_key(a), training.selection_key(b))

    def test_combined_candidate_requires_two_qualified_factors(self):
        def result(spec, scores):
            return {'candidate': spec, 'metrics': {'macro_f1': float(np.mean(scores)), 'balanced_accuracy': .6,
                                                   'roc_auc': .5, 'log_loss': .7},
                    'fits': [{'validation_metrics': {'macro_f1': score}} for score in scores]}
        ref = result(selection.candidate(9, .1), [.5]*3)
        weight = result(selection.candidate(9, .1, weighting='B2'), [.52]*3)
        norm = result(selection.candidate(9, .1, normalization='N1'), [.52]*3)
        self.assertIsNone(selection.combine(ref, [weight], self.cfg))
        combined = selection.combine(ref, [weight, norm], self.cfg)
        self.assertEqual(combined['weighting'], 'B2')
        self.assertEqual(combined['normalization'], 'N1')
        self.assertFalse(combined['augmentation'])

    def test_calibration_uses_three_separate_development_roles(self):
        c = SyntheticCorpus()
        for split in c.splits:
            train, cal, assess = calibration_roles(c, split)
            self.assertEqual([len(train), len(cal), len(assess)], [6, 2, 2])
            separated(c.select(train), c.select(cal), c.select(assess), c.select(split.test))
        report = diagnostic(c, c.splits[0], SyntheticCache(), self.cfg, self.root/'calibration', 'synthetic')
        self.assertEqual(report['threshold'], .5)
        self.assertIn('excluded', report['label'])

    def test_preservation_mismatch_is_fail_closed(self):
        before = {'original': {'raw': {'size': 1}}, 'accuracy_hubert': {}, 'accuracy_hubert_v2': {}}
        after = copy.deepcopy(before)
        after['original']['raw']['size'] = 2
        with patch.object(common, 'read', return_value=before), patch.object(common, 'snapshot', return_value=after), \
             patch.object(common, 'write_json'):
            with self.assertRaisesRegex(ValueError, 'Protected data differ'): common.verify_preservation()

    def test_launcher_defaults_development_and_requires_explicit_outer(self):
        text = (common.ROOT/'run_all.ps1').read_text()
        self.assertIn("$Mode = 'Develop'", text)
        self.assertIn('[switch]$AuthorizeOuter', text)
        self.assertIn('.venv-accuracy', text)
        self.assertIn('PYTHONDONTWRITEBYTECODE', text)
        with self.assertRaises(ValueError): pipeline.main(['outer'])

    def test_csv_serialization_stable_across_frozen_json_key_sorting(self):
        rows = [{'z_metric': .25, 'a_candidate': 'SYNTHETIC', 'evidence_kind': 'SYNTHETIC_TEST_FIXTURE'}]
        path = self.root/'stable.csv'
        reporting.csv_publish(path, rows)
        before = path.read_bytes()
        reloaded = json.loads(json.dumps(rows, sort_keys=True))
        reporting.csv_publish(path, reloaded)
        self.assertEqual(path.read_bytes(), before)

    def test_synthetic_end_to_end_selection_freeze_outer_reports_and_resume(self):
        c, cache = SyntheticCorpus(), SyntheticCache()
        env = {'versions': common.versions(), 'gpu': 'SYNTHETIC'}
        (self.root/'logs').mkdir()
        (self.root/'V2_LINEAGE.json').write_bytes((common.ROOT/'V2_LINEAGE.json').read_bytes())
        with patch.object(pipeline, 'ROOT', self.root), patch.object(pipeline, 'AugmentProvider', SyntheticProvider):
            frozen = pipeline.develop(c, cache, None, {'features': {}}, {}, self.cfg, env)
        self.assertEqual(len(frozen['selections']), 3)
        self.assertFalse((self.root/'results/outer').exists())
        self.assertIn('gpu_peak_reserved_bytes', frozen['runtime_evidence'])
        # Simulate interruption between decision publication and its seal: no outer claim yet.
        (self.root/'results/FROZEN_DECISIONS.seal.json').unlink()
        with patch.object(pipeline, 'ROOT', self.root), patch.object(pipeline, 'select_fold', side_effect=AssertionError('Repeated development')):
            self.assertEqual(pipeline.develop(c, cache, None, {'features': {}}, {}, self.cfg, env), frozen)
        self.assertTrue((self.root/'results/FROZEN_DECISIONS.seal.json').exists())
        results = pipeline.evaluate_frozen(c, cache, frozen, self.root/'results', authorize_outer=True)
        self.assertEqual(len(results), 6)
        self.assertEqual(results[0]['predictions'][0]['evidence_kind'], 'SYNTHETIC_TEST_FIXTURE')
        seal_path = self.root/'results/FROZEN_DECISIONS.seal.json'
        correct_seal = common.read(seal_path)
        common.write_json(seal_path, {'sha256': 'tampered'}, replace=True)
        with self.assertRaisesRegex(ValueError, 'seal mismatch'):
            pipeline.evaluate_frozen(c, cache, frozen, self.root/'results', authorize_outer=True)
        common.write_json(seal_path, correct_seal, replace=True)
        with patch.object(pipeline, 'predict', side_effect=AssertionError('Repeated outer forward')):
            self.assertEqual(pipeline.evaluate_frozen(c, cache, frozen, self.root/'results', authorize_outer=True), results)
        baselines = {'SYNTHETIC baseline': {m: {'mean': .5, 'sample_std': .1} for m in ('accuracy', 'macro_f1')}}
        output, figures = self.root/'reports', self.root/'figures'
        report = reporting.final_report(c, results, frozen['selections'], frozen, output, figures, baselines,
                                        data_kind='SYNTHETIC_TEST_FIXTURE')
        self.assertEqual(report['data_kind'], 'SYNTHETIC_TEST_FIXTURE')
        self.assertNotIn('accuracy is not yet known', (output/'DEVELOPMENT_REPORT.md').read_text())
        self.assertEqual(len(list(figures.glob('*.png'))), 5)
        self.assertEqual(reporting.final_report(c, results, frozen['selections'], frozen, output, figures, baselines,
                                                data_kind='SYNTHETIC_TEST_FIXTURE'), report)
        broken = copy.deepcopy(results)
        broken[0]['predictions'].pop()
        with self.assertRaises(ValueError):
            reporting.final_report(c, broken, frozen['selections'], frozen, self.root/'bad', figures, baselines,
                                   data_kind='SYNTHETIC_TEST_FIXTURE')


if __name__ == '__main__':
    unittest.main()
