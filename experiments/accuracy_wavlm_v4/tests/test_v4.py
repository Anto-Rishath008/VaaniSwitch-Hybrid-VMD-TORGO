import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np
import torch
import copy
import json
import random
import uuid
from types import SimpleNamespace
from experiments.accuracy_wavlm_v4.src.common import (ROOT, atomic, checkpoint, config, digest,
    inside, load_checkpoint, publish, read, sha)
from experiments.accuracy_wavlm_v4.src.model import (load_backbone, AttentionPool, Classifier, frame_mask, pool, inputs)
from experiments.accuracy_wavlm_v4.src.data import (Corpus, Record, Partition, audit, development_cv,
    segment, separated)
from experiments.accuracy_wavlm_v4.src.numerics import finite, step
from experiments.accuracy_wavlm_v4.src.training import (fit_lr, fit_neural, predict_lr,
    restore_delta, delta_state, rng_state, restore_rng, metrics)
from experiments.accuracy_wavlm_v4.src.features import Cache
from experiments.accuracy_wavlm_v4.src.selection import candidates, selection_key, aggregate
from experiments.accuracy_wavlm_v4.src.pipeline import require_authorization
from experiments.accuracy_wavlm_v4.src.evaluation import seal, evaluate


def output(name):
    path = ROOT/'validation/unit'/f'{name}-{uuid.uuid4().hex}'
    path.mkdir(parents=True)
    return path


def record(speaker, label, n=0):
    return Record(f'{speaker}/session/p{n}', speaker, 'dysarthric' if label else 'control',
        f'data/raw/{speaker}/{n}.wav', digest([speaker, n]), 1., 16000, 16000, 'head')


def panels():
    training = tuple(record(s, y, i) for s, y, count in [('a', 0, 3), ('b', 1, 1)] for i in range(count))
    validation = (record('c', 0), record('d', 1))
    return (Partition(training, ('a', 'b'), ('outer',), 'train', 1),
        Partition(validation, ('c', 'd'), ('outer',), 'validation', 1))


class DataContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = Corpus()

    def test_real_core(self):
        self.assertEqual(len(self.corpus.records), 8673)
        self.assertEqual(len(self.corpus.mapping), 15)
        self.assertEqual(sum(self.corpus.select([s])[0].label for s in self.corpus.mapping), 8)

    def test_real_cv_coverage_and_isolation(self):
        result = audit(self.corpus)
        self.assertEqual(len(result['development_cv']), 3)
        for split in self.corpus.splits:
            pairs = development_cv(self.corpus, split)
            self.assertEqual(sorted(s for _, v in pairs for s in v), list(split.development))

    def test_outer_role_rejected(self):
        train, _ = panels()
        with self.assertRaisesRegex(ValueError, 'Outer'):
            Partition(train.records, train.allowed, train.forbidden, 'outer', 1).check()

    def test_forbidden_speaker_rejected(self):
        train, _ = panels()
        with self.assertRaises(ValueError):
            Partition(train.records, train.allowed, ('a',), 'train', 1).check()

    def test_duplicate_record_rejected(self):
        train, _ = panels()
        with self.assertRaises(ValueError):
            Partition(train.records+train.records[:1], train.allowed, train.forbidden, 'train', 1).check()

    def test_audio_hash_duplicate_across_roles_rejected(self):
        a, b = record('a', 0), record('b', 1)
        b = Record(**{**vars(b), 'primary_audio_sha256': a.primary_audio_sha256})
        with self.assertRaisesRegex(ValueError, 'primary_audio_sha256'):
            separated((a,), (b,))

    def test_speaker_weight_mass(self):
        train, _ = panels()
        w = train.weights()
        self.assertAlmostEqual(w.mean(), 1.)
        self.assertAlmostEqual(w[:3].sum(), w[3])

    def test_validation_weight_fit_rejected(self):
        _, val = panels()
        with self.assertRaises(ValueError):
            val.weights()

    def test_exact_center_crop(self):
        wave = np.arange(160000, dtype=np.float32)
        out, spec = segment(wave, 16000, 4)
        self.assertEqual(len(out), 64000)
        self.assertEqual(spec['start_sample_16k'], 48000)
        np.testing.assert_array_equal(out, wave[48000:112000])

    def test_resampling_and_stereo(self):
        wave = np.ones((8000, 2), np.float32)
        out, spec = segment(wave, 8000, 4)
        self.assertEqual(out.shape, (16000,))
        self.assertEqual(spec['right_padding_samples'], 0)

    def test_silence_is_finite(self):
        out, _ = segment(np.zeros(400, np.float32), 16000, 4)
        self.assertTrue(np.isfinite(out).all())

    def test_short_audio_fails_without_exclusion(self):
        with self.assertRaisesRegex(ValueError, 'stop without discarding'):
            segment(np.ones(399), 16000, 4)

    def test_nonfinite_audio_fails(self):
        wave = np.ones(400); wave[0] = np.nan
        with self.assertRaises(ValueError):
            segment(wave, 16000, 4)


class ModelContracts(unittest.TestCase):
    def test_masked_mean_std_ignores_padding(self):
        h = torch.tensor([[[1., 3.], [3., 7.], [1000., 1000.]]])
        result = pool(h, torch.tensor([2]), 'mean_std')
        torch.testing.assert_close(result, torch.tensor([[2., 5., 1., 2.]]))

    def test_mask_invalid_lengths(self):
        for length in (0, 4):
            with self.assertRaises(ValueError):
                pool(torch.ones(1, 3, 2), torch.tensor([length]))

    def test_attention_padding_and_gradient(self):
        head = AttentionPool(2)
        h = torch.tensor([[[1., 3.], [2., 5.], [900., 900.]]], requires_grad=True)
        p = head(h, torch.tensor([2]))
        p.sum().backward()
        self.assertTrue(torch.equal(h.grad[:, 2], torch.zeros(1, 2)))
        self.assertTrue(all(p.grad is not None for p in head.parameters()))

    def test_tiny_real_wavlm_selective_gradient(self):
        from transformers import WavLMConfig, WavLMModel
        cfg = WavLMConfig(hidden_size=16, num_hidden_layers=4, num_attention_heads=2,
            intermediate_size=32, conv_dim=(8, 8, 8), conv_kernel=(3, 3, 3), conv_stride=(2, 2, 2),
            num_conv_pos_embeddings=8, num_conv_pos_embedding_groups=2, num_buckets=16, max_bucket_distance=32)
        model = Classifier(WavLMModel(cfg), 2, 'mean', True).train()
        self.assertFalse(model.backbone.feature_extractor.training)
        self.assertFalse(model.backbone.encoder.layers[0].training)
        self.assertTrue(model.backbone.encoder.layers[3].training)
        x = torch.randn(2, 160)
        logits = model(x, torch.ones_like(x, dtype=torch.long))
        self.assertEqual(logits.shape, (2,))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.]))
        loss.backward()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        step(model, optimizer, 1.)
        self.assertEqual(model.parameter_report()['upper_layers_zero_based'], [2, 3])

    def test_unapproved_depth_rejected(self):
        with self.assertRaises(ValueError):
            Classifier(torch.nn.Linear(2, 2), 12, 'mean')

    def test_nonfinite_pool_fails(self):
        with self.assertRaises(ValueError):
            pool(torch.full((1, 2, 3), float('inf')), torch.tensor([2]))

    def test_delta_restore_requires_exact_keys(self):
        model = torch.nn.Linear(2, 1)
        state = delta_state(model)
        state.pop('bias')
        with self.assertRaises(ValueError):
            restore_delta(model, state)


class CacheCheckpointContracts(unittest.TestCase):
    def test_atomic_no_overwrite(self):
        p = output('atomic')/'file'
        atomic(p, b'first')
        with self.assertRaises(FileExistsError):
            atomic(p, b'second')
        self.assertEqual(p.read_bytes(), b'first')

    def test_output_escape_rejected(self):
        with self.assertRaises(ValueError):
            atomic(ROOT.parent/'bad.txt', b'x')

    def test_checkpoint_checksum_and_identity(self):
        ref = checkpoint(output('checkpoint'), {'identity': 'x', 'model': torch.arange(3)})
        torch.testing.assert_close(load_checkpoint(ref, 'x')['model'], torch.arange(3))
        with self.assertRaises(ValueError):
            load_checkpoint(ref, 'different')
        with Path(ref['path']).open('ab') as stream:
            stream.write(b'corruption-fixture')
        with self.assertRaises(ValueError):
            load_checkpoint(ref, 'x')

    def test_unpublished_partial_checkpoint_ignored(self):
        root = output('partial')
        (root/'state.pt.partial').write_bytes(b'incomplete')
        self.assertFalse((root/'latest.json').exists())

    def test_rng_restoration(self):
        state = rng_state()
        a = (random.random(), np.random.rand(), torch.rand(4))
        restore_rng(state)
        b = (random.random(), np.random.rand(), torch.rand(4))
        self.assertEqual(a[:2], b[:2]); self.assertTrue(torch.equal(a[2], b[2]))

    def test_cache_roundtrip_and_corruption(self):
        cache = Cache(output('cache'), {'model': 'synthetic'})
        r = record('a', 0)
        x = np.ones((2, 1536), np.float32)
        cache.save(r, x, {'start': 0})
        np.testing.assert_array_equal(cache.load(r), x)
        with self.assertRaises(ValueError):
            cache.save(r, x*2, {})
        with (cache.path(r)/'features.npy').open('ab') as f:
            f.write(b'corruption')
        with self.assertRaises(ValueError):
            cache.load(r)

    def test_cache_source_identity_bound(self):
        cache = Cache(output('source'), {'model': 'synthetic'})
        r = record('a', 0)
        cache.save(r, np.ones((2, 1536), np.float32), {})
        changed = Record(**{**vars(r), 'primary_audio_sha256': 'changed'})
        with self.assertRaises(ValueError):
            cache.load(changed)


class TrainingContracts(unittest.TestCase):
    def test_v4_requires_deterministic_attention_backend(self):
        from experiments.accuracy_wavlm_v4.src.common import environment
        environment()
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
        self.assertTrue(torch.backends.cuda.math_sdp_enabled())
        self.assertFalse(torch.backends.cuda.mem_efficient_sdp_enabled())
        self.assertFalse(torch.backends.cuda.flash_sdp_enabled())
        self.assertFalse(torch.backends.cuda.cudnn_sdp_enabled())

    def test_lr_train_only_scaler_and_resume(self):
        train, val = panels()
        x = np.array([[-2, 1], [-1, 2], [-3, 3], [2, 4]], np.float32)
        vx = np.array([[100, 200], [300, 400]], np.float32)
        cfg = config(); spec = candidates(cfg)[0][0]
        root = output('lr')
        with self.assertRaises(InterruptedError):
            fit_lr(train, val, x, vx, spec, cfg, root, 'unit', interrupt='after_start')
        result = fit_lr(train, val, x, vx, spec, cfg, root, 'unit')
        payload = load_checkpoint(result['checkpoint'], result['identity'])
        np.testing.assert_allclose(payload['scaler'].mean_, np.average(x, axis=0, weights=train.weights()), atol=1e-7)
        self.assertEqual(result, fit_lr(train, val, x, vx, spec, cfg, root, 'unit'))

    def test_lr_rejects_shared_speaker(self):
        train, _ = panels()
        val = Partition(train.records, train.allowed, train.forbidden, 'validation', 1)
        with self.assertRaises(ValueError):
            fit_lr(train, val, np.ones((4, 3)), np.ones((4, 3)), {}, config(), output('leak'), 'unit')

    def test_lr_publication_recovery(self):
        train, val = panels()
        x = np.arange(12, dtype=np.float32).reshape(4, 3)
        vx = x[:2]
        spec = candidates(config())[0][0]; root = output('lrpublish')
        with self.assertRaises(InterruptedError):
            fit_lr(train, val, x, vx, spec, config(), root, 'unit', interrupt='after_checkpoint')
        with patch('sklearn.linear_model.LogisticRegression.fit', side_effect=AssertionError('must not refit')):
            result = fit_lr(train, val, x, vx, spec, config(), root, 'unit')
        self.assertTrue((root/'complete.json').exists())

    def test_oom_stops_and_keeps_checkpoint(self):
        train, val = panels(); root = output('oom')
        fake = torch.nn.Linear(3, 1)
        optimizer = torch.optim.AdamW(fake.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: 1.)
        def oom(_):
            raise torch.cuda.OutOfMemoryError('deliberate test injection')
        with patch('experiments.accuracy_wavlm_v4.src.training.build', return_value=(fake, None, {})), \
             patch('experiments.accuracy_wavlm_v4.src.training.optimizer_for', return_value=(optimizer, scheduler)):
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                fit_neural(train, val, candidates(config())[1][0], config(), None, root, 'unit', fault=oom)
        ref = read(root/'latest.json')
        self.assertEqual(load_checkpoint(ref, ref['identity'])['step'], 0)
        failures = list((root/'events').glob('*.json'))
        self.assertEqual(len(failures), 1)
        self.assertIn('OutOfMemoryError', read(failures[0])['type'])
        self.assertFalse((root/'complete.json').exists())

    def test_nonfinite_optimizer_rejected(self):
        with self.assertRaisesRegex(ValueError, 'optimizer'):
            finite({'exp_avg': torch.tensor([float('nan')])}, 'optimizer')

    def test_metrics_known_confusion(self):
        m = metrics([0, 0, 1, 1], [.1, .8, .3, .9])
        self.assertEqual(m['accuracy'], .5)
        self.assertEqual(m['macro_f1'], .5)
        self.assertEqual(m['confusion_matrix'], [[1, 1], [1, 1]])

    def test_single_class_specificity_is_null(self):
        self.assertIsNone(metrics([1, 1], [.2, .8])['specificity'])


class ExecutionGates(unittest.TestCase):
    def test_full_selection_orchestration_uses_only_development(self):
        from experiments.accuracy_wavlm_v4.src.selection import select_fold
        from experiments.accuracy_wavlm_v4.src.training import validate_roles
        corpus = Corpus(); split = corpus.splits[0]; calls = []
        metric = metrics([0, 1], [.2, .8])
        def fake_fit(train, val, candidate):
            validate_roles(train, val)
            self.assertFalse({r.speaker_id for r in train.records} & set(split.test))
            if val:
                self.assertFalse({r.speaker_id for r in val.records} & set(split.test))
            calls.append((candidate['family'], val is not None))
            return {'candidate': candidate, 'validation_metrics': metric if val else None,
                'best_epoch': 1, 'checkpoint': {'synthetic': True}}
        def lr(train, val, x, vx, candidate, *args, **kwargs):
            return fake_fit(train, val, candidate)
        def neural(train, val, candidate, *args, **kwargs):
            return fake_fit(train, val, candidate)
        provider = SimpleNamespace(get=lambda p,c: np.zeros((len(p.records), 3)), close=lambda: None)
        with patch('experiments.accuracy_wavlm_v4.src.selection.fit_lr', side_effect=lr), \
             patch('experiments.accuracy_wavlm_v4.src.selection.fit_neural', side_effect=neural):
            result = select_fold(corpus, split, provider, None, config(), output('selection'), 'synthetic')
        self.assertEqual(len(result['comparisons']), 11)
        self.assertEqual(calls[:24], [('A', True)]*24)
        self.assertEqual(calls[24:33], [('B', True)]*9)
        self.assertEqual(calls[33:], [('A', False)])

    def test_fold_sample_sd_uses_ddof_one(self):
        candidate = candidates(config())[0][0]
        values = []
        for p in (.2, .5, .8):
            values.append({'validation_metrics': {k:p for k in ('macro_f1','balanced_accuracy','accuracy','roc_auc','sensitivity','specificity','log_loss')}})
        summary = aggregate(candidate, values)
        self.assertAlmostEqual(summary['sample_std']['accuracy'], .3)

    def test_small_grid(self):
        a, b = candidates(config())
        self.assertEqual((len(a), len(b)), (8, 3))
        self.assertEqual(len({c['id'] for c in a+b}), 11)

    def test_development_requires_authorization(self):
        with self.assertRaises(ValueError):
            require_authorization('Develop')

    def test_outer_requires_separate_authorization(self):
        with self.assertRaises(ValueError):
            require_authorization('Outer', True, False)
        with self.assertRaises(ValueError):
            require_authorization('Develop', True, True)

    def test_outer_gate_precedes_any_data_access(self):
        with self.assertRaisesRegex(ValueError, 'authorization'):
            evaluate(None, None, None, None, None, None)

    def test_consumed_outer_claim_cannot_retry(self):
        root = output('outerclaim')
        calls = []
        def fail():
            calls.append(1)
            raise RuntimeError('deliberate interruption')
        with self.assertRaises(RuntimeError):
            seal(root, {'synthetic': True}, fail)
        with self.assertRaises(ValueError):
            seal(root, {'synthetic': True}, fail)
        self.assertEqual(len(calls), 1)

    def test_sealed_result_reused_without_inference(self):
        root = output('outerseal')
        value = seal(root, {'synthetic': True}, lambda: {'fixture': 1})
        again = seal(root, {'synthetic': True}, lambda: self.fail('no second forward'))
        self.assertEqual(value, again)

    def test_outer_claim_race_cannot_forward_twice(self):
        from experiments.accuracy_hubert_v2.src.common import require as original_require
        root = output('claimrace'); identity = {'synthetic': True}
        def inject(condition, message):
            original_require(condition, message)
            if message.startswith('Outer claim consumed'):
                publish(root/'claim.json', {'identity': identity, 'status': 'CONSUMED_BEFORE_FORWARD'})
        with patch('experiments.accuracy_wavlm_v4.src.evaluation.require', side_effect=inject):
            with self.assertRaises((ValueError, FileExistsError)):
                seal(root, identity, lambda: self.fail('Concurrent claim winner already owns inference'))

    def test_selection_rejects_outer_metric(self):
        with self.assertRaises(ValueError):
            selection_key({'evidence_kind': 'OUTER'})

    def test_deterministic_tie_prefers_frozen(self):
        a, b = candidates(config())
        metric = {'macro_f1': .5, 'balanced_accuracy': .5, 'accuracy': .5, 'roc_auc': .5, 'log_loss': .7}
        values = [{'candidate': c, 'mean': metric, 'evidence_kind': 'DEVELOPMENT_GROUPED_CV'} for c in [b[0], a[0]]]
        self.assertEqual(min(values, key=selection_key)['candidate']['family'], 'A')

    def test_preservation_baseline_excludes_only_v4_study_work(self):
        before = read(ROOT/'validation/protected_before.json')
        self.assertFalse(any(k.startswith('experiments/accuracy_wavlm_v4/') for k in before['study']))
        for name in ('accuracy_hubert', 'accuracy_hubert_v2', 'accuracy_hubert_v3'):
            values = [v for k, v in before['study'].items() if k.startswith('experiments/'+name+'/')]
            self.assertGreater(len(values), 100)
            self.assertTrue(all('sha256' in v for v in values))


class ProcessorRegression(unittest.TestCase):
    def test_official_processor_normalization_is_preserved(self):
        from transformers import Wav2Vec2FeatureExtractor
        processor = Wav2Vec2FeatureExtractor.from_pretrained(
            ROOT/'cache/model/4c66d4806a428f2e922ccfa1a962776e232d487b', local_files_only=True)
        self.assertFalse(processor.do_normalize)
        backbone = MagicMock()
        backbone.config.hidden_size = 768
        backbone.encoder.layers = [None]*12
        info = {'missing_keys': [], 'mismatched_keys': [], 'error_msgs': [], 'unexpected_keys': []}
        with patch('transformers.WavLMModel.from_pretrained', return_value=(backbone, info)):
            _, actual, _ = load_backbone(ROOT/'cache/model/4c66d4806a428f2e922ccfa1a962776e232d487b')
        self.assertFalse(actual.do_normalize)


if __name__ == '__main__':
    unittest.main()
