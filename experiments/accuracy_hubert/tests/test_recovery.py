"""CUDA state recovery on synthetic development records, never real outer data."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, config, read, write_json
from src.data import Record
from src.features import Extractor, acquire
from src.reporting import aggregate, speaker_analysis
from src.training import Classifier, Partition, load_checkpoint, seed_all, train_neural


def synthetic():
    records = tuple(Record(f"{s}/Session1/{i:04d}", s, "control" if i < 2 else "dysarthric",
                           f"synthetic/{i}.wav", "a"*64, 1.0, 16000, 16000, "head")
                    for i, s in enumerate(("A", "B", "C", "D")))
    return Partition(records, np.random.default_rng(1).normal(size=(4,1536)).astype(np.float32),
                     ("A","B","C","D"), ("T",), "development")


class Recovery(unittest.TestCase):
    def setUp(self):
        (ROOT/"smoke").mkdir(exist_ok=True)
        torch.set_num_threads(2)

    def test_fold_mean_sample_std_not_pooled(self):
        from src.training import metrics
        rows = []
        for fold, scores in enumerate(([.1,.9],[.9,.1],[.1,.9]), 1):
            rows.append({"outer_fold": fold, "metrics": metrics([0,1], scores),
                         "training_seconds": fold, "parameter_count": 1537})
        result = aggregate(rows)
        self.assertAlmostEqual(result["accuracy"]["mean"], 2/3)
        self.assertAlmostEqual(result["accuracy"]["sample_std"], np.std([1,0,1], ddof=1))
        with self.assertRaises(ValueError):
            aggregate(rows[:2])

    def test_speaker_mean_probability_at_fixed_threshold(self):
        rows = [{"model":"E1", "outer_fold":1, "speaker_id":"A", "label":0, "probability":p}
                for p in [.4,.7,.4]]
        rows += [{"model":"E1", "outer_fold":2, "speaker_id":"B", "label":1, "probability":.6}]
        speakers, summary = speaker_analysis(rows)
        self.assertEqual(speakers[0]["probability"], .5)
        self.assertEqual(summary["incorrect_speakers"], ["A"])
        self.assertEqual(summary["metrics"]["confusion_matrix"], [[0,1],[0,1]])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_e3_cuda_backward_under_declared_seed_policy(self):
        seed_all()
        model = Classifier("E3", .3).cuda()
        speech = torch.randn(2,1536, device="cuda")
        rp = torch.randint(0,2,(2,1,224,224), dtype=torch.uint8, device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(speech, rp)
        output.float().square().mean().backward()
        self.assertTrue(torch.isfinite(model.rp.blocks[0][0].weight.grad).all())
        model.eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            self.assertTrue(torch.equal(model(speech,rp), model(speech,rp)))
        del model
        torch.cuda.empty_cache()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for checkpoint interruption regression")
    def test_terminal_epoch_checkpoint_resumes_before_summary_publication(self):
        cfg = config()
        cfg["training"]["batch_cap"] = 2
        train = synthetic()
        candidate = cfg["training"]["neural_candidates"][0]
        with tempfile.TemporaryDirectory(dir=ROOT/"smoke") as directory:
            root = Path(directory)
            def interrupted(path, value, **kwargs):
                if Path(path).name == "complete.json":
                    raise RuntimeError("interrupted before publication")
                return write_json(path, value, **kwargs)
            with patch("src.training.write_json", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "interrupted before publication"):
                    train_neural(train, None, "E2", candidate, root, "terminal-unit", 6, cfg, fixed_epochs=1)
            previous = read(root/"latest.json")
            result = train_neural(train, None, "E2", candidate, root, "terminal-unit", 6, cfg, fixed_epochs=1)
            self.assertEqual(result["checkpoint"], previous)
            self.assertEqual(result["best_epoch"], 1)
            payload = load_checkpoint(previous, "terminal-unit")
            self.assertIn("optimizer", payload)
            self.assertIn("cuda", payload["rng"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_epoch_boundary_resume_matches_uninterrupted_parameters(self):
        cfg = config()
        cfg["training"]["batch_cap"] = 2
        train = synthetic()
        candidate = cfg["training"]["neural_candidates"][0]
        with tempfile.TemporaryDirectory(dir=ROOT/"smoke") as directory:
            base = Path(directory)
            reference = train_neural(train, None, "E2", candidate, base/"reference", "resume-unit", 6, cfg, fixed_epochs=2)
            from src.training import save_checkpoint
            def interrupted(root, payload):
                ref = save_checkpoint(root, payload)
                if payload["epoch"] == 1:
                    raise RuntimeError("epoch boundary interruption")
                return ref
            with patch("src.training.save_checkpoint", side_effect=interrupted):
                with self.assertRaises(RuntimeError):
                    train_neural(train, None, "E2", candidate, base/"interrupted", "resume-unit", 6, cfg, fixed_epochs=2)
            resumed = train_neural(train, None, "E2", candidate, base/"interrupted", "resume-unit", 6, cfg, fixed_epochs=2)
            a = load_checkpoint(reference["checkpoint"], "resume-unit")
            b = load_checkpoint(resumed["checkpoint"], "resume-unit")
            for name in a["model"]:
                torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)

    @unittest.skipUnless(os.environ.get("ACCURACY_CUDA_FEATURE_TEST") == "1", "Opt-in real checkpoint test")
    def test_actual_hubert_dimensions_masks_and_frozen_cuda(self):
        cfg = config()
        checkpoint, _ = acquire(cfg)
        extractor = Extractor(cfg, checkpoint)
        wave = np.sin(np.arange(16000, dtype=np.float32)*.01)
        first = extractor.infer([wave])
        second = extractor.infer([wave])
        self.assertEqual(first.shape, (1,3,1536))
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_array_equal(first, second)
        self.assertFalse(extractor.model.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in extractor.model.parameters()))
        with self.assertRaisesRegex(ValueError, "exact-length"):
            extractor.infer([wave, wave[:8000]])
        extractor.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
