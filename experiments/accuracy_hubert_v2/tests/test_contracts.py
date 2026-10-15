"""Focused scientific contracts; live RP reads are Fold 1 development only."""
import ast
import copy
import dataclasses
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, ORIGINAL, Stages, digest, read, sha, write_json
from src.data import Corpus, Record, guard_records, load_rp, training_weights, validate_microphone
from src.features import FeatureCache, acquire, audio, feature_lengths, masked_pool
from src.models import Classifier
from src.training import (Partition, fit_scaler, load_checkpoint, save_checkpoint,
                          seal_outer, selection_key, metrics)


def fake(speaker="A", number=0, condition="control"):
    return Record(f"{speaker}/Session1/{number:04d}", speaker, condition,
                  f"data/raw/{speaker}/{number}.wav", "a"*64, 1.0, 16000, 16000, "head")


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT/"smoke").mkdir(exist_ok=True)
        cls.corpus = Corpus()

    def test_dataset_identities_and_core_counts(self):
        self.assertEqual(len(self.corpus.records), 8673)
        self.assertEqual(len(self.corpus.by_id), 8673)
        self.assertEqual(self.corpus.audit()["cropped_recordings"], 235)
        self.assertEqual(self.corpus.audit()["primary_microphones"], {"head": 7503, "array": 1170})

    def test_original_nested_speakers(self):
        self.assertEqual([s.validation for s in self.corpus.splits],
                         [("M03", "MC02"), ("F03", "FC03"), ("F03", "MC02")])
        self.assertEqual([len(self.corpus.select(s.test)) for s in self.corpus.splits], [2819, 2873, 2981])
        for split in self.corpus.splits:
            self.assertFalse(set(split.train) & set(split.validation))
            self.assertFalse(set(split.development) & set(split.test))

    def test_primary_head_preference_and_array_fallback(self):
        row = {"head_audio_status": "usable", "head_mic_path": "head.wav",
               "array_audio_status": "usable", "array_mic_path": "array.wav",
               "primary_audio_path": "head.wav", "primary_microphone": "head",
               "primary_selection_reason": "head_structurally_usable"}
        validate_microphone(row)
        row["primary_audio_path"] = "array.wav"
        with self.assertRaises(ValueError):
            validate_microphone(row)
        for status, reason in [("missing", "array_structurally_usable_head_missing"),
                               ("unusable", "head_unusable_array_fallback")]:
            row.update(head_audio_status=status, primary_microphone="array", primary_selection_reason=reason)
            validate_microphone(row)

    def test_rp_nested_source_metadata_regression(self):
        # AH-001: serialized RP metadata stores source under representation_metadata.
        split = self.corpus.splits[0]
        record = self.corpus.select(split.train)[0]
        values = load_rp(record)
        self.assertEqual(values.shape, (224, 224))
        self.assertEqual(values.dtype, np.uint8)

    def test_native_b2_code_is_exact_except_isolated_imports(self):
        expected = (ORIGINAL/"src/vaaniswitch/m3_encoders/recurrence_encoder.py").read_text(encoding="utf-8")
        expected = expected.replace("from vaaniswitch.m3_encoders.config import RecurrenceEncoderConfig",
                                    "from .native_b2_contract import RecurrenceEncoderConfig")
        expected = expected.replace("from vaaniswitch.m3_encoders.preprocessing import preprocess_recurrence",
                                    "from .native_b2_preprocessing import preprocess_recurrence")
        self.assertEqual(ast.dump(ast.parse(expected)), ast.dump(ast.parse(
            (ROOT/"src/native_b2_encoder.py").read_text(encoding="utf-8-sig"))))
        self.assertEqual(sha(ORIGINAL/"src/vaaniswitch/m3_encoders/preprocessing.py"),
                         sha(ROOT/"src/native_b2_preprocessing.py"))

    def test_masked_pool_excludes_padding_population_std(self):
        hidden = torch.tensor([[[1., 2.], [3., 4.], [1000., 2000.]]])
        result = masked_pool(hidden, torch.tensor([2]))
        torch.testing.assert_close(result, torch.tensor([[2., 3., 1., 1.]]))
        with self.assertRaises(ValueError):
            masked_pool(hidden, torch.tensor([0]))

    def test_correct_convolution_mask_lengths(self):
        result = feature_lengths(torch.tensor([16000, 128000]), [10,3,3,3,3,2,2], [5,2,2,2,2,2,2])
        self.assertEqual(result.tolist(), [49, 399])

    def test_feature_cache_resume_and_corruption(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"smoke") as directory:
            cache = FeatureCache(Path(directory), {"model_revision": "fixed"})
            record, values = fake(), np.zeros((3, 1536), dtype=np.float32)
            self.assertIsNone(cache.load(record))
            cache.save(record, values, {"cropped": False})
            metadata = cache.location(record)/"complete.json"
            original = metadata.read_bytes()
            np.testing.assert_array_equal(cache.load(record), values)
            cache.save(record, values, {"cropped": False})
            self.assertEqual(metadata.read_bytes(), original)
            with self.assertRaises(ValueError):
                cache.save(record, values+1, {})
            document = read(metadata)
            (metadata.parent/document["filename"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "corruption"):
                cache.load(record)

    def test_cache_invalidated_source_identity_fails_closed(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"smoke") as directory:
            cache = FeatureCache(Path(directory), {"model_revision": "fixed"})
            record = fake()
            cache.save(record, np.zeros((3, 1536), np.float32), {})
            with self.assertRaisesRegex(ValueError, "identity"):
                cache.load(dataclasses.replace(record, primary_audio_sha256="b"*64))

    def test_scaler_train_only_and_outer_fit_rejected(self):
        records = (fake("A", 0), fake("B", 1, "dysarthric"))
        part = Partition(records, np.array([[1., 2.], [3., 4.]]), ("A", "B"), ("T",), "inner_train")
        scaler, provenance = fit_scaler(part)
        np.testing.assert_array_equal(scaler.mean_, [2., 3.])
        scaler.transform([[10000., 10000.]])
        np.testing.assert_array_equal(scaler.mean_, [2., 3.])
        self.assertEqual(provenance["speakers"], ["A", "B"])
        with self.assertRaises(ValueError):
            fit_scaler(dataclasses.replace(part, role="outer_test"))
        with self.assertRaises(ValueError):
            fit_scaler(dataclasses.replace(part, role="inner_validation"))

    def test_weights_training_speakers_only_equal_mass(self):
        records = (fake("A", 0), fake("A", 1), fake("B", 2, "dysarthric"))
        weights = training_weights(records, ("A", "B"), ("T",))
        self.assertAlmostEqual(weights[:2].sum(), weights[2])
        with self.assertRaises(ValueError):
            training_weights(records+(fake("T"),), ("A", "B"), ("T",))

    def test_selection_uses_inner_metrics_without_outer_field(self):
        a = {"validation_metrics": {"macro_f1": .7, "balanced_accuracy": .7, "roc_auc": .7},
             "validation_loss": .5, "layer": 6, "candidate": {"id": "a"}}
        b = copy.deepcopy(a)
        b["validation_metrics"]["macro_f1"] = .6
        b["outer_accuracy"] = 1.0
        self.assertLess(selection_key(a), selection_key(b))
        a["outer_accuracy"] = 0
        self.assertLess(selection_key(a), selection_key(b))

    def test_outer_partition_cannot_supply_selection_labels(self):
        class PoisonRecord:
            logical_id, speaker_id = "T/Session1/1", "T"
            @property
            def label(self):
                raise AssertionError("OUTER LABEL READ")
        part = Partition((PoisonRecord(),), np.zeros((1, 2)), ("T",), ("T",), "outer_test")
        with self.assertRaises(ValueError):
            fit_scaler(part)
        with self.assertRaises(ValueError):
            _ = part.labels

    def test_checkpoint_checksum_and_state_roundtrip(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"smoke") as directory:
            value = {"identity": "unit", "epoch": 1, "model": {"weight": torch.ones(2)},
                     "optimizer": {"step": 1}, "rng": torch.get_rng_state()}
            ref = save_checkpoint(Path(directory), value)
            loaded = load_checkpoint(ref, "unit")
            torch.testing.assert_close(loaded["model"]["weight"], torch.ones(2))
            self.assertEqual(loaded["optimizer"]["step"], 1)
            with self.assertRaises(ValueError):
                load_checkpoint(ref, "different")

    def test_stage_resume_skips_success_and_resumes_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"smoke") as directory:
            stages = Stages(Path(directory), "unit")
            with self.assertRaises(RuntimeError):
                stages.execute("work", lambda: (_ for _ in ()).throw(RuntimeError("interrupt")))
            self.assertEqual(stages.execute("work", lambda: (42, [])), 42)
            self.assertEqual(stages.execute("work", lambda: self.fail("repeated stage")), 42)

    def test_outer_seal_skips_inference_after_completion(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"smoke") as directory:
            result = seal_outer(Path(directory), "unit", lambda: {"score": .5})
            self.assertEqual(seal_outer(Path(directory), "unit", lambda: self.fail("repeated outer")), result)
            with self.assertRaises(ValueError):
                seal_outer(Path(directory), "changed", lambda: {})

    def test_ambiguous_outer_claim_blocks_reexecution(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"smoke") as directory:
            root = Path(directory)
            with self.assertRaises(RuntimeError):
                seal_outer(root, "unit", lambda: (_ for _ in ()).throw(RuntimeError("interrupted")))
            with self.assertRaisesRegex(ValueError, "AMBIGUOUS OUTER CLAIM"):
                seal_outer(root, "unit", lambda: self.fail("outer retry"))

    def test_sealed_outer_corruption_blocks_reexecution(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"smoke") as directory:
            root = Path(directory)
            seal_outer(root, "unit", lambda: {"score": .5})
            (root/"outer_result.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "seal invalid"):
                seal_outer(root, "unit", lambda: self.fail("outer retry"))

    def test_neural_forward_backward_and_deterministic_evaluation(self):
        torch.set_num_threads(2)
        for name in ("E2", "E3"):
            model = Classifier(name, .3)
            speech = torch.randn(2, 1536)
            rp = torch.randint(0, 2, (2, 1, 224, 224), dtype=torch.uint8) if name == "E3" else None
            output = model(speech, rp)
            self.assertTrue(torch.isfinite(output).all())
            output.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            if name == "E3":
                self.assertIsNotNone(model.rp.blocks[0][0].weight.grad)
            model.eval()
            with torch.inference_mode():
                self.assertTrue(torch.equal(model(speech, rp), model(speech, rp)))

    def test_metrics_known_confusion_and_auc(self):
        m = metrics([0, 0, 1, 1], [.1, .8, .2, .9])
        self.assertEqual(m["accuracy"], .5)
        self.assertEqual(m["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(m["roc_auc"], .75)

    def test_audio_center_crop_and_downmix_are_recorded(self):
        length = 300001
        row = dataclasses.replace(fake(),frame_count=length,duration_seconds=length/16000)
        values = np.column_stack((np.linspace(-.2,.2,length,dtype=np.float32),
                                  np.linspace(-.1,.1,length,dtype=np.float32)))
        with patch("src.features.sha", return_value=row.primary_audio_sha256), \
             patch("soundfile.read",return_value=(values,16000)):
            cropped, metadata = audio(row)
        self.assertTrue(metadata["cropped"])
        self.assertEqual(metadata["start_sample_16k"],86000)
        self.assertEqual(metadata["end_sample_16k"],214000)
        np.testing.assert_array_equal(cropped,values.mean(axis=1,dtype=np.float32)[86000:214000])

    def test_pinned_model_configuration_hash_rejected_when_changed(self):
        from src.common import config
        cfg = config()
        cfg["features"]["config_sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"configuration checksum"):
            acquire(cfg)

    def test_rp_primary_audio_binding_corruption_rejected(self):
        import src.data as data_module
        record = self.corpus.select(self.corpus.splits[0].train)[0]
        real_read = data_module.read
        def changed(path):
            doc = real_read(path)
            if "source_audio_sha256" in doc:
                doc["source_audio_sha256"]="0"*64
            return doc
        with patch.object(data_module,"read",side_effect=changed):
            with self.assertRaisesRegex(ValueError,"VMD source identity"):
                load_rp(record)

    def test_completed_stage_artifact_corruption_blocks_resume(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"smoke") as directory:
            root=Path(directory)
            artifact=root/"artifact.txt"
            artifact.write_text("original")
            stage=Stages(root/"stages","unit")
            stage.execute("test",lambda: (1,[artifact]))
            artifact.write_text("changed")
            with self.assertRaisesRegex(ValueError,"Sealed artifact changed"):
                stage.execute("test",lambda: self.fail("completed action repeated"))


if __name__ == "__main__":
    (ROOT/"smoke").mkdir(exist_ok=True)
    unittest.main(verbosity=2)
