"""Regression guards for numerical failures and immutable V1 reuse."""
import dataclasses
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.common import ROOT, config, read, write_json
from src.features import FeatureCache
from src.reuse import ReadOnlyFeatureCache, acquire_readonly
from src.numerics import NumericalError, checked_forward, optimizer_step
from src.training import Classifier, train_neural, transform, load_checkpoint
from src import training
from test_contracts import fake
from test_recovery import synthetic


class NumericalStability(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.model = Classifier("E2", .3).cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=.0003)
        self.batch = {"speech": torch.randn(2,1536,device="cuda"),
                      "labels": torch.tensor([0.,1.],device="cuda"),
                      "weights": torch.ones(2,device="cuda")}

    def tearDown(self):
        del self.model, self.optimizer, self.batch
        torch.cuda.empty_cache()

    def test_fp32_update_even_inside_autocast_no_scaler(self):
        before = self.model.head[-1].weight.detach().clone()
        with patch("torch.amp.GradScaler", side_effect=AssertionError("GradScaler forbidden")):
            with torch.autocast("cuda", dtype=torch.float16):
                output = checked_forward(self.model, self.batch)
                loss, norm = optimizer_step(self.model,self.optimizer,self.batch)
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(np.isfinite([loss,norm]).all())
        self.assertFalse(torch.equal(before,self.model.head[-1].weight))

    def test_nonfinite_input_rejected_before_step(self):
        self.batch["speech"][0,0] = float("nan")
        with patch.object(self.optimizer,"step",side_effect=AssertionError("invalid step")):
            with self.assertRaisesRegex(NumericalError,"input.speech"):
                optimizer_step(self.model,self.optimizer,self.batch)

    def test_nonfinite_parameters_rejected_before_forward(self):
        with torch.no_grad():
            self.model.head[-1].weight[0,0] = float("inf")
        with self.assertRaisesRegex(NumericalError,"parameters.before"):
            optimizer_step(self.model,self.optimizer,self.batch)

    def test_nonfinite_logits_rejected(self):
        with patch.object(self.model,"forward",return_value=torch.full((2,),float("nan"),device="cuda")):
            with self.assertRaisesRegex(NumericalError,"forward.logits"):
                optimizer_step(self.model,self.optimizer,self.batch)

    def test_finite_inputs_overflow_loss_rejected(self):
        logits = torch.tensor([1.e20,-1.e20],device="cuda",requires_grad=True)
        self.batch["weights"].fill_(1.e30)
        with patch.object(self.model,"forward",return_value=logits):
            with self.assertRaisesRegex(NumericalError,"loss"):
                optimizer_step(self.model,self.optimizer,self.batch)

    def test_nonfinite_gradient_never_steps_or_hides_error(self):
        handle = self.model.head[-1].weight.register_hook(lambda g: torch.full_like(g,float("inf")))
        try:
            with patch.object(self.optimizer,"step",side_effect=AssertionError("invalid step")):
                with self.assertRaisesRegex(NumericalError,"gradient.head.3.weight"):
                    optimizer_step(self.model,self.optimizer,self.batch,{"record_ids":["fixture"]})
        finally:
            handle.remove()

    def test_finite_gradients_with_overflowed_norm_are_rejected(self):
        handle = self.model.head[-1].weight.register_hook(lambda g: torch.full_like(g,1.e30))
        try:
            with patch.object(self.optimizer,"step",side_effect=AssertionError("invalid step")):
                with self.assertRaisesRegex(NumericalError,"gradient_norm"):
                    optimizer_step(self.model,self.optimizer,self.batch)
        finally:
            handle.remove()

    def test_optimizer_corruption_rejected(self):
        optimizer_step(self.model,self.optimizer,self.batch)
        state = next(iter(self.optimizer.state.values()))
        state["exp_avg"].fill_(float("nan"))
        with self.assertRaisesRegex(NumericalError,"optimizer.before"):
            optimizer_step(self.model,self.optimizer,self.batch)

    def test_clipping_oom_retains_recoverable_exception_type(self):
        with patch("torch.nn.utils.clip_grad_norm_",side_effect=torch.cuda.OutOfMemoryError("clip OOM")):
            with patch.object(self.optimizer,"step",side_effect=AssertionError("invalid step")):
                with self.assertRaises(torch.cuda.OutOfMemoryError):
                    optimizer_step(self.model,self.optimizer,self.batch)

    def test_corrupt_post_step_state_detected(self):
        real_step = self.optimizer.step
        def corrupt():
            real_step()
            with torch.no_grad():
                self.model.head[-1].weight.fill_(float("nan"))
        with patch.object(self.optimizer,"step",side_effect=corrupt):
            with self.assertRaisesRegex(NumericalError,"parameters.after"):
                optimizer_step(self.model,self.optimizer,self.batch)

    def test_invalid_scaler_and_rp_are_rejected(self):
        with self.assertRaisesRegex(ValueError,"input/scaler"):
            transform(np.ones((1,2)),{"mean":[0,0],"scale":[0,1]})
        with self.assertRaisesRegex(ValueError,"recurrence"):
            dataclasses.replace(synthetic(),rp=np.full((4,224,224),np.nan)).check(fitting=True)

    def test_failure_diagnostic_and_no_checkpoint_publication(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"smoke") as directory:
            error = NumericalError("regression","gradient.injected",{"epoch":1,"record_ids":["fixture"]})
            with patch("src.training.optimizer_step",side_effect=error):
                with self.assertRaises(NumericalError):
                    train_neural(synthetic(),None,"E2",config()["training"]["neural_candidates"][0],
                                 Path(directory),"numeric-unit",6,config(),fixed_epochs=1)
            self.assertFalse((Path(directory)/"latest.json").exists())
            self.assertFalse((Path(directory)/"complete.json").exists())
            failures = list(Path(directory).glob("numerical-failure-*.json"))
            self.assertEqual(len(failures),1)
            self.assertEqual(read(failures[0])["details"]["record_ids"],["fixture"])

    def test_readonly_cache_rejects_write_missing_and_corrupt_entries(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"smoke") as directory:
            root, record = Path(directory), fake()
            writable = FeatureCache(root,{"unit":"readonly"})
            cache = ReadOnlyFeatureCache(root,{"unit":"readonly"})
            with self.assertRaisesRegex(ValueError,"no automatic repair"):
                cache.load(record)
            with self.assertRaisesRegex(RuntimeError,"writes"):
                cache.save(record,np.zeros((3,1536),np.float32),{})
            writable.save(record,np.zeros((3,1536),np.float32),{})
            np.testing.assert_array_equal(cache.load(record),np.zeros((3,1536),np.float32))
            meta = read(writable.location(record)/"complete.json")
            (writable.location(record)/meta["filename"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError,"corruption"):
                cache.load(record)

    def test_v1_preprocessing_drift_rejected_without_download(self):
        cfg = config()
        cfg["features"]["crop"] = "different"
        with self.assertRaisesRegex(ValueError,"preprocessing identity"):
            acquire_readonly(cfg)

    def test_partial_epoch_restart_restores_all_state(self):
        cfg=config()
        cfg["training"]["batch_cap"]=2
        part=synthetic()
        candidate=cfg["training"]["neural_candidates"][0]
        with tempfile.TemporaryDirectory(dir=ROOT/"smoke") as directory:
            base=Path(directory)
            reference=train_neural(part,None,"E2",candidate,base/"reference","partial-unit",6,cfg,fixed_epochs=2)
            real_step=training.optimizer_step
            calls=0
            def interrupt(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==4:
                    raise RuntimeError("Partial epoch interruption after a successful step")
                return real_step(*args,**kwargs)
            with patch.object(training,"optimizer_step",side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,"Partial epoch"):
                    train_neural(part,None,"E2",candidate,base/"interrupted","partial-unit",6,cfg,fixed_epochs=2)
            resumed=train_neural(part,None,"E2",candidate,base/"interrupted","partial-unit",6,cfg,fixed_epochs=2)
            a=load_checkpoint(reference["checkpoint"],"partial-unit")
            b=load_checkpoint(resumed["checkpoint"],"partial-unit")
            for key in a["model"]:
                self.assertTrue(torch.equal(a["model"][key],b["model"][key]))
            self.assertEqual(a["history"],b["history"])
            self.assertTrue(torch.equal(a["rng"]["torch"],b["rng"]["torch"]))
            for key in a["optimizer"]["state"]:
                for field in a["optimizer"]["state"][key]:
                    self.assertTrue(torch.equal(a["optimizer"]["state"][key][field],
                                                b["optimizer"]["state"][key][field]))

    def test_fp16_checkpoint_cannot_resume_v2(self):
        cfg=config()
        candidate=cfg["training"]["neural_candidates"][0]
        with tempfile.TemporaryDirectory(dir=ROOT/"smoke") as directory:
            root=Path(directory)
            train_neural(synthetic(),None,"E2",candidate,root,"precision-unit",6,cfg,fixed_epochs=1)
            real_load=training.load_checkpoint
            def wrong_precision(*args):
                state=real_load(*args)
                state["precision"]="fp16"
                return state
            with patch.object(training,"load_checkpoint",side_effect=wrong_precision):
                with self.assertRaisesRegex(ValueError,"checkpoint precision"):
                    train_neural(synthetic(),None,"E2",candidate,root,"precision-unit",6,cfg,fixed_epochs=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
