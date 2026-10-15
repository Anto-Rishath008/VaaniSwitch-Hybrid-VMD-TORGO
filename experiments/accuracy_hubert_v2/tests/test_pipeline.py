"""Synthetic orchestration/exports; no reserved TORGO signal or test-label access."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pipeline, reporting, training
from src.common import ROOT, Stages, config, read, sha, write_json
from src.data import Record, Split
from src.features import Extractor, FeatureCache, validate_cache
from src.training import Partition, load_checkpoint, train_neural


class SyntheticCorpus:
    def __init__(self):
        speakers = [f"D{i}" for i in range(8)] + [f"C{i}" for i in range(7)]
        # Both classes in each outer fold; identities are entirely synthetic.
        self.mapping = {s: i%3+1 for i,s in enumerate(speakers)}
        self.records = tuple(Record(f"{s}/Session1/{j:04d}", s,
                            "dysarthric" if s.startswith("D") else "control",
                            f"synthetic/{s}/{j}.wav", "a"*64, 1., 16000, 16000, "head")
                             for s in speakers for j in (1,2))
        self.by_id = {r.logical_id:r for r in self.records}
        # Distinct synthetic primary identities keep the inherited leakage checker meaningful.
        self.records = tuple(dataclasses.replace(r, primary_audio_sha256=f"{i:064x}")
                             for i,r in enumerate(self.records))
        self.by_id = {r.logical_id:r for r in self.records}
        self.splits = tuple(Split(self,f) for f in (1,2,3))
    def select(self, speakers):
        return tuple(r for r in self.records if r.speaker_id in speakers)
    def audit(self):
        return {"synthetic":True,"count":len(self.records)}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        (ROOT/"smoke").mkdir(exist_ok=True)
        torch.set_num_threads(2)

    def test_split_identity_is_json_roundtrip_stable(self):
        corpus = SyntheticCorpus()
        for split in corpus.splits:
            self.assertEqual(json.loads(json.dumps(split.identity)), split.identity)

    def test_logistic_fit_bounds_actual_blas_thread_pools(self):
        from sklearn.linear_model import LogisticRegression
        from threadpoolctl import threadpool_info
        from test_recovery import synthetic
        calls=[]
        real_fit=LogisticRegression.fit
        def checked(model,*args,**kwargs):
            pools=threadpool_info()
            calls.extend(pools)
            self.assertTrue(all(p["num_threads"]<=2 for p in pools))
            return real_fit(model,*args,**kwargs)
        with tempfile.TemporaryDirectory(dir=ROOT/"smoke") as directory, \
             patch.object(LogisticRegression,"fit",checked):
            training.logistic_fit(synthetic(),None,.1,Path(directory),"threads-unit",6,config())
        self.assertTrue(calls)

    def test_selection_final_fit_and_outer_seal_synthetic(self):
        corpus, cfg = SyntheticCorpus(), config()
        cfg["training"]["logistic_C"] = [.01,.1]
        arrays = {"index":{r.logical_id:i for i,r in enumerate(corpus.records)},
                  "values":np.random.default_rng(3).normal(size=(30,3,1536)).astype(np.float32)}
        with tempfile.TemporaryDirectory(dir=ROOT/"smoke") as directory, patch.object(pipeline,"ROOT",Path(directory)):
            stages = Stages(Path(directory)/"stages", "synthetic")
            split = corpus.splits[0]
            selected = pipeline.choose_and_fit(corpus,split,"E1",cfg,arrays,None,"synthetic",stages)
            ref = load_checkpoint(selected["final"]["checkpoint"], selected["final"]["checkpoint"]["identity"])
            self.assertEqual(set(ref["scaler"]["provenance"]["speakers"]),set(split.development))
            self.assertFalse(set(ref["scaler"]["provenance"]["speakers"]) & set(split.test))
            with patch.object(pipeline, "logistic_fit", side_effect=AssertionError("fit repeated")):
                reused = pipeline.choose_and_fit(corpus,split,"E1",cfg,arrays,None,"synthetic",stages)
            self.assertEqual(reused, selected)
            first = pipeline.evaluate(corpus,split,"E1",selected,arrays,None,"freeze-unit","synthetic")
            with patch.object(pipeline,"load_checkpoint",side_effect=AssertionError("inference repeated")):
                self.assertEqual(first, pipeline.evaluate(corpus,split,"E1",selected,arrays,None,"freeze-unit","synthetic"))
            self.assertEqual(len(first["predictions"]),10)

    def test_extraction_resume_after_summary_before_stage_completion(self):
        records = SyntheticCorpus().records[:2]
        corpus = SimpleNamespace(records=records)
        with tempfile.TemporaryDirectory(dir=ROOT/"smoke") as directory, patch.object(pipeline,"ROOT",Path(directory)):
            base = Path(directory)
            cache = FeatureCache(base/"features", {"test":"extraction"})
            for record in records:
                cache.save(record,np.zeros((3,1536),np.float32),{"cropped":False})
            expected = {"extraction":{"newly_extracted":2},"benchmark":{"selected_batch_size":1}}
            write_json(base/"logs/extraction_summary.json",expected)
            with patch.object(pipeline,"Extractor",side_effect=AssertionError("model recreated")):
                value, paths = pipeline.extraction_stage(config(),corpus,cache,None)
            self.assertEqual(value,expected)
            self.assertEqual(len(paths),1)

    def test_extraction_oom_halves_batch_and_preserves_cache(self):
        records = SyntheticCorpus().records[:2]
        cfg = config()
        extractor = Extractor.__new__(Extractor)
        extractor.cfg, extractor.batch_size = cfg, 2
        extractor.model = torch.nn.Linear(1,1)  # Disposable stub for backbone parameter accounting.
        calls = []
        def infer(waves):
            calls.append(len(waves))
            if len(waves)>1:
                raise torch.cuda.OutOfMemoryError("synthetic OOM")
            return np.zeros((len(waves),3,1536),np.float32)
        extractor.infer = infer
        with tempfile.TemporaryDirectory(dir=ROOT/"smoke") as directory:
            cache = FeatureCache(Path(directory),{"unit":"oom"})
            with patch("src.features.audio",return_value=(np.zeros(16000,np.float32),{"cropped":False})):
                result = extractor.extract(records,cache)
                self.assertEqual(result["newly_extracted"],2)
                before = validate_cache(records,cache)
                self.assertEqual(extractor.extract(records,cache)["newly_extracted"],0)
                self.assertEqual(before,validate_cache(records,cache))
            self.assertEqual(calls,[2,1,1])

    def test_failed_stage_event_survives_successful_resume(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"smoke") as directory:
            stages = Stages(Path(directory),"unit")
            with self.assertRaises(RuntimeError):
                stages.execute("example",lambda: (_ for _ in ()).throw(RuntimeError("failure")))
            stages.execute("example",lambda: (1,[]))
            events = [read(p) for p in (Path(directory)/"events").glob("*.json")]
            self.assertTrue(any(e["status"]=="FAILED" and e["error"]=="RuntimeError: failure" for e in events))
            self.assertEqual(read(Path(directory)/"example.json")["status"],"COMPLETE")

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA required")
    def test_training_oom_rolls_back_optimizer_scaler_and_rng(self):
        from test_recovery import synthetic
        train = synthetic()
        records = train.records + tuple(dataclasses.replace(r, logical_id=r.logical_id+"x") for r in train.records)
        train = dataclasses.replace(train,records=records,features=np.tile(train.features,(2,1)))
        cfg, candidate = config(), config()["training"]["neural_candidates"][0]
        with tempfile.TemporaryDirectory(dir=ROOT/"smoke") as directory:
            base = Path(directory)
            cfg["training"]["batch_cap"]=2
            reference = train_neural(train,None,"E2",candidate,base/"reference","oom-unit",6,cfg,fixed_epochs=1)
            cfg["training"]["batch_cap"]=4
            real_batch = training.cuda_batch
            calls = 0
            def once(tensors,indices):
                nonlocal calls
                calls+=1
                if calls==2:
                    raise torch.cuda.OutOfMemoryError("synthetic after optimizer step")
                return real_batch(tensors,indices)
            with patch.object(training,"cuda_batch",side_effect=once):
                resumed = train_neural(train,None,"E2",candidate,base/"interrupted","oom-unit",6,cfg,fixed_epochs=1)
            self.assertEqual(resumed["batch_size"],2)
            a=load_checkpoint(reference["checkpoint"],"oom-unit")
            b=load_checkpoint(resumed["checkpoint"],"oom-unit")
            for name in a["model"]:
                torch.testing.assert_close(a["model"][name],b["model"][name],rtol=0,atol=0)

    def test_all_result_files_and_figures_from_synthetic_predictions(self):
        corpus = SyntheticCorpus()
        results, selections = [], {}
        for model in ("E1","E2","E3"):
            for split in corpus.splits:
                records = corpus.select(split.test)
                probabilities = [.8 if r.label else .2 for r in records]
                predictions = [{"model":model,"outer_fold":split.fold,"logical_id":r.logical_id,
                                "speaker_id":r.speaker_id,"label":r.label,"probability":p,"prediction":int(p>=.5)}
                               for r,p in zip(records,probabilities)]
                results.append({"model":model,"outer_fold":split.fold,"metrics":training.metrics(
                    [r.label for r in records],probabilities),"predictions":predictions,
                    "training_seconds":1.,"parameter_count":1537,"layer":9,"selected_epochs":1})
                selections[f"fold{split.fold}_{model}"]={"winner":{"history":[{
                    "epoch":1,"training_loss":.5,"validation_macro_f1":.5}]}}
        baseline={m:{"accuracy":{"mean":.6,"sample_std":.1},"macro_f1":{"mean":.6,"sample_std":.1}}
                  for m in ("B2","C2")}
        with tempfile.TemporaryDirectory(dir=ROOT/"smoke") as directory, patch.object(reporting,"ROOT",Path(directory)):
            summary, paths = reporting.generate_report(corpus,results,selections,{"synthetic":True},baseline)
            self.assertEqual(len(paths),8)
            self.assertTrue(all(p.exists() and p.stat().st_size>0 for p in paths))
            report_text=(Path(directory)/"results/FINAL_EXPERIMENT_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("accuracy_hubert_v2",report_text)
            self.assertNotIn("accuracy_hubert\\run_all.ps1",report_text)
            self.assertEqual(summary["models"]["E3"]["speaker_descriptive"]["metrics"]["accuracy"],1.)
            checksums={str(p):sha(p) for p in paths}
            reporting.generate_report(corpus,results,selections,{"synthetic":True},baseline)
            self.assertEqual(checksums,{str(p):sha(p) for p in paths})


if __name__=="__main__":
    unittest.main(verbosity=2)
