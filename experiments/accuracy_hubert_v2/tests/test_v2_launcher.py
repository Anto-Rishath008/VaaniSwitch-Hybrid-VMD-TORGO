"""Synthetic orchestration only: all nine decisions precede any evaluation."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from src import pipeline
from src.common import ROOT, config, read, write_json
from test_pipeline import SyntheticCorpus


class V2Launcher(unittest.TestCase):
    def test_three_folds_all_models_freeze_before_outer_and_never_extract(self):
        corpus, events = SyntheticCorpus(), []
        cache = type("Cache",(),{"load":lambda self,r:np.zeros((3,1536),np.float32)})()
        identity={"source_files":{"synthetic":"fixed"}}
        def choose(corpus,split,model,*args):
            events.append(("fit",split.fold,model))
            return {"winner":{"layer":6,"candidate":{"id":"fixture"},"best_epoch":1},
                    "selection_sha256":"fixture","final":{"checkpoint":{"synthetic":True}}}
        def evaluate(corpus,split,model,*args):
            self.assertEqual(sum(e[0]=="fit" for e in events),9)
            self.assertTrue((pipeline.ROOT/"results/FROZEN_DECISIONS.json").exists())
            events.append(("evaluate",split.fold,model))
            return {"synthetic":True}
        with tempfile.TemporaryDirectory(dir=ROOT/"smoke") as directory, ExitStack() as stack:
            base=Path(directory)
            patches={
                "ROOT":base,"native_contract":lambda:{},
                "preservation_snapshot":lambda:{},
                "verify_preservation":lambda before:{"status":"PASS"},
                "verify_inputs":lambda records:None,
                "acquire_readonly":lambda cfg:(None,{"synthetic":True}),
                "freeze_run":lambda *args:identity,
                "reference_cache":lambda *args:cache,
                "verify_reuse":lambda *args:({"status":"PASS"},[]),
                "validate_cache":lambda *args:{"cropped_recordings":0},
                "load_rp":lambda r:np.zeros((224,224),np.uint8),
                "benchmark_training":lambda *args:{"selected_batch_size":8},
                "choose_and_fit":choose,"evaluate":evaluate,
                "generate_report":lambda *args:({"models":{"E1":{},"E2":{},"E3":{}}},[]),
                "source_identity":lambda:identity["source_files"],
            }
            for name,value in patches.items():
                stack.enter_context(patch.object(pipeline,name,value))
            stack.enter_context(patch.object(pipeline,"Extractor",side_effect=AssertionError("Extraction forbidden")))
            stack.enter_context(patch.object(corpus,"audit",return_value={"cropped_recordings":0}))
            stack.enter_context(patch.object(pipeline,"read",side_effect=lambda path:
                {"ablations":{m:{"summary":{}} for m in ("B2","C2")}}
                if Path(path).name=="cross_fold_metrics.json" else read(path)))
            pipeline.full_run(config(),{},corpus)
        expected=[(fold,model) for fold in (1,2,3) for model in ("E1","E2","E3")]
        self.assertEqual([e[1:] for e in events[:9]],expected)
        self.assertEqual([e[1:] for e in events[9:]],expected)

    def test_frozen_v2_identity_reuses_and_rejects_source_drift(self):
        env={k:"fixture" for k in ("python","torch","numpy","transformers","cuda_runtime","gpu")}
        with tempfile.TemporaryDirectory(dir=ROOT/"smoke") as directory:
            root=Path(directory)
            write_json(root/"V1_LINEAGE.json",{"repair_of":"fixture"})
            with patch.object(pipeline,"ROOT",root),patch.object(pipeline,"source_identity",return_value={"code":"a"}):
                first=pipeline.freeze_run(config(),env,{}, {})
                self.assertEqual(first,pipeline.freeze_run(config(),env,{}, {}))
            with patch.object(pipeline,"ROOT",root),patch.object(pipeline,"source_identity",return_value={"code":"b"}):
                with self.assertRaisesRegex(ValueError,"identity changed"):
                    pipeline.freeze_run(config(),env,{}, {})
