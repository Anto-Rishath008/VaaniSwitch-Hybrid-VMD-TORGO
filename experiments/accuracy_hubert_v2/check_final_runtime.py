"""Small real-development check bound to the finalized V2 source identity."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from src import pipeline
import numpy as np
from src.common import ROOT,config,digest,read,require,source_identity,write_json
from src.data import Corpus,load_rp
from src.reuse import acquire_readonly,reference_cache
from src.training import Partition,benchmark_training,train_neural,load_checkpoint
from src.numerics import finite_state

cfg=config()
pipeline.environment(cfg)
sources=source_identity()
require(read(ROOT/"logs/frozen_run.json")["source_files"]==sources,"Final frozen source mismatch")
corpus=Corpus()
split=corpus.splits[0]
records=tuple(r for s in split.train for r in corpus.select([s])[:8])
_,lock=acquire_readonly(cfg)
cache=reference_cache(cfg,lock)
values=np.stack([cache.load(r)[0] for r in records])
rp=np.stack([load_rp(r) for r in records])
report={"status":"PASS","source_identity":sources,"development_records":len(records),
        "speakers":list(split.train),"outer_forwards":0,"classifiers":{}}
for model in ("E2","E3"):
    part=Partition(records,values,split.train,split.test,"inner_train",rp if model=="E3" else None)
    benchmark=benchmark_training(part,model,cfg)
    cfg["training"]["measured_batch_sizes"]={model:benchmark["selected_batch_size"]}
    identity=digest({"final_validation":sources,"model":model})
    result=train_neural(part,None,model,cfg["training"]["neural_candidates"][0],
                        ROOT/"validation/final_source"/model,identity,6,cfg,fixed_epochs=2)
    payload=load_checkpoint(result["checkpoint"],identity)
    finite_state(payload["model"],"final.model")
    finite_state(payload["optimizer"],"final.optimizer")
    report["classifiers"][model]={"status":"PASS","epochs":2,
        "optimization_steps":sum(r["optimization_steps"] for r in result["history"]),
        "benchmark":benchmark,"checkpoint":result["checkpoint"]}
require(source_identity()==sources,"Source changed during final check")
write_json(ROOT/"logs/final_runtime_check.json",report)
print("Final frozen-source CUDA E2/E3 and FP32 benchmarks PASS")
