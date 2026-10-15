"""Bounded development-only V2 validation; never calls full_run/evaluate."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from src import pipeline  # Sets CUDA/CPU environment before importing torch.
import copy
import dataclasses
import gc
import time
import uuid
from unittest.mock import patch
import numpy as np
import torch
from src.common import ROOT, config, digest, progress, read, require, source_identity, write_json
from src.data import Corpus, load_rp, verify_inputs
from src.reuse import acquire_readonly, reference_cache
from src import training
from src.training import (Classifier, Partition, cpu_tensors, load_checkpoint, neural_predict,
                          save_checkpoint, train_neural)
from src.numerics import finite_state


def equal_tree(a,b):
    if torch.is_tensor(a):
        return torch.equal(a,b)
    if isinstance(a,np.ndarray):
        return np.array_equal(a,b)
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):
        return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b


def interrupted_fit(train, model, cfg, directory, identity):
    candidate=cfg["training"]["neural_candidates"][0]
    def interrupt(root,payload):
        ref=save_checkpoint(root,payload)
        if payload["epoch"]==2:
            raise RuntimeError("Intentional validation interruption at epoch 2")
        return ref
    with patch.object(training,"save_checkpoint",side_effect=interrupt):
        try:
            train_neural(train,None,model,candidate,directory,identity,6,cfg,fixed_epochs=4)
        except RuntimeError as exc:
            require("Intentional validation interruption" in str(exc),str(exc))
        else:
            raise AssertionError("Interruption did not occur")
    state=load_checkpoint(read(directory/"latest.json"),identity)
    require(state["epoch"]==2 and state["precision"]=="fp32" and state["amp"] is None,
            "Wrong interruption checkpoint")
    rng=state["rng"]
    training.restore_rng(rng)
    require(equal_tree(training.rng_state(),rng),"RNG save/load mismatch")
    result=train_neural(train,None,model,candidate,directory,identity,6,cfg,fixed_epochs=4)
    # Sealed completion must not retrain.
    with patch.object(training,"optimizer_step",side_effect=AssertionError("Completed fit retrained")):
        require(train_neural(train,None,model,candidate,directory,identity,6,cfg,fixed_epochs=4)==result,
                "Completed fit changed")
    return result


def main():
    cfg=config()
    env=pipeline.environment(cfg)
    corpus=Corpus()
    require(read(ROOT/"logs/feature_reuse.json")["record_count"]==8673,"Verify reuse first")
    _,lock=acquire_readonly(cfg)
    cache=reference_cache(cfg,lock)
    split=corpus.splits[0]
    records=corpus.select(split.train)
    progress("Bounded validation: verify development-only primary/RP inputs")
    verify_inputs(records)
    values=np.stack([cache.load(r) for r in records])
    rp=np.stack([load_rp(r) for r in records])
    cfg["training"]["batch_cap"]=8
    directory=ROOT/"validation"/("cuda-"+uuid.uuid4().hex)
    directory.mkdir(parents=True)
    report={"status":"RUNNING","environment":env,"source_identity":source_identity(),
            "development_only":True,"outer_test_forwards":0,"training_speakers":list(split.train),
            "reserved_fold1_speakers":list(split.test),"training_records":len(records),
            "training_ids_sha256":digest([r.logical_id for r in records]),"classifiers":{}}
    write_json(directory/"attempt.json",report)
    started=time.perf_counter()
    for model in ("E2","E3"):
        part=Partition(records,values[:,0],split.train,split.test,"inner_train",
                       rp if model=="E3" else None)
        identity=digest({"validation":str(directory),"model":model,"source":source_identity()})
        progress(f"CUDA {model}: four epochs on all {len(records)} Fold 1 inner-training records")
        reference=None
        if model=="E2":
            reference=train_neural(part,None,model,cfg["training"]["neural_candidates"][0],
                                   directory/"E2_uninterrupted",identity,6,cfg,fixed_epochs=4)
        result=interrupted_fit(part,model,cfg,directory/model,identity)
        payload=load_checkpoint(result["checkpoint"],identity)
        finite_state(payload["model"],"validated.model")
        finite_state(payload["optimizer"],"validated.optimizer")
        exact=None
        if reference:
            other=load_checkpoint(reference["checkpoint"],identity)
            exact=all(equal_tree(other[k],payload[k]) for k in
                      ("model","optimizer","rng","best","history","scaler"))
            require(exact,"Real development E2 resumed state differs")
        panel=dataclasses.replace(part,records=records[:8],features=values[:8,0],
                                  rp=rp[:8] if model=="E3" else None)
        network=Classifier(model,.3).cuda()
        network.load_state_dict(payload["best"])
        tensors=cpu_tensors(panel,payload["scaler"],fitting=False)
        first=neural_predict(network,tensors,8)
        second=neural_predict(network,tensors,8)
        require(np.array_equal(first,second),"Fixed checkpoint predictions differ")
        report["classifiers"][model]={
            "status":"PASS","precision":"fp32","epochs":4,
            "optimization_steps":sum(r["optimization_steps"] for r in result["history"]),
            "additional_uninterrupted_comparison_steps":sum(r["optimization_steps"] for r in reference["history"]) if reference else 0,
            "maximum_gradient_norm":max(r["maximum_gradient_norm"] for r in result["history"]),
            "training_losses":[r["training_loss"] for r in result["history"]],
            "peak_allocated_bytes":result["peak_allocated_bytes"],
            "peak_reserved_bytes":result["peak_reserved_bytes"],
            "checkpoint":result["checkpoint"],"save_load_resume":"PASS",
            "rng_roundtrip":"PASS","completed_fit_reuse":"PASS",
            "fixed_checkpoint_repeat_prediction":"PASS","bitwise_resume_equality":exact,
            "nondeterminism_note":"Native B2 CUDA backward warns; E3 bitwise resumed training equivalence not claimed" if model=="E3" else None}
        del network,tensors,payload,part
        gc.collect()
        torch.cuda.empty_cache()
        write_json(directory/(model+"_report.json"),report["classifiers"][model])
    # Small development panels exercise layer 9/12, regularized setting and inner validation.
    train_indices=[]
    for speaker in split.train:
        train_indices += [i for i,r in enumerate(records) if r.speaker_id==speaker][:8]
    val_records=tuple(r for speaker in split.validation for r in corpus.select([speaker])[:8])
    val_values=np.stack([cache.load(r) for r in val_records])
    val_rp=np.stack([load_rp(r) for r in val_records])
    checks=[]
    for model in ("E2","E3"):
        for layer,index in ((9,1),(12,2)):
            tiny=copy.deepcopy(cfg)
            tiny["training"]["max_epochs"]=2
            train=Partition(tuple(records[i] for i in train_indices),values[train_indices,index],
                            split.train,split.test,"inner_train",rp[train_indices] if model=="E3" else None)
            val=Partition(val_records,val_values[:,index],split.validation,split.test,"inner_validation",
                          val_rp if model=="E3" else None)
            result=train_neural(train,val,model,cfg["training"]["neural_candidates"][1],
                                directory/f"{model}_layer{layer}_panel",f"panel-{model}-{layer}",layer,tiny)
            checks.append({"model":model,"layer":layer,"candidate":"regularized","status":"PASS",
                           "epochs":len(result["history"]),"optimization_steps":sum(r["optimization_steps"] for r in result["history"])})
    report.update(status="PASS",panel_checks=checks,seconds=time.perf_counter()-started)
    write_json(directory/"CUDA_VALIDATION.json",report)
    write_json(ROOT/"logs/cuda_validation.json",{"path":str(directory/"CUDA_VALIDATION.json"),
                                               "status":"PASS","outer_test_forwards":0})
    progress(f"Bounded CUDA validation PASS: {directory}")


if __name__=="__main__":
    main()
