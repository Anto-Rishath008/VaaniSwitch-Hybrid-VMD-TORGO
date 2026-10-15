"""Read-only replay of the last V1 development checkpoint; writes V2 evidence only."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "2"
import sys
from pathlib import Path
OUT = Path(__file__).resolve().parent
V1 = OUT.with_name("accuracy_hubert")
sys.path.insert(0, str(V1))
import numpy as np
import torch
from src.common import read, sha, write_json, config, progress
from src.data import Corpus
from src.features import FeatureCache, extraction_identity
from src.training import (Classifier, Partition, cpu_tensors, cuda_batch, load_checkpoint,
                          restore_rng, rng_state, seed_all)

def finite_tree(x):
    if torch.is_tensor(x):
        return bool(torch.isfinite(x).all())
    if isinstance(x, dict):
        return all(finite_tree(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return all(finite_tree(v) for v in x)
    return True

def main():
    torch.set_num_threads(2)
    cfg, corpus = config(), Corpus()
    frozen = read(V1/"logs/frozen_run.json")
    cache = FeatureCache(V1/"cache/features", extraction_identity(cfg, frozen["model_lock"]))
    split = corpus.splits[0]
    records = corpus.select(split.train)
    progress("Loading verified Fold 1 inner-training features for V1 diagnosis")
    values = np.stack([cache.load(r)[0] for r in records])
    part = Partition(records, values, split.train, split.test, "inner_train")
    directory = V1/"checkpoints/fold1_E2_layer6_baseline"
    ref = read(directory/"latest.json")
    state = load_checkpoint(ref, ref["identity"])
    seed_all()
    model = Classifier("E2", state["candidate"]["dropout"]).cuda()
    model.load_state_dict(state["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                 weight_decay=state["candidate"]["weight_decay"])
    optimizer.load_state_dict(state["optimizer"])
    amp = torch.amp.GradScaler("cuda")
    amp.load_state_dict(state["amp"])
    tensors = cpu_tensors(part, state["scaler"], fitting=True)
    report = {"checkpoint":ref, "last_completed_epoch":state["epoch"],
              "replay_epoch":state["epoch"]+1, "initial_amp":state["amp"],
              "parameters_finite":finite_tree(state["model"]),
              "optimizer_finite":finite_tree(state["optimizer"]),
              "inputs_finite":finite_tree(tensors), "training_records":len(records),
              "speakers":list(split.train), "outer_forwards":0}
    restore_rng(state["rng"])
    model.train()
    order = torch.randperm(len(records))
    bs = state["batch_size"]
    for step, start in enumerate(range(0, len(order), bs), 1):
        idx = order[start:start+bs]
        batch = cuda_batch(tensors, idx)
        optimizer.zero_grad(set_to_none=True)
        before_rng = rng_state()
        scale = amp.get_scale()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(batch["speech"])
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(), batch["labels"], reduction="none")*batch["weights"]).mean()
        assert finite_tree(logits) and finite_tree(loss), "Non-finite forward"
        amp.scale(loss).backward()
        amp.unscale_(optimizer)
        bad = [n for n,p in model.named_parameters() if not finite_tree(p.grad)]
        if bad:
            report["overflow"] = {"step":step, "scale":scale, "loss":float(loss.detach()),
                "logits_finite":True, "parameters_finite":finite_tree(model.state_dict()),
                "bad_gradient_parameters":bad, "record_ids":[records[i].logical_id for i in idx.tolist()]}
            for name, dtype in (("fp16_scale_one", torch.float16), ("fp32", None)):
                optimizer.zero_grad(set_to_none=True)
                restore_rng(before_rng)
                with torch.autocast("cuda", dtype=dtype or torch.float32, enabled=dtype is not None):
                    output = model(batch["speech"])
                    trial_loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                        output.float(),batch["labels"],reduction="none")*batch["weights"]).mean()
                trial_loss.backward()
                finite = all(finite_tree(p.grad) for p in model.parameters())
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)) if finite else None
                report[name] = {"loss":float(trial_loss.detach()),"logits_finite":finite_tree(output),
                                "gradients_finite":finite,"gradient_norm":norm}
            report["diagnosis"] = "FP16 scaled-backward overflow with finite inputs, parameters, logits and loss" 
            write_json(OUT/"logs/v1_diagnosis.json", report)
            progress(f"Reproduced V1 overflow: epoch {state['epoch']+1}, step {step}, loss scale {scale}")
            return
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        amp.step(optimizer)
        amp.update()
    report["diagnosis"] = "Not reproduced in one bounded resumed epoch"
    write_json(OUT/"logs/v1_diagnosis.json",report)
    raise RuntimeError(report["diagnosis"])

if __name__ == "__main__":
    main()
