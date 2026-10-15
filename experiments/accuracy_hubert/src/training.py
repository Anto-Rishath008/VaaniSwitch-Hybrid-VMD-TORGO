"""Development-only selection, train-only scaling, resumable CUDA fits and sealed outer claims."""
from __future__ import annotations

import copy
import gc
import io
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .common import atomic_bytes, digest, progress, read, require, sha, write_json
from .data import guard_records, training_weights
from .models import Classifier


@dataclass(frozen=True)
class Partition:
    records: tuple
    features: np.ndarray
    allowed: tuple
    forbidden: tuple
    role: str
    rp: np.ndarray | None = None

    def check(self, fitting=False):
        guard_records(self.records, self.allowed, self.forbidden)
        require(len(self.features) == len(self.records) and np.isfinite(self.features).all(),
                "Features differ from assigned records")
        if fitting:
            require(self.role in {"inner_train", "development"}, "Fitting requires a training role")
        require(self.role in {"inner_train", "inner_validation", "development"},
                "Outer labels/data forbidden in development selection")

    @property
    def labels(self):
        self.check()
        return np.array([r.label for r in self.records], dtype=np.float32)


def fit_scaler(partition):
    partition.check(fitting=True)
    weights = training_weights(partition.records, partition.allowed, partition.forbidden)
    scaler = StandardScaler().fit(partition.features, sample_weight=weights)
    return scaler, {"role": partition.role, "record_ids_sha256": digest([r.logical_id for r in partition.records]),
                    "speakers": sorted({r.speaker_id for r in partition.records}),
                    "count": len(partition.records), "sample_weight": "training speaker inverse frequency"}


def scaler_dict(scaler, provenance):
    return {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "provenance": provenance}


def transform(features, spec):
    return np.asarray((features - np.asarray(spec["mean"])) / np.asarray(spec["scale"]), dtype=np.float32)


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Native B2 adaptive_avg_pool2d CUDA backward has no deterministic implementation.
    # Preserve the signed architecture, seed all RNGs, and expose PyTorch's warning.
    # Repeated fixed-checkpoint evaluation remains checked separately for exact equality.
    torch.use_deterministic_algorithms(True, warn_only=True)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(root, payload):
    root = Path(root)
    filename = f"epoch-{payload['epoch']:03d}-{uuid.uuid4().hex}.pt"
    stream = io.BytesIO()
    torch.save(payload, stream)
    atomic_bytes(root / filename, stream.getvalue())
    reference = {"path": str(root / filename), "sha256": sha(root / filename),
                 "epoch": payload["epoch"], "identity": payload["identity"]}
    write_json(root / "latest.json", reference, replace=True)
    return reference


def load_checkpoint(reference, identity):
    require(reference["identity"] == identity and sha(reference["path"]) == reference["sha256"],
            "Checkpoint identity/checksum mismatch")
    # Only locally created, checksum-bound study checkpoints; never original checkpoint deserialization.
    payload = torch.load(reference["path"], map_location="cpu", weights_only=False)
    require(payload["identity"] == identity, "Checkpoint payload identity mismatch")
    return payload


def metrics(labels, scores):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                 f1_score, precision_score, recall_score, roc_auc_score)
    labels, scores = np.asarray(labels, int), np.asarray(scores, float)
    require(len(labels) == len(scores) > 0 and np.isfinite(scores).all()
            and ((scores >= 0) & (scores <= 1)).all(), "Invalid evaluation arrays")
    predictions = scores >= 0.5
    tn, fp, fn, tp = map(int, confusion_matrix(labels, predictions, labels=[0, 1]).ravel())
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
            "specificity": tn/(tn+fp) if tn+fp else None,
            "roc_auc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
            "confusion_matrix": [[tn, fp], [fn, tp]], "record_count": len(labels), "threshold": 0.5}


def binary_loss(labels, probabilities):
    p = np.clip(np.asarray(probabilities, float), 1e-7, 1-1e-7)
    return float(-(labels * np.log(p) + (1-labels)*np.log1p(-p)).mean())


def selection_key(result):
    m = result["validation_metrics"]
    return (-m["macro_f1"], -m["balanced_accuracy"], -(m["roc_auc"] or 0),
            result["validation_loss"], result["layer"], result["candidate"]["id"])


def logistic_fit(train, validation, C, root, identity, layer, cfg):
    from sklearn.exceptions import ConvergenceWarning
    import warnings
    from threadpoolctl import threadpool_limits
    train.check(fitting=True)
    if validation:
        validation.check()
        require(validation.role == "inner_validation", "Only inner validation may select E1")
    complete = Path(root) / "complete.json"
    if complete.exists():
        result = read(complete)
        load_checkpoint(result["checkpoint"], identity)
        return result
    started = time.perf_counter()
    scaler, provenance = fit_scaler(train)
    weights = training_weights(train.records, train.allowed, train.forbidden)
    model = LogisticRegression(C=C, penalty="l2", solver="lbfgs", class_weight=None,
                               max_iter=cfg["training"]["logistic_max_iter"], random_state=42)
    with warnings.catch_warnings(), threadpool_limits(limits=cfg["training"]["cpu_threads"]):
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(scaler.transform(train.features), train.labels, sample_weight=weights)
    payload = {"identity": identity, "epoch": 0, "model_name": "E1", "model": model,
               "scaler": scaler_dict(scaler, provenance), "layer": layer, "rng": rng_state()}
    reference = save_checkpoint(root, payload)
    result = {"model_name": "E1", "layer": layer, "candidate": {"id": f"C_{C}", "C": C},
              "checkpoint": reference, "best_epoch": 0, "parameter_count": train.features.shape[1]+1,
              "training_seconds": time.perf_counter()-started, "history": []}
    if validation:
        probabilities = model.predict_proba(scaler.transform(validation.features))[:, 1]
        result.update(validation_metrics=metrics(validation.labels, probabilities),
                      validation_loss=binary_loss(validation.labels, probabilities))
    write_json(complete, result)
    return result


def cpu_tensors(partition, scaler, *, fitting):
    partition.check(fitting=fitting)
    result = {"speech": torch.from_numpy(transform(partition.features, scaler)).pin_memory(),
              "labels": torch.from_numpy(partition.labels).pin_memory()}
    if partition.rp is not None:
        result["rp"] = torch.from_numpy(partition.rp[:, None, :, :]).pin_memory()
    if fitting:
        result["weights"] = torch.from_numpy(training_weights(partition.records,
                           partition.allowed, partition.forbidden).astype(np.float32)).pin_memory()
    return result


def cuda_batch(tensors, indices):
    return {key: value[indices].pin_memory().cuda(non_blocking=True) for key, value in tensors.items()}


@torch.inference_mode()
def neural_predict(model, tensors, batch_size):
    model.eval()
    probabilities = []
    for start in range(0, len(tensors["speech"]), batch_size):
        batch = cuda_batch(tensors, slice(start, start+batch_size))
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(batch["speech"], batch.get("rp"))
        require(bool(torch.isfinite(logits).all()), "Non-finite evaluation logit")
        probabilities.append(logits.float().sigmoid().cpu().numpy())
    return np.concatenate(probabilities)


def train_neural(train, validation, model_name, candidate, root, identity, layer, cfg,
                 *, fixed_epochs=None):
    train.check(fitting=True)
    require((validation is None) == (fixed_epochs is not None), "Final fit requires frozen epoch count")
    if validation:
        validation.check()
        require(validation.role == "inner_validation", "Only inner validation may select epoch")
    root = Path(root)
    complete = root / "complete.json"
    if complete.exists():
        result = read(complete)
        load_checkpoint(result["checkpoint"], identity)
        return result
    seed_all()
    scaler, provenance = fit_scaler(train)
    scaler_spec = scaler_dict(scaler, provenance)
    tensors = cpu_tensors(train, scaler_spec, fitting=True)
    val = cpu_tensors(validation, scaler_spec, fitting=False) if validation else None
    model = Classifier(model_name, candidate["dropout"]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                 weight_decay=candidate["weight_decay"])
    amp = torch.amp.GradScaler("cuda")
    batch_size = cfg["training"].get("measured_batch_sizes", {}).get(model_name, cfg["training"]["batch_cap"])
    maximum = fixed_epochs or cfg["training"]["max_epochs"]
    history, best, best_epoch, best_metrics, best_loss = [], None, 0, None, float("inf")
    best_f1, stale, elapsed, start_epoch = -1.0, 0, 0.0, 1
    latest = root / "latest.json"
    if latest.exists():
        state = load_checkpoint(read(latest), identity)
        require(state["scaler"] == scaler_spec, "Resumed training scaler changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        amp.load_state_dict(state["amp"])
        restore_rng(state["rng"])
        history, best = state["history"], state["best"]
        best_epoch, best_metrics, best_loss = state["best_epoch"], state["best_metrics"], state["best_loss"]
        best_f1, stale, elapsed = state["best_f1"], state["stale"], state["elapsed"]
        start_epoch, batch_size = state["epoch"]+1, state["batch_size"]
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, maximum+1):
        if validation and stale >= cfg["training"]["patience"]:
            break
        # Roll back the whole unfinished epoch on OOM, including optimizer, scaler and RNG.
        rollback = {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "optimizer": copy.deepcopy(optimizer.state_dict()),
                    "amp": copy.deepcopy(amp.state_dict()), "rng": rng_state()}
        started = time.perf_counter()
        while True:
            try:
                model.train()
                order = torch.randperm(len(train.records))
                loss_total = 0.0
                for start in range(0, len(order), batch_size):
                    batch = cuda_batch(tensors, order[start:start+batch_size])
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.float16):
                        logits = model(batch["speech"], batch.get("rp"))
                        loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                            logits.float(), batch["labels"], reduction="none") * batch["weights"]).mean()
                    require(bool(torch.isfinite(loss)), "Non-finite training loss")
                    amp.scale(loss).backward()
                    amp.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                    require(bool(torch.isfinite(norm)), "Non-finite training gradient")
                    amp.step(optimizer)
                    amp.update()
                    loss_total += float(loss.detach()) * len(batch["labels"])
                probabilities = neural_predict(model, val, batch_size) if val else None
                break
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size//2)
                optimizer.zero_grad(set_to_none=True)
                model.load_state_dict(rollback["model"])
                optimizer.load_state_dict(rollback["optimizer"])
                amp.load_state_dict(rollback["amp"])
                restore_rng(rollback["rng"])
                gc.collect()
                torch.cuda.empty_cache()
                progress(f"Training OOM; restart unfinished epoch with batch {batch_size}")
        row = {"epoch": epoch, "training_loss": loss_total/len(train.records)}
        if validation:
            m = metrics(validation.labels, probabilities)
            loss = binary_loss(validation.labels, probabilities)
            row.update(validation_loss=loss, validation_macro_f1=m["macro_f1"])
            if m["macro_f1"] > best_f1 + cfg["training"]["min_delta"]:
                best_f1, stale = m["macro_f1"], 0
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch, best_metrics, best_loss = epoch, m, loss
            else:
                stale += 1
        else:
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch, best_loss = epoch, 0.0
        history.append(row)
        elapsed += time.perf_counter()-started
        payload = {"identity": identity, "epoch": epoch, "model_name": model_name,
                   "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "amp": amp.state_dict(), "rng": rng_state(), "scaler": scaler_spec,
                   "layer": layer, "candidate": candidate, "history": history, "best": best,
                   "best_epoch": best_epoch, "best_metrics": best_metrics, "best_loss": best_loss,
                   "best_f1": best_f1, "stale": stale, "elapsed": elapsed, "batch_size": batch_size}
        save_checkpoint(root, payload)
        progress(f"{model_name} layer {layer} {candidate['id']}: epoch {epoch}/{maximum}")
    reference = read(latest)
    result = {"model_name": model_name, "layer": layer, "candidate": candidate,
              "checkpoint": reference, "best_epoch": best_epoch,
              "parameter_count": model.parameter_count(), "training_seconds": elapsed,
              "history": history, "batch_size": batch_size,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    if validation:
        result.update(validation_metrics=best_metrics, validation_loss=best_loss)
    write_json(complete, result)
    del model, optimizer, tensors, val
    gc.collect()
    torch.cuda.empty_cache()
    return result


def benchmark_training(train, model_name, cfg):
    """Measure a disposable model on development inputs before choosing a batch cap."""
    train.check(fitting=True)
    scaler, provenance = fit_scaler(train)
    tensors = cpu_tensors(train, scaler_dict(scaler, provenance), fitting=True)
    timings = []
    for size in (1, 2, 4, 8, 16):
        if size > cfg["training"]["batch_cap"]:
            break
        seed_all()
        model = Classifier(model_name, cfg["training"]["neural_candidates"][0]["dropout"]).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
        amp = torch.amp.GradScaler("cuda")
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            indices = torch.arange(size) % len(train.records)
            for _ in range(2):
                batch = cuda_batch(tensors, indices)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(batch["speech"], batch.get("rp"))
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), batch["labels"])
                amp.scale(loss).backward()
                amp.step(optimizer)
                amp.update()
            torch.cuda.synchronize()
            timings.append({"batch_size": size, "seconds_per_step": (time.perf_counter()-started)/2,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        except torch.cuda.OutOfMemoryError:
            break
        finally:
            del model, optimizer, amp
            gc.collect()
            torch.cuda.empty_cache()
    memory = torch.cuda.get_device_properties(0).total_memory
    safe = [t for t in timings if t["peak_reserved_bytes"] <= memory*.8]
    require(safe, "No safe neural training batch")
    chosen = min(8, max(safe, key=lambda t: t["batch_size"]/t["seconds_per_step"])["batch_size"])
    return {"model": model_name, "timings": timings, "selected_batch_size": chosen,
            "policy": "development-only disposable model; two steps; conservative cap eight"}


def seal_outer(root, identity, infer):
    """Any consumed claim without a checksum-bound result stops; never retries inference."""
    root = Path(root)
    claim, result, seal = root/"claim.json", root/"outer_result.json", root/"seal.json"
    if claim.exists():
        require(read(claim)["identity"] == identity, "Outer claim identity differs")
        require(result.exists() and seal.exists(),
                "AMBIGUOUS OUTER CLAIM: explicit review required; automatic inference retry forbidden")
        require(read(seal)["identity"] == identity and read(seal)["result_sha256"] == sha(result),
                "Outer result seal invalid; explicit review required")
        return read(result)
    require(not result.exists() and not seal.exists(), "Orphan outer result/seal; explicit review required")
    # Exclusive creation + fsync before the first outer data/label access or model forward.
    write_json(claim, {"identity": identity, "status": "CLAIM_CONSUMED", "time_ns": time.time_ns()})
    value = infer()
    write_json(result, value)
    write_json(seal, {"identity": identity, "result_sha256": sha(result), "status": "SEALED"})
    return value
