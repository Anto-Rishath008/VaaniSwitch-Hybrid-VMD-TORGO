"""Noninteractive launcher. Only the explicit `run` command executes the full study."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.dont_write_bytecode = True
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import argparse
import gc
import platform
import shutil
import time
import traceback
import uuid

import numpy as np
import torch

from .common import (BASELINE, ORIGINAL, ROOT, STUDY, TITLE, ProcessLock, Stages,
                     config, digest, git, preservation_snapshot, progress, read, require,
                     sha, source_identity, verify_preservation, write_json)
from .data import Corpus, guard_records, load_rp, verify_inputs
from .features import (Extractor, FeatureCache, acquire, audio, extraction_identity,
                       validate_cache)
from .reporting import generate_report
from .reuse import acquire_readonly, reference_cache, verify_reuse
from .training import (Classifier, Partition, benchmark_training, load_checkpoint, logistic_fit, metrics,
                       neural_predict, seal_outer, seed_all, selection_key, train_neural,
                       transform)


def environment(cfg):
    import transformers
    require(sys.version_info[:2] == (3, 12), "Use the existing Python 3.12 environment")
    require(Path(sys.prefix).resolve() == (STUDY/".venv-accuracy").resolve(), "Wrong Python environment")
    require(torch.__version__ == "2.13.0+cu126" and torch.cuda.is_available(), "Expected existing CUDA torch")
    require(np.__version__ == "1.26.4" and transformers.__version__ == "4.57.6", "Verified dependency versions differ")
    require("4070 Laptop" in torch.cuda.get_device_name(0), "Expected RTX 4070 Laptop GPU")
    require(git(STUDY, "branch", "--show-current") == "experiment/dysarthria-accuracy-improvement",
            "Wrong experimental branch")
    require(git(STUDY, "rev-parse", "HEAD") == BASELINE
            and git(ORIGINAL, "rev-parse", "HEAD") == BASELINE, "Baseline commit differs")
    free = shutil.disk_usage(ROOT).free
    require(free > cfg["execution"]["minimum_free_disk_gib"] * 1024**3, "Insufficient free disk")
    torch.set_num_threads(cfg["training"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(cfg["execution"]["vram_fraction"])
    seed_all()
    for directory in ("cache", "checkpoints", "results", "logs", "plots", "smoke"):
        (ROOT/directory).mkdir(exist_ok=True)
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "free_disk_bytes": free, "baseline": BASELINE,
            "branch": git(STUDY, "branch", "--show-current"),
            "python_executable": sys.executable, "cpu_threads": torch.get_num_threads()}


def native_contract():
    import ast
    import yaml
    expected = (ORIGINAL/"src/vaaniswitch/m3_encoders/recurrence_encoder.py").read_text(encoding="utf-8")
    expected = expected.replace("from vaaniswitch.m3_encoders.config import RecurrenceEncoderConfig",
                                "from .native_b2_contract import RecurrenceEncoderConfig")
    expected = expected.replace("from vaaniswitch.m3_encoders.preprocessing import preprocess_recurrence",
                                "from .native_b2_preprocessing import preprocess_recurrence")
    require(ast.dump(ast.parse(expected)) == ast.dump(ast.parse(
            (ROOT/"src/native_b2_encoder.py").read_text(encoding="utf-8-sig"))), "B2 recurrence code drift")
    require(sha(ORIGINAL/"src/vaaniswitch/m3_encoders/preprocessing.py") ==
            sha(ROOT/"src/native_b2_preprocessing.py"), "B2 preprocessing differs")
    original_cfg = yaml.safe_load((ORIGINAL/"configs/models/encoders.yaml").read_text())
    require(original_cfg["recurrence"]["channels"] == [16,32,64]
            and original_cfg["common_token_dim"] == 128
            and original_cfg["regularization"]["dropout"] == .2, "B2 architecture contract differs")
    paths = ["data/manifests/torgo_manifest.jsonl", "data/splits/speaker_folds.json",
             "configs/models/encoders.yaml", "configs/representations/default.yaml",
             "artifacts/metrics/module2_full_signoff_audit.json",
             "src/vaaniswitch/m3_encoders/recurrence_encoder.py",
             "src/vaaniswitch/m3_encoders/preprocessing.py"]
    closure = ORIGINAL/"artifacts/module5/677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9/final_closure"
    manifest = closure/"evidence_manifest.json"
    require(sha(manifest) == "f9d4e56917b2bfc63abbf312e2b389457aa2ad29734424f01da5ee03109a8a6d",
            "Original signed Module 5 closure manifest changed")
    package = read(manifest)
    for name, spec in package["files"].items():
        require(sha(closure/name) == spec["sha256"], f"Signed closure artifact changed: {name}")
        paths.append((closure/name).relative_to(ORIGINAL).as_posix())
    paths.append(manifest.relative_to(ORIGINAL).as_posix())
    return {p: sha(ORIGINAL/p) for p in paths}


def make_partition(corpus, speakers, split, role, arrays, layer, rp_arrays=None):
    records = corpus.select(speakers)
    guard_records(records, speakers, split.test)
    indices = [arrays["index"][r.logical_id] for r in records]
    layer_index = (6, 9, 12).index(layer)
    return Partition(records, arrays["values"][indices, layer_index], tuple(speakers), split.test,
                     role, rp_arrays[indices] if rp_arrays is not None else None)


def choose_and_fit(corpus, split, model, cfg, arrays, rp_arrays, run_identity, stages):
    candidates = []
    for layer in cfg["features"]["layers"]:
        train = make_partition(corpus, split.train, split, "inner_train", arrays, layer, rp_arrays)
        val = make_partition(corpus, split.validation, split, "inner_validation", arrays, layer, rp_arrays)
        settings = ([{"id": f"C_{c}", "C": c} for c in cfg["training"]["logistic_C"]]
                    if model == "E1" else cfg["training"]["neural_candidates"])
        for candidate in settings:
            key = f"fold{split.fold}_{model}_layer{layer}_{candidate['id']}"
            identity = digest({"run": run_identity, "stage": key, "split": split.identity,
                               "train": [r.logical_id for r in train.records],
                               "validation": [r.logical_id for r in val.records]})
            directory = ROOT/"checkpoints"/key

            def action(train=train, val=val, candidate=candidate, directory=directory,
                       identity=identity, layer=layer):
                result = (logistic_fit(train, val, candidate["C"], directory, identity, layer, cfg)
                          if model == "E1" else
                          train_neural(train, val, model, candidate, directory, identity, layer, cfg))
                return result, [directory/"complete.json", Path(result["checkpoint"]["path"])]
            candidates.append(stages.execute(key, action))
    winner = min(candidates, key=selection_key)
    selection_path = ROOT/"checkpoints"/f"fold{split.fold}_{model}_selection.json"
    selection = {"model": model, "outer_fold": split.fold, "split": split.identity,
                 "winner": winner, "candidates": candidates, "selection_role": "inner_validation",
                 "threshold": .5, "run_identity": run_identity}
    if selection_path.exists():
        require(read(selection_path) == selection, "Completed selection changed")
    else:
        write_json(selection_path, selection)
    dev = make_partition(corpus, split.development, split, "development", arrays, winner["layer"], rp_arrays)
    key = f"fold{split.fold}_{model}_final"
    directory = ROOT/"checkpoints"/key
    identity = digest({"run": run_identity, "stage": key, "selection_sha256": sha(selection_path),
                       "development_ids": [r.logical_id for r in dev.records]})

    def final():
        result = (logistic_fit(dev, None, winner["candidate"]["C"], directory, identity,
                               winner["layer"], cfg) if model == "E1" else
                  train_neural(dev, None, model, winner["candidate"], directory, identity,
                               winner["layer"], cfg, fixed_epochs=winner["best_epoch"]))
        return result, [directory/"complete.json", Path(result["checkpoint"]["path"])]
    result = stages.execute(key, final)
    selection["final"] = result
    selection["selection_sha256"] = sha(selection_path)
    selection["training_seconds"] = sum(c["training_seconds"] for c in candidates) + result["training_seconds"]
    return selection


def evaluate(corpus, split, model, selected, arrays, rp_arrays, freeze_sha, run_identity):
    result = selected["final"]
    outer_identity = {"run_identity": run_identity, "freeze_sha256": freeze_sha,
                      "checkpoint": result["checkpoint"], "split": split.identity,
                      "selection_sha256": selected["selection_sha256"], "threshold": .5}

    def infer():
        # First test feature/label access happens only after seal_outer durably consumes its claim.
        records = corpus.select(split.test)
        indices = [arrays["index"][r.logical_id] for r in records]
        values = arrays["values"][indices, (6,9,12).index(result["layer"])]
        checkpoint = load_checkpoint(result["checkpoint"], result["checkpoint"]["identity"])
        values = transform(values, checkpoint["scaler"])
        started = time.perf_counter()
        if model == "E1":
            probabilities = checkpoint["model"].predict_proba(values)[:, 1]
        else:
            network = Classifier(model, result["candidate"]["dropout"]).cuda().eval()
            network.load_state_dict(checkpoint["best"])
            tensors = {"speech": torch.from_numpy(values).pin_memory()}
            if model == "E3":
                tensors["rp"] = torch.from_numpy(rp_arrays[indices, None, :, :]).pin_memory()
            # Conservative, measured batch; any post-claim OOM stops for review, no retry.
            probabilities = neural_predict(network, tensors, min(4, result["batch_size"]))
            del network, tensors
            gc.collect()
            torch.cuda.empty_cache()
        predictions = [{"model": model, "outer_fold": split.fold, "logical_id": r.logical_id,
                        "speaker_id": r.speaker_id, "label": r.label, "probability": float(p),
                        "prediction": int(p >= .5)} for r, p in zip(records, probabilities, strict=True)]
        return {"study_label": TITLE, "model": model, "outer_fold": split.fold,
                "metrics": metrics([r.label for r in records], probabilities), "predictions": predictions,
                "training_seconds": selected["training_seconds"],
                "final_training_seconds": result["training_seconds"],
                "parameter_count": result["parameter_count"], "layer": result["layer"],
                "selected_epochs": result["best_epoch"], "candidate": result["candidate"],
                "evaluation_seconds": time.perf_counter()-started, "identity": outer_identity}
    return seal_outer(ROOT/"results/outer"/f"fold{split.fold}_{model}", outer_identity, infer)


def smoke(cfg, env, corpus, namespace=None):
    namespace = (Path(namespace).resolve() if namespace else ROOT/"smoke"/("cuda-"+uuid.uuid4().hex))
    require(namespace.is_relative_to((ROOT/"smoke").resolve()), "Smoke output must stay in smoke namespace")
    require(not (namespace/"SMOKE_REPORT.json").exists(), "Completed smoke sealed; choose a new namespace")
    namespace.mkdir(parents=True, exist_ok=True)
    write_json(namespace/("attempt-"+uuid.uuid4().hex+".json"),
               {"source_identity": source_identity(), "config": cfg, "time_ns": time.time_ns()})
    split = corpus.splits[0]
    # Four training records from two inner-training speakers; one per validation speaker.
    training = []
    for condition in ("control", "dysarthric"):
        speaker = min(s for s in split.train if any(r.speaker_id == s and r.condition == condition
                                                   for r in corpus.records))
        available = [r for r in corpus.records if r.speaker_id == speaker]
        training += [min(available, key=lambda r: (abs(r.duration_seconds-1), r.logical_id)),
                     max(available, key=lambda r: (r.duration_seconds, r.logical_id))]
    validation = [min(corpus.select([s]), key=lambda r: (abs(r.duration_seconds-2), r.logical_id))
                  for s in split.validation]
    records = tuple(training+validation)
    guard_records(records, split.development, split.test)
    verify_inputs(records)
    checkpoint, lock = acquire(cfg)
    cache = FeatureCache(namespace/"features", extraction_identity(cfg, lock))
    extractor = Extractor(cfg, checkpoint)
    waves = [audio(r)[0] for r in records[:4]]
    benchmark = extractor.benchmark(waves)
    extraction = extractor.extract(records, cache)
    validation_cache = validate_cache(records, cache)
    cache_before = digest(validation_cache)
    reused = extractor.extract(records, cache)
    require(reused["newly_extracted"] == 0
            and cache_before == digest(validate_cache(records, cache)), "Smoke cache resume failed")
    extractor.close()
    values = np.stack([cache.load(r)[1] for r in records])  # hidden_states[9] for smoke only
    rp = np.stack([load_rp(r) for r in records])
    train = Partition(tuple(training), values[:4], split.train, split.test, "inner_train", rp[:4])
    val = Partition(tuple(validation), values[4:], split.validation, split.test, "inner_validation", rp[4:])
    tiny = read_yaml_copy(cfg)
    tiny["training"].update(max_epochs=1, batch_cap=4)
    training_benchmarks = {m: benchmark_training(train, m, cfg) for m in ("E2", "E3")}
    tiny["training"]["measured_batch_sizes"] = {m: min(4, b["selected_batch_size"])
                                                for m, b in training_benchmarks.items()}
    stages = Stages(namespace/"stages", "smoke-only")
    candidates = {}
    for model in ("E1", "E2", "E3"):
        directory = namespace/"checkpoints"/model
        identity = digest({"smoke": str(namespace), "model": model})

        def action(model=model, directory=directory, identity=identity):
            result = (logistic_fit(train, val, .1, directory, identity, 9, tiny) if model == "E1" else
                      train_neural(train, val, model, cfg["training"]["neural_candidates"][0],
                                   directory, identity, 9, tiny))
            return result, [directory/"complete.json", Path(result["checkpoint"]["path"])]
        result = stages.execute(model, action)
        require(stages.execute(model, lambda: (_ for _ in ()).throw(AssertionError("repeated stage"))) == result,
                "Smoke stage resume failed")
        load_checkpoint(result["checkpoint"], identity)
        candidates[model] = result
    measured = {m: {k: r[k] for k in ("parameter_count", "training_seconds", "best_epoch")}
                for m, r in candidates.items()}
    all_timings = benchmark["timings"] + [t for b in training_benchmarks.values() for t in b["timings"]]
    peak = max([t["peak_allocated_bytes"] for t in all_timings]
               + [candidates[m]["peak_allocated_bytes"] for m in ("E2", "E3")])
    result = {"status": "PASS", "namespace": str(namespace), "environment": env,
              "development_only": True, "outer_test_forwards": 0,
              "logical_ids": [r.logical_id for r in records], "benchmark": benchmark,
              "training_benchmarks": training_benchmarks,
              "extraction": extraction, "feature_shape": [6,3,1536],
              "cropped_in_smoke": validation_cache["cropped_recordings"],
              "classifiers": measured, "peak_allocated_bytes": peak,
              "peak_reserved_bytes": max([t["peak_reserved_bytes"] for t in all_timings]
                                        + [candidates[m]["peak_reserved_bytes"] for m in ("E2", "E3")]),
              "model_lock": lock, "cache_resume": "PASS", "stage_resume": "PASS",
              "note": "Development smoke metrics intentionally omitted; not final accuracy"}
    write_json(namespace/"SMOKE_REPORT.json", result)
    write_json(ROOT/"logs/latest_smoke.json", {"path": str(namespace/"SMOKE_REPORT.json"),
               "sha256": sha(namespace/"SMOKE_REPORT.json")}, replace=True)
    progress(f"CUDA smoke PASS: {namespace}")
    return result


def read_yaml_copy(cfg):
    import copy
    return copy.deepcopy(cfg)


def extraction_stage(cfg, corpus, cache, checkpoint):
    path = ROOT/"logs/extraction_summary.json"
    if path.exists():
        validate_cache(corpus.records, cache)
        return read(path), [path]
    extractor = Extractor(cfg, checkpoint)
    try:
        dev = corpus.select(corpus.splits[0].train)
        panel = [max(dev, key=lambda r: r.duration_seconds), min(dev, key=lambda r: abs(r.duration_seconds-2))]
        benchmark = extractor.benchmark([audio(r)[0] for r in panel])
        result = extractor.extract(corpus.records, cache)
    finally:
        extractor.close()
    value = {"extraction": result, "benchmark": benchmark}
    write_json(path, value)
    return value, [path]


def freeze_run(cfg, env, sources, lock):
    identity = {"source_files": source_identity(), "inputs": sources, "config": cfg,
                "model_lock": lock, "environment_versions": {k: env[k] for k in
                ("python", "torch", "numpy", "transformers", "cuda_runtime", "gpu")},
                "repair_lineage": read(ROOT/"V1_LINEAGE.json")}
    protocol_path = ROOT/"logs/frozen_run.json"
    if protocol_path.exists():
        require(read(protocol_path) == identity, "Frozen study identity changed; explicit new study required")
    else:
        write_json(protocol_path, identity)
    return identity


def prepare(cfg, env, corpus):
    """Freeze V2 and verify reuse without any classifier fit or outer forward."""
    _, lock = acquire_readonly(cfg)
    identity = freeze_run(cfg, env, native_contract(), lock)
    stages = Stages(ROOT/"logs/stages", digest(identity))
    cache = reference_cache(cfg, lock)
    # Always verify actual bytes, including on a repeated prepare.
    report, paths = verify_reuse(corpus, cfg, cache)
    stages.execute("features", lambda: (report, paths))
    progress("V2 prepared; no full training or outer inference executed")


def full_run(cfg, env, corpus):
    # This function is invoked only by the user's manual `run_all.ps1` command.
    sources = native_contract()
    preservation_path = ROOT/"logs/full_run_preservation_before.json"
    if not preservation_path.exists():
        write_json(preservation_path, preservation_snapshot())
    before = read(preservation_path)
    verify_preservation(before)
    progress("Full input integrity verification")
    verify_inputs(corpus.records)  # Always rechecked on resume, even when earlier stages completed.
    _, lock = acquire_readonly(cfg)
    identity = freeze_run(cfg, env, sources, lock)
    run_identity = digest(identity)
    stages = Stages(ROOT/"logs/stages", run_identity)
    cache = reference_cache(cfg, lock)

    reuse, artifacts = verify_reuse(corpus, cfg, cache)
    stages.execute("features", lambda: (reuse, artifacts))
    validation = validate_cache(corpus.records, cache)  # Never trust a stage marker instead of actual cache bytes.
    validation_path = ROOT/"logs/feature_validation.json"
    if validation_path.exists():
        require(read(validation_path) == validation, "Feature cache bindings changed")
    else:
        write_json(validation_path, validation)
    require(validation["cropped_recordings"] == corpus.audit()["cropped_recordings"], "Crop count mismatch")
    arrays = {"index": {r.logical_id: i for i, r in enumerate(corpus.records)},
              "values": np.stack([cache.load(r) for r in corpus.records])}
    progress("Loading existing signed RP into CPU memory once")
    rp_arrays = np.stack([load_rp(r) for r in corpus.records])
    fit_cfg = read_yaml_copy(cfg)
    def measure_training():
        path = ROOT/"logs/training_benchmarks.json"
        if path.exists():
            return read(path), [path]
        split = corpus.splits[0]
        panel = make_partition(corpus, split.train, split, "inner_train", arrays, 9, rp_arrays)
        value = {m: benchmark_training(panel, m, cfg) for m in ("E2", "E3")}
        write_json(path, value)
        return value, [path]
    measured = stages.execute("training_benchmarks", measure_training)
    fit_cfg["training"]["measured_batch_sizes"] = {m: b["selected_batch_size"] for m, b in measured.items()}
    selections = {}
    # Finish every model/fold decision and final development fit before ANY outer inference.
    for split in corpus.splits:
        for model in ("E1", "E2", "E3"):
            key = f"fold{split.fold}_{model}"
            selections[key] = choose_and_fit(corpus, split, model, fit_cfg, arrays,
                                             rp_arrays if model == "E3" else None, run_identity, stages)
    frozen = {"study_label": TITLE, "run_identity": run_identity, "config": fit_cfg,
              "model_lock": lock, "feature_validation_sha256": sha(validation_path),
              "selections": {k: {"selection_sha256": v["selection_sha256"],
                                   "final_checkpoint": v["final"]["checkpoint"],
                                   "layer": v["winner"]["layer"], "candidate": v["winner"]["candidate"],
                                   "epochs": v["winner"]["best_epoch"]} for k, v in selections.items()}}
    freeze_path = ROOT/"results/FROZEN_DECISIONS.json"
    if freeze_path.exists():
        require(read(freeze_path) == frozen, "Decisions changed after freezing")
    else:
        write_json(freeze_path, frozen)
    results = []
    for split in corpus.splits:
        for model in ("E1", "E2", "E3"):
            require(read(freeze_path) == frozen and source_identity() == identity["source_files"],
                    "Frozen decisions/source changed before outer evaluation")
            progress(f"Outer fold {split.fold}/3: {model}; decisions already frozen")
            results.append(evaluate(corpus, split, model, selections[f"fold{split.fold}_{model}"],
                                    arrays, rp_arrays, sha(freeze_path), run_identity))
    baseline_path = ORIGINAL/"artifacts/module5/677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9/final_closure/cross_fold_metrics.json"
    baseline = {m: read(baseline_path)["ablations"][m]["summary"] for m in ("B2", "C2")}

    def report():
        summary, paths = generate_report(corpus, results, selections, frozen, baseline)
        return {"status": "PASS", "models": list(summary["models"])}, paths
    stages.execute("reports", report)
    preservation = verify_preservation(before)
    require(source_identity() == identity["source_files"], "Experimental source changed during execution")
    final_path = ROOT/"logs/final_preservation.json"
    if final_path.exists():
        require(read(final_path) == preservation, "Preservation evidence changed")
    else:
        write_json(final_path, preservation)
    progress("FULL EXPLORATORY STUDY COMPLETE; outputs sealed")


class Tee:
    def __init__(self, terminal, stream):
        self.terminal, self.stream = terminal, stream
    def write(self, text):
        self.terminal.write(text)
        self.stream.write(text)
        self.stream.flush()
    def flush(self):
        self.terminal.flush()
        self.stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "smoke", "prepare", "run"])
    parser.add_argument("--smoke-namespace", type=Path, help="Resume a specific unfinished smoke namespace")
    args = parser.parse_args()
    (ROOT/"logs").mkdir(exist_ok=True)
    logfile = ROOT/"logs"/(args.command+"-"+time.strftime("%Y%m%dT%H%M%S")+"-"+uuid.uuid4().hex[:8]+".log")
    with logfile.open("x", encoding="utf-8") as stream, ProcessLock(ROOT/"logs/process.lock"):
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(stdout, stream), Tee(stderr, stream)
        try:
            cfg = config()
            env = environment(cfg)
            corpus = Corpus()
            audit = {"environment": env, "dataset": corpus.audit(), "native_sources": native_contract()}
            progress(f"Environment/data audit PASS: {env['gpu']}, {len(corpus.records)} CORE records")
            if args.command == "audit":
                path = ROOT/"logs"/("audit-"+uuid.uuid4().hex+".json")
                write_json(path, audit)
                print(path, flush=True)
            elif args.command == "smoke":
                smoke(cfg, env, corpus, args.smoke_namespace)
            elif args.command == "prepare":
                prepare(cfg, env, corpus)
            else:
                full_run(cfg, env, corpus)
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            sys.stdout, sys.stderr = stdout, stderr


if __name__ == "__main__":
    main()
