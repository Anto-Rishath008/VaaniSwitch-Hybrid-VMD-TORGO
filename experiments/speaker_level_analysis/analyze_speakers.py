"""Post-hoc analysis of immutable outer predictions; no project/model imports.

Run with an existing Python containing NumPy, SciPy, scikit-learn and Matplotlib.
All writes, including Matplotlib's cache, stay beside this script. Existing
published results are never overwritten. --verify checks retained outputs only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
STUDY = OUT.parent.parent
ORIGINAL = STUDY.parent / "VaaniSwitch_FINAL_Hybrid_VMD_TORGO_Project"
BASELINE = "2dee81f7656e5b5cae800d9a3c8692e5e8107bf9"
PROTOCOL = "677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9"
CODE = "7696cad010914ec37f494d9f1a11a2e1761e46f353e61d97a07399f746a34ff8"
ANCHOR = "f9d4e56917b2bfc63abbf312e2b389457aa2ad29734424f01da5ee03109a8a6d"
BASE = ORIGINAL / "artifacts/module5" / PROTOCOL
PACKAGE = BASE / "final_closure"
TITLE = "POST-HOC EXPLORATORY SPEAKER-LEVEL ANALYSIS"
MODELS = ("B2", "C2")
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "precision",
           "sensitivity", "specificity", "roc_auc")
THRESHOLD = 0.5
BOOTSTRAPS = 20000
SEED = 20261011
EXCLUDED = {".git", ".venv", ".venv-m6", "__pycache__", ".pytest_cache",
            ".ruff_cache", ".mypy_cache"}
SCIENCE_DOC = "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def output_path(name):
    path = OUT / name
    require(path.resolve().is_relative_to(OUT), "Output escapes analysis directory")
    return path


def write_text(name, text):
    with output_path(name).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def write_json(name, value):
    write_text(name, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(name, rows):
    require(bool(rows), "Cannot publish an empty table")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    write_text(name, stream.getvalue())


def git(root, *args):
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    return subprocess.check_output(["git", "-C", str(root), *args], env=env).decode().strip()


def git_state(root):
    return {"head": git(root, "rev-parse", "HEAD"),
            "branch": git(root, "branch", "--show-current"),
            "status_porcelain_v1": git(root, "status", "--porcelain=v1"),
            "release_ref": git(root, "rev-parse", "refs/heads/release/vaaniswitch-final-m0-m6"),
            "experiment_ref": git(root, "rev-parse", "refs/heads/experiment/dysarthria-accuracy-improvement")}


def inventory(root):
    """Hash project/evidence files; inventory raw and processed corpora by metadata.

    Virtual environments, Git internals and tool caches are outside this audit.
    No raw waveform or processed tensor is opened.
    """
    result = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED)
        for name in sorted(files):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            stat = path.stat()
            entry = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if not relative.startswith(("data/raw/", "data/processed/")):
                entry["sha256"] = sha(path)
            result[relative] = entry
    return result


def tracked_inventory(root):
    return {name: sha(root / name) for name in git(root, "ls-files").splitlines()}


def digest_ids(ids):
    return hashlib.sha256(json.dumps(ids, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def folds_fingerprint(document):
    """Match Module0 leakage.py: canonical JSON, rather than file byte hash."""
    payload = json.dumps(document, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def exact_interval(successes, trials):
    from scipy.stats import binomtest
    if not trials:
        return None
    ci = binomtest(successes, trials).proportion_ci(confidence_level=0.95, method="exact")
    return {"low": float(ci.low), "high": float(ci.high),
            "confidence_level": 0.95, "method": "Clopper-Pearson exact binomial",
            "successes": successes, "trials": trials}


def metrics(labels, scores):
    import numpy as np
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                 f1_score, precision_score, recall_score, roc_auc_score)
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predicted = scores >= THRESHOLD
    tn, fp, fn, tp = [int(x) for x in confusion_matrix(labels, predicted, labels=[0, 1]).ravel()]
    result = {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, labels=[0, 1], average="macro", zero_division=0)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "sensitivity": float(recall_score(labels, predicted, pos_label=1, zero_division=0)) if tp + fn else None,
        "specificity": float(recall_score(labels, predicted, pos_label=0, zero_division=0)) if tn + fp else None,
        "roc_auc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
        "confusion_matrix": [[tn, fp], [fn, tp]], "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "correct": tn + tp, "incorrect": fp + fn, "count": len(labels), "threshold": THRESHOLD,
    }
    result["confidence_intervals_95"] = {
        "accuracy": exact_interval(tn + tp, len(labels)),
        "precision": exact_interval(tp, tp + fp),
        "sensitivity": exact_interval(tp, tp + fn),
        "specificity": exact_interval(tn, tn + fp),
    }
    # Independent arithmetic check on all threshold metrics and tie-aware AUC.
    require(math.isclose(result["accuracy"], (tn + tp) / len(labels)), "Accuracy mismatch")
    if tp + fn and tn + fp:
        require(math.isclose(result["balanced_accuracy"], (tp / (tp + fn) + tn / (tn + fp)) / 2), "Balanced accuracy mismatch")
        positive, negative = scores[labels == 1], scores[labels == 0]
        pairwise_auc = float(((positive[:, None] > negative).sum() +
                              0.5 * (positive[:, None] == negative).sum()) / (len(positive) * len(negative)))
        require(math.isclose(pairwise_auc, result["roc_auc"]), "AUC mismatch")
    f1p = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
    f1n = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0
    require(math.isclose(result["macro_f1"], (f1p + f1n) / 2), "Macro-F1 mismatch")
    return result


def add_bootstrap_intervals(result, rows):
    """Resample speakers within true class; never resample individual recordings."""
    import numpy as np
    positives = np.array([r["mean_dysarthria_probability"] for r in rows if r["true_label"] == 1])
    negatives = np.array([r["mean_dysarthria_probability"] for r in rows if r["true_label"] == 0])
    rng = np.random.default_rng(SEED)
    ps = positives[rng.integers(len(positives), size=(BOOTSTRAPS, len(positives)))]
    ns = negatives[rng.integers(len(negatives), size=(BOOTSTRAPS, len(negatives)))]
    tp, tn = (ps >= THRESHOLD).sum(axis=1), (ns < THRESHOLD).sum(axis=1)
    fn, fp = len(positives) - tp, len(negatives) - tn
    values = {
        "balanced_accuracy": (tp / len(positives) + tn / len(negatives)) / 2,
        "macro_f1": (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn)) / 2,
        "roc_auc": ((ps[:, :, None] > ns[:, None, :]) +
                    0.5 * (ps[:, :, None] == ns[:, None, :])).mean(axis=(1, 2)),
    }
    for name, samples in values.items():
        low, high = np.quantile(samples, [0.025, 0.975], method="linear")
        result["confidence_intervals_95"][name] = {
            "low": float(low), "high": float(high), "confidence_level": 0.95,
            "method": "class-stratified speaker bootstrap percentile",
            "resamples": BOOTSTRAPS, "seed": SEED,
        }


def validate_sources(before):
    require(sha(PACKAGE / "evidence_manifest.json") == ANCHOR, "Signed closure anchor differs")
    evidence = read(PACKAGE / "evidence_manifest.json")
    for name, expected in evidence["files"].items():
        require(sha(PACKAGE / name) == expected["sha256"], f"Closure package drift: {name}")
    for name, expected in evidence["input_evidence"].items():
        normalized = Path(name).as_posix()
        require(normalized in before, f"Missing signed input: {name}")
        require(before[normalized].get("sha256") == expected["sha256"], f"Signed input drift: {name}")
    closure = read(PACKAGE / "module5_final_closure.json")
    require(closure["status"] == "COMPLETE / PASSED / SIGNED OFF", "Unsigned closure")
    audit = read(PACKAGE / "final_outer_audit.json")
    require(audit["protocol_hash"] == PROTOCOL and audit["frozen_code_hash"] == CODE, "Wrong scientific identity")
    protocol = read(BASE / "protocol.json")
    manifest_path = ORIGINAL / "data/manifests/torgo_manifest.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    manifest = {r["logical_id"]: r for r in records}
    require(len(manifest) == len(records), "Duplicate manifest logical ID")
    # Exact existing Module1 CORE rule, read from src/vaaniswitch/m1_vmd/manifest.py.
    core = sorted((r for r in records if r["normal_speech_eligible"]
                   and r["usability_status"] == "usable"
                   and r["prompt_type"] in {"word", "sentence"}), key=lambda r: r["logical_id"])
    require(len(core) == protocol["core_records"] == 8673, "CORE count mismatch")
    folds_path = ORIGINAL / "data/splits/speaker_folds.json"
    folds = read(folds_path)
    mapping = folds["speaker_to_evaluation_fold"]
    require(len(mapping) == 15, "Expected 15 speakers")
    manifest_hash, folds_hash = sha(manifest_path), folds_fingerprint(folds)
    source_rows, provenance = [], []
    for model in MODELS:
        seen = set()
        cells = sorted((c for c in audit["cells"] if c["ablation"] == model), key=lambda c: c["outer_fold"])
        require([c["outer_fold"] for c in cells] == [1, 2, 3], "Missing/duplicate outer cell")
        for cell in cells:
            source = ORIGINAL / cell["result_path"]
            require(source.resolve().is_relative_to(ORIGINAL), "Source escapes original project")
            require(sha(source) == cell["result_sha256"], "Prediction file hash differs")
            result = read(source)
            identity, fold = result["identity"], cell["outer_fold"]
            split = identity["split"]
            speakers = {s for s, f in mapping.items() if f == fold}
            expected_ids = [r["logical_id"] for r in core if r["speaker_id"] in speakers]
            observed_ids = [r["logical_id"] for r in result["predictions"]]
            require(cell["provenance"] == "NATIVE" and result["status"] == "OUTER_EVALUATED", "Non-native or non-outer result")
            require(result["ablation"] == model and result["outer_fold"] == fold, "Result model/fold differs")
            require(identity["config_hash"] == PROTOCOL and identity["code_hash"] == CODE
                    and identity["mode"] == "production", "Result identity differs")
            require(result["threshold"] == identity["threshold"] == result["metrics"]["threshold"] == THRESHOLD, "Threshold differs")
            require(split["manifest_fingerprint"] == manifest_hash and split["folds_fingerprint"] == folds_hash, "Manifest/split hash differs")
            require(set(split["test_speakers"]) == speakers and split["outer_test_fold"] == fold, "Wrong test speakers")
            require(not speakers.intersection(split["train_speakers"] + split["validation_speakers"]), "Development/test speaker overlap")
            require(set(split["train_speakers"] + split["validation_speakers"]) == set(mapping) - speakers, "Development membership differs")
            require(observed_ids == expected_ids and len(observed_ids) == len(set(observed_ids)), "Missing/duplicate/selected records")
            require(digest_ids(observed_ids) == cell["record_hash"], "Outer record hash differs")
            require(not seen.intersection(observed_ids), "Repeated evaluation record across folds")
            seen.update(observed_ids)
            require(result["source_load_count"] == result["forward_count"] == len(observed_ids), "Saved evaluation count differs")
            checkpoint = source.parent / "final" / result["checkpoint"]["filename"]
            require(sha(checkpoint) == result["checkpoint"]["sha256"] == cell["checkpoint_sha256"], "Checkpoint hash differs")
            for prediction in result["predictions"]:
                record = manifest[prediction["logical_id"]]
                probability = prediction["probability"]
                require(type(probability) in (int, float) and math.isfinite(probability)
                        and 0 <= probability <= 1, "Missing/non-finite/out-of-range probability")
                require(prediction["speaker_id"] == record["speaker_id"] and prediction["speaker_id"] in speakers, "Unreliable speaker identity")
                require(type(prediction["label"]) is int and prediction["label"] == int(record["condition"] == "dysarthric"), "True label differs")
                source_rows.append({
                    "model_variant": model, "outer_fold": fold, "logical_id": prediction["logical_id"],
                    "speaker_id": prediction["speaker_id"], "true_label": prediction["label"],
                    "dysarthria_probability": probability, "predicted_label": int(probability >= THRESHOLD),
                    "predicted_label_source": "derived from saved probability using original >=0.5 rule",
                    "session_id": record["session_id"], "prompt_id": record["prompt_id"],
                    "primary_microphone": record["primary_microphone"],
                    "source_relative_path": source.relative_to(ORIGINAL).as_posix(),
                    "source_sha256": cell["result_sha256"],
                })
            computed = metrics([p["label"] for p in result["predictions"]],
                               [p["probability"] for p in result["predictions"]])
            for key in (*METRICS, "confusion_matrix"):
                actual, saved = computed[key], result["metrics"][key]
                require(actual == saved if key == "confusion_matrix" else math.isclose(actual, saved, abs_tol=1e-12), f"Saved utterance metric differs: {key}")
                require(saved == cell["metrics"][key], "Signed audit metric differs")
            provenance.append({
                "model_variant": model, "outer_fold": fold, "experiment_id": result["experiment_id"],
                "provenance": cell["provenance"], "source_relative_path": source.relative_to(ORIGINAL).as_posix(),
                "source_sha256": cell["result_sha256"], "outer_record_hash": cell["record_hash"],
                "checkpoint_sha256": cell["checkpoint_sha256"], "evaluation_speakers": sorted(speakers),
                "record_count": len(observed_ids), "saved_status": result["status"],
                "all_probabilities_valid": True, "exact_core_record_coverage": True,
                "development_test_speakers_disjoint": True, "utterance_metrics_reproduced": True,
            })
        require(seen == {r["logical_id"] for r in core}, "Pooled CORE coverage differs")
    return source_rows, provenance, evidence, closure


def aggregate(source_rows):
    groups = defaultdict(list)
    for row in source_rows:
        groups[row["model_variant"], row["speaker_id"]].append(row)
    speaker_rows = []
    for (model, speaker), group in sorted(groups.items()):
        require(len({r["true_label"] for r in group}) == len({r["outer_fold"] for r in group}) == 1, "Speaker label/fold not unique")
        score = statistics.mean(r["dysarthria_probability"] for r in group)
        label, predicted = group[0]["true_label"], int(score >= THRESHOLD)
        speaker_rows.append({
            "model_variant": model, "speaker_id": speaker, "outer_fold": group[0]["outer_fold"],
            "true_label": label, "true_condition": "dysarthric" if label else "control",
            "recording_count": len(group), "mean_dysarthria_probability": score,
            "threshold": THRESHOLD, "predicted_label": predicted,
            "predicted_condition": "dysarthric" if predicted else "control", "correct": predicted == label,
            "source_relative_path": group[0]["source_relative_path"], "source_sha256": group[0]["source_sha256"],
        })
    require(len(speaker_rows) == 30, "Expected 15 speakers per model")
    summary, fold_csv = {}, []
    original_summary = read(PACKAGE / "cross_fold_metrics.json")["ablations"]
    for model in MODELS:
        rows = [r for r in speaker_rows if r["model_variant"] == model]
        require(len({r["speaker_id"] for r in rows}) == 15, "Duplicate pooled speaker")
        pooled = metrics([r["true_label"] for r in rows], [r["mean_dysarthria_probability"] for r in rows])
        add_bootstrap_intervals(pooled, rows)
        folds = {}
        for fold in (1, 2, 3):
            selected = [r for r in rows if r["outer_fold"] == fold]
            require(len(selected) == 5, "Expected five speakers per fold")
            fold_metrics = metrics([r["true_label"] for r in selected], [r["mean_dysarthria_probability"] for r in selected])
            folds[str(fold)] = {**fold_metrics, "speakers": [r["speaker_id"] for r in selected]}
            fold_csv.append({"model_variant": model, "outer_fold": fold,
                             **{key: fold_metrics[key] for key in METRICS},
                             **{key: fold_metrics[key] for key in ("TN", "FP", "FN", "TP", "correct", "incorrect", "count")}})
        fold_summary = {key: {"mean": statistics.mean(f[key] for f in folds.values()),
                              "sample_std_ddof1": statistics.stdev(f[key] for f in folds.values())}
                        for key in METRICS}
        baseline = original_summary[model]["summary"]
        summary[model] = {
            "pooled_speaker_metrics": pooled, "fold_speaker_metrics": folds,
            "fold_speaker_mean_sample_std": fold_summary,
            "original_signed_utterance_fold_summary": baseline,
            "speaker_minus_utterance_fold_mean_accuracy_percentage_points":
                100 * (pooled["accuracy"] - baseline["accuracy"]["mean"]),
            "speaker_minus_utterance_fold_mean_macro_f1_percentage_points":
                100 * (pooled["macro_f1"] - baseline["macro_f1"]["mean"]),
            "incorrect_speakers": [r["speaker_id"] for r in rows if not r["correct"]],
            "achieved_90_percent": pooled["accuracy"] >= 0.9,
            "exceeded_90_percent": pooled["accuracy"] > 0.9,
        }
    return speaker_rows, summary, fold_csv


def plot(summary):
    os.environ["MPLCONFIGDIR"] = str(output_path(".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    for ax, model in zip(axes, MODELS):
        m = summary[model]["pooled_speaker_metrics"]
        matrix = np.array(m["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=8)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=25,
                        color="white" if matrix[i, j] >= 5 else "#143955")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Control (0)", "Dysarthric (1)"],
               yticklabels=["Control (0)", "Dysarthric (1)"], xlabel="Predicted speaker label",
               ylabel="True speaker label", title=f"{model}: {m['correct']}/15 correct ({100*m['accuracy']:.2f}%)")
    fig.suptitle(TITLE + "\n15 unique speakers per model; fixed threshold 0.5", fontsize=12)
    fig.savefig(output_path("confusion_matrix.png"), dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10.8, 5.7), constrained_layout=True)
    positions, width = np.arange(2), 0.34
    baseline = [100 * summary[m]["original_signed_utterance_fold_summary"]["accuracy"]["mean"] for m in MODELS]
    stds = [100 * summary[m]["original_signed_utterance_fold_summary"]["accuracy"]["sample_std"] for m in MODELS]
    speaker = [100 * summary[m]["pooled_speaker_metrics"]["accuracy"] for m in MODELS]
    ci = [summary[m]["pooled_speaker_metrics"]["confidence_intervals_95"]["accuracy"] for m in MODELS]
    intervals = np.array([[v - 100*c["low"] for v, c in zip(speaker, ci)],
                          [100*c["high"] - v for v, c in zip(speaker, ci)]])
    bars1 = ax.bar(positions - width/2, baseline, width, yerr=stds, capsize=5,
                   color="#899ba8", label="Original: utterance accuracy, fold mean +/- sample SD")
    bars2 = ax.bar(positions + width/2, speaker, width, yerr=intervals, capsize=5,
                   color="#196b86", label="Exploratory: pooled speaker accuracy, 95% exact interval")
    ax.bar_label(bars1, labels=[f"{v:.2f}%" for v in baseline], label_type="center", color="white", fontsize=12)
    ax.bar_label(bars2, labels=[f"{v:.2f}%\n11/15" for v in speaker], label_type="center", color="white", fontsize=12)
    ax.axhline(90, color="#aa3c42", linestyle="--", label="90% target")
    ax.set(xticks=positions, xticklabels=["B2: Recurrence Plot CNN", "C2: Full Hybrid Cross-Attention"],
           ylabel="Accuracy (%)", ylim=(0, 105))
    ax.set_title(TITLE + "\nDifferent evaluation units; uncertainty bars use different methods", fontsize=12)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    fig.savefig(output_path("comparison_chart.png"), dpi=180)
    plt.close(fig)


def pct(value):
    return f"{100 * value:.2f}%"


def report(rows, summary, preserved):
    lines = ["# " + TITLE, "", "Date: 2026-10-11 (Asia/Calcutta).", "",
             "Both B2 and C2 classify **11/15 speakers correctly (73.33%)**, with **4/15 incorrect**. "
             "Neither achieves or exceeds 90%. At least 14/15 (93.33%) would be needed to reach 90% with 15 speakers. "
             "They tie on accuracy and macro-F1; their errors and class recalls differ.", "",
             "## Inputs and provenance", "",
             f"Original read-only checkout: `{ORIGINAL}`.", "",
             f"Writable checkout: `{STUDY}`, branch `experiment/dysarthria-accuracy-improvement`, baseline `{BASELINE}`.", "",
             f"Frozen protocol: `{PROTOCOL}`. Frozen code: `{CODE}`.", "",
             "Six NATIVE / OUTER_EVALUATED results are checksum-bound to the final Module5 audit. "
             "Each model covers the same 8,673 signed CORE logical records and 15 unique speakers "
             "(8 dysarthric, 7 control). Per-fold recording counts: 2,819 / 2,873 / 2,981; five speakers each. "
             "Ordered record IDs, speaker IDs, true labels, CORE membership, test/development disjointness, "
             "split and manifest hashes, checkpoints and original utterance metrics were independently verified.", "",
             "`PROJECT_ARCHITECTURE.md` does not exist in this checkout. `PROJECT_SCOPE_LOCK.md`, "
             "`ARCHITECTURE_RATIONALE.md`, `MODULE_TRACKER.md`, and the Module5 final handoff/report were read. "
             "Module6 is currently signed off; its earlier NOT STARTED status in the historical Module5 handoff is historical.", "",
             "Original prediction entries contain `logical_id`, `speaker_id`, `label`, and `probability`; "
             "`label` is the true label, verified against the manifest. Per-record predicted labels are "
             "deterministically derived with the original `probability >= 0.5` rule, explicitly marked in "
             "`record_predictions.csv`; they were not separately stored. No probability, identity or label was imputed.", "",
             "## Fixed aggregation", "",
             "For each model independently, take the arithmetic mean of every existing outer evaluation "
             "recording probability for each speaker, across all that speaker's sessions and prompts. "
             "Predict dysarthric when the mean is >=0.5, otherwise control. Each recording has equal weight "
             "within its speaker; each speaker contributes exactly one decision to that model's pooled assessment. "
             "There are 30 CSV rows because two models each assess the same 15 speakers. "
             "No threshold search, majority-vote alternative, filtering, new inference or model selection was performed. "
             "All available predictions in the frozen CORE evaluation are included; records outside its original "
             "eligibility policy have no predictions and were not reconstructed.", "",
             "## Pooled speaker metrics", "",
             "Positive class = dysarthric (1); negative class = control (0). Precision and sensitivity refer to dysarthria.", "",
             "| Metric | B2 | C2 |", "|---|---:|---:|"]
    for key in METRICS:
        values = [summary[m]["pooled_speaker_metrics"][key] for m in MODELS]
        lines.append(f"| {key} | {pct(values[0])} | {pct(values[1])} |")
    lines += ["| Correct / incorrect | 11 / 4 | 11 / 4 |", "",
              "Confusion matrices use rows = true [control, dysarthric], columns = predicted [control, dysarthric]; "
              "the order is `[[TN, FP], [FN, TP]]`.", ""]
    for model in MODELS:
        m = summary[model]["pooled_speaker_metrics"]
        lines += [f"- **{model}**: `{m['confusion_matrix']}`. Incorrect speakers: "
                  + ", ".join(summary[model]["incorrect_speakers"]) + "."]
    lines += ["", "## Uncertainty", "",
              "95% intervals below are descriptive and conditional on the retained models and speakers. "
              "Accuracy, precision, sensitivity and specificity use Clopper-Pearson exact binomial intervals; "
              "balanced accuracy, macro-F1 and AUC use 20,000 class-stratified speaker bootstrap resamples "
              f"with seed {SEED}. Bootstrap sampling occurs over speaker means, retaining 8 dysarthric and "
              "7 control speakers per resample; it does not treat 8,673 correlated recordings as independent. "
              "The class-stratified bootstrap conditions on the observed class composition. "
              "Binomial intervals assume independent Bernoulli units; overlapping training folds and prior "
              "development exposure weaken population coverage claims. These intervals exclude model refitting, "
              "training randomness, fold design and post-hoc selection uncertainty. A wide interval containing "
              "90% is not a measured 90% result.", "",
              f"Method reference: [SciPy exact binomial interval documentation]({SCIENCE_DOC}).", "",
              "| Metric, 95% interval | B2 | C2 |", "|---|---:|---:|"]
    for key in METRICS:
        cells = []
        for model in MODELS:
            ci = summary[model]["pooled_speaker_metrics"]["confidence_intervals_95"][key]
            cells.append(f"{pct(ci['low'])} to {pct(ci['high'])}")
        lines.append(f"| {key} | {' | '.join(cells)} |")
    lines += ["", "## Fold speaker results", "",
              "Each speaker appears in exactly one test fold. Fold metrics assess five speakers, "
              "so accuracy changes in steps of 20 percentage points. All folds have both true classes; "
              "fold AUC is therefore defined. Full metrics, confusion matrices and binomial intervals "
              "are in `speaker_metrics.json`; `fold_speaker_metrics.csv` provides the flat metrics.", "",
              "| Model | Fold | Correct | Accuracy | Balanced accuracy | Macro-F1 | Precision | Sensitivity | Specificity | AUC | [[TN,FP],[FN,TP]] |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for model in MODELS:
        for fold, fm in summary[model]["fold_speaker_metrics"].items():
            lines.append(f"| {model} | {fold} | {fm['correct']}/5 | " + " | ".join(pct(fm[k]) for k in METRICS) + f" | {fm['confusion_matrix']} |")
    lines += ["", "Fold speaker accuracy mean +/- sample SD (ddof=1): "
              + "; ".join(f"{model} {pct(summary[model]['fold_speaker_mean_sample_std']['accuracy']['mean'])} +/- "
                          f"{pct(summary[model]['fold_speaker_mean_sample_std']['accuracy']['sample_std_ddof1'])}" for model in MODELS)
              + ". These happen to equal the pooled speaker accuracies because every fold contains five speakers. "
              "Other mean fold metrics need not equal pooled metrics.", "",
              "## Exact per-speaker results", "",
              "Values below are displayed to 12 decimal places; full numeric precision is retained in the CSV. "
              "The mean uses all listed recordings. C = control, D = dysarthric.", "",
              "| Speaker | Fold | True | Recordings | B2 mean probability | B2 prediction | C2 mean probability | C2 prediction |",
              "|---|---:|---|---:|---:|---|---:|---|"]
    by = {(r["model_variant"], r["speaker_id"]): r for r in rows}
    for speaker in sorted({r["speaker_id"] for r in rows}):
        b, c = by["B2", speaker], by["C2", speaker]
        lines.append(f"| {speaker} | {b['outer_fold']} | {'D' if b['true_label'] else 'C'} | {b['recording_count']} | "
                     f"{b['mean_dysarthria_probability']:.12f} | {'D' if b['predicted_label'] else 'C'} "
                     f"({'correct' if b['correct'] else 'incorrect'}) | {c['mean_dysarthria_probability']:.12f} | "
                     f"{'D' if c['predicted_label'] else 'C'} ({'correct' if c['correct'] else 'incorrect'}) |")
    lines += ["", "## Comparison with signed-off utterance results", "",
              "Original values are equal-weight means +/- sample SD over three folds of utterance metrics. "
              "New values are pooled metrics over 15 speaker decisions. The numerical differences below "
              "change the assessment unit and weighting; they do not show an improvement on the original endpoint.", "",
              "| Model | Original utterance accuracy | Speaker accuracy | Numerical difference | Original utterance macro-F1 | Speaker macro-F1 |",
              "|---|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        s = summary[model]
        a, f = s["original_signed_utterance_fold_summary"]["accuracy"], s["original_signed_utterance_fold_summary"]["macro_f1"]
        m = s["pooled_speaker_metrics"]
        lines.append(f"| {model} | {pct(a['mean'])} +/- {pct(a['sample_std'])} | {pct(m['accuracy'])} | "
                     f"{s['speaker_minus_utterance_fold_mean_accuracy_percentage_points']:+.2f} pp | "
                     f"{pct(f['mean'])} +/- {pct(f['sample_std'])} | {pct(m['macro_f1'])} |")
    lines += ["", "## What explains the observed results", "",
              "Averaging yields one decision per speaker and gives a speaker with 121 recordings the same "
              "pooled assessment weight as one with 1,237 recordings. A speaker may be counted correctly "
              "even with many individually incorrect recordings if the mean probability stays on the correct "
              "side of 0.5. This changes weighting and can reduce recording-level variation, but cannot "
              "remove a speaker's systematic score bias. The observed score patterns explain the fixed-rule "
              "decisions; they do not establish why the models learned those patterns.", "",
              "Both models miss F03 and F04 (dysarthric means below 0.5) and misclassify MC04 as dysarthric "
              "(mean above 0.5). B2 additionally misses M03, while C2 correctly classifies M03 but "
              "misclassifies FC03. Consequently both have four errors. B2 has higher specificity and "
              "precision; C2 has higher sensitivity. Their balanced accuracy and AUC should be read "
              "alongside this tied accuracy; no overall superiority is established.", "",
              "B2's F04 and M03 means are about 0.4921 and 0.4919, close to 0.5. "
              "They remain errors under the predeclared rule. No post-hoc threshold adjustment is justified by these test labels.", "",
              "## Limitations and scope", "",
              "- Post-hoc exploratory analysis of the same outer predictions used in the original study; "
              "no new independent cohort and no confirmatory speaker endpoint.",
              "- Only 15 independent speaker identities, three folds and one training seed. "
              "One changed decision moves pooled accuracy by 6.67 percentage points.",
              "- Each fold uses a different frozen model. Pooled speaker AUC compares scores across "
              "models without a common calibration; fold AUC is included for context. Averaged scores "
              "are not claimed to be calibrated probabilities of a speaker's condition.",
              "- Module4 development previously exposed Fold2/3 speakers. Within each frozen evaluation "
              "cell test speakers are disjoint from that cell's development speakers, but the wider study "
              "is not pristine external validation.",
              "- Unequal recording counts, prompts and sessions per speaker affect the stability of means. "
              "No recordings were removed; one primary logical record avoids head/array copy duplication.",
              "- B2 and C2 use the original C2-selected settings; comparing these two requested models "
              "does not authorize architecture, deployment-checkpoint or threshold reselection.",
              "- Fold1/A1 remains RECOVERED_PROTOCOL_EXCEPTION with prior native forwards UNKNOWN. "
              "It is outside these six NATIVE B2/C2 inputs; its original scientific disclosure is preserved.",
              "- No clinical diagnostic, causal, state-of-the-art or guaranteed-performance inference follows.", "",
              "## Preservation and validation", "",
              f"Preservation: **{preserved['status']}**. Original project inventory, content hashes for "
              f"{preserved['original_hashed_files']} files, raw/processed metadata for "
              f"{preserved['raw_processed_metadata_files']} files, tracked experimental files, both branch "
              "refs and original dirty/untracked status are unchanged. The pre-existing modified "
              "`artifacts/metrics/module2_representation_summary.json` was preserved. "
              "Raw audio and processed tensors were not opened or rehashed; this is a metadata comparison "
              "for those corpora, not a new full corpus integrity audit. Virtual environments, tool caches "
              "and Git internals are excluded from file inventories; branch refs and status are checked separately.", "",
              f"All {preserved['signed_input_hashes_verified']} signed closure input hashes and eight closure "
              "package files match. Original utterance metrics were independently reproduced with scikit-learn; "
              "threshold metrics and AUC were cross-checked by arithmetic. Exact record equality, class labels, "
              "uniqueness, aggregation counts and output checksums are validated. "
              "No Modules0-6 edit, training, model forward, checkpoint modification, label/split/prediction "
              "change, sign-off change, commit, push or release-branch change occurred. "
              "No original model or project Python module was imported.", "",
              "See `PRESERVATION_REPORT.md`, `preservation_before.json`, `preservation_after.json`, "
              "`source_provenance.json`, and `output_manifest.json` for auditable evidence.", "",
              "**STOP: analysis complete. Further experiments await user review.**", ""]
    return "\n".join(lines)


def verify():
    manifest = read(output_path("output_manifest.json"))
    for name, expected in manifest["files"].items():
        require(sha(output_path(name)) == expected["sha256"], f"Output hash mismatch: {name}")
    before, after = read(output_path("preservation_before.json")), read(output_path("preservation_after.json"))
    require(before["original_files"] == after["original_files"], "Original inventory changed")
    require(before["original_git"] == after["original_git"], "Original Git state changed")
    require(before["study_tracked_files"] == after["study_tracked_files"], "Experimental tracked file changed")
    require(before["study_git"]["head"] == after["study_git"]["head"] == BASELINE, "Study commit changed")
    with output_path("speaker_predictions.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    data = read(output_path("speaker_metrics.json"))
    for model in MODELS:
        selected = [r for r in rows if r["model_variant"] == model]
        require(len(selected) == len({r["speaker_id"] for r in selected}) == 15, "Speaker CSV count differs")
        recomputed = metrics([int(r["true_label"]) for r in selected],
                             [float(r["mean_dysarthria_probability"]) for r in selected])
        saved = data["models"][model]["pooled_speaker_metrics"]
        for key in (*METRICS, "confusion_matrix", "correct", "incorrect"):
            require(recomputed[key] == saved[key], f"CSV/JSON mismatch: {model} {key}")
    print(json.dumps({"verification": "PASS", "output_files_verified": len(manifest["files"]),
                      "speakers_per_model": 15, "no_model_imports_or_forwards": True}))


def main():
    require(STUDY.name == "VaaniSwitch_Accuracy_Study" and ORIGINAL.is_dir(), "Wrong workspace")
    require(OUT.is_relative_to(STUDY) and not OUT.is_relative_to(ORIGINAL), "Unsafe output root")
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--resume", action="store_true",
                      help="Resume from the original retained preservation snapshot after an analysis-only failure")
    args = parser.parse_args()
    if args.verify:
        verify()
        return
    require(not output_path("speaker_metrics.json").exists(), "Existing analysis: use --verify; no overwrite")
    original_git, study_git = git_state(ORIGINAL), git_state(STUDY)
    require(original_git["head"] == study_git["head"] == BASELINE, "Baseline commit differs")
    require(original_git["branch"] == "release/vaaniswitch-final-m0-m6", "Original release branch differs")
    require(study_git["branch"] == "experiment/dysarthria-accuracy-improvement", "Experimental branch differs")
    if args.resume:
        print("Reusing the immutable before-analysis preservation snapshot...", flush=True)
        before = read(output_path("preservation_before.json"))
        before_files, study_files = before["original_files"], before["study_tracked_files"]
        require(original_git == before["original_git"], "Original Git state changed since snapshot")
        require(tracked_inventory(STUDY) == study_files, "Study tracked files changed since snapshot")
        study_git = before["study_git"]
    else:
        require(not output_path("preservation_before.json").exists(), "Existing snapshot: use --resume")
        print("Taking read-only original project inventory and hashes...", flush=True)
        before_files, study_files = inventory(ORIGINAL), tracked_inventory(STUDY)
        before = {"original_files": before_files, "original_git": original_git,
                  "study_tracked_files": study_files, "study_git": study_git,
                  "scope_exclusions": sorted(EXCLUDED),
                  "raw_processed_policy": "path, size and mtime_ns only; no content read or full integrity claim"}
        write_json("preservation_before.json", before)
    print("Verifying frozen provenance and computing speaker metrics...", flush=True)
    source_rows, provenance, evidence, closure = validate_sources(before_files)
    rows, summary, folds = aggregate(source_rows)
    timestamp = datetime.now(ZoneInfo("Asia/Calcutta")).isoformat()
    data = {
        "analysis_label": TITLE, "analysis_date": "2026-10-11", "generated_at": timestamp,
        "original_project": str(ORIGINAL), "experimental_worktree": str(STUDY),
        "baseline_commit": BASELINE, "protocol_hash": PROTOCOL, "frozen_code_hash": CODE,
        "signed_closure_manifest_sha256": ANCHOR,
        "rule": "Within each model and speaker: arithmetic mean of every saved outer CORE probability; predict 1 iff mean >=0.5; one decision per speaker",
        "class_mapping": {"0": "control", "1": "dysarthric"},
        "confusion_matrix_order": "rows true [0,1], columns predicted [0,1]; [[TN,FP],[FN,TP]]",
        "unique_speakers_per_model": 15, "class_counts": {"control": 7, "dysarthric": 8},
        "records_per_model": 8673, "outer_folds": 3,
        "missing_probabilities": 0, "unreliable_speaker_ids": 0, "excluded_saved_predictions": 0,
        "threshold_optimized": False, "new_model_forwards": 0, "training_runs": 0,
        "comparison": "B2 and C2 tie on accuracy and macro-F1; neither achieves 90%",
        "minimum_correct_for_90_percent_with_15_speakers": 14,
        "uncertainty_limitations": "Descriptive conditional intervals; independence/representativeness uncertain. No model-refitting, fold-design, prior-exposure or post-hoc selection uncertainty captured. Bootstrap retains observed class composition. Pooled AUC combines uncalibrated fold models.",
        "original_study_limitations": closure["limitations"], "models": summary,
        "source_predictions": provenance,
        "runtime_versions": {"python": sys.version, **{name: importlib.metadata.version(name)
                              for name in ("numpy", "scipy", "scikit-learn", "matplotlib")}},
    }
    write_json("speaker_metrics.json", data)
    write_csv("speaker_predictions.csv", rows)
    write_csv("record_predictions.csv", source_rows)
    write_csv("fold_speaker_metrics.csv", folds)
    write_json("source_provenance.json", {"analysis_label": TITLE, "sources": provenance,
               "signed_closure_manifest_sha256": ANCHOR,
               "verified_signed_input_hashes": len(evidence["input_evidence"]),
               "verified_closure_package_files": len(evidence["files"]),
               "manifest_sha256": sha(ORIGINAL / "data/manifests/torgo_manifest.jsonl"),
               "folds_file_sha256": sha(ORIGINAL / "data/splits/speaker_folds.json"),
               "folds_canonical_json_fingerprint": folds_fingerprint(read(ORIGINAL / "data/splits/speaker_folds.json"))})
    plot(summary)
    print("Checking original project preservation after analysis...", flush=True)
    after_files, after_original_git = inventory(ORIGINAL), git_state(ORIGINAL)
    after_study_files, after_study_git = tracked_inventory(STUDY), git_state(STUDY)
    require(after_files == before_files, "Original project changed during analysis")
    require(after_original_git == original_git, "Original Git state changed")
    require(after_study_files == study_files, "Experimental tracked files changed")
    for key in ("head", "branch", "release_ref", "experiment_ref"):
        require(after_study_git[key] == study_git[key], f"Experimental Git {key} changed")
    write_json("preservation_after.json", {"original_files": after_files, "original_git": after_original_git,
               "study_tracked_files": after_study_files, "study_git": after_study_git})
    counts = Counter("content_hash" if "sha256" in v else "raw_processed_metadata" for v in before_files.values())
    preserved = {"status": "PASS", "original_hashed_files": counts["content_hash"],
                 "raw_processed_metadata_files": counts["raw_processed_metadata"],
                 "original_inventory_files": len(before_files),
                 "experimental_tracked_files": len(study_files),
                 "signed_input_hashes_verified": len(evidence["input_evidence"]),
                 "original_git_status_preserved": True, "both_branch_refs_preserved": True,
                 "modules_0_to_6_unchanged": True, "checkpoints_unchanged": True,
                 "raw_processed_content_rehashed": False, "new_model_forwards": 0,
                 "training_runs": 0, "commits": 0, "pushes": 0}
    write_json("preservation_summary.json", preserved)
    write_text("ANALYSIS_REPORT.md", report(rows, summary, preserved))
    write_text("PRESERVATION_REPORT.md", f"# {TITLE} — preservation\n\n"
               f"**PASS.** All new files are confined to `{OUT}`.\n\n"
               f"- Original project: {len(before_files):,} inventoried files unchanged before/after.\n"
               f"- {counts['content_hash']:,} project/evidence files: SHA-256, size and modification time unchanged. "
               "This includes all inventoried source, tests, configs, documents, checkpoints, ledgers, "
               "predictions and sign-off artifacts.\n"
               f"- {counts['raw_processed_metadata']:,} raw/processed files: path, size and modification time unchanged. "
               "Contents were not opened or rehashed; this does not assert a new full corpus integrity audit.\n"
               f"- All {len(evidence['input_evidence'])} signed Module5 closure input hashes and "
               f"{len(evidence['files'])} package files match their pinned hashes; the manifest itself "
               f"matches `{ANCHOR}`.\n"
               f"- All {len(study_files)} tracked files in the experimental checkout remain byte-for-byte unchanged.\n"
               f"- Both HEADs remain `{BASELINE}`; original release and experimental branch refs unchanged.\n"
               "- Original pre-existing modified/untracked status preserved, including "
               "`artifacts/metrics/module2_representation_summary.json`. No commit or push.\n"
               "- No original project Python imports, model loading/deserialization, forwards, training, "
               "fine-tuning, original metric replacement or Modules0-6 edits. Checkpoints were read only "
               "as bytes for hashing. No raw WAV or processed representation loaded.\n"
               "- Exclusions: virtual environments, Git internals, Python bytecode and tool caches. "
               "Git branch refs/commit/status checked separately using GIT_OPTIONAL_LOCKS=0. "
               "Matplotlib cache is under this analysis directory.\n\n"
               "Evidence: `preservation_before.json`, `preservation_after.json`, "
               "`preservation_summary.json`, `source_provenance.json`, `output_manifest.json`. "
               "The before/after original-file mappings are identical.\n\n"
               "No existing module defect was fixed, so original module tracker, changelog, defect logs and test "
               "reports were preserved. An analysis-only fold fingerprint check was corrected with a "
               "regression test; TEST_REPORT.md, ANALYSIS_TRACKER.md, CHANGELOG.md and DEFECT_LOG.md "
               "record that work inside this analysis directory.\n\n"
               "**STOP. Further experiments require user review.**\n")
    write_text("README.md", f"# {TITLE}\n\n"
               "B2 and C2 both achieve **73.33% (11/15 speakers correct)**. Both make four errors; "
               "neither reaches 90%. These exploratory results do not replace the signed utterance-level results.\n\n"
               "Start with [ANALYSIS_REPORT.md](ANALYSIS_REPORT.md).\n\n"
               "| File | Contents |\n|---|---|\n"
               "| speaker_metrics.json | All pooled/fold metrics, intervals, original comparisons and provenance |\n"
               "| speaker_predictions.csv | Exactly 15 rows per model, full precision speaker means and decisions |\n"
               "| record_predictions.csv | All 17,346 model-record rows; original IDs, labels, probabilities and derived decisions |\n"
               "| fold_speaker_metrics.csv | Five-speaker metrics for each of six model-fold cells |\n"
               "| confusion_matrix.png | Both complete speaker confusion matrices |\n"
               "| comparison_chart.png | Original utterance fold means vs pooled speaker accuracy, explicitly distinct intervals |\n"
               "| source_provenance.json | Six original native result paths and checksums, verified source identities |\n"
               "| PRESERVATION_REPORT.md | Verification scope, counts and limitations |\n"
               "| preservation_before.json / preservation_after.json | Original inventory/hashes and Git states |\n"
               "| preservation_summary.json | Machine-readable preservation verdict |\n"
               "| output_manifest.json | SHA-256 and byte size of published analysis files |\n"
               "| analyze_speakers.py | Standalone reproducible analysis; imports no project/model code |\n\n"
               "Class mapping: 0 = control, 1 = dysarthric. Matrix order: [[TN,FP],[FN,TP]]. "
               "Aggregation: all saved evaluation probabilities per speaker, arithmetic mean, >=0.5. "
               "All 8,673 frozen CORE records are retained per model.\n\n"
               "To verify the retained package from this experimental worktree, using the existing global Python:\n\n"
               "```powershell\npython -B .\\experiments\\speaker_level_analysis\\analyze_speakers.py --verify\n```\n\n"
               "For reproducibility, default execution generates a new package only when outputs do not exist; "
               "it refuses overwrites. It expects the unchanged original checkout and its frozen artifacts "
               "beside this experimental worktree. Exact library versions are in `speaker_metrics.json`.\n\n"
               "No project code, checkpoint, original prediction, sign-off or release branch changed. "
               "No training, model forwards, commit or push. **STOP: await user review.**\n")
    published = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "output_manifest.json")
    write_json("output_manifest.json", {"analysis_label": TITLE,
               "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in published},
               "self_excluded": True, "runtime_cache_excluded": ".matplotlib/"})
    verify()
    print(json.dumps({m: summary[m]["pooled_speaker_metrics"] for m in MODELS}, indent=2))


if __name__ == "__main__":
    main()
