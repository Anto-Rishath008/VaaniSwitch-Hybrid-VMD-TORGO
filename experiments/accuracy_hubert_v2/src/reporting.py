"""Fold arithmetic/sample SD, descriptive speaker aggregation and reproducible figures."""
from __future__ import annotations

import csv
import io
import json
import statistics

import numpy as np

from .common import ROOT, TITLE, atomic_bytes, digest, require, sha, write_json
from .training import metrics

METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "precision", "sensitivity", "specificity", "roc_auc")


def aggregate(fold_results):
    require(len(fold_results) == 3 and {r["outer_fold"] for r in fold_results} == {1, 2, 3},
            "Aggregation requires exactly the three original outer folds")
    output = {}
    for name in (*METRICS, "training_seconds", "parameter_count"):
        values = [r["metrics"][name] if name in METRICS else r[name] for r in fold_results]
        output[name] = {"mean": statistics.mean(values), "sample_std": statistics.stdev(values),
                        "fold_values": values} if all(v is not None for v in values) else {
                            "mean": None, "sample_std": None, "fold_values": values,
                            "reason": "At least one fold has undefined metric"}
    return output


def speaker_analysis(predictions):
    groups = {}
    for row in predictions:
        groups.setdefault(row["speaker_id"], []).append(row)
    rows = []
    for speaker, records in sorted(groups.items()):
        require(len({r["label"] for r in records}) == 1
                and len({r["outer_fold"] for r in records}) == 1, "Speaker spans test folds or labels")
        probability = statistics.mean(r["probability"] for r in records)
        rows.append({"model": records[0]["model"], "outer_fold": records[0]["outer_fold"],
                     "speaker_id": speaker, "label": records[0]["label"],
                     "probability": probability, "prediction": int(probability >= .5),
                     "recording_count": len(records), "unit": "speaker_descriptive"})
    m = metrics([r["label"] for r in rows], [r["probability"] for r in rows])
    return rows, {"metrics": m, "incorrect_speakers": [r["speaker_id"] for r in rows
                   if r["label"] != r["prediction"]],
                   "aggregation": "All held-out recording probabilities per speaker, arithmetic mean >= 0.5"}


def csv_bytes(rows):
    require(bool(rows), "Cannot publish empty result table")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def publish(path, value):
    if path.exists():
        require(path.read_bytes() == value, f"Refusing to overwrite historical output: {path}")
    else:
        atomic_bytes(path, value)


def figures(summary, folds, histories, output):
    import os
    os.environ["MPLCONFIGDIR"] = str(ROOT/"cache/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []

    def finish(figure, filename):
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        plt.close(figure)
        path = output/filename
        publish(path, buffer.getvalue())
        paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for model in ("E2", "E3"):
        for fold in (1, 2, 3):
            history = histories[f"fold{fold}_{model}"]
            axes[0].plot([r["epoch"] for r in history], [r["training_loss"] for r in history],
                         label=f"{model} fold {fold}")
            axes[1].plot([r["epoch"] for r in history], [r["validation_macro_f1"] for r in history],
                         label=f"{model} fold {fold}")
    axes[0].set(title="Selected inner candidate: training loss", xlabel="Epoch", ylabel="Weighted BCE")
    axes[1].set(title="Inner validation only", xlabel="Epoch", ylabel="Recording macro-F1")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    finish(fig, "training_curves.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    names = ["Original B2", "Original C2", "E1", "E2", "E3"]
    means = [summary["baselines"][m]["accuracy"]["mean"] for m in ("B2", "C2")]
    stds = [summary["baselines"][m]["accuracy"]["sample_std"] for m in ("B2", "C2")]
    means += [summary["models"][m]["fold_aggregation"]["accuracy"]["mean"] for m in ("E1", "E2", "E3")]
    stds += [summary["models"][m]["fold_aggregation"]["accuracy"]["sample_std"] for m in ("E1", "E2", "E3")]
    ax.bar(names, np.asarray(means)*100, yerr=np.asarray(stds)*100, capsize=4,
           color=["#888888", "#aaaaaa", "#3787a8", "#6174b5", "#397c56"])
    ax.set(ylabel="Recording accuracy (%)", ylim=(0, 110),
           title="Post-hoc exploratory study; fold mean ± sample SD")
    finish(fig, "accuracy_comparison.png")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for col, model in enumerate(("E1", "E2", "E3")):
        pooled = np.sum([f["metrics"]["confusion_matrix"] for f in folds if f["model"] == model], axis=0)
        speaker = summary["models"][model]["speaker_descriptive"]["metrics"]["confusion_matrix"]
        for row, (matrix, title) in enumerate(((pooled, "Pooled held-out recordings"),
                                               (speaker, "Descriptive speakers"))):
            ax = axes[row, col]
            ax.imshow(matrix, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(matrix[i][j]), ha="center", va="center")
            ax.set(title=f"{model}: {title}", xlabel="Predicted", ylabel="True",
                   xticks=[0, 1], yticks=[0, 1], xticklabels=["Control", "Dysarthric"],
                   yticklabels=["Control", "Dysarthric"])
    fig.tight_layout()
    finish(fig, "confusion_matrix.png")
    return paths


def generate_report(corpus, results, selections, frozen, baseline):
    rows, fold_rows, speakers, histories = [], [], [], {}
    summary = {"study_label": TITLE, "dataset_provenance": corpus.audit(),
               "frozen_decisions": frozen, "baselines": baseline, "models": {},
               "limitations": ["Post-hoc research direction informed by original outer results",
                   "Only 15 speakers; overlapping development sets and one fixed seed",
                   "One two-speaker inner validation split per outer fold",
                   "Centered eight-second HuBERT segment; RP summarizes complete VMD reconstruction",
                   "Pretrained LibriSpeech speech representations may retain speaker/channel cues",
                   "Seeded E3 CUDA training has nondeterministic native adaptive-pooling backward; evaluation is fixed",
                   "No pristine independent benchmark, clinical reliability or SOTA claim"]}
    for model in ("E1", "E2", "E3"):
        cells = [r for r in results if r["model"] == model]
        require(len(cells) == 3, "Incomplete model folds")
        predictions = [p for cell in cells for p in cell["predictions"]]
        require(len(predictions) == len(corpus.by_id)
                and {p["logical_id"] for p in predictions} == set(corpus.by_id),
                "Outer predictions do not cover exact CORE")
        speaker_rows, descriptive = speaker_analysis(predictions)
        require(len(speaker_rows) == len(corpus.mapping), "Speaker aggregation lost a speaker")
        require(all(p["speaker_id"] == corpus.by_id[p["logical_id"]].speaker_id
                    and p["label"] == corpus.by_id[p["logical_id"]].label
                    and p["prediction"] == int(p["probability"] >= .5) for p in predictions),
                "Report prediction identity/label/threshold mismatch")
        speakers += speaker_rows
        rows += predictions
        stats = aggregate(cells)
        comparison = {m: {"accuracy_difference_pp": 100*(stats["accuracy"]["mean"]-baseline[m]["accuracy"]["mean"]),
                          "macro_f1_difference_pp": 100*(stats["macro_f1"]["mean"]-baseline[m]["macro_f1"]["mean"])}
                      for m in ("B2", "C2")}
        summary["models"][model] = {"fold_aggregation": stats, "folds": cells,
                                    "speaker_descriptive": descriptive, "comparison": comparison}
        for cell in cells:
            fold_rows.append({"model": model, "outer_fold": cell["outer_fold"],
                              **{k: cell["metrics"][k] for k in METRICS},
                              "confusion_matrix": json.dumps(cell["metrics"]["confusion_matrix"]),
                              "training_seconds": cell["training_seconds"], "parameter_count": cell["parameter_count"],
                              "layer": cell["layer"], "selected_epochs": cell["selected_epochs"]})
            histories[f"fold{cell['outer_fold']}_{model}"] = selections[f"fold{cell['outer_fold']}_{model}"]["winner"]["history"]
    lines = [f"# {TITLE}", "", "All results below are observed held-out results from this post-hoc study.",
             "The original outer results informed this research direction. This is not an independent benchmark.",
             "", "Positive class = dysarthric. Threshold = 0.5. Matrix rows/columns = control, dysarthric.",
             "Arithmetic mean and sample SD (ddof=1) weight the three folds equally.", "",
             "| Model | Recording accuracy mean ± SD | Macro-F1 mean ± SD | Descriptive speaker accuracy |",
             "|---|---:|---:|---:|"]
    for model, content in summary["models"].items():
        stats = content["fold_aggregation"]
        a, f = stats["accuracy"], stats["macro_f1"]
        sm = content["speaker_descriptive"]["metrics"]
        lines.append(f"| {model} | {a['mean']*100:.2f}% ± {a['sample_std']*100:.2f}% | "
                     f"{f['mean']*100:.2f}% ± {f['sample_std']*100:.2f}% | {sm['accuracy']*100:.2f}% |")
        lines += []
    lines += ["", "## Dataset, checkpoint and training provenance", "", "```json",
              json.dumps({"dataset": corpus.audit(), "frozen_decisions": frozen}, indent=2), "```", "",
              "Full fold-specific metrics, parameter counts, times, selected layers/epochs and baseline differences",
              f"are in final_results.json and fold_results.csv. All {len(rows):,} model-record predictions and {len(speakers)}",
              "model-speaker decisions are retained in the CSVs. Training curves use inner validation only.", "",
              "## Incorrectly classified speakers", ""]
    for model, content in summary["models"].items():
        lines.append(f"- {model}: {', '.join(content['speaker_descriptive']['incorrect_speakers']) or 'none'}")
    lines += ["", "## Limitations", ""] + [f"- {s}" for s in summary["limitations"]]
    lines += ["", "No direct recording-versus-speaker accuracy improvement claim is made.",
              "The original signed B2/C2 and earlier descriptive analysis remain unchanged.", "",
              "## Reproducibility", "", "From the experimental worktree:", "```powershell",
              ".\\experiments\\accuracy_hubert_v2\\run_all.ps1", "```", "",
              "The same command validates and reuses sealed artifacts. Consumed outer claims without valid",
              "sealed results stop for explicit review. Feature/model/code identities are immutable for this run.",
              "Checkpoint attribution: https://huggingface.co/facebook/hubert-base-ls960", ""]
    published = []
    for filename, value in [("final_results.json", (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()),
                            ("fold_results.csv", csv_bytes(fold_rows)), ("record_predictions.csv", csv_bytes(rows)),
                            ("speaker_predictions.csv", csv_bytes(speakers)),
                            ("FINAL_EXPERIMENT_REPORT.md", "\n".join(lines).encode())]:
        path = ROOT/"results"/filename
        publish(path, value)
        published.append(path)
    published += figures(summary, results, histories, ROOT/"plots")
    return summary, published
