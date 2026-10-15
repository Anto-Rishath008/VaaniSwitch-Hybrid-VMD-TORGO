"""Authoritative TORGO cross-validation metrics and evaluation data.

All values are verbatim signed records from Module 5 and Module 6 reports:
- docs/modules/5_accuracy_robustness/FINAL_CLOSURE_MODULE6_HANDOFF.md
- docs/modules/6_english_asr_tts/FINAL_EVALUATION_REPORT.md
- artifacts/metrics/module0_dataset_summary.json
"""

DATASET_SUMMARY = {
    "name": "TORGO Acoustic-Articulatory Dysarthria Corpus",
    "total_manifest_records": 9494,
    "core_clean_records": 8673,
    "speakers_total": 15,
    "dysarthric_speakers": ["F01", "F03", "F04", "M01", "M02", "M04", "M05"],
    "control_speakers": ["FC01", "FC02", "FC03", "MC01", "MC02", "MC03", "MC04"],
    "folds_count": 3,
    "speaker_overlap": 0,
    "sampling_rate_hz": 16000,
    "evaluation_protocol": "Speaker-Disjoint Stratified Group Cross-Validation",
}

# 7 Ablation experiment benchmark (Mean ± SD over 3 speaker-disjoint folds)
BENCHMARK_ABLATIONS = [
    {
        "ablation_id": "A0",
        "description": "Raw Speech Baseline (No VMD, standard Log-Mel Spectrogram)",
        "accuracy": "68.91% ± 8.07%",
        "balanced_acc": "65.54% ± 10.65%",
        "macro_f1": "64.80% ± 9.93%",
        "precision": "51.37% ± 12.32%",
        "sensitivity": "55.51% ± 19.74%",
        "specificity": "75.57% ± 6.01%",
        "roc_auc": "0.7365 ± 0.1185",
    },
    {
        "ablation_id": "A1*",
        "description": "Proposed Model (Shared VMD + Log-Mel + NVG + Recurrence + Token Cross-Attention)",
        "accuracy": "68.21% ± 12.93%",
        "balanced_acc": "66.07% ± 14.49%",
        "macro_f1": "65.17% ± 13.95%",
        "precision": "50.94% ± 16.96%",
        "sensitivity": "59.77% ± 19.54%",
        "specificity": "72.38% ± 10.17%",
        "roc_auc": "0.7352 ± 0.1411",
    },
    {
        "ablation_id": "B1",
        "description": "Ablation: Mel Spectrogram Only (Nonlinear tokens excluded)",
        "accuracy": "35.46% ± 4.34%",
        "balanced_acc": "50.99% ± 0.49%",
        "macro_f1": "30.37% ± 7.27%",
        "precision": "32.79% ± 1.65%",
        "sensitivity": "94.64% ± 8.44%",
        "specificity": "7.34% ± 9.43%",
        "roc_auc": "0.5645 ± 0.0919",
    },
    {
        "ablation_id": "B2",
        "description": "Ablation: Nonlinear Tokens Only (NVG Graph + Recurrence Plot, Mel excluded)",
        "accuracy": "69.72% ± 16.07%",
        "balanced_acc": "69.93% ± 12.02%",
        "macro_f1": "67.83% ± 14.73%",
        "precision": "56.36% ± 17.52%",
        "sensitivity": "70.92% ± 4.71%",
        "specificity": "68.94% ± 24.10%",
        "roc_auc": "0.7881 ± 0.1206",
    },
    {
        "ablation_id": "B3",
        "description": "Ablation: One-Vector Concatenation (Cross-attention mechanism disabled)",
        "accuracy": "57.67% ± 13.08%",
        "balanced_acc": "61.82% ± 12.68%",
        "macro_f1": "53.91% ± 14.31%",
        "precision": "41.10% ± 8.10%",
        "sensitivity": "73.31% ± 38.76%",
        "specificity": "50.33% ± 30.27%",
        "roc_auc": "0.7541 ± 0.1500",
    },
    {
        "ablation_id": "C1",
        "description": "Ablation: VMD Mode Deletion / Selective Mode Filtering",
        "accuracy": "67.87% ± 11.50%",
        "balanced_acc": "65.93% ± 12.33%",
        "macro_f1": "65.02% ± 11.88%",
        "precision": "50.60% ± 13.53%",
        "sensitivity": "60.47% ± 15.18%",
        "specificity": "71.40% ± 9.72%",
        "roc_auc": "0.7197 ± 0.1401",
    },
    {
        "ablation_id": "C2",
        "description": "Ablation: Duration Centering & Temporal Crop Robustness",
        "accuracy": "65.02% ± 9.03%",
        "balanced_acc": "65.17% ± 12.99%",
        "macro_f1": "62.57% ± 11.07%",
        "precision": "46.01% ± 11.87%",
        "sensitivity": "64.97% ± 26.06%",
        "specificity": "65.36% ± 6.32%",
        "roc_auc": "0.7333 ± 0.1380",
    },
]

# Per-fold confusion matrices for the proposed A1 architecture
PROPOSED_FOLD_RESULTS = [
    {
        "fold": 1,
        "test_speakers": "F03, FC01, M04, MC03, MC04",
        "macro_f1": 0.4937,
        "roc_auc": 0.5742,
        "tn": 1163,
        "fp": 721,
        "fn": 587,
        "tp": 348,
        "total": 2819,
    },
    {
        "fold": 2,
        "test_speakers": "F04, FC02, M01, M02, MC02",
        "macro_f1": 0.7578,
        "roc_auc": 0.8375,
        "tn": 1575,
        "fp": 346,
        "fn": 281,
        "tp": 671,
        "total": 2873,
    },
    {
        "fold": 3,
        "test_speakers": "F01, FC03, M03, M05, MC01",
        "macro_f1": 0.7036,
        "roc_auc": 0.7939,
        "tn": 1519,
        "fp": 550,
        "fn": 259,
        "tp": 653,
        "total": 2981,
    },
]

# Module 6 Speech Recognition Benchmark on 130-record frozen panel
ASR_RECONSTRUCTION_BENCHMARK = {
    "panel_size": 130,
    "speakers_count": 13,
    "overall_wer": "26.12%",
    "overall_cer": "18.68%",
    "subgroups": [
        {"group": "Control Speakers (60 records)", "wer": "6.71%", "cer": "6.17%"},
        {"group": "Dysarthric Speakers (70 records)", "wer": "44.95%", "cer": "30.60%"},
        {"group": "Continuous Sentences (65 records)", "wer": "22.22%", "cer": "16.73%"},
        {"group": "Isolated Words (65 records)", "wer": "58.46%", "cer": "34.66%"},
    ],
    "human_listening_confirmation": "PASS — User confirmed both M03 and MC02 synthesized samples produce clear, intelligible English speech.",
}
