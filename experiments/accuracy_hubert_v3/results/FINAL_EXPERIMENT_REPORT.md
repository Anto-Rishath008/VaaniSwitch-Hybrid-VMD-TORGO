# V3 POST-HOC EXPLORATORY STUDY

Evidence: REAL_SEALED_OUTER

Same recording-level endpoint; arithmetic outer-fold mean and sample SD (ddof=1).
Positive class: dysarthric. Fixed threshold: 0.5. No independent confirmation.

| Model | Recording accuracy mean ± SD | Macro-F1 mean ± SD | Descriptive speaker accuracy |
|---|---:|---:|---:|
| reference | 76.43% ± 5.82% | 74.09% ± 6.19% | 66.67% |
| selected | 76.70% ± 5.95% | 74.23% ± 6.44% | 73.33% |

Historical comparisons (same recording endpoint, no independent significance claim):

| Historical model | Accuracy mean ± SD | Macro-F1 mean ± SD |
|---|---:|---:|
| Original B2 | 69.72% ± 16.07% | 67.83% ± 14.73% |
| Original C2 | 65.02% ± 9.03% | 62.57% ± 11.07% |
| V2 E1 | 74.79% ± 8.17% | 72.66% ± 8.33% |
| V2 E2 | 74.16% ± 9.38% | 71.81% ± 9.66% |
| V2 E3 | 72.39% ± 10.22% | 70.27% ± 10.27% |

Full per-fold, per-speaker, class rates, calibration/reliability bins, microphone diagnostics,
between-speaker variation, runtimes, trainable parameter counts and all predictions are retained
in JSON/CSV. Missing class-specific speaker rates are null, not invented zeros.
ablation_results.csv contains development-only evidence; selected_configurations.json freezes choices.
training_curves.png explains that no iterative loss histories are available for sklearn LBFGS.

Original/V1/V2 outer results informed this direction; not an independent benchmark
15 speakers, overlapping development sets, single fixed seed, selection optimism
Channel differences are observational and confounded; not causal
Calibration diagnostics are separate and do not alter primary outer probabilities
No clinical, significance, SOTA, or guaranteed improvement claim
