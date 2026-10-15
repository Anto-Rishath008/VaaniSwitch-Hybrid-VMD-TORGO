# POST-HOC EXPLORATORY SPEAKER-LEVEL ANALYSIS

Date: 2026-10-11 (Asia/Calcutta).

Both B2 and C2 classify **11/15 speakers correctly (73.33%)**, with **4/15 incorrect**. Neither achieves or exceeds 90%. At least 14/15 (93.33%) would be needed to reach 90% with 15 speakers. They tie on accuracy and macro-F1; their errors and class recalls differ.

## Inputs and provenance

Original read-only checkout: `C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_FINAL_Hybrid_VMD_TORGO_Project`.

Writable checkout: `C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_Accuracy_Study`, branch `experiment/dysarthria-accuracy-improvement`, baseline `2dee81f7656e5b5cae800d9a3c8692e5e8107bf9`.

Frozen protocol: `677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9`. Frozen code: `7696cad010914ec37f494d9f1a11a2e1761e46f353e61d97a07399f746a34ff8`.

Six NATIVE / OUTER_EVALUATED results are checksum-bound to the final Module5 audit. Each model covers the same 8,673 signed CORE logical records and 15 unique speakers (8 dysarthric, 7 control). Per-fold recording counts: 2,819 / 2,873 / 2,981; five speakers each. Ordered record IDs, speaker IDs, true labels, CORE membership, test/development disjointness, split and manifest hashes, checkpoints and original utterance metrics were independently verified.

`PROJECT_ARCHITECTURE.md` does not exist in this checkout. `PROJECT_SCOPE_LOCK.md`, `ARCHITECTURE_RATIONALE.md`, `MODULE_TRACKER.md`, and the Module5 final handoff/report were read. Module6 is currently signed off; its earlier NOT STARTED status in the historical Module5 handoff is historical.

Original prediction entries contain `logical_id`, `speaker_id`, `label`, and `probability`; `label` is the true label, verified against the manifest. Per-record predicted labels are deterministically derived with the original `probability >= 0.5` rule, explicitly marked in `record_predictions.csv`; they were not separately stored. No probability, identity or label was imputed.

## Fixed aggregation

For each model independently, take the arithmetic mean of every existing outer evaluation recording probability for each speaker, across all that speaker's sessions and prompts. Predict dysarthric when the mean is >=0.5, otherwise control. Each recording has equal weight within its speaker; each speaker contributes exactly one decision to that model's pooled assessment. There are 30 CSV rows because two models each assess the same 15 speakers. No threshold search, majority-vote alternative, filtering, new inference or model selection was performed. All available predictions in the frozen CORE evaluation are included; records outside its original eligibility policy have no predictions and were not reconstructed.

## Pooled speaker metrics

Positive class = dysarthric (1); negative class = control (0). Precision and sensitivity refer to dysarthria.

| Metric | B2 | C2 |
|---|---:|---:|
| accuracy | 73.33% | 73.33% |
| balanced_accuracy | 74.11% | 73.21% |
| macro_f1 | 73.21% | 73.21% |
| precision | 83.33% | 75.00% |
| sensitivity | 62.50% | 75.00% |
| specificity | 85.71% | 71.43% |
| roc_auc | 94.64% | 85.71% |
| Correct / incorrect | 11 / 4 | 11 / 4 |

Confusion matrices use rows = true [control, dysarthric], columns = predicted [control, dysarthric]; the order is `[[TN, FP], [FN, TP]]`.

- **B2**: `[[6, 1], [3, 5]]`. Incorrect speakers: F03, F04, M03, MC04.
- **C2**: `[[5, 2], [2, 6]]`. Incorrect speakers: F03, F04, FC03, MC04.

## Uncertainty

95% intervals below are descriptive and conditional on the retained models and speakers. Accuracy, precision, sensitivity and specificity use Clopper-Pearson exact binomial intervals; balanced accuracy, macro-F1 and AUC use 20,000 class-stratified speaker bootstrap resamples with seed 20261011. Bootstrap sampling occurs over speaker means, retaining 8 dysarthric and 7 control speakers per resample; it does not treat 8,673 correlated recordings as independent. The class-stratified bootstrap conditions on the observed class composition. Binomial intervals assume independent Bernoulli units; overlapping training folds and prior development exposure weaken population coverage claims. These intervals exclude model refitting, training randomness, fold design and post-hoc selection uncertainty. A wide interval containing 90% is not a measured 90% result.

Method reference: [SciPy exact binomial interval documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html).

| Metric, 95% interval | B2 | C2 |
|---|---:|---:|
| accuracy | 44.90% to 92.21% | 44.90% to 92.21% |
| balanced_accuracy | 53.57% to 93.75% | 47.32% to 93.75% |
| macro_f1 | 49.76% to 93.33% | 46.43% to 93.33% |
| precision | 35.88% to 99.58% | 34.91% to 96.81% |
| sensitivity | 24.49% to 91.48% | 34.91% to 96.81% |
| specificity | 42.13% to 99.64% | 29.04% to 96.33% |
| roc_auc | 78.57% to 100.00% | 62.50% to 100.00% |

## Fold speaker results

Each speaker appears in exactly one test fold. Fold metrics assess five speakers, so accuracy changes in steps of 20 percentage points. All folds have both true classes; fold AUC is therefore defined. Full metrics, confusion matrices and binomial intervals are in `speaker_metrics.json`; `fold_speaker_metrics.csv` provides the flat metrics.

| Model | Fold | Correct | Accuracy | Balanced accuracy | Macro-F1 | Precision | Sensitivity | Specificity | AUC | [[TN,FP],[FN,TP]] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B2 | 1 | 3/5 | 60.00% | 58.33% | 58.33% | 50.00% | 50.00% | 66.67% | 83.33% | [[2, 1], [1, 1]] |
| B2 | 2 | 4/5 | 80.00% | 83.33% | 80.00% | 100.00% | 66.67% | 100.00% | 100.00% | [[2, 0], [1, 2]] |
| B2 | 3 | 4/5 | 80.00% | 83.33% | 80.00% | 100.00% | 66.67% | 100.00% | 100.00% | [[2, 0], [1, 2]] |
| C2 | 1 | 3/5 | 60.00% | 58.33% | 58.33% | 50.00% | 50.00% | 66.67% | 83.33% | [[2, 1], [1, 1]] |
| C2 | 2 | 4/5 | 80.00% | 83.33% | 80.00% | 100.00% | 66.67% | 100.00% | 100.00% | [[2, 0], [1, 2]] |
| C2 | 3 | 4/5 | 80.00% | 75.00% | 76.19% | 75.00% | 100.00% | 50.00% | 100.00% | [[1, 1], [0, 3]] |

Fold speaker accuracy mean +/- sample SD (ddof=1): B2 73.33% +/- 11.55%; C2 73.33% +/- 11.55%. These happen to equal the pooled speaker accuracies because every fold contains five speakers. Other mean fold metrics need not equal pooled metrics.

## Exact per-speaker results

Values below are displayed to 12 decimal places; full numeric precision is retained in the CSV. The mean uses all listed recordings. C = control, D = dysarthric.

| Speaker | Fold | True | Recordings | B2 mean probability | B2 prediction | C2 mean probability | C2 prediction |
|---|---:|---|---:|---:|---|---:|---|
| F01 | 3 | D | 121 | 0.666479285591 | D (correct) | 0.991541596484 | D (correct) |
| F03 | 1 | D | 550 | 0.460261871977 | C (incorrect) | 0.121120998024 | C (incorrect) |
| F04 | 2 | D | 326 | 0.492082473896 | C (incorrect) | 0.449274034714 | C (incorrect) |
| FC01 | 1 | C | 153 | 0.267757477908 | C (correct) | 0.023180091023 | C (correct) |
| FC02 | 2 | C | 1237 | 0.283773796228 | C (correct) | 0.336859794913 | C (correct) |
| FC03 | 3 | C | 977 | 0.410462921022 | C (correct) | 0.590825538516 | D (incorrect) |
| M01 | 2 | D | 300 | 0.710806463361 | D (correct) | 0.795608968996 | D (correct) |
| M02 | 2 | D | 326 | 0.664941589114 | D (correct) | 0.939063995208 | D (correct) |
| M03 | 3 | D | 410 | 0.491875147020 | C (incorrect) | 0.622746900593 | D (correct) |
| M04 | 1 | D | 385 | 0.734122308392 | D (correct) | 0.713305923140 | D (correct) |
| M05 | 3 | D | 381 | 0.789924011378 | D (correct) | 0.961162855619 | D (correct) |
| MC01 | 3 | C | 1092 | 0.371770522349 | C (correct) | 0.304663495262 | C (correct) |
| MC02 | 2 | C | 684 | 0.330460823953 | C (correct) | 0.331883308455 | C (correct) |
| MC03 | 1 | C | 882 | 0.333529974556 | C (correct) | 0.097846835458 | C (correct) |
| MC04 | 1 | C | 849 | 0.627866011657 | D (incorrect) | 0.648153304185 | D (incorrect) |

## Comparison with signed-off utterance results

Original values are equal-weight means +/- sample SD over three folds of utterance metrics. New values are pooled metrics over 15 speaker decisions. The numerical differences below change the assessment unit and weighting; they do not show an improvement on the original endpoint.

| Model | Original utterance accuracy | Speaker accuracy | Numerical difference | Original utterance macro-F1 | Speaker macro-F1 |
|---|---:|---:|---:|---:|---:|
| B2 | 69.72% +/- 16.07% | 73.33% | +3.62 pp | 67.83% +/- 14.73% | 73.21% |
| C2 | 65.02% +/- 9.03% | 73.33% | +8.32 pp | 62.57% +/- 11.07% | 73.21% |

## What explains the observed results

Averaging yields one decision per speaker and gives a speaker with 121 recordings the same pooled assessment weight as one with 1,237 recordings. A speaker may be counted correctly even with many individually incorrect recordings if the mean probability stays on the correct side of 0.5. This changes weighting and can reduce recording-level variation, but cannot remove a speaker's systematic score bias. The observed score patterns explain the fixed-rule decisions; they do not establish why the models learned those patterns.

Both models miss F03 and F04 (dysarthric means below 0.5) and misclassify MC04 as dysarthric (mean above 0.5). B2 additionally misses M03, while C2 correctly classifies M03 but misclassifies FC03. Consequently both have four errors. B2 has higher specificity and precision; C2 has higher sensitivity. Their balanced accuracy and AUC should be read alongside this tied accuracy; no overall superiority is established.

B2's F04 and M03 means are about 0.4921 and 0.4919, close to 0.5. They remain errors under the predeclared rule. No post-hoc threshold adjustment is justified by these test labels.

## Limitations and scope

- Post-hoc exploratory analysis of the same outer predictions used in the original study; no new independent cohort and no confirmatory speaker endpoint.
- Only 15 independent speaker identities, three folds and one training seed. One changed decision moves pooled accuracy by 6.67 percentage points.
- Each fold uses a different frozen model. Pooled speaker AUC compares scores across models without a common calibration; fold AUC is included for context. Averaged scores are not claimed to be calibrated probabilities of a speaker's condition.
- Module4 development previously exposed Fold2/3 speakers. Within each frozen evaluation cell test speakers are disjoint from that cell's development speakers, but the wider study is not pristine external validation.
- Unequal recording counts, prompts and sessions per speaker affect the stability of means. No recordings were removed; one primary logical record avoids head/array copy duplication.
- B2 and C2 use the original C2-selected settings; comparing these two requested models does not authorize architecture, deployment-checkpoint or threshold reselection.
- Fold1/A1 remains RECOVERED_PROTOCOL_EXCEPTION with prior native forwards UNKNOWN. It is outside these six NATIVE B2/C2 inputs; its original scientific disclosure is preserved.
- No clinical diagnostic, causal, state-of-the-art or guaranteed-performance inference follows.

## Preservation and validation

Preservation: **PASS**. Original project inventory, content hashes for 22175 files, raw/processed metadata for 169658 files, tracked experimental files, both branch refs and original dirty/untracked status are unchanged. The pre-existing modified `artifacts/metrics/module2_representation_summary.json` was preserved. Raw audio and processed tensors were not opened or rehashed; this is a metadata comparison for those corpora, not a new full corpus integrity audit. Virtual environments, tool caches and Git internals are excluded from file inventories; branch refs and status are checked separately.

All 636 signed closure input hashes and eight closure package files match. Original utterance metrics were independently reproduced with scikit-learn; threshold metrics and AUC were cross-checked by arithmetic. Exact record equality, class labels, uniqueness, aggregation counts and output checksums are validated. No Modules0-6 edit, training, model forward, checkpoint modification, label/split/prediction change, sign-off change, commit, push or release-branch change occurred. No original model or project Python module was imported.

See `PRESERVATION_REPORT.md`, `preservation_before.json`, `preservation_after.json`, `source_provenance.json`, and `output_manifest.json` for auditable evidence.

**STOP: analysis complete. Further experiments await user review.**
