# Module 5 final closure and Module 6 handoff

Date: 2026-10-07 (Asia/Calcutta).

**Module 5: COMPLETE / PASSED / SIGNED OFF (exception-aware).**
**Module 6: NOT STARTED; handoff prepared only.**

All 21 frozen outer cells are independently verified: 20 NATIVE and Fold1/A1
RECOVERED_PROTOCOL_EXCEPTION. Its original prior native-forward count remains UNKNOWN;
its original consumed claim, native state and absent native result remain intact.
The eight-entry remaining sequence is COMPLETE with no next target.
Frozen foldwise mean/sample-SD aggregation and C2 descriptive robustness are complete.
No integrity blocker remains. This sign-off does not assert uninterrupted native
once-only compliance for Fold1/A1.

## Scientific identity and evidence

- Protocol: 677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9.
- Frozen Modules0–5 code: 7696cad010914ec37f494d9f1a11a2e1761e46f353e61d97a07399f746a34ff8.
- Runtime: seed 42, torch 2.14.0+cpu, CPU threads 2, threshold 0.5.
- Final completion: acdafedcab014c98f61e747558daadb0805ce32c233a1c342fbe093972be98a7.
- Package: artifacts/module5/677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9/final_closure/.
- Native ledger: 30 runs, 20 OUTER_EVALUATED, one FINAL_TRAIN_COMPLETE
  (original Fold1/A1), nine INNER_COMPLETE.
- 21 unique consumed native claims; 20 completion events. The sole unfinished native
  claim is the documented Fold1/A1 exception; no unexpected unfinished claims.
- 168 native files, 173 matching ledger/CSV events, 189 recovery files and
  247 protected predecessor files preserved.
- All 21 checkpoint/selection/candidate/epoch identities, signed ordered record IDs,
  speakers, labels, probabilities, record hashes and threshold verified.
  Every confusion matrix and metric recomputed from retained predictions.
- The original Fold1 closure, all earlier completions and all eight linked completion
  JSONs match their pinned hashes. No retained native result changed.
- Full raw/cache corpora were not rescanned; existing signed audits and protected
  fingerprints were verified. Frozen source and all previously used helpers unchanged.

## Primary cross-fold results

Arithmetic mean ± sample SD (ddof=1), equal weight for each of the three folds.
Values are proportions. No utterance-pooled metric replaces these estimates.
A1* includes Fold1/A1 = RECOVERED_PROTOCOL_EXCEPTION.

| Ablation | Accuracy | Balanced accuracy | Macro-F1 | Precision | Sensitivity | Specificity | ROC-AUC |
|---|---|---|---|---|---|---|---|
| A0 | 0.6891 ± 0.0807 | 0.6554 ± 0.1065 | 0.6480 ± 0.0993 | 0.5137 ± 0.1232 | 0.5551 ± 0.1974 | 0.7557 ± 0.0601 | 0.7365 ± 0.1185 |
| A1* | 0.6821 ± 0.1293 | 0.6607 ± 0.1449 | 0.6517 ± 0.1395 | 0.5094 ± 0.1696 | 0.5977 ± 0.1954 | 0.7238 ± 0.1017 | 0.7352 ± 0.1411 |
| B1 | 0.3546 ± 0.0434 | 0.5099 ± 0.0049 | 0.3037 ± 0.0727 | 0.3279 ± 0.0165 | 0.9464 ± 0.0844 | 0.0734 ± 0.0943 | 0.5645 ± 0.0919 |
| B2 | 0.6972 ± 0.1607 | 0.6993 ± 0.1202 | 0.6783 ± 0.1473 | 0.5636 ± 0.1752 | 0.7092 ± 0.0471 | 0.6894 ± 0.2410 | 0.7881 ± 0.1206 |
| B3 | 0.5767 ± 0.1308 | 0.6182 ± 0.1268 | 0.5391 ± 0.1431 | 0.4110 ± 0.0810 | 0.7331 ± 0.3876 | 0.5033 ± 0.3027 | 0.7541 ± 0.1500 |
| C1 | 0.6787 ± 0.1150 | 0.6593 ± 0.1233 | 0.6502 ± 0.1188 | 0.5060 ± 0.1353 | 0.6047 ± 0.1518 | 0.7140 ± 0.0972 | 0.7197 ± 0.1401 |
| C2 | 0.6502 ± 0.0903 | 0.6517 ± 0.1299 | 0.6257 ± 0.1107 | 0.4601 ± 0.1187 | 0.6497 ± 0.2606 | 0.6536 ± 0.0632 | 0.7333 ± 0.1380 |

## Per-fold metrics and confusion matrices

Full seven-metric rows, candidates, epochs, trainable parameters and retained
training times are in cross_fold_metrics.csv and cross_fold_metrics.json.
Here each matrix is [[TN,FP],[FN,TP]], control=0 and dysarthric=1.

| Ablation | Fold | Provenance | Macro-F1 | ROC-AUC | TN | FP | FN | TP |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| A0 | 1 | NATIVE | 0.5334 | 0.6002 | 1360 | 524 | 613 | 322 |
| A0 | 2 | NATIVE | 0.7082 | 0.7950 | 1585 | 336 | 395 | 557 |
| A0 | 3 | NATIVE | 0.7025 | 0.8144 | 1490 | 579 | 241 | 671 |
| A1 | 1 | RECOVERED_PROTOCOL_EXCEPTION | 0.4937 | 0.5742 | 1163 | 721 | 587 | 348 |
| A1 | 2 | NATIVE | 0.7578 | 0.8375 | 1575 | 346 | 281 | 671 |
| A1 | 3 | NATIVE | 0.7036 | 0.7939 | 1519 | 550 | 259 | 653 |
| B1 | 1 | NATIVE | 0.3877 | 0.4769 | 343 | 1541 | 141 | 794 |
| B1 | 2 | NATIVE | 0.2639 | 0.6602 | 26 | 1895 | 0 | 952 |
| B1 | 3 | NATIVE | 0.2595 | 0.5565 | 51 | 2018 | 9 | 903 |
| B2 | 1 | NATIVE | 0.5158 | 0.6495 | 780 | 1104 | 259 | 676 |
| B2 | 2 | NATIVE | 0.8030 | 0.8700 | 1656 | 265 | 240 | 712 |
| B2 | 3 | NATIVE | 0.7162 | 0.8447 | 1639 | 430 | 313 | 599 |
| B3 | 1 | NATIVE | 0.5143 | 0.5830 | 1402 | 482 | 666 | 269 |
| B3 | 2 | NATIVE | 0.4100 | 0.8160 | 314 | 1607 | 6 | 946 |
| B3 | 3 | NATIVE | 0.6930 | 0.8633 | 1246 | 823 | 75 | 837 |
| C1 | 1 | NATIVE | 0.5132 | 0.5597 | 1135 | 749 | 531 | 404 |
| C1 | 2 | NATIVE | 0.7252 | 0.8203 | 1458 | 463 | 269 | 683 |
| C1 | 3 | NATIVE | 0.7121 | 0.7792 | 1615 | 454 | 306 | 606 |
| C2 | 1 | NATIVE | 0.5019 | 0.5740 | 1227 | 657 | 605 | 330 |
| C2 | 2 | NATIVE | 0.7152 | 0.8181 | 1379 | 542 | 233 | 719 |
| C2 | 3 | NATIVE | 0.6599 | 0.8077 | 1224 | 845 | 145 | 767 |

## Frozen robustness executed

Exact defining sources:

- src/vaaniswitch/m5_experiments/robustness.py, robustness() (line 12):
  dimensions at line 32; sample-count <30 flag at line 53.
- docs/modules/5_accuracy_robustness/MODULE.md, **Reporting and limitations**:
  C2 descriptive subgroups and separately labelled pooled summaries.
- tests/m5_experiments/test_workflow_outer_robustness.py,
  test_robustness_counts_single_class_and_no_selection_feedback.
- Aggregation: src/vaaniswitch/m5_experiments/aggregation.py, aggregate() (line 48),
  sample SD at line 71; config.py identity fixes unweighted three-fold ddof=1.

Called the unchanged frozen functions directly with verified retained results.
The frozen native loader expects 21 native OUTER_EVALUATED rows, so the external helper
adapts only the recovered A1 schema in memory, annotates its exception in JSON/CSV,
and passes 21 numerical reporting views to the unchanged aggregation function.
It never creates a native A1 result, edits its ledger row, or removes the disclosure.
This follows the explicitly authorized exceptional inclusion rule.

C2 robustness uses its three NATIVE prediction sets only: 8,673 distinct CORE records,
15 speakers, 70 observed groups (45 per-fold and 25 pooled-descriptive).
Dimensions: gender, primary microphone, word/sentence, duration <=8s/>8s,
padded/cropped/exact Mel handling, and speaker. Counts, class composition, confusion,
fixed-threshold metrics and null-with-reason AUC are retained.
There are no observed exact-eight-second records, so the unchanged implementation
emits padded/cropped groups only; no empty group's performance is fabricated.
Pooled results below are secondary descriptive summaries from different fold models,
not the primary cross-fold estimate or calibrated common scores.

- gender/female: n=3364, speakers=6, control/dysarthric=2367/997; macro-F1 0.4571, sensitivity 0.2949, specificity 0.6189.
- gender/male: n=5309, speakers=9, control/dysarthric=3507/1802; macro-F1 0.7252, sensitivity 0.8446, specificity 0.6744.
- primary_microphone/array: n=1170, speakers=9, control/dysarthric=864/306; macro-F1 0.3355, sensitivity 0.5621, specificity 0.2569.
- primary_microphone/head: n=7503, speakers=15, control/dysarthric=5010/2493; macro-F1 0.6779, sensitivity 0.6594, specificity 0.7202.
- duration/>8s: n=235, speakers=15, control/dysarthric=30/205; macro-F1 0.6867, sensitivity 0.9561, specificity 0.3667.
- duration/<=8s: n=8438, speakers=15, control/dysarthric=5844/2594; macro-F1 0.6187, sensitivity 0.6245, specificity 0.6535.

## Conservative interpretation

- RP-only B2 has the highest observed mean macro-F1 (0.6783) and ROC-AUC (0.7881),
  but its macro-F1 SD is 0.1473. Three folds establish no superiority or significance.
- A0 and A1 have similar mean macro-F1 (0.6480 and 0.6517). These data do not show a
  consistent VMD benefit; A1 includes the explicitly exceptional Fold1 result.
- Graph-only B1 predicts dysarthria very often: sensitivity 0.9464 versus specificity
  0.0734, balanced accuracy 0.5099. Its low accuracy SD is not evidence of a useful model.
- Graph+RP B3 is unstable: sensitivity SD 0.3876 and specificity SD 0.3027. It is worse
  than RP-only B2 in each fold's macro-F1. Mel/nonlinear C1 and C2 also remain below
  B2 in each fold. Nonlinear RP carries useful descriptive evidence; adding branches
  does not establish complementary predictive benefit under these frozen settings.
- Attention C2 does not improve macro-F1 over pooled-concatenation C1 in any fold:
  C2-minus-C1 = -0.011315, -0.010001, -0.052160; mean -0.024492.
  Its mean ROC-AUC is slightly higher (0.7333 vs 0.7197), but score ranking and
  classification at threshold 0.5 measure different behavior. No threshold tuning
  or primary-model reselection follows these results.
- C2 sensitivity varies strongly (SD 0.2606). Its speaker-level dysarthric recall
  ranges from F03 0.0564 to F01 1.0000; control specificity ranges from MC04 0.3051
  to FC01 0.9935. Speaker groups contain only one class: their AUC is undefined and
  their reported balanced accuracy is the frozen present-class convention.
- Secondary C2 subgroup gaps for gender and microphone deserve disclosure, but
  they are confounded with speaker, class composition and fold. They are not causal
  estimates of gender or microphone effects. The >8s/cropped group has 205 positive
  and only 30 control records, so its high accuracy does not establish crop robustness.
- Only 15 speakers, one seed and historical Module4 exposure to Fold2/3 development
  limit generalization claims. Ablations share C2-selected settings and differ in
  capacity. No clinical, state-of-the-art or guaranteed-performance claim is warranted.

## Validation and acceptance

27 focused final-closure tests PASS (1.87s retained run; initial run also passed).
The relevant full Module5 suite passes **830 tests in 28.74s**.
Ruff and helper CLI help pass. Synthetic tests exercise models in temporary fixtures;
the real closure audit/reporting blocks model construction/calls, training and native
Ledger opening. Production model forwards/training/native-scientific writes are ZERO.
No outer rerun, retraining, retuning, claim recreation, or source change occurred.

Focused coverage includes incomplete/duplicate cells, preserved exception disclosure,
unweighted sample SD, subgroup and null-AUC behavior, exact claim lifecycle,
checksum corruption, no-overwrite publication, changed evidence refusal, and
forbidden model/Ledger calls. No unresolved integrity blocker exists.
All final acceptance items are recorded in module5_final_closure.json.

## Artifact SHA-256 manifest

Paths below are relative to the immutable package directory above.

| File | SHA-256 |
|---|---|
| ablation_summary.csv | c561e2a755e56d9cda01abaebfa3caa975bba75f4874db60e61415db0bb6c467 |
| cross_fold_metrics.csv | 2caa53e973a01f0a41dc74618850107b20512fd8c7b75367156f939199587a91 |
| cross_fold_metrics.json | 41d1d316841c7779a8d11e5ab70ed58b1fb7a44a473585cbc271e9b981369d9f |
| evidence_manifest.json | f9d4e56917b2bfc63abbf312e2b389457aa2ad29734424f01da5ee03109a8a6d |
| final_outer_audit.json | d14a0eb4ce5d54e295b0e0c4866c040439aef0adeacdd02fd2b6aeb63eeb6331 |
| module5_final_closure.json | 449b1a8237efbb21e7912868d4a0586bdca30e85806248ca646b6d647f5683e2 |
| module6_handoff.json | 36f5a276137922fd585dfe052e2402218fe36a4b0569384e0c4bd4e241b2235f |
| robustness_summary.json | a0ac89fc49f9be536cfed686ddbbcd5be08c41e6b903911876071bf607c1499a |
| validation_checks.json | 6416d9963d3620a4e3d333ec75b0d9484d82b15387aa03986d34e529b03ec8e1 |

## Module 6 handoff — preparation only

The immutable module6_handoff.json binds this closure and cross-fold report hashes,
all limitations and the three retained C2 checkpoint identities.

Keep C2 as the predeclared primary architecture. Preserve each fold's pre-outer final
development checkpoint and its inner-selected epoch. Do not choose a deployment
checkpoint, ensemble, or new architecture from these outer scores. Module6's separate
ASR/TTS utilities do not require a classifier refit.

| Fold | Candidate | Epoch | Final C2 checkpoint SHA-256 |
|---:|---|---:|---|
| 1 | stronger_dropout | 1 | d32bc51dce71ab37afd02471c5cfe874ce03f737e8191acad7f31663c6fe4e24 |
| 2 | stronger_decay | 2 | cf54f92b2e06c6c544e4b9efb91d3e248f9207f9ce773fc510fd4f23b3649d8d |
| 3 | baseline | 2 | a0c0b3520ef6f04a11e29c32dd11a65fda363b3bbd8793c82f75f1e53ac37508 |

Next objective, from PROJECT_SCOPE_LOCK.md, FINAL_WORKFLOW.md and
docs/modules/6_english_asr_tts/MODULE.md: frozen English ASR, preserved raw transcript,
auxiliary WER/CER on authorized references, explicit optional correction, and standard
English TTS. No voice cloning or ASR research training. A separately authorized
Module6 session must read its local AGENTS/MODULE/TEST_PLAN first.

**STOP: Module 6 NOT STARTED. No Module6 code, inference, model download or execution.**
