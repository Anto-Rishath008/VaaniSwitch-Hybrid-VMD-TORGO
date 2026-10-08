# Phase 5B closure and Phase 5C manual handoff

Closure date: 2026-09-23.

- Phase 5A: COMPLETE / PASSED (retained; not rerun).
- Phase 5B: COMPLETE / PASSED.
- Phase 5C: NOT STARTED / READY FOR MANUAL EXECUTION.
- Module 5: MANUAL EXPERIMENTS IN PROGRESS / NOT SIGNED OFF.
- Module 6: NOT STARTED.

## Authoritative audit

[Read-only audit artifact](../../../artifacts/module5/677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9/phase5b_audit/20260923T180221073595Z/phase5b_audit.json)

The external `scripts/audit_module5_phase5b.py` passed in 9.44 seconds. It compared
the actual SQLite ledger, all 39 events and the exact CSV export, three stored
reports and selections, and all 12 checksummed inner checkpoints. Production
evidence SHA-256, sizes and mtimes match before/after. All 247 protected predecessor
files match the retained Phase 5A fingerprints; raw/cache tree inventories and
A0 transforms were not rerun.

Frozen protocol:
`677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9`.

Frozen Modules 0–5 Python code:
`7696cad010914ec37f494d9f1a11a2e1761e46f353e61d97a07399f746a34ff8`.

The code hash covers `src/vaaniswitch/m[012345]*/*.py`. Only external audit/tests
and run-state documentation were added/updated. Package, protocol, candidates,
splits, architecture and preprocessing are unchanged.

## Phase 5B results (inner validation only)

Exactly 12 production C2 runs: three SELECTED and nine INNER_COMPLETE.
For each fold the named winner is SELECTED and the other three candidates
(baseline, lower_lr, stronger_decay, stronger_dropout) are INNER_COMPLETE.
No FAILED/INVALIDATED run, final stage, outer result, outer claim or outer event exists.

| Fold | Winner | Best epoch | Loss | Accuracy | Balanced accuracy | Macro-F1 | ROC-AUC | TN / FP / FN / TP |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | stronger_dropout | 1 | 0.499713 | 0.905850 | 0.883184 | 0.895718 | 0.958800 | 666 / 18 / 85 / 325 |
| 2 | stronger_decay | 2 | 2.154629 | 0.586117 | 0.556968 | 0.556113 | 0.559946 | 646 / 331 / 301 / 249 |
| 3 | baseline | 2 | 1.731122 | 0.664506 | 0.641980 | 0.636592 | 0.686579 | 581 / 103 / 311 / 239 |

| Fold | Precision | Sensitivity | Specificity | Inner-train-only pos_weight | Validation speakers / records |
|---|---:|---:|---:|---:|---|
| 1 | 0.947522 | 0.792683 | 0.973684 | 2.2737276478679505 | M03, MC02 / 1094 |
| 2 | 0.429310 | 0.452727 | 0.661208 | 2.294525828835775 | F03, FC03 / 1527 |
| 3 | 0.698830 | 0.434545 | 0.849415 | 2.3343305908750933 | F03, MC02 / 1234 |

Unrounded metrics, identities, all four comparisons and timing are retained in
`experiments/reports/fold{1,2,3}_phase5b.json` under the protocol namespace.

Reporter recomputation agrees exactly with all three reports and selections.
All winners beat every other candidate on macro-F1, the first frozen criterion.
Remaining tie-break criteria remain balanced accuracy, valid ROC-AUC, lower loss,
fewer trainable parameters and ascending candidate ID.

All 106 saved epochs have exact signed inner-training/validation role, count and
record hashes, threshold 0.5 and correct stopping history. All 12 checkpoints pass
strict identity, tensor shape/dtype/finiteness, AdamW settings/state/step-count,
architecture and parameter checks (291,921 trainable parameters).
Each matches all 120 frozen ResNet tensors/buffers in the signed Module 4 reference.
The reference is used only for checking the frozen ImageNet backbone, not to
initialize learned Module 5 branch/head weights.

Outer model forwards: **0 supported by retained evidence** (frozen guards, exact
role/record histories, no claims/events/results). This is a derived audit finding;
the checkpoints do not contain an independent historical forward counter.
The audit itself prohibits model forward and performed zero forwards.
Loss and ROC-AUC were checked for consistency with retained evidence, not
recomputed by forwarding validation or outer data.

There are only 15 speakers and each inner-validation pair differs. The observed
score variation describes heterogeneous held-out speaker difficulty; variation
alone does not establish an implementation defect. It does not estimate outer-test
performance or statistical significance. Preserve the frozen splits/candidates.
The protocol's prior Module 4 development exposure limitation still applies.

## Phase 5C scope

This handoff narrows the older phase scheduling: Phase 5C now ends after matched
**inner** ablation selection. Final retraining, outer evaluation, aggregation and
robustness require later separate authorization. The scientific protocol is unchanged.

Each fold uses A0, A1, B1, B2, B3, C1, C2 with its selected C2 candidate:

| Fold | Common candidate | Learning rate | Weight decay | Classifier dropout |
|---|---|---:|---:|---:|
| 1 | stronger_dropout | 3e-4 | 1e-4 | 0.40 |
| 2 | stronger_decay | 3e-4 | 1e-3 | 0.30 |
| 3 | baseline | 3e-4 | 1e-4 | 0.30 |

All retain seed 42, frozen backbone, AdamW, batch size 16, maximum 30 epochs,
patience 5, min_delta 1e-4, threshold 0.5 and no augmentation. Only inner-training
records fit weights/pos_weight; only inner validation selects each ablation's own epoch.
No independent ablation hyperparameter search is permitted.

The existing `prepare_ablations` explicitly skips C2 training and incorporates
the original selected C2 identity after checking its ledger selection.
Thus **21 matched inner results = 3 reused C2 + 18 new runs**.
There is a whole-fold command; no separate manual winner/epoch choice is needed.

## Exact PowerShell commands

Run one fold at a time and wait for success before proceeding. Concurrent CPU
execution against this shared ledger/checkpoint namespace has not been validated.

```powershell
Set-Location 'C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_FINAL_Hybrid_VMD_TORGO_Project'

uv run --python 3.12 --extra dev vaaniswitch-m5 ablation-inner --outer-fold 1 --resume
uv run --python 3.12 --extra dev vaaniswitch-m5 ablation-inner --outer-fold 2 --resume
uv run --python 3.12 --extra dev vaaniswitch-m5 ablation-inner --outer-fold 3 --resume

uv run --python 3.12 --extra dev vaaniswitch-m5 ledger
```

The `ablation-inner --help` syntax was checked against the actual installed CLI:
`--outer-fold {1,2,3} [--resume] [--retry-failed REASON]`.

Rerun the same fold command after interruption. Exact experiment identities and
checkpoint checks prevent identity drift; completed runs are reused. Resume loads
the latest completed epoch's model/AdamW/RNG state; interrupted unsaved epoch work
can repeat. Existing selection bundles must match exactly and cannot be overwritten.

A recorded FAILED run requires correcting the cause and explicitly recording its
reason, for example (replace fold/reason with the actual case):

```powershell
uv run --python 3.12 --extra dev vaaniswitch-m5 ablation-inner --outer-fold 1 --resume --retry-failed 'Actual failure cause corrected; describe the correction'
```

Do not use this to bypass a scientific identity or provenance failure.

## Expected artifacts and return package

All paths below are under:

`artifacts/module5/677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9/experiments/`.

After each fold:

- `selection/foldN_ablations.json`: common candidate and all seven selections,
  including original C2 identity; each stores its own selected `epoch`,
  inner-validation metrics, split/record hashes and trainable parameter count.
- Six new `runs/foldN_<ablation>_<candidate>_seed42_677ad640faf5/inner/latest.json`
  pointers plus checksum-named `.pt` files.
- Checkpoints retain latest resumable weights/optimizer/RNG, full epoch history,
  train-only pos_weight and best-epoch metadata. They are not best-epoch weight exports.
- Updated `ledger.sqlite3` and `events.csv` with six new SELECTED ablations;
  the existing C2 evidence remains reusable.

After all three panels expect **30 total runs: 21 SELECTED + 9 INNER_COMPLETE**.
No final checkpoint, outer claim/event/result should exist.

Return the three final fold-console JSON outputs (or their
`fold1_ablations.json`, `fold2_ablations.json`, `fold3_ablations.json` files),
the final `ledger` console JSON, updated `events.csv`, and any failure/interruption
messages. Keep checkpoint directories and the SQLite ledger intact locally for
the next audit; no checkpoint upload is required now.

## Verification executed for this closure

- Read-only production audit: PASS, 9.44 seconds.
- 14 focused tests: PASS, 4.66 seconds, including metadata/metric rejection,
  mocked interruption recovery, own epoch selection, C2 reuse and completed-run reuse.
  Evidence: `artifacts/module5/phase5b_closure_tests.xml`.
- Ruff passed for the new external audit and test file.
- Existing CLI help inspected; no real Phase 5C command was executed.
- No full suite, A0 rebuild, training, final retraining, outer evaluation or robustness.

**STOP after the three Phase 5C inner panels and return their evidence.
Module 5 is NOT SIGNED OFF. Module 6 is NOT STARTED.**

