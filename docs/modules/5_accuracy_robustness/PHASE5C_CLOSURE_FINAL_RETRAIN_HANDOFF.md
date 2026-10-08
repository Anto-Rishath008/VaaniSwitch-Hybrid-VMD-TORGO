# Phase 5C closure and final development retraining handoff

Date: 2026-09-25.

- Phases 5A, 5B, 5C: COMPLETE / PASSED.
- Next phase: FINAL RETRAINING READY FOR MANUAL EXECUTION; NOT STARTED.
- Module 5: MANUAL EXPERIMENTS IN PROGRESS / NOT SIGNED OFF.
- Module 6: NOT STARTED.

## Read-only audit

[Audit artifact](../../../artifacts/module5/677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9/phase5c_audit/20260925T040400520005Z/phase5c_audit.json)

The external `scripts/audit_module5_phase5c.py` passed in 9.94 seconds.
It validates actual retained files and compares the user's pasted ledger with
live read-only SQLite evidence; the two match exactly.

Protocol: `677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9`.

Frozen code: `7696cad010914ec37f494d9f1a11a2e1761e46f353e61d97a07399f746a34ff8`.

The package hash covers `src/vaaniswitch/m[012345]*/*.py`; the new external audit,
tests and run-state documentation are outside it. No frozen package, protocol,
architecture, candidate, split, preprocessing or runtime changes were made.
CPU remains `torch 2.14.0+cpu`, two threads.

### Hard gates

- Exactly 30 production runs: 21 SELECTED, nine INNER_COMPLETE (losing C2 candidates).
- Seven SELECTED ablations per fold; one common candidate per fold; no independent
  A0/A1/B1/B2/B3/C1 search. Every selection matches its own checkpoint/ledger epoch.
- Three selected C2 runs reused exactly. All 30 retained Phase 5B C2 checkpoint,
  pointer, selection and reporting files match prior SHA-256/size/mtime fingerprints.
- All 111 ledger events match `events.csv`; only PLANNED, RUNNING, INNER_COMPLETE,
  SELECTED occur.
- All 30 checkpoint checksums, exact identities, architecture/tensor keys, shapes,
  dtypes and finiteness, AdamW settings/state/step counts and RNG state pass.
  Python/NumPy/CPU torch RNG states round-trip through isolated generators.
- All 281 saved epochs match exact signed inner-training and inner-validation
  record hashes/counts/roles, threshold 0.5 and frozen stopping/selection rules.
- Train-only pos_weight is verified for all runs: Fold 1 2.2737276478679505;
  Fold 2 2.294525828835775; Fold 3 2.3343305908750933.
- Every Mel-containing checkpoint (21 including losing C2 candidates) matches all
  120 frozen ResNet tensors/buffers in the signed reference. The nine B-only
  checkpoints have no Mel backbone.
- All 247 protected predecessor files match prior SHA-256/size/mtime fingerprints.
  Accepted raw/A0/Module 1/Module 2 full-tree audits were not rerun.
- FAILED/INVALIDATED = 0; final-training stages = 0; outer claims/events/results = 0.
- Retained evidence supports zero outer model forwards through frozen guards and
  exact inner histories. This is derived evidence, not an independent historical
  runtime counter. The audit itself prohibited forward and performed zero forwards.
- Production evidence and frozen protocol match their pre-audit fingerprints.
  Loss/ROC-AUC consistency uses saved evidence; no validation predictions were rerun.

## Exact selected inner-validation results

Common candidates: Fold 1 **stronger_dropout**, Fold 2 **stronger_decay**,
Fold 3 **baseline**. Unrounded values below come directly from the selection bundles.
The audit JSON also retains loss, accuracy, precision, sensitivity, specificity,
confusion counts, experiment/split identities and full checkpoint summaries.

| Fold | Ablation | Epoch | Macro-F1 | Balanced accuracy | ROC-AUC |
|---|---|---:|---:|---:|---:|
| 1 | A0 | 1 | 0.7757554288989366 | 0.7590393667094566 | 0.9328590785907859 |
| 1 | A1 | 1 | 0.8360875886485466 | 0.8258878904578519 | 0.944084296106119 |
| 1 | B1 | 1 | 0.6328339796001208 | 0.6868421052631579 | 0.7876051918413921 |
| 1 | B2 | 3 | 0.7364270603208656 | 0.754535729567822 | 0.7735950648980174 |
| 1 | B3 | 1 | 0.7788265804076853 | 0.7864427328483812 | 0.8235451433461703 |
| 1 | C1 | 8 | 0.8423428585318763 | 0.8532520325203252 | 0.9291256596776494 |
| 1 | C2 | 1 | 0.895718403968331 | 0.883183568677792 | 0.9588004564256168 |
| 2 | A0 | 11 | 0.5332362204392427 | 0.5418609844607798 | 0.5622536521820043 |
| 2 | A1 | 2 | 0.5453317472559553 | 0.5454042988741044 | 0.5682795198660091 |
| 2 | B1 | 1 | 0.26958909697486216 | 0.488754070903508 | 0.42701405043267887 |
| 2 | B2 | 9 | 0.6015224830938766 | 0.6013641016097515 | 0.6154740857913836 |
| 2 | B3 | 5 | 0.5271298134323804 | 0.529446357122918 | 0.514947427188983 |
| 2 | C1 | 2 | 0.5604344504600258 | 0.5610616916348748 | 0.5594119289103936 |
| 2 | C2 | 2 | 0.5561128180597207 | 0.5569675258211594 | 0.5599460314506374 |
| 3 | A0 | 5 | 0.7076090494963323 | 0.7073498139287613 | 0.7921504518872939 |
| 3 | A1 | 5 | 0.6149739703347159 | 0.6157123870281764 | 0.6564673046251993 |
| 3 | B1 | 9 | 0.608552214186017 | 0.6270388091440723 | 0.6413237639553429 |
| 3 | B2 | 6 | 0.6347093211260306 | 0.6350372142477405 | 0.5868421052631579 |
| 3 | B3 | 10 | 0.6463324114503676 | 0.6478415736310473 | 0.6891307814992026 |
| 3 | C1 | 5 | 0.6159786313497833 | 0.6156007442849548 | 0.6525358851674641 |
| 3 | C2 | 2 | 0.6365920420659671 | 0.6419803296119087 | 0.6865789473684211 |

## Descriptive development comparisons

Differences below are **macro-F1**, left ablation minus right. The audit also
stores the corresponding balanced-accuracy and ROC-AUC differences.

| Difference | Fold 1 | Fold 2 | Fold 3 |
|---|---:|---:|---:|
| A1-A0 | 0.06033215974961004 | 0.012095526816712665 | -0.0926350791616164 |
| B3-B1 | 0.1459926008075645 | 0.25754071645751825 | 0.03778019726435056 |
| B3-B2 | 0.042399520086819686 | -0.07439266966149616 | 0.011623090324336949 |
| C1-A1 | 0.006255269883329695 | 0.015102703204070478 | 0.001004661015067354 |
| C2-C1 | 0.05337554543645473 | -0.004321632400305142 | 0.020613410716183833 |

Unweighted mean and **sample** standard deviation across the three inner
validation folds (`ddof=1`):

| Ablation | Macro-F1 mean | Sample SD |
|---|---:|---:|
| A0 | 0.6722002329448372 | 0.1250768956888609 |
| A1 | 0.6654644354130727 | 0.15181141302237453 |
| B1 | 0.5036584302536666 | 0.2030732395444915 |
| B2 | 0.6575529548469242 | 0.07029356258135536 |
| B3 | 0.650762935096811 | 0.12590686153380004 |
| C1 | 0.6729186467805618 | 0.14933087977154696 |
| C2 | 0.6961410880313396 | 0.17746141876932622 |

These are selected development results, not final generalization estimates:

- VMD improves macro-F1 in Folds 1 and 2 but reduces it in Fold 3 (A1 minus A0).
- B1 is highly unstable across these speaker sets; B2 has the smallest observed
  macro-F1 sample SD among the seven ablations and is comparatively stable.
- B3 exceeds B1 in all three folds; it exceeds B2 in Folds 1/3 but trails it in Fold 2.
- C1 modestly exceeds A1 in all three folds under pooled fusion.
- C2 exceeds C1 in Folds 1/3 and is slightly below it in Fold 2.
- C2 has the highest descriptive mean macro-F1 and substantial fold variability.
- Fold winners differ: Fold 1 C2, Fold 2 B2, Fold 3 A0.
- Fifteen speakers, two validation speakers per fold and overlapping development
  exposure limit interpretation. No significance or universal-superiority claim.
  The frozen prior Module 4 exposure limitation remains applicable.
- Do not change candidates, splits, architecture, epochs or threshold in response.

## Final retraining design: manual CPU only

Run 21 final stages: all seven ablations in each fold. These use the existing
21 SELECTED experiment identities, not 21 additional ledger rows.

The real CLI dispatches `retrain` to `retrain_fold`. It reads the immutable
`selection/foldN_ablations.json` bundle and requires each pending run's selection
to match the ledger. For each ablation:

1. Construct a fresh model using the same frozen ImageNet/random initialization
   policy (seed 42 and fixed branch/head substreams). Do not load learned inner weights.
2. Train on all ten development speakers, combining the former inner training and
   validation records. Outer speakers remain excluded.
3. Recompute pos_weight from development labels only.
4. Run exactly the frozen selected number of epochs. No validation pass,
   early stopping or replacement of short epoch counts.
5. Save final-only checkpoints with full identity, selected configuration,
   development/outer speaker lists, record hashes, optimizer, RNG and epoch history.

| Fold | Common candidate | Development records | Development-only pos_weight |
|---|---|---:|---:|
| 1 | stronger_dropout | 5854 | 2.140557939914163 |
| 2 | stronger_decay | 5800 | 2.1402273957769355 |
| 3 | baseline | 5692 | 2.0164281928987813 |

Selected final epoch counts:

| Fold | A0 | A1 | B1 | B2 | B3 | C1 | C2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 1 | 3 | 1 | 8 | 1 |
| 2 | 11 | 2 | 1 | 9 | 5 | 2 | 2 |
| 3 | 5 | 5 | 9 | 6 | 10 | 5 | 2 |

The total is 90 ablation-epochs across 21 final stages (16 / 32 / 42 per fold).
This is a workload count, not a runtime estimate.

## Exact PowerShell commands

Run each fold **sequentially**, proceeding only after success. Concurrent CPU
execution against the shared ledger/checkpoint namespace has not been validated.

```powershell
Set-Location 'C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_FINAL_Hybrid_VMD_TORGO_Project'

uv run --python 3.12 --extra dev vaaniswitch-m5 retrain --outer-fold 1 --resume
uv run --python 3.12 --extra dev vaaniswitch-m5 retrain --outer-fold 2 --resume
uv run --python 3.12 --extra dev vaaniswitch-m5 retrain --outer-fold 3 --resume

uv run --python 3.12 --extra dev vaaniswitch-m5 ledger
```

Actual installed `retrain --help`:
`--outer-fold {1,2,3} [--resume] [--retry-failed REASON]`.

No GPU flags, torch/CUDA changes or new protocol. The whole-fold command has
no outer evaluation call. No outer, aggregate or robustness command is authorized.

### Resume and failure recovery

Rerun the same command for an interrupted fold. Completed final runs are skipped
using their FINAL_TRAIN_COMPLETE ledger state. A pending final stage resumes only
from `final/latest.json`; its checksum and exact checkpoint identity are validated.
There is no path that resumes final training from an `inner/` checkpoint.

The last completed epoch's model/AdamW/RNG state is restored; unsaved epoch work may
repeat. Code/protocol drift is rejected by the frozen gate. Pending selection
drift is rejected against ledger selection; resumed identity drift is rejected by
the checkpoint loader.

If the ledger records FAILED, correct the cause, preserve the evidence and supply
an explicit factual reason. Example syntax (replace fold and reason as needed):

```powershell
uv run --python 3.12 --extra dev vaaniswitch-m5 retrain --outer-fold 1 --resume --retry-failed 'Actual failure cause corrected; describe the correction'
```

Do not alter frozen code/config or bypass provenance failures to continue.
The command does not re-audit checkpoints for runs it skips as already completed;
the subsequent retained-evidence audit verifies completed final artifacts.

## Expected artifacts and user return package

Base namespace:

`artifacts/module5/677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9/experiments/`.

For each selected run:

- `runs/foldN_<ablation>_<candidate>_seed42_677ad640faf5/final/latest.json`.
- Corresponding checksum-named `final/<sha256>.pt`.
- Ledger fields `final_checkpoint`, `final_identity`, `final_epoch`,
  `final_training_seconds`, `final_pos_weight` and parameter counts.
- Training history contains only `development` epoch records, with no validation entry.
- Existing inner checkpoints and immutable C2/ablation selections remain retained.

The command prints one completion JSON per fold:
`{"outer_fold": N, "status": "FINAL_TRAIN_COMPLETE", "outer_evaluations": 0}`.
It does not create a separate per-fold final-report JSON file.

After all three commands expect **30 ledger runs: 21 FINAL_TRAIN_COMPLETE and
nine INNER_COMPLETE**, 21 final checkpoint pointers/generations and no outer
claim/event/result. No additional model selection is performed.

Return:

- All three final-retraining completion console JSONs and any failures/interruptions.
- The final `ledger` console JSON and updated `events.csv`.
- Keep the 21 final checkpoint directories, inner evidence and SQLite ledger intact
  locally for the next read-only audit. No checkpoint upload is required now.

## Verification executed during this closure

- Read-only production audit PASS: 9.94 seconds.
- Seventeen focused tests PASS: 1.75 seconds. Coverage includes isolated RNG
  restoration/rejection, immutable bundle/C2 binding, sample SD/difference calculation,
  mocked fresh final initialization, final-only resume, all-development fitting,
  no validation, fixed epoch counts, completed-final reuse, explicit failed recovery,
  and checksum-valid checkpoint identity rejection.
- Evidence: `artifacts/module5/phase5c_closure_tests.xml`.
- Ruff passed on the new external audit and tests; installed CLI help checked.
- No full suite, raw/cache rebuild, actual training, model forward, outer evaluation,
  outer aggregation or robustness analysis was executed.

**STOP after final retraining and return the evidence. Outer evaluation is a
separate future phase. Module 5 is NOT SIGNED OFF; Module 6 is NOT STARTED.**

