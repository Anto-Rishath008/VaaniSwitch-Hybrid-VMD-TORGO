# Final retraining closure and once-only outer-evaluation handoff

Closure date: 2026-09-29 (local; audit timestamp is UTC).

- Phases 5A, 5B, 5C: COMPLETE / PASSED.
- Final development retraining: COMPLETE / PASSED.
- Next: OUTER EVALUATION READY FOR MANUAL EXECUTION; NOT STARTED.
- Module 5: MANUAL EXPERIMENTS IN PROGRESS / NOT SIGNED OFF.
- Module 6: NOT STARTED.

## Authoritative audit

[Read-only audit](../../../artifacts/module5/677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9/final_retraining_audit/20260928T195038869875Z/final_retraining_audit.json)

External `scripts/audit_module5_final_retraining.py`: PASS in 43.32 seconds.
The user's pasted ledger exactly matches live read-only SQLite evidence.

Protocol: `677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9`.
Frozen code: `7696cad010914ec37f494d9f1a11a2e1761e46f353e61d97a07399f746a34ff8`.
CPU runtime: `torch 2.14.0+cpu`, two threads, seed 42, threshold 0.5.

Exactly 30 identities: 21 FINAL_TRAIN_COMPLETE and nine INNER_COMPLETE.
There are no remaining SELECTED, FAILED, INVALIDATED or OUTER_EVALUATED runs.
All 132 events match the CSV. Exactly 21 final directories contain one checksum
generation and one latest pointer each (seven per fold; 574,566,122 checkpoint bytes).

All 21 final checkpoints pass exact identity, selected-configuration, shape/dtype,
finite model tensors, architecture/parameter counts, AdamW settings/state/step-count
and isolated Python/NumPy/CPU torch RNG restoration checks. All 12 Mel-containing
final checkpoints match all 120 signed frozen ResNet tensors/buffers; the nine
B-only models contain no Mel backbone.

All 90 final epochs contain development training only, covering the exact complete
development record count/hash once per epoch under the frozen trainer's coverage
guard. No validation metrics, inner/outer role, early-stopping state or altered epoch
count appears. All 69 retained inner checkpoint/pointer/selection/report files are
unchanged; their prior strict Phase 5C audit remains valid. All 247 protected predecessor
files match SHA-256/size/mtime. Accepted raw/A0/Module 1/Module 2 full-tree evidence
was inspected by reference rather than recomputed.

### Exact final epochs and fitting populations

| Fold | Candidate | A0 | A1 | B1 | B2 | B3 | C1 | C2 | Development records | Development-only pos_weight |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | stronger_dropout | 1 | 1 | 1 | 3 | 1 | 8 | 1 | 5854 | 2.140557939914163 |
| 2 | stronger_decay | 11 | 2 | 1 | 9 | 5 | 2 | 2 | 5800 | 2.1402273957769355 |
| 3 | baseline | 5 | 5 | 9 | 6 | 10 | 5 | 2 | 5692 | 2.0164281928987813 |

Each population contains all ten development speakers and excludes that fold's
five outer speakers. Exact speaker lists and record hashes are in the audit.

Fresh initialization and final-only resume are supported by the unchanged
`train_stage` path, exact final identities/optimizer steps/histories, and focused
mocked regressions. A missing `final/latest.json` starts from the original
pretrained/random policy; an existing pointer loads only final state, never learned
inner weights. There is no separate historical initialization/resume event trace.
The precise Fold 3 interruption position is user-reported; its completed B3 epoch 10,
C1 epoch 5 and C2 epoch 2 states pass all retained-evidence checks.

Outer claims/events/results are zero; robustness has not started. Frozen guards
and exact histories support zero outer forwards, without an independent historical
counter. The audit itself performed no data/model forwards.

## Exact native evaluator behavior

Installed help confirms:

```powershell
uv run --python 3.12 --extra dev vaaniswitch-m5 outer --outer-fold 1 --ablation A0
```

There is **one command per ablation per fold**, no native whole-fold outer command.
The accepted ablations are A0, A1, B1, B2, B3, C1, C2. No resume/retry flags exist.

The native order is:

1. Require the unchanged frozen protocol/code and selection bundle.
2. Require all seven selected models in this fold to be FINAL_TRAIN_COMPLETE or
   already OUTER_EVALUATED.
3. Lock the target experiment. An already completed target returns its checksum-
   verified retained JSON, with no new forward, provided the fold gate still passes.
4. Require target FINAL_TRAIN_COMPLETE, exact ledger selection, and no prior claim.
5. Load the ledger-bound final checkpoint with checksum, exact identity, tensor,
   optimizer and selected-epoch checks.
6. Commit an atomic one-use SQLite claim (`BEGIN IMMEDIATE`; update only when
   state is FINAL_TRAIN_COMPLETE and claim is NULL) and an OUTER_CLAIMED event.
7. Only then construct the permitted outer dataset. Source/batch guards constrain
   every sample to that fold; inference uses eval/no_grad and threshold 0.5.
8. Require ordered predictions to cover exactly the expected logical records.
9. Publish `outer_result.json` atomically without replacement, record its SHA-256,
   and transition to OUTER_EVALUATED with a completion event.

No fit, pos_weight computation, tuning or threshold-selection path exists in this
evaluator. There is no automatic call to aggregate or robustness.

### Additional external evidence checks

The frozen result schema contains predictions, their count via `forward_count`
and `source_load_count`, and split/checkpoint provenance, but no explicit
outer-record-hash field. The frozen package was not changed.

External `scripts/module5_outer_evidence.py` provides:

- `preflight`: read-only check against this successful closure audit. Before each
  target, it verifies all seven audited final checkpoint files for the fold,
  all selection files, exact experiment identities and immutable inner/final
  ledger fields, CPU/code/protocol, the expected outer IDs/count/hash, and no prior
  target claim/event/result. No model construction or forward.
- `report`: after all seven results exist, checksum/provenance-check each result,
  verify exact predicted IDs/speakers/labels, recompute per-result metrics from
  retained predictions, and store explicit per-result record counts/hashes in a
  separate immutable fold sidecar. No model forward or across-fold aggregation.

Read-only preflights passed for all three folds during this closure.
Evidence: sibling `outer_preflight_checks.json`. Future reports have not been generated.

| Fold | Outer records | Exact outer record hash |
|---|---:|---|
| 1 | 2819 | 2936df532e7ada33d7b571841c1026ddb30219fda3029cd89b3325f0f32d0569 |
| 2 | 2873 | e85a74cb8dd33249ac0f27d42521d5dc0cd2d9abd283970c978f278d9fb97353 |
| 3 | 2981 | 72eedd08aa36215e3d998e0d19bdf95f655c9cc7686f83f707a5cb053fde266a |

## Manual execution: one fold at a time, sequentially

Do not run concurrent commands. Shared-namespace concurrent evaluation has not
been validated. Keep the frozen CPU runtime.

Set the fold to 1 and run this block manually. Return its outputs before proceeding
to Fold 2, then Fold 3 using the same block with the fold number changed.

```powershell
Set-Location 'C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_FINAL_Hybrid_VMD_TORGO_Project'
$fold = 1
$audit = 'artifacts/module5/677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9/final_retraining_audit/20260928T195038869875Z/final_retraining_audit.json'

foreach ($ablation in @('A0', 'A1', 'B1', 'B2', 'B3', 'C1', 'C2')) {
    uv run --python 3.12 --extra dev python scripts/module5_outer_evidence.py preflight --audit $audit --outer-fold $fold --ablation $ablation
    if ($LASTEXITCODE -ne 0) { throw "Preflight failed: fold $fold / $ablation. Stop and preserve evidence." }

    uv run --python 3.12 --extra dev vaaniswitch-m5 outer --outer-fold $fold --ablation $ablation
    if ($LASTEXITCODE -ne 0) { throw "Outer command failed: fold $fold / $ablation. Stop; do not retry a claim." }
}

uv run --python 3.12 --extra dev python scripts/module5_outer_evidence.py report --audit $audit --outer-fold $fold --save
if ($LASTEXITCODE -ne 0) { throw "Retained result report failed. Preserve evidence; do not reevaluate." }

uv run --python 3.12 --extra dev vaaniswitch-m5 ledger
```

For the subsequent fold panels use `$fold = 2`, then `$fold = 3`. This is a
PowerShell loop over the real per-ablation CLI, not an invented whole-fold command.

No settings may change after outer results: candidates, epochs, architecture,
splits, threshold 0.5 and seed 42 remain frozen. No retraining or model selection.

### Failure and repeated execution

- A rejection before the one-use claim creates no outer forward. Diagnose the
  rejected metadata/checkpoint condition; never change the frozen scientific identity.
- After claim, an ordinary exception records FAILED and retains the claim.
  A hard interruption may leave FINAL_TRAIN_COMPLETE with a non-null claim;
  the CSV export may lag the authoritative SQLite event history.
- Both cases require stopping for evidence/defect review. No automatic retry,
  clearing/deleting claims, rebuilding ledger rows or overwriting results.
- Publication followed by interrupted ledger completion can leave an immutable
  result with an unfinished claim: preserve it, do not rerun inference.
- A fully completed result can be read again without a forward via the native
  completed-result path (when its fold gate passes), or the external fold report
  after all seven are complete. The manual preflight deliberately refuses a
  previously evaluated target, preventing accidental repeated panel execution.

## Expected artifacts and return after EACH fold

Base:
`artifacts/module5/677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9/`.

For each of seven ablations:

- `experiments/runs/<experiment_id>/outer_result.json`: immutable predictions
  (logical ID, speaker, binary label, probability), selection, run identity,
  checkpoint pointer/checksum, threshold, timing and parameter counts.
- Metrics: accuracy, balanced accuracy, macro-F1, precision, sensitivity,
  specificity, ROC-AUC, TN/FP/FN/TP and `[[TN, FP], [FN, TP]]`.
- Exact count via prediction list/source_load_count/forward_count. Here
  forward_count counts predicted records, not Python model-call batches.
- SHA-256 bound in ledger; OUTER_CLAIMED then OUTER_EVALUATED events.

Additional sidecar:
`outer_evidence/foldN_outer_report.json`, containing explicit record count/hash,
verified per-result metrics and final checkpoint identity/provenance.

Expected cumulative production states:

| After fold | OUTER_EVALUATED | FINAL_TRAIN_COMPLETE | INNER_COMPLETE |
|---|---:|---:|---:|
| 1 | 7 | 14 | 9 |
| 2 | 14 | 7 | 9 |
| 3 | 21 | 0 | 9 |

Return after **each** fold:

1. Seven outer-command console JSON outputs and any error/interruption text.
2. `outer_evidence/foldN_outer_report.json`.
3. Updated ledger console JSON and `experiments/events.csv`.
4. Preserve the seven immutable result JSONs, checkpoint directories and SQLite
   ledger locally for audit; do not delete or regenerate them.

Only after all 21 immutable results are independently verified does a later phase
aggregate unweighted three-fold means/sample SD (`ddof=1`). Pooled confusion is
secondary. Robustness follows immutable evaluation; neither phase is authorized
by this handoff. Three folds do not establish significance.

## Verification and stop boundary

37 focused tests passed; two unrelated mocked workflow tests were deselected.
Tests covered final development-only histories and epochs, RNG/bundle integrity,
mocked fresh/final-only resume, one-use claims in an isolated test ledger, and
synthetic retained-result corruption/report checks. No outer evaluator was called.
Evidence: `artifacts/module5/final_retraining_closure_tests.xml`.
Ruff and actual native/helper CLI help passed. No full suite or predecessor rebuild.

**No real training, outer evaluation, outer aggregation or robustness was executed
in this session. Module 5 remains NOT SIGNED OFF. Module 6 remains NOT STARTED.**

