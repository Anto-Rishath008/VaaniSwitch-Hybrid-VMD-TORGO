# Fold 1 post-recovery continuation: B1 only

Date: 2026-10-06 (Asia/Calcutta).

**READY FOR MANUAL B1 EXECUTION. B1 has not run.**
Module 5: MANUAL EXPERIMENTS IN PROGRESS / NOT SIGNED OFF.
Module 6: NOT STARTED.

## Preserved provenance and scope

| Target | Provenance | Current state |
|---|---|---|
| Fold 1 A0 | NATIVE | OUTER_EVALUATED; verified and unchanged |
| Fold 1 A1 | RECOVERED_PROTOCOL_EXCEPTION | External recovery COMPLETE / VERIFIED; native FINAL_TRAIN_COMPLETE with consumed claim and no native result |
| Fold 1 B1 | Intended NATIVE | FINAL_TRAIN_COMPLETE; unclaimed; read-only preflight PASS |
| Fold 1 B2, B3, C1, C2 | Intended NATIVE | FINAL_TRAIN_COMPLETE; unclaimed; future targets only |
| Folds 2 and 3 | Unchanged | No outer evaluation authorized here |

The predeclared future order is B1 → B2 → B3 → C1 → C2, one target at a time.
Only B1 is handed off now. Return its evidence before the next target is authorized.
Do not use a loop. Observed A0/A1 metrics do not change this matrix or justify
tuning, threshold changes, epoch changes, model changes, or performance-based stopping.
No ranking, aggregation or robustness analysis is part of this handoff.
Future aggregation must retain A1's RECOVERED_PROTOCOL_EXCEPTION provenance flag.
Its original native prior-forward count remains UNKNOWN.

Protocol SHA-256:
677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9

Frozen code SHA-256:
7696cad010914ec37f494d9f1a11a2e1761e46f353e61d97a07399f746a34ff8

The new helper, tests and documentation are outside the Modules 0–5 Python-source
fingerprint. The scientific source, frozen protocol and both original external
helpers are unchanged. None of the documentation edits overlap the 247 protected
predecessor files or captured frozen source files.

## Why native continuation is safe

Inspected the actual evaluator, ledger, data guards, CLI and reporting code:

- evaluator.py:27–31 accepts either FINAL_TRAIN_COMPLETE or OUTER_EVALUATED for
  all seven selected models in the requested fold. A1 satisfies this gate.
- evaluator.py:32–45 locks and checks only the requested target's claim. It does
  not reject another run's unfinished claim. It does not require A1 to be evaluated.
- evaluator.py:48–66 selects that fold, restores that target's final checkpoint,
  and creates the target-bound OuterPermit and dataset. B1 does not read A1
  predictions or evaluate Fold 2/3 data. Shared signed split/manifest metadata
  are read normally.
- ledger.py:165–175 claims only the requested run using a conditional SQL update.
  Its transition likewise updates only the requested run. A1's state, claim,
  event and missing result remain untouched.
- evaluator.py:86–113 writes only the target's immutable result, changes its
  ledger row and exports the global event CSV. It does not write selections.
  The CSV legitimately gains the previously unexported original A1 claim event.
- evaluator.py:37–40 reuses a checksum-valid completed result without new forwards.
  An interrupted consumed target claim is not resumable through this native CLI.

The old external module5_outer_evidence.py:168 rejects every unfinished claim,
which is the current blocker. It is unchanged. Its full-fold report remains
blocked. Native aggregation requires all 21 native cells and cannot consume
external A1 automatically; no exception-aware aggregation consumer was added.

Focused tests call the actual native evaluator with fake models, datasets and
ledger for each remaining target. They verify target-only mutation and completed
result reuse while preserving A1. These are mocked control-flow checks, not real
inference or an assertion that a future run cannot encounter an operational error.

## Exact exception accepted by the new guard

All conditions must hold; any change fails closed:

1. A1 experiment is fold1_A1_stronger_dropout_seed42_677ad640faf5, still
   FINAL_TRAIN_COMPLETE with claim cb66ac287b054a8b8cd3e68d3d8ee385. Native A1
   outer_result.json remains absent. Original OUTER_CLAIMED event 135 is unchanged.
2. Original forensics, recovery authorization and its input bindings remain valid.
   The pinned acceptance records a fresh PASS from the original recovery verifier
   and proves production evidence was unchanged by recovery.
3. Attempt is a1_20261005T110007Z_af06fd40be7c. All 189 recovery namespace files,
   including authorization, journal and predictions, remain unchanged.
4. Recovery result SHA-256 is
   86279a525cec7960fb1e395db86364571c990229d10ba4a491b91bf7511c1659.
   Recovery manifest SHA-256 is
   03a51736746620d07319364ec5ffbf321f10d1a72caf793345948b7fdcfc1ba6.
5. Provenance flag is RECOVERED_PROTOCOL_EXCEPTION; retained result fields are
   status RECOVERED_AFTER_INTERRUPTION and classification PROTOCOL_EXCEPTION.
   Exactly 2,819 ordered records / 177 batches match record hash
   2936df532e7ada33d7b571841c1026ddb30219fda3029cd89b3325f0f32d0569.
6. A1 checkpoint remains
   5c2cc1de2f8e46d9729829f1eeb303e03d4733e86dab862ce6e85ad1c1eb34a8.
   Threshold 0.5, seed 42, torch 2.14.0+cpu and two CPU threads remain fixed.
   The verifier recomputes the recovered result from its journal/predictions.
7. No other unfinished claim is allowed. Requests are limited to Fold 1
   B1/B2/B3/C1/C2, with exactly the preceding targets completed and verified.
   A0/A1 and Fold 2/3 are rejected as targets.
8. A preflight target has no prior claim, result or outer event. Its signed final
   checkpoint, selected epoch, selection, identities and runtime must match.
   All non-target evidence remains fixed; no later target or partial file is allowed.

After legitimate native continuation, the original recovery verifier's strict
whole-production snapshot check would fail. It remains unchanged. The new helper
verifies the pinned original recovery proof, unchanged recovery evidence and
original native events, then allows only the exact completed native target prefix.
It does not relabel A1 as native or erase its exception.

## Verification actually performed

- 54 focused synthetic/mocked tests PASS in 0.31 seconds; Ruff PASS.
- New helper help and the installed native outer --help PASS.
- Instrumented real B1 read-only preflight PASS in 3.547 seconds. Model calls,
  model/dataset construction, native Ledger creation and native evaluation were
  blocked. Actual scientific model forwards this session: ZERO.
- Production inventory: 149 files unchanged; 30 runs, 135 ledger events and
  the original 134 exported CSV rows. A0 is the sole native OUTER_EVALUATED run;
  20 remain FINAL_TRAIN_COMPLETE and nine INNER_COMPLETE.
- Recovery inventory: 189 files unchanged. Frozen source/protocol, original
  helpers and all 247 captured predecessor files unchanged.
- No full suite, corpus audit, training, inference or aggregation was executed.

Evidence under artifacts/module5/<protocol-hash>/outer_continuation/fold1/:

- acceptance.json: pinned original native/recovery fingerprints and recovery PASS;
- authorization_request.txt: exact current user instruction;
- b1_preflight.json: measured preflight and instrumentation result;
- handoff_checks.json: focused check outputs and final preservation checks.

B1 identity: fold1_B1_stronger_dropout_seed42_677ad640faf5.
Selection: stronger_dropout, epoch 1.
Final checkpoint SHA-256:
fb7b9d3df0c3fff1de45275c21ef244b844de9d1e8887a6c1d68f744d86f4742

## Exact manual PowerShell commands

Run the following commands sequentially. Stop on any failure. Keep this terminal
and host running during native evaluation; do not start another evaluation.

~~~powershell
Set-Location -LiteralPath 'C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_FINAL_Hybrid_VMD_TORGO_Project'
$env:PYTHONDONTWRITEBYTECODE = '1'

# 1. Read-only B1 preflight. Continue only on PASS.
.\.venv\Scripts\python.exe -B scripts/module5_outer_exception_continuation.py preflight --outer-fold 1 --ablation B1
if ($LASTEXITCODE -ne 0) { throw 'B1 preflight failed. Stop and return its output.' }

# 2. Native B1 evaluation. This is the only inference command authorized here.
.\.venv\Scripts\vaaniswitch-m5.exe outer --outer-fold 1 --ablation B1
if ($LASTEXITCODE -ne 0) { throw 'B1 native evaluation failed. Preserve evidence; do not retry.' }

# 3. Read-only verification of the completed native B1 result.
.\.venv\Scripts\python.exe -B scripts/module5_outer_exception_continuation.py verify --outer-fold 1 --ablation B1
if ($LASTEXITCODE -ne 0) { throw 'B1 verification failed. Stop and return its output.' }

# STOP. Return B1 evidence for inspection before any further target.
~~~

The guard only prints JSON; it writes no scientific artifacts. Native evaluation
prints its result at completion and need not show batch progress. Do not infer
failure from a quiet terminal. There is no native post-claim interruption resume:
preserve the error, claim, lock and partial evidence and stop for review. Do not
clear/retry a claimed incomplete run. This guard rejects a second exception.

## Expected B1 output and return package

Native result, relative to artifacts/module5/<protocol-hash>/:

experiments/runs/fold1_B1_stronger_dropout_seed42_677ad640faf5/outer_result.json

The result retains ordered predictions, metrics, selection, identity, checkpoint,
threshold, one-use claim and counts. Expected source_load_count / forward_count
are 2,819; the frozen batch size implies 177 batches. The native schema counts
records as forwards, not a separately recorded count of model batch calls.

On successful completion only B1 becomes OUTER_EVALUATED. The expected new events
are 136 OUTER_CLAIMED and 137 OUTER_EVALUATED. The ledger and refreshed events.csv
then have 137 events/rows; states are two OUTER_EVALUATED, 19 FINAL_TRAIN_COMPLETE,
nine INNER_COMPLETE. The three non-null claims are A0, unchanged A1 and new B1.
The only new production file is B1's outer_result.json.

The post-target verifier checks its checksum, exact ordered records/labels,
record hash, threshold, checkpoint, claim lifecycle and recomputed metrics. It
also checks unchanged A0, A1 native and recovered evidence, later targets and
Fold 2/3. It neither forwards a model nor ranks/aggregates results.

Return all three console outputs (including errors if any), B1 outer_result.json,
and the current experiments/events.csv. Keep experiments/ledger.sqlite3 and all
checkpoints in place for local read-only inspection. Do not run B2 yet.
