# Fold-1 exception-aware closure and Fold-2/A0 handoff

Date: 2026-10-06 (Asia/Calcutta).

**FOLD 1 PANEL COMPLETE / EXCEPTION-AWARE CLOSURE PASS.**
Six NATIVE results plus one RECOVERED_PROTOCOL_EXCEPTION.
**FOLD 2 NOT STARTED; A0 READ-ONLY PREFLIGHT PASS.**
Module 5 NOT SIGNED OFF. Module 6 NOT STARTED.

## Scientific disclosure

Fold 1/A1 does NOT satisfy uninterrupted native once-only outer evaluation.
Its original native model-forward count cannot be established and remains UNKNOWN.
Its native state is FINAL_TRAIN_COMPLETE, its original claim remains
cb66ac287b054a8b8cd3e68d3d8ee385, event 135 is unchanged, and its native result is
absent. External recovery remains RECOVERED_PROTOCOL_EXCEPTION. Neither this
closure nor any future consumer may relabel it NATIVE or erase this limitation.

Observed Fold-1 performance did not change selections, thresholds, epochs, model
architecture or the execution plan. No ranking, winner selection, across-fold
mean/SD, tuning or robustness analysis was performed. All future reporting must
retain each cell's provenance and this disclosure.

## Immutable identity and evidence

Protocol:
677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9

Frozen Modules 0–5 Python code:
7696cad010914ec37f494d9f1a11a2e1761e46f353e61d97a07399f746a34ff8

Runtime unchanged: seed 42, torch 2.14.0+cpu, two CPU threads, threshold 0.5,
frozen pretrained Mel backbone. Checkpoint file/identity checks preserve the
previously audited frozen-backbone tensors without constructing a model.

All following paths are relative to artifacts/module5/<protocol-hash>/:

| Artifact | SHA-256 |
|---|---|
| outer_exception_report/fold1/fold1_outer_panel.json | df03075219e2aa584664b3d340d1fe537c60b6c6c588bb2451995e2b0eea362b |
| outer_continuation/fold2/a0_preflight.json | 5f66f330590a11c9eff7fa7e632834f9023848ab893b1e67857c5cbcda5b555b |

The closure binds all seven cells, result/checkpoint hashes, exact identities,
claims, metrics, the current native files/runs/events snapshot, and A1 recovery
inventory. Its new external writer uses atomic no-replace publication; an exact
repeat is reused without overwrite. Changed existing content fails.
The Fold-2 guard pins this closure's SHA-256 and requires PASS before proceeding.
Final preservation and test outputs are retained in
outer_exception_report/fold1/handoff_checks.json.

A1 recovery attempt: a1_20261005T110007Z_af06fd40be7c.
Result SHA: 86279a525cec7960fb1e395db86364571c990229d10ba4a491b91bf7511c1659.
Manifest SHA: 03a51736746620d07319364ec5ffbf321f10d1a72caf793345948b7fdcfc1ba6.
All 189 recovery files, authorization, journal/predictions and original forensics
remain unchanged. Recovery metrics were recomputed read-only.

## Descriptive Fold-1 table, in declared ablation order

| Ablation | Provenance | Macro-F1 | ROC-AUC |
|---|---|---:|---:|
| A0 | NATIVE | 0.5334029569460835 | 0.6001504365498371 |
| A1 | RECOVERED_PROTOCOL_EXCEPTION | 0.49368571607473 | 0.5742429919275179 |
| B1 | NATIVE | 0.3876614286304653 | 0.4768776184474948 |
| B2 | NATIVE | 0.5158362362642382 | 0.6495407427591766 |
| B3 | NATIVE | 0.5143063139644897 | 0.5830239449572533 |
| C1 | NATIVE | 0.5132048998974691 | 0.5596546203889778 |
| C2 | NATIVE | 0.5018899065715767 | 0.5740232977962465 |

All seven cells cover exactly 2,819 ordered signed records, hash
2936df532e7ada33d7b571841c1026ddb30219fda3029cd89b3325f0f32d0569.
Every native result passes exact checksum/claim lifecycle/checkpoint identity,
record order/labels, threshold and recomputed metrics. No second exception exists.
This is one-fold description only, with no performance-based decision.

## Checks actually executed

- 117 focused tests PASS in 0.81 seconds: 63 new report/Fold-2 cases and 54 retained
  continuation cases. Ruff and both new CLI help checks plus native outer help PASS.
- Instrumented real Fold-1 closure PASS in 3.908 seconds.
- Instrumented real Fold-2/A0 preflight PASS in 3.603 seconds.
- Model construction/forward, dataset construction and native ledger/evaluator
  execution were blocked during both real inspections.
- Actual scientific model forwards this session: ZERO. Native/scientific writes:
  ZERO. Only new external report/evidence and implementation/documentation files.
- All 154 production files, 30 runs and 145 ledger/CSV events remain unchanged
  during this work: six OUTER_EVALUATED, 15 FINAL_TRAIN_COMPLETE, nine INNER_COMPLETE.
  Seven consumed claims comprise six completed native cells and original A1.
- All 247 captured predecessor files, frozen scientific source/protocol, original
  evidence/recovery helpers, original Fold-1 continuation helper, and A1 artifacts
  are preserved. No expensive raw/cache corpus audit or full suite was repeated.
- Fold 2 and Fold 3 have no outer claim/result/event. No inference was executed.

## Native Fold-2 continuation proof and external gate

The unchanged evaluator reads selection/fold{fold}_ablations.json and checks only
that fold's seven selected runs (evaluator.py:25–31). It locks and inspects the
requested target's claim (32–45). Fold1/A1 is outside this Fold-2 readiness panel.
Ledger.claim_outer uses a conditional UPDATE WHERE id=? and claim IS NULL;
transition also updates only that target. The dataset uses context.split(2) and a
target-bound OuterPermit. Shared signed metadata are read, but Fold-1/3 model
evaluation does not occur. Selections and other run rows are not changed.

Seven focused tests exercise each Fold-2 target through the actual native
evaluator, SQLite ledger/lock and OuterPermit using temporary synthetic evidence
and mocked model/data. They verify target-only claims/transitions/results,
unchanged Fold1/A1 and other runs, Fold-2-only split access, unchanged selections,
and reuse of completed results with no additional mocked forward.

Only the old external evidence helper's global unfinished-claim rule blocks this
state. It and the native evaluator are unchanged. The new separate Fold-2 guard
requires the pinned Fold-1 PASS closure and preserves its entire native history,
results and recovery evidence. It permits only the exact A0 claim/completion
delta, rechecks the A1 exception, and rejects another unfinished claim, changed
checkpoint/selection, unexpected file/event, later Fold-2 target or Fold-3 activity.
Its post-target verifier recomputes only that target's retained metrics.

The order A0 → A1 → B1 → B2 → B3 → C1 → C2 agrees with protocol.yaml and the earlier
final-retraining handoff. This handoff enables only A0; later targets are rejected
by its CLI and require a later reviewed handoff. Do not use the historical loop.

## Frozen Fold-2 preparation

All targets retain candidate stronger_decay:

| Target | Selected epoch |
|---|---:|
| A0 | 11 |
| A1 | 2 |
| B1 | 1 |
| B2 | 9 |
| B3 | 5 |
| C1 | 2 |
| C2 | 2 |

Outer records: 2,873, in the signed order, hash
e85a74cb8dd33249ac0f27d42521d5dc0cd2d9abd283970c978f278d9fb97353.
A0 experiment: fold2_A0_stronger_decay_seed42_677ad640faf5.
A0 checkpoint:
3dc65c5e1b57860fcd28ddad921ea7f0f4110fefd35781e3f796b5eb5269cc41.
Current state FINAL_TRAIN_COMPLETE, null claim, no outer result/event.

## Exact manual PowerShell commands: Fold-2/A0 only

Run sequentially, with no other evaluation process. Continue only after PASS.

~~~powershell
Set-Location -LiteralPath 'C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_FINAL_Hybrid_VMD_TORGO_Project'
$env:PYTHONDONTWRITEBYTECODE = '1'

.\.venv\Scripts\python.exe -B scripts/module5_outer_fold2_continuation.py preflight --outer-fold 2 --ablation A0
if ($LASTEXITCODE -ne 0) { throw 'Fold-2/A0 preflight failed. Stop and return output.' }

.\.venv\Scripts\vaaniswitch-m5.exe outer --outer-fold 2 --ablation A0
if ($LASTEXITCODE -ne 0) { throw 'Native evaluation failed. Preserve evidence; do not retry.' }

.\.venv\Scripts\python.exe -B scripts/module5_outer_fold2_continuation.py verify --outer-fold 2 --ablation A0
if ($LASTEXITCODE -ne 0) { throw 'Fold-2/A0 verification failed. Stop and return output.' }

# STOP and return A0 evidence before another target.
~~~

The native inference command above was NOT run in this session. Native completed
evidence can be reused, but there is no automatic resume after a consumed,
unfinished claim. On interruption, preserve the claim/result/journal and return
the error. Do not clear or retry it. A second exception fails this gate.

Expected result under the protocol namespace:
experiments/runs/fold2_A0_stronger_decay_seed42_677ad640faf5/outer_result.json.
Expected source/forward record counts: 2,873 (180 batches at batch size 16).
On successful native completion, only A0 changes state, with events 146 and 147;
the global CSV is refreshed. Fold-1 evidence and A1 exception stay unchanged.

Return all three console outputs, A0 outer_result.json and current events.csv.
Keep ledger.sqlite3/checkpoints in place. Stop before Fold-2/A1, aggregation,
robustness or Module 6.
