# C1 closure and fixed remaining outer sequence

Prepared 2026-10-07. **C1 independently VERIFIED; Fold2/C2 preflight PASS.**
C2 and all Fold3 native inference NOT STARTED by this handoff.
Module 5 NOT SIGNED OFF. Module 6 NOT STARTED.

## Immutable C1 anchor

Project-relative artifact:
~~~text
artifacts/module5/677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9/outer_remaining_sequence/c1_completion.json
~~~
SHA-256: a132ee00cb7398dc10d3626f90cb8523846f1111650a172b92ffaf1cc347d8d9

This artifact binds C1 identity/claim/result, recomputed metrics, checkpoint,
candidate/epoch, ordered signed records, frozen protocol/code/runtime, the eight
discovered final checkpoints, five earlier completion files, ten protected helpers
and the complete current native snapshot. It is hash-pinned in the new guard.

Production: 30 runs (12 OUTER_EVALUATED, 9 FINAL_TRAIN_COMPLETE, 9 INNER_COMPLETE),
160 native files and 157 events. All 189 recovery files and 247 captured protected
predecessor files pass preservation checks. No repeated raw/cache corpus audit.

Fold1/A1 remains RECOVERED_PROTOCOL_EXCEPTION. Its original native claim/event
and absent native result are unchanged. Original prior native forwards UNKNOWN;
no uninterrupted native once-only claim is made for that cell.

## Frozen scientific identity

Protocol: 677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9
Frozen Modules 0–5 code: 7696cad010914ec37f494d9f1a11a2e1761e46f353e61d97a07399f746a34ff8
Runtime: seed 42, torch 2.14.0+cpu, CPU threads 2, threshold 0.5.
Sequence SHA-256: 3cf6ef447c7791df2eec2823e68e36d8ad3b7719b090b6089e10ac8d3499cf48

The new external guard/tests/docs are outside ExperimentContext.code_hash, which
covers src/vaaniswitch/m[012345]*/*.py. All previously used helpers, frozen source,
datasets, splits, candidates, selected epochs and native evaluation semantics remain unchanged.

## Exact remaining order

| Step | Fold | Ablation | Candidate | Selected epoch |
|---:|---:|---|---|---:|
| 1 | 2 | C2 | stronger_decay | 2 |
| 2 | 3 | A0 | baseline | 5 |
| 3 | 3 | A1 | baseline | 5 |
| 4 | 3 | B1 | baseline | 9 |
| 5 | 3 | B2 | baseline | 6 |
| 6 | 3 | B3 | baseline | 10 |
| 7 | 3 | C1 | baseline | 5 |
| 8 | 3 | C2 | baseline | 2 |

Discovered and verified final checkpoint SHA-256 values:

~~~text
Fold2/C2: cf54f92b2e06c6c544e4b9efb91d3e248f9207f9ce773fc510fd4f23b3649d8d
Fold3/A0: 0e89f15f10dbf3998873db86251530220e9955ea9e2760192c075f194e862b4a
Fold3/A1: 4bb1a331a24082e265f885eceb58e9d94ea4eb07ce21bfa8634e7fa55ef56146
Fold3/B1: 2b2d66805322907fb06f4d47f05c4a790264d55b2cbfc96a8002bfad3088baa3
Fold3/B2: 9123550a57819f9df7237267e65da05f502fef433abaada9534ae799da9b3f89
Fold3/B3: fcff3972a783a92e137b62c50da8aa438d0508ef11e94f6babef688dc9e8bdaa
Fold3/C1: f2928c87c74eca11e8df9820afb7fe4246838725e6ddc47dc577410d02a922ae
Fold3/C2: a0c0b3520ef6f04a11e29c32dd11a65fda363b3bbd8793c82f75f1e53ac37508
~~~

Fold2: 2,873 records; hash e85a74cb8dd33249ac0f27d42521d5dc0cd2d9abd283970c978f278d9fb97353.
Fold3: 2,981 records; hash 72eedd08aa36215e3d998e0d19bdf95f655c9cc7686f83f707a5cb053fde266a.

## Generic guard and completion chain

The same script supports only status, preflight and verify. It never creates or
clears native claims, trains or evaluates a model. All invocations are sequential;
concurrent execution against this namespace is not supported.

Every invocation validates the anchor, earlier completions, protected helpers,
frozen protocol/code/runtime, Fold1 closure/exception, signed Fold2/Fold3 record
orders and final checkpoints. Every completed Fold2/Fold3 result is independently
checked for identity, claim, checksum, checkpoint, candidate/epoch, threshold,
signed IDs/speakers/labels, finite probabilities and recomputed metrics.

Preflight permits only the exact next untouched target. Verify permits exactly
that target's one claim event, one completion event, one result and required
ledger/events updates. Every unrelated run/file stays unchanged.

Successful verify publishes one external JSON at:

~~~text
artifacts/module5/677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9/outer_remaining_sequence/completions/00/<completion-sha256>.json
...
artifacts/module5/677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9/outer_remaining_sequence/completions/07/<completion-sha256>.json
~~~

Each JSON includes the preceding completion SHA, fixed target/next target,
verified result and metrics, full production snapshot, runtime hash and protected
helper hashes including this generic guard. Its filename pins its own SHA.
A completed directory is published atomically and never replaced. Missing,
modified, duplicate, unexpected or partial chain entries fail closed.
The next preflight verifies all links back to the hard-coded C1 anchor.

State handling:

- READY: run preflight for exactly next_target, then one manual native evaluation.
- AWAITING_VERIFICATION: native evidence already completed; run verify only.
- COMPLETE: all eight external completions verified. This is not Module 5 signoff.
- FAIL / STOP_FOR_REVIEW: preserve all evidence and return the output. No automatic
  claim cleanup, native retry, staging cleanup or recovery is authorized.

If verification publication was interrupted, status identifies whether the
completion was published. If it is still AWAITING_VERIFICATION with no partial
chain error, verify can be run again without inference. Never rerun the native
evaluator because verification was interrupted.

## Reusable manual PowerShell commands

Start with Fold2/C2. After successful verify, inspect next_target and manually
change only the two target variables according to the fixed table. Reuse the
same three commands; no helper rewrite or evaluation loop is needed.

~~~powershell
Set-Location -LiteralPath 'C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_FINAL_Hybrid_VMD_TORGO_Project'
$env:PYTHONDONTWRITEBYTECODE = '1'

.\.venv\Scripts\python.exe -B scripts/module5_outer_remaining_sequence.py status
if ($LASTEXITCODE -ne 0) { throw 'Status failed. Preserve evidence and stop.' }

$outerFold = 2
$ablation = 'C2'

.\.venv\Scripts\python.exe -B scripts/module5_outer_remaining_sequence.py preflight --outer-fold $outerFold --ablation $ablation
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed. Stop and return output.' }

.\.venv\Scripts\vaaniswitch-m5.exe outer --outer-fold $outerFold --ablation $ablation
if ($LASTEXITCODE -ne 0) { throw 'Native evaluation failed. Preserve evidence; do not retry.' }

.\.venv\Scripts\python.exe -B scripts/module5_outer_remaining_sequence.py verify --outer-fold $outerFold --ablation $ablation
if ($LASTEXITCODE -ne 0) { throw 'Verification failed. Preserve evidence and return output.' }
~~~

Save each console output, native outer_result.json and completion path/SHA.
After the eighth verify, return final status and all eight completion paths/SHAs.
Do not aggregate/rank results, run robustness, sign off Module 5 or start Module 6.

## Checks actually run

- 88 focused tests PASS in 3.56 seconds, including the full eight-step synthetic
  sequence, out-of-order/completed-target rejection, corruption and interruption.
- Ruff on both new Python files PASS.
- Generic --help and native outer --help PASS.
- Independent C1 verification plus eight checkpoint discoveries PASS in 12.787 seconds.
- Real status PASS in 3.867 seconds.
- Instrumented Fold2/C2 preflight PASS in 4.138 seconds.
- Actual scientific model forwards ZERO; native/scientific writes ZERO.

Only external scripts/tests/evidence/docs were written. No full suite, inference,
training, tuning, performance ranking, aggregation or robustness ran.

## Complete real Fold2/C2 preflight

~~~json
{
  "anchor_completion_sha256": "a132ee00cb7398dc10d3626f90cb8523846f1111650a172b92ffaf1cc347d8d9",
  "checked_target": {
    "ablation": "C2",
    "candidate": "stronger_decay",
    "checkpoint_sha256": "cf54f92b2e06c6c544e4b9efb91d3e248f9207f9ce773fc510fd4f23b3649d8d",
    "experiment_id": "fold2_C2_stronger_decay_seed42_677ad640faf5",
    "outer_fold": 2,
    "record_count": 2873,
    "record_hash": "e85a74cb8dd33249ac0f27d42521d5dc0cd2d9abd283970c978f278d9fb97353",
    "selected_epoch": 2,
    "selection_sha256": "ff1c4e5a100e503dbc8e95f1772cd92dcf613a50d4f34cb90fd7b5db091cb8d5"
  },
  "completed_fold2_cells_unchanged": true,
  "completed_fold2_prefix_verified": [
    "fold2_A0_stronger_decay_seed42_677ad640faf5",
    "fold2_A1_stronger_decay_seed42_677ad640faf5",
    "fold2_B1_stronger_decay_seed42_677ad640faf5",
    "fold2_B2_stronger_decay_seed42_677ad640faf5",
    "fold2_B3_stronger_decay_seed42_677ad640faf5",
    "fold2_C1_stronger_decay_seed42_677ad640faf5"
  ],
  "completed_remaining_results_verified": 0,
  "cross_fold_aggregation": false,
  "execution_plan_changed_by_metrics": false,
  "external_completion_written": false,
  "fold1_a1_exception_unchanged": true,
  "fold1_a1_original_prior_forward_count": "UNKNOWN",
  "fold1_a1_provenance": "RECOVERED_PROTOCOL_EXCEPTION",
  "fold1_closure_sha256": "df03075219e2aa584664b3d340d1fe537c60b6c6c588bb2451995e2b0eea362b",
  "fold1_unchanged": true,
  "fold3_fully_untouched": true,
  "frozen_code_hash": "7696cad010914ec37f494d9f1a11a2e1761e46f353e61d97a07399f746a34ff8",
  "frozen_protocol_code_runtime_unchanged": true,
  "later_targets_untouched": true,
  "mode": "preflight",
  "model_forwards": 0,
  "native_event_count": 157,
  "native_file_count": 160,
  "native_run_count": 30,
  "native_writes": 0,
  "next_target": {
    "ablation": "C2",
    "candidate": "stronger_decay",
    "checkpoint_sha256": "cf54f92b2e06c6c544e4b9efb91d3e248f9207f9ce773fc510fd4f23b3649d8d",
    "experiment_id": "fold2_C2_stronger_decay_seed42_677ad640faf5",
    "outer_fold": 2,
    "record_count": 2873,
    "record_hash": "e85a74cb8dd33249ac0f27d42521d5dc0cd2d9abd283970c978f278d9fb97353",
    "selected_epoch": 2,
    "selection_sha256": "ff1c4e5a100e503dbc8e95f1772cd92dcf613a50d4f34cb90fd7b5db091cb8d5"
  },
  "previous_completion_sha256": "a132ee00cb7398dc10d3626f90cb8523846f1111650a172b92ffaf1cc347d8d9",
  "production_unchanged_during_check": true,
  "protocol_hash": "677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9",
  "remaining_order": [
    "Fold2/C2",
    "Fold3/A0",
    "Fold3/A1",
    "Fold3/B1",
    "Fold3/B2",
    "Fold3/B3",
    "Fold3/C1",
    "Fold3/C2"
  ],
  "runtime": {
    "cpu_threads": 2,
    "seed": 42,
    "threshold": 0.5,
    "torch": "2.14.0+cpu"
  },
  "sequence_sha256": "3cf6ef447c7791df2eec2823e68e36d8ad3b7719b090b6089e10ac8d3499cf48",
  "sequence_state": "READY",
  "status": "PASS",
  "target_claim": null,
  "target_state": "FINAL_TRAIN_COMPLETE",
  "verified_chain_completions": 0
}
~~~

Retained external evidence in outer_remaining_sequence:
c1_completion.json, initial_status.json, fold2_c2_preflight.json,
preflight_instrumentation.json and handoff_checks.json.
No real remaining-target completion has been created yet.
