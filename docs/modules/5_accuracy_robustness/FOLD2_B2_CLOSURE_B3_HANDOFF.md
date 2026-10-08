# Fold2/B2 closure and B3-only manual handoff

Prepared 2026-10-06. **B2 VERIFIED; B3 preflight PASS; B3 inference NOT STARTED.**
Module 5 MANUAL EXPERIMENTS IN PROGRESS / NOT SIGNED OFF. Module 6 NOT STARTED.

## Immutable B2 completion

Artifact (project-relative):
~~~text
artifacts/module5/677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9/outer_continuation/fold2/b3_handoff/b2_completion.json
~~~
SHA-256: 17eb5b4bd05fe961440bbcb141f9d01419ebf4e1c77a61b43363ef162ee2a64e

Retained B2 probabilities are finite; ordered signed IDs/labels and recomputed
metrics match the supplied evidence. Exact claim, result, checkpoint,
stronger_decay epoch 9 and threshold 0.5 verified. A0/A1/B1 independently reverified.

Snapshot: 30 runs (10 OUTER_EVALUATED, 11 FINAL_TRAIN_COMPLETE, 9 INNER_COMPLETE),
158 native files and 153 native events. All 189 recovery files and 247 captured
protected predecessor files pass preservation checks. This is a bounded evidence
check, not a repeated raw/cache corpus audit.

Fold1 closure remains pinned. Fold1/A1 remains RECOVERED_PROTOCOL_EXCEPTION:
original native claim/event, FINAL_TRAIN_COMPLETE state and absent native result
preserved. Original prior native forwards UNKNOWN; no uninterrupted native
once-only claim is made for this cell.

## Frozen identity and discovered target

Protocol: 677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9
Frozen code: 7696cad010914ec37f494d9f1a11a2e1761e46f353e61d97a07399f746a34ff8
Runtime: seed 42; torch 2.14.0+cpu; CPU threads 2; threshold 0.5.

B3 experiment: fold2_B3_stronger_decay_seed42_677ad640faf5
Candidate: stronger_decay. Selected epoch: 5.
Checkpoint SHA-256: 2787a6fb201c79ef2031593f0432438c19e0d9f7f5d5254b01a880b95547bd72

The actual final/latest.json pointer, checkpoint file checksum and saved identity/
progress were independently verified against the native run and frozen selection.
No scientific model was constructed.

All used helpers remain byte-identical. ExperimentContext.code_hash covers
src/vaaniswitch/m[012345]*/*.py. The new external helper/tests/docs are outside that
hash; actual frozen protocol/code/runtime checked again. No scientific edits.

## Guard and validation

Preflight permits no change from the B2 snapshot. Verification permits only B3's
OUTER_CLAIMED then OUTER_EVALUATED events, native result and ledger/events updates.
Earlier results are independently reverified, including exact ordered signed
records, labels, finite probabilities and metrics. C1/C2 and Fold3 remain untouched.

99 focused tests PASS in 0.76 seconds; Ruff and both CLI help checks PASS.
B2 verification PASS in 5.991 seconds. Instrumented real B3 preflight PASS in
4.895 seconds. Model construction/calls, dataset construction, native evaluator
and ledger construction blocked. Actual scientific forwards/native writes ZERO.
Only external helper/tests/reports and run-state docs were written. No inference,
training, tuning, ranking, aggregation, robustness or full historical suite ran.

## Exact sequential manual commands

Run native evaluation only after successful preflight. After any evaluation failure
or interruption, preserve evidence and return output; do not retry or clear claims.

~~~powershell
Set-Location -LiteralPath 'C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_FINAL_Hybrid_VMD_TORGO_Project'
$env:PYTHONDONTWRITEBYTECODE = '1'

.\.venv\Scripts\python.exe -B scripts/module5_outer_fold2_b3_continuation.py preflight --outer-fold 2 --ablation B3
if ($LASTEXITCODE -ne 0) { throw 'B3 preflight failed. Stop and return output.' }

.\.venv\Scripts\vaaniswitch-m5.exe outer --outer-fold 2 --ablation B3
if ($LASTEXITCODE -ne 0) { throw 'B3 evaluation failed. Preserve evidence; do not retry.' }

.\.venv\Scripts\python.exe -B scripts/module5_outer_fold2_b3_continuation.py verify --outer-fold 2 --ablation B3
if ($LASTEXITCODE -ne 0) { throw 'B3 verification failed. Stop and return output.' }
# STOP before C1.
~~~

Expected successful delta: event 154 (claim), 155 (completion), native result
runs/fold2_B3_stronger_decay_seed42_677ad640faf5/outer_result.json and corresponding
ledger/events updates. No unrelated changes. Preflight refuses the completed
target; use verify afterward.

Return all three console outputs and the B3 outer_result.json path/content,
including claim/checkpoint/result checksums and verification metrics. Preserve
ledger.sqlite3 and events.csv. Stop before C1/C2, Fold3, aggregation, robustness,
Module 5 signoff or Module 6.

## Complete real B3 preflight output

~~~json
{
  "ablation": "B3",
  "b2_completion_sha256": "17eb5b4bd05fe961440bbcb141f9d01419ebf4e1c77a61b43363ef162ee2a64e",
  "candidate": "stronger_decay",
  "checkpoint_sha256": "2787a6fb201c79ef2031593f0432438c19e0d9f7f5d5254b01a880b95547bd72",
  "cross_fold_aggregation": false,
  "currently_authorized_target": "Fold2/B3",
  "execution_plan_changed_by_metrics": false,
  "experiment_id": "fold2_B3_stronger_decay_seed42_677ad640faf5",
  "fold1_a1_exception_unchanged": true,
  "fold1_a1_original_prior_forward_count": "UNKNOWN",
  "fold1_a1_provenance": "RECOVERED_PROTOCOL_EXCEPTION",
  "fold1_closure_sha256": "df03075219e2aa584664b3d340d1fe537c60b6c6c588bb2451995e2b0eea362b",
  "fold1_unchanged": true,
  "fold2_a0_unchanged": true,
  "fold2_a1_unchanged": true,
  "fold2_b1_unchanged": true,
  "fold2_b2_unchanged": true,
  "fold3_outer_untouched": true,
  "frozen_code_hash": "7696cad010914ec37f494d9f1a11a2e1761e46f353e61d97a07399f746a34ff8",
  "frozen_protocol_code_runtime_unchanged": true,
  "later_fold2_targets_untouched": true,
  "mode": "preflight",
  "model_forwards": 0,
  "native_event_count": 153,
  "native_file_count": 158,
  "native_run_count": 30,
  "outer_fold": 2,
  "prior_native_results": {
    "A0": {
      "candidate": "stronger_decay",
      "checkpoint_sha256": "3dc65c5e1b57860fcd28ddad921ea7f0f4110fefd35781e3f796b5eb5269cc41",
      "claim": "950f7ce40de0443c85a26ed1f74ea13f",
      "result_sha256": "745db46496016e77586094ce1939ae3eff25926b476f4fbffaa92f3dc2c278bf",
      "selected_epoch": 11,
      "verification": "PASS"
    },
    "A1": {
      "candidate": "stronger_decay",
      "checkpoint_sha256": "6840e0688780b28ce5a103776993224880ed8137ce9122c7ba676b954ed69c67",
      "claim": "a7e181ffc4de46c39582f2720c020df6",
      "result_sha256": "13252f3660e5066c31dbd2ad0b2d0a9e6f85fd664c7a07cecce1f4f8389ff76b",
      "selected_epoch": 2,
      "verification": "PASS"
    },
    "B1": {
      "candidate": "stronger_decay",
      "checkpoint_sha256": "0325643987a235655f39da418622a09049f1ee2fc94f0c09421daac0666132d4",
      "claim": "cd7f1b81a31e41bfa737670b05f502f0",
      "result_sha256": "bfbd24a84b7eecd6685e8de1127306d230200f6a61b17500ffa0a15fae0f5375",
      "selected_epoch": 1,
      "verification": "PASS"
    },
    "B2": {
      "candidate": "stronger_decay",
      "checkpoint_sha256": "3e8a11c49869ab5ba5cc797d79c521eb5934612c38dee407e73b959e638d0910",
      "claim": "50e280d7b8f2491a9966aaeb6263c6df",
      "result_sha256": "ec05eb39d0dd1ef7e6a18cb74a5c02b96b6cd5e47900e811f24f9e2a8fffe1b2",
      "selected_epoch": 9,
      "verification": "PASS"
    }
  },
  "production_unchanged_during_check": true,
  "production_writes": 0,
  "protocol_hash": "677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9",
  "provenance": "NATIVE",
  "record_count": 2873,
  "record_hash": "e85a74cb8dd33249ac0f27d42521d5dc0cd2d9abd283970c978f278d9fb97353",
  "runtime": {
    "cpu_threads": 2,
    "seed": 42,
    "threshold": 0.5,
    "torch": "2.14.0+cpu"
  },
  "selected_epoch": 5,
  "selection_sha256": "ff1c4e5a100e503dbc8e95f1772cd92dcf613a50d4f34cb90fd7b5db091cb8d5",
  "status": "PASS",
  "stop_after_verification": true,
  "target_claim": null,
  "target_state": "FINAL_TRAIN_COMPLETE",
  "threshold": 0.5
}
~~~

Immutable evidence in the B3 handoff directory: b2_completion.json,
b3_preflight.json, b3_preflight_instrumentation.json, handoff_checks.json.
