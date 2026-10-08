# Fold2/B1 closure and B2-only manual handoff

Prepared 2026-10-06. **B1 independently VERIFIED; B2 preflight PASS; B2 inference NOT STARTED.**
Module 5 MANUAL EXPERIMENTS IN PROGRESS / NOT SIGNED OFF. Module 6 NOT STARTED.

## Immutable B1 completion

Artifact (project-relative):
~~~text
artifacts/module5/677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9/outer_continuation/fold2/b2_handoff/b1_completion.json
~~~

SHA-256: 49f27c1cca33dcb83ef98c03c68d30665744c5638eda7fe308441f29f4fe8c59

Independent verification checked the actual retained native result, claim, checkpoint,
stronger_decay epoch 1, threshold 0.5, 2,873 ordered signed records and labels,
finite probabilities and recomputed metrics. Supplied metrics matched exactly.
No inference or model construction was used. A0/A1 were independently reverified.

The snapshot contains 30 runs (9 OUTER_EVALUATED, 12 FINAL_TRAIN_COMPLETE,
9 INNER_COMPLETE), 151 native ledger events and 157 native files. All 189 recovery
files and 247 captured protected predecessor files passed preservation checks.
This was a bounded evidence check, not a repeated raw/cache corpus audit.

Fold1 closure remains pinned. Fold1/A1 remains RECOVERED_PROTOCOL_EXCEPTION:
its consumed native claim, original event, FINAL_TRAIN_COMPLETE state and absence
of native result are unchanged. Original prior native forwards remain UNKNOWN.
No uninterrupted native once-only claim is made for that cell.

## Scope and unchanged scientific identity

The new script permits only Fold2/B2 preflight or verification. Preflight requires
no native delta from the B1 snapshot. Verification requires exactly one B2
OUTER_CLAIMED event, then one OUTER_EVALUATED event, one native B2 result and
corresponding ledger/events changes. It revalidates ordered predictions, signed
labels, metrics and all prior identities; any other change fails closed.

Used A0/A1/B1 guards are byte-identical. Frozen source hashing covers
src/vaaniswitch/m[012345]*/*.py, as implemented by ExperimentContext.code_hash.
The new helper/tests and run-state documentation are outside that hash. The actual
frozen hash and protocol were checked again. No scientific protocol edits occurred.

Protocol: 677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9
Frozen code: 7696cad010914ec37f494d9f1a11a2e1761e46f353e61d97a07399f746a34ff8
Runtime: seed 42; torch 2.14.0+cpu; CPU threads 2; threshold 0.5.

B2 experiment: fold2_B2_stronger_decay_seed42_677ad640faf5
Candidate: stronger_decay. Selected epoch: 9.
Checkpoint SHA-256: 3e8a11c49869ab5ba5cc797d79c521eb5934612c38dee407e73b959e638d0910

The checkpoint was discovered from final/latest.json, verified against the native
run pointer and selection, SHA-256 checked and its saved identity/progress loaded
on CPU without constructing a scientific model.

## Validation performed

- 92 focused B2 tests PASS in 0.69 seconds.
- Ruff on the two new Python files PASS.
- New helper --help and native outer --help PASS.
- B1 verification PASS in 6.327 seconds.
- Instrumented real B2 CLI preflight PASS in 4.566 seconds.
- Actual scientific forwards: ZERO. Native/scientific writes: ZERO.

Only external helper/tests/reports and documentation were written.
Test cases use synthetic namespaces and block model construction/calls, dataset
construction, evaluator execution and native ledger construction. No full suite,
training, outer inference, ranking, tuning, aggregation or robustness ran.

## Exact sequential manual commands

Run from this project. Execute native evaluation only if preflight exits successfully.
Do not retry a failed/interrupted native evaluation or clear its claim. Preserve
all evidence and return the console output for review.

~~~powershell
Set-Location -LiteralPath 'C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_FINAL_Hybrid_VMD_TORGO_Project'
$env:PYTHONDONTWRITEBYTECODE = '1'

.\.venv\Scripts\python.exe -B scripts/module5_outer_fold2_b2_continuation.py preflight --outer-fold 2 --ablation B2
if ($LASTEXITCODE -ne 0) { throw 'B2 preflight failed. Stop and return output.' }

.\.venv\Scripts\vaaniswitch-m5.exe outer --outer-fold 2 --ablation B2
if ($LASTEXITCODE -ne 0) { throw 'B2 evaluation failed. Preserve evidence; do not retry.' }

.\.venv\Scripts\python.exe -B scripts/module5_outer_fold2_b2_continuation.py verify --outer-fold 2 --ablation B2
if ($LASTEXITCODE -ne 0) { throw 'B2 verification failed. Stop and return output.' }
# STOP before B3.
~~~

After successful B2 execution, the native namespace should add only
runs/fold2_B2_stronger_decay_seed42_677ad640faf5/outer_result.json, with the expected
ledger/events updates (events 152 claim and 153 completion). No other run changes.
Preflight then intentionally refuses the already-completed target; use verify.

Return all three console outputs, including result/claim/checkpoint SHA values
and metrics, and the B2 outer_result.json path plus verify JSON. Keep the full
ledger and events.csv. Do not run B3/C1/C2, Fold3, aggregation, robustness or Module 6.

## Complete real B2 preflight output

~~~json
{
  "ablation": "B2",
  "b1_completion_sha256": "49f27c1cca33dcb83ef98c03c68d30665744c5638eda7fe308441f29f4fe8c59",
  "candidate": "stronger_decay",
  "checkpoint_sha256": "3e8a11c49869ab5ba5cc797d79c521eb5934612c38dee407e73b959e638d0910",
  "cross_fold_aggregation": false,
  "currently_authorized_target": "Fold2/B2",
  "execution_plan_changed_by_metrics": false,
  "experiment_id": "fold2_B2_stronger_decay_seed42_677ad640faf5",
  "fold1_a1_exception_unchanged": true,
  "fold1_a1_original_prior_forward_count": "UNKNOWN",
  "fold1_a1_provenance": "RECOVERED_PROTOCOL_EXCEPTION",
  "fold1_closure_sha256": "df03075219e2aa584664b3d340d1fe537c60b6c6c588bb2451995e2b0eea362b",
  "fold1_unchanged": true,
  "fold2_a0_unchanged": true,
  "fold2_a1_unchanged": true,
  "fold2_b1_unchanged": true,
  "fold3_outer_untouched": true,
  "frozen_code_hash": "7696cad010914ec37f494d9f1a11a2e1761e46f353e61d97a07399f746a34ff8",
  "frozen_protocol_code_runtime_unchanged": true,
  "later_fold2_targets_untouched": true,
  "mode": "preflight",
  "model_forwards": 0,
  "native_event_count": 151,
  "native_file_count": 157,
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
  "selected_epoch": 9,
  "selection_sha256": "ff1c4e5a100e503dbc8e95f1772cd92dcf613a50d4f34cb90fd7b5db091cb8d5",
  "status": "PASS",
  "stop_after_verification": true,
  "target_claim": null,
  "target_state": "FINAL_TRAIN_COMPLETE",
  "threshold": 0.5
}
~~~

Retained external evidence in the B2 handoff directory:
b1_completion.json, b2_preflight.json, b2_preflight_instrumentation.json,
handoff_checks.json. The first three are published immutably with no replacement.
