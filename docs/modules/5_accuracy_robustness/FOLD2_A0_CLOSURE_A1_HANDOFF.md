# Fold2/A0 closure and Fold2/A1-only handoff

Date: 2026-10-06 (Asia/Calcutta).

**FOLD2/A0 INDEPENDENT VERIFICATION PASS / COMPLETION EVIDENCE FROZEN.**
**FOLD2/A1 READ-ONLY PREFLIGHT PASS / INFERENCE NOT STARTED.**
Module 5 MANUAL EXPERIMENTS IN PROGRESS / NOT SIGNED OFF.
Module 6 NOT STARTED.

## Scope and preserved scientific identity

The only next manual target is Fold2/A1. Order remains A0 complete, A1 next,
B1/B2/B3/C1/C2 later. No later target is enabled in this handoff. Observed outer
metrics did not alter the frozen candidate, epochs, threshold or execution plan.
No training, retuning, inference, cross-fold aggregation or robustness was run
in this session. A0 was verified from retained predictions, not evaluated again.

Protocol:
677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9

Frozen Modules 0–5 code:
7696cad010914ec37f494d9f1a11a2e1761e46f353e61d97a07399f746a34ff8

Runtime: seed 42, torch 2.14.0+cpu, two CPU threads, threshold 0.5.
The frozen source/protocol/runtime and all previously used helpers are unchanged.
The A0 helper was not extended or weakened; A1 uses a new external helper.

## Fold2/A0 independent verification

Experiment: fold2_A0_stronger_decay_seed42_677ad640faf5.
Provenance NATIVE; native state OUTER_EVALUATED.
Claim: 950f7ce40de0443c85a26ed1f74ea13f.
Candidate stronger_decay; selected epoch 11.
Checkpoint:
3dc65c5e1b57860fcd28ddad921ea7f0f4110fefd35781e3f796b5eb5269cc41.
Result:
745db46496016e77586094ce1939ae3eff25926b476f4fbffaa92f3dc2c278bf.

The read-only verifier checked ledger/CSV claim lifecycle, exact file checksum,
checkpoint identity, selection, threshold, all 2,873 ordered records/labels and
recomputed metrics. These match the user's supplied values. Elapsed verification:
7.624 seconds. The resulting external completion record includes the unchanged
native files/runs/events snapshot and the hashes of the already-used helpers.
The new helper pins this completion record and independently recomputes A0's
metrics on every A1 preflight and post-verification.

## Fold1 prerequisite and exceptional provenance

Fold1 closure SHA:
df03075219e2aa584664b3d340d1fe537c60b6c6c588bb2451995e2b0eea362b.

Fold1/A1 remains RECOVERED_PROTOCOL_EXCEPTION, native FINAL_TRAIN_COMPLETE with
claim cb66ac287b054a8b8cd3e68d3d8ee385, unchanged original event 135, and absent
native result. Its original prior native-forward count remains UNKNOWN.
It does not satisfy uninterrupted native once-only outer evaluation.
All recovered authorization/journal/prediction/result/manifest evidence is unchanged.
No second unfinished claim is accepted. Fold2/A1 is a separate intended NATIVE cell.

## Fold2/A1 preflight result

Experiment: fold2_A1_stronger_decay_seed42_677ad640faf5.
Candidate: stronger_decay. Selected epoch: 2.
Checkpoint:
6840e0688780b28ce5a103776993224880ed8137ce9122c7ba676b954ed69c67.

Current state FINAL_TRAIN_COMPLETE, null claim, no native outer result and no
A1 outer claim/evaluation event. Records: exactly 2,873 in signed order, hash
e85a74cb8dd33249ac0f27d42521d5dc0cd2d9abd283970c978f278d9fb97353.
Preflight PASS in 3.728 seconds. It rechecked A0, Fold1 closure and the exact
Fold1/A1 exception, all selections, A1 checkpoint and untouched later targets.

## Focused verification and preservation

- 64 focused synthetic/mocked tests PASS in 0.46 seconds; Ruff and new CLI help PASS.
- Tests reject changed A0 claim/result/checkpoint, closure/completion hash drift,
  A1 checkpoint/selection changes, prior claim, later Fold2/Fold3 activity, a second
  unfinished claim, duplicate tokens, incorrect event deltas, prediction/metric
  corruption and wrong record order/hash. Both inspection modes preserve sentinels.
- Model construction/calls, dataset construction and native Ledger/evaluator were
  blocked during both real inspections. Actual scientific forwards: ZERO.
  Native/scientific writes: ZERO; only new external evidence and code/docs.
- All 155 production files, 30 runs, 147 ledger/CSV events and 189 recovery files
  remain unchanged during this work. States: seven OUTER_EVALUATED,
  14 FINAL_TRAIN_COMPLETE, nine INNER_COMPLETE.
- All 247 captured predecessor files and frozen scientific identities remain
  unchanged. No full suite or raw/cache corpus audit was repeated.
- B1/B2/B3/C1/C2 in Fold2 and all Fold3 outer targets are untouched.

## New immutable evidence

Directory, relative to the project root:
artifacts/module5/677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9/outer_continuation/fold2/a1_handoff/

| Artifact | SHA-256 |
|---|---|
| a0_completion.json | 34fecfc92758919c8b1a22a5a59e9f0680b1da1311f609cd1f55af00df21609a |
| a1_preflight.json | 4dfd9a74396b400576640f9038c8fa18c0ad3af6deed1b1ae610013f11b09686 |

handoff_checks.json retains actual test/check output and final preservation results.
The new helper itself only prints JSON; it has no inference or save command.

## Exact manual PowerShell commands

Run sequentially, one target only. Continue only after PASS.

~~~powershell
Set-Location -LiteralPath 'C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_FINAL_Hybrid_VMD_TORGO_Project'
$env:PYTHONDONTWRITEBYTECODE = '1'

# 1. Read-only A1 preflight
.\.venv\Scripts\python.exe -B scripts/module5_outer_fold2_a1_continuation.py preflight --outer-fold 2 --ablation A1
if ($LASTEXITCODE -ne 0) { throw 'A1 preflight failed. Stop and return output.' }

# 2. Native A1 evaluation (not executed by the agent)
.\.venv\Scripts\vaaniswitch-m5.exe outer --outer-fold 2 --ablation A1
if ($LASTEXITCODE -ne 0) { throw 'A1 evaluation failed. Preserve evidence; do not retry.' }

# 3. Read-only post-evaluation verification
.\.venv\Scripts\python.exe -B scripts/module5_outer_fold2_a1_continuation.py verify --outer-fold 2 --ablation A1
if ($LASTEXITCODE -ne 0) { throw 'A1 verification failed. Stop and return output.' }

# STOP and return A1 evidence before any later target.
~~~

After successful manual inference, the new verifier accepts exactly one A1 claim
and completion (events 148 and 149), with only A1's run/result and global ledger/CSV
advanced. It requires unchanged A0, earlier native history, Fold1 exception and
all later targets. It verifies the A1 checksum/provenance, checkpoint, threshold,
ordered predictions/labels, counts and recomputed metrics without forwarding.
The older A0-only guard intentionally rejects later A1 activity; use this new
A1 verifier after A1 completes.

Expected native result under the protocol namespace:
experiments/runs/fold2_A1_stronger_decay_seed42_677ad640faf5/outer_result.json.
Expected source/forward record counts: 2,873. The native result counts records
as forwards; batch size 16 implies 180 batches.

On interruption, preserve the claim and all evidence. There is no automatic
post-claim retry/resume, and this handoff does not authorize another exception.
Return the three console outputs, A1 outer_result.json and updated events.csv.
Keep the ledger/checkpoints in place. Stop before B1, aggregation or Module 6.
