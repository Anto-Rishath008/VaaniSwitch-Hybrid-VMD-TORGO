# Fold 1 / A1 external recovery exception — authorization and manual handoff

**Implementation READY; metadata preflight PASS; recovery NOT EXECUTED.**
**Module 5 NOT SIGNED OFF. Module 6 NOT STARTED.**

This supersedes the earlier outer-panel handoff for the current action only.
Run the single recovery below, return its evidence, and stop. Other Fold 1 models,
Fold 2/3, aggregation and robustness remain outside the current authorization.

## Incident and pre-performance authorization

A0 completed normally. A1 consumed native claim
cb66ac287b054a8b8cd3e68d3d8ee385 at event 135,
2026-09-29T04:44:19.996521+00:00. Its result is absent. Windows records a restart
request about 153 seconds later; the exact inference interruption point is unknown.
The retained A1 process-lock journal is preserved.

The completed forensic audit established that **zero prior A1 outer forwards cannot
be proven**. Original native prior-forward count: **UNKNOWN**. The native evaluator
must not be rerun and its claim must not be cleared.

On 2026-10-05 the user authorized implementation of an external recovery mechanism,
before any recovered A1 performance existed, and required inference to be manual.
The original instruction is retained as authorization_request.txt, with its hash in
authorization_policy.json. This is a **PROTOCOL_EXCEPTION**. A completed output will
be marked **RECOVERED_AFTER_INTERRUPTION**, with the explicit notice:

> THIS IS NOT A NATIVE OUTER_EVALUATED RESULT

No recovered metrics exist at preparation time.

## Exact target and immutable inputs

| Field | Required value |
|---|---|
| Experiment | fold1_A1_stronger_dropout_seed42_677ad640faf5 |
| Outer fold / ablation / candidate | 1 / A1 / stronger_dropout |
| Original state | FINAL_TRAIN_COMPLETE |
| Original claim | cb66ac287b054a8b8cd3e68d3d8ee385 |
| Final epoch | 1 |
| Checkpoint SHA-256 | 5c2cc1de2f8e46d9729829f1eeb303e03d4733e86dab862ce6e85ad1c1eb34a8 |
| Outer records | 2,819, in original signed order |
| Speakers | F03, FC01, M04, MC03, MC04 |
| Ordered record hash | 2936df532e7ada33d7b571841c1026ddb30219fda3029cd89b3325f0f32d0569 |
| Threshold / seed | 0.5 / 42 |
| Runtime | torch 2.14.0+cpu; CPU threads 2 |
| Loader | batch 16; no shuffle; zero workers; no dropped records |
| Model | Frozen A1 architecture and exact final weights; evaluation mode; no gradients |

Protocol: 677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9.
Frozen code: 7696cad010914ec37f494d9f1a11a2e1761e46f353e61d97a07399f746a34ff8.

The checkpoint is the SHA-named .pt file under the original A1 final directory.
Its checksum, payload identity, selected epoch, pointer, selection and frozen
protocol are verified. Inference loads weights strictly; it never constructs an
optimizer, loads optimizer state into an optimizer, computes pos_weight, trains,
selects a model, changes thresholds or augments data.

Forensic anchors under forensics/fold1_A1_20261005T060808Z:

- forensic_audit.json: 319c562c0509142f23d4d300d3157aa0b209f5a9e1de049cbfb816b3a5562c73
- FORENSIC_REPORT.md: 6d69eed61b0c58ef701f6bf35364d75f1eaf5ed28bf785bc160f92f020918ccf
- host_evidence.json: 4f90ece527d1b88c02120010be7355d4736db7446b7dcbe45686a5c11de3ca84

## External namespace and attempt reservation

Base namespace:

~~~text
artifacts/module5/677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9/outer_recovery/fold1_A1/
~~~

Only this reserved attempt ID is accepted:

~~~text
a1_20261005T110007Z_af06fd40be7c
~~~

The ID combines a UTC preparation timestamp and random UUID suffix. It is fixed
in the authorization policy; the CLI cannot create a second ID, choose another
target, or change scientific settings. The per-attempt directory does not yet exist.

Preparation creates only policy, authorization-request copy and preparation evidence.
Preflight creates no attempt, authorization, journal, model or dataset. Immediately
before manual inference, run exclusively reserves the attempt directory, atomically
publishes authorization.json and fsyncs RECOVERY_AUTHORIZED.

The immutable authorization binds the forensic report/JSON, original ledger/CSV,
surviving lock journal, checkpoint, selection, protocol files, implementation script,
authorization policy/request hashes, production fingerprint digest, full final
identity, original event/token, target records/settings, UNKNOWN original forward
count, reason, classification and pre-performance authorization decision.

The recovery policy binds the external script checksum. Changed recovery code fails
preflight; it cannot silently resume an existing attempt.

## Preservation and preflight

Production SQLite is opened with mode=ro and query_only. No production Ledger,
evaluator, process lock or CSV exporter is instantiated. A read-only adapter supplies
the original OuterPermit get/verify methods to the unchanged data/provenance guards.

Preflight checks all 149 production files against the forensic snapshot, including
A0 result, both A1 checkpoints, all final checkpoints/selections, ledger, CSV and
the surviving journal. It checks frozen source fingerprints, frozen protocol,
247 protected predecessor file fingerprints, runtime, exact signed ordered records,
speakers, native state and claim. Unexpected production changes stop recovery.

The full raw/cache corpus is not re-audited. Each record consumed during manual
recovery still passes the original signed Module 2 source validation and batch guards.
Current source-load counts belong only to recovery, never to the unknown native pass.

No native A1 outer_result.json or OUTER_EVALUATED event will be created. Native A1
remains FINAL_TRAIN_COMPLETE with its original claim, even after external success.

## Durable journal and interruption behavior

The only execution lock is external execution.lock, released by the OS on process
exit. Concurrent writers are rejected. Native SQLite lock files are never opened.

For each batch, the script:

1. Appends BATCH_INTENT and flushes/fsyncs it **before loading or forwarding the batch**.
2. Uses the original record/speaker guards and performs the one inference batch.
3. Atomically publishes immutable batches/0000.json (and subsequent numbered files).
4. Appends BATCH_COMMITTED with the fragment checksum, then flushes/fsyncs it.

All events include attempt ID, batch index, ordered logical IDs and their hash,
checkpoint hash, timestamp and prior cumulative committed count. A hash chain links
events. RECOVERY_AUTHORIZED uses index -1 and the complete ordered record list.

**Unmatched intent, truncated journal, orphan/partial artifact or integrity mismatch:
STOP FOR REVIEW. Never replay the batch, delete evidence, edit the journal or choose
another attempt ID.** A fragment without its matching committed event remains
ambiguous even when its contents are readable.

If interruption occurs strictly between fully committed batches, explicit run
--resume skips those records without loading or forwarding them again. Model
construction failure before any intent can similarly resume with the same ID.
If all batches committed but final publication was interrupted, clean partial
finalization can finish without constructing a model; leftover temporary files
still require review. Completed attempts are verified/reused without new forwards.

An inference operation and a filesystem commit cannot form one atomic transaction.
This journal identifies uncertainty; it does not make unknown prior native exposure
or an ambiguous recovery batch disappear.

## Exact manual commands

Run from the VaaniSwitch repository, using its existing .venv. These commands do
not change dependencies. The script checks the required torch runtime.

~~~powershell
Set-Location 'C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_FINAL_Hybrid_VMD_TORGO_Project'

.\.venv\Scripts\python.exe -B scripts/module5_recover_outer_exception.py preflight --attempt-id a1_20261005T110007Z_af06fd40be7c
~~~

Only after preflight returns PASS, execute the single authorized recovery:

~~~powershell
.\.venv\Scripts\python.exe -B scripts/module5_recover_outer_exception.py run --attempt-id a1_20261005T110007Z_af06fd40be7c
~~~

After success, run the independent read-only verification command:

~~~powershell
.\.venv\Scripts\python.exe -B scripts/module5_recover_outer_exception.py verify --attempt-id a1_20261005T110007Z_af06fd40be7c
~~~

If interrupted, first run preflight again. Only if it passes with every prior batch
committed may the same attempt continue explicitly:

~~~powershell
.\.venv\Scripts\python.exe -B scripts/module5_recover_outer_exception.py run --attempt-id a1_20261005T110007Z_af06fd40be7c --resume
~~~

Any STOP FOR REVIEW means preserve the entire namespace and return the error.
Do not run the original native A1 command or the earlier Fold 1 loop.

## Workload and outputs

One CPU pass over 2,819 signed records: **177 batches = 176 x 16 + 3**.
No training. Runtime was deliberately not benchmarked on outer records; allow an
uninterrupted session, potentially minutes to hours depending on cache/OneDrive I/O.
The script prints each durable batch commit without publishing interim metrics.

On completion the attempt directory contains:

~~~text
authorization.json
journal.jsonl
batches/0000.json ... batches/0176.json
predictions.jsonl
recovery_result.json
recovery_manifest.json
~~~

Result: explicit exception classification, original identity/token/state and UNKNOWN
native forward count; recovery-only source/forward/batch counts; exact record hash;
checkpoint/selection/runtime provenance; timestamps; complete ordered predictions;
accuracy, balanced accuracy, macro-F1, precision, sensitivity, specificity, ROC-AUC,
TN/FP/FN/TP and confusion matrix; artifact hashes and native-result disclaimer.

Verifier: checks all artifacts and hash chains, ordered IDs exactly once, speakers,
labels, finite probabilities in [0,1], checkpoint identity, threshold and recomputed
metrics. It checks production fingerprints and the continued absence of any native
A1 result/event. The manifest's own checksum is returned by the verifier.

Return both run and verifier console outputs, authorization.json, journal.jsonl,
recovery_result.json and recovery_manifest.json; preserve predictions and all batch
fragments locally. Return any error and the entire attempt inventory if interrupted.

## Focused verification

- 48 tests passed in 1.70 seconds; forwards are mocked over synthetic records.
- Real torch Module calls, native evaluator/Ledger, AdamW and training were blocked.
- Tests cover exact target/settings/claim, missing forensics, checkpoint/order rejection,
  read-only SQLite, immutable publication, durable intent/commit, crash windows,
  safe resume, completed reuse, locking, loader settings, metric/provenance tampering
  and unchanged synthetic production fingerprints.
- Ruff passed for the external script and focused tests.
- Actual help command passed.
- Live metadata preflight passed in 6.0 seconds, with model calls, construction,
  dataset construction, native Ledger and AdamW instrumented to fail if invoked.
- Actual model forwards this session: **0**. Recovery attempts started: **0**.
- Evidence: the namespace's preparation/ directory.

## Future consumer rule — design only

Allow normal immutable native OUTER_EVALUATED results as NATIVE. Permit exactly
the authorization-pinned Fold 1/A1 external result as RECOVERED_PROTOCOL_EXCEPTION,
only after independent verification. Reject duplicate native/exceptional cells,
other exceptional identities, missing authorization, changed artifacts and incomplete
record coverage. Retain the flag in every table/report and attach the incident notice.

Current verification deliberately requires production to match the incident snapshot.
A future consumer, after separately authorized later native evaluations, must check
the retained verified snapshot plus append-only legitimate later events, preserving
the original A1 event/state/claim and absence of a fabricated native A1 result.
Do not weaken the recovery preflight to accommodate later evaluations.

No consumer implementation or aggregation is included here. Native full-fold report
commands remain blocked on the missing native A1 result; do not bypass that gate.

## Recommended report/paper disclosure

> Fold 1/A1 outer evaluation was interrupted after its one-use native claim.
> A host restart was documented shortly afterward, but the number of prior model
> forwards could not be established. Before observing recovered A1 performance,
> we authorized one external recovery as a protocol exception using the unchanged
> checkpoint, ordered test records, threshold and runtime. The original claim and
> evidence were preserved. If completed and independently verified, the recovered
> result is reported as RECOVERED_PROTOCOL_EXCEPTION, not as an uninterrupted native
> OUTER_EVALUATED result.

After verified completion, replace the conditional final sentence with an accurate
past-tense completion statement. Do not claim strict once-only compliance for A1.
Preserve all existing study limitations, including prior Module 4 exposure to
Fold 2/3 development speakers.

**STOP after returning recovery evidence. Module 5 is not signed off. Module 6 has
not started.**

