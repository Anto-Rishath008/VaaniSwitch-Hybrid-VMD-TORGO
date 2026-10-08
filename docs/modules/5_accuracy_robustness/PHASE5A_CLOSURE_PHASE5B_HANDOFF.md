# Module 5 — Phase 5A closure and Phase 5B manual handoff

## Current state (2026-09-22)

- Phase 5A: **COMPLETE / PASSED**.
- Phase 5B: **NOT STARTED / READY FOR MANUAL EXECUTION**.
- Module 5: **MANUAL EXPERIMENTS IN PROGRESS / NOT SIGNED OFF**.
- Module 6: **NOT STARTED**.

## Actual Phase 5A evidence

Audit command executed:

```powershell
uv run --python 3.12 --extra dev python scripts/audit_module5_phase5a.py
```

The audit is read-only for A0, raw TORGO, Modules 1–4 and frozen protocol/package code.
It writes only new Module 5 audit evidence. No A0 transform was recomputed.
The successful audit took 290.1164698 seconds.

Artifact:
`artifacts/module5/677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9/phase5a_audit/20260922T155051Z/phase5a_audit.json`.
Sibling artifacts: `a0_protocol_bindings.csv`, `protected_inputs.json`.
Console: `artifacts/module5/phase5a_audit_console.log`.

| Identity | Exact value |
|---|---|
| Protocol | `677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9` |
| Frozen Modules 0–5 code | `7696cad010914ec37f494d9f1a11a2e1761e46f353e61d97a07399f746a34ff8` |
| A0 config | `9b57afd31f1b5d7d7c03c4d82828f3479bf0e115a0a35d3a262c0d190337f8d4` |
| Architecture | `signed_D128_heads4_all_modes_K5_nodes256_canvas8s` |

A0 namespace:
`data/processed/module5_ablation/no_vmd_mel/9b57afd31f1b5d7d7c03c4d82828f3479bf0e115a0a35d3a262c0d190337f8d4/`.

| Check | Verified result |
|---|---|
| Retained build | selected 8,673; computed 8,667; cache hits 6; validated 8,673; failed 0; 370.9191407999897 s |
| Retained validation | selected/validated 8,673; computed/cache hits/failed 0; 108.2358358999918 s |
| Logical directories / Mel / metadata | 8,673 / 8,673 / 8,673 |
| Files / total bytes | 17,346 / 1,763,168,040 |
| Unexpected or partial / zero-byte required files | 0 / 0 |
| Separate directory check | 8,718 actual and expected directories including speaker/session containers; no extras |
| All Mel arrays | float32 [224,224], all finite; min -100.0, max 2.316357374191284 |
| Exact signed-record bindings / live waveform SHA-256 checks | 8,673 / 8,673 |
| Both production failure CSVs | zero rows |
| Production ledger runs / outer claims / outer events | 0 / 0 / 0 |
| Outer model forwards / audit model forwards | 0 / 0 |
| M5-CODE-001 | CLOSED; original exact-primary and changed-waveform regressions passed |

The frozen schema uses existence of `protocol.json` plus complete equality with
`require_frozen`; it has no standalone frozen boolean. That exact lock passed.
Per-entry A0 metadata directly binds the signed record/primary, manifest and A0 config.
Its experiment-protocol binding is transitive through the retained full-run reports
and the frozen protocol. The new checksummed, 8,673-row audit binding table makes
that chain explicit without modifying any original cache metadata.

Protected inputs:

| Input | Files | Bytes | Result |
|---|---:|---:|---|
| Raw TORGO | 82,772 | 15,845,129,970 | Unchanged |
| Module 1 cache | 26,019 | 19,981,366,416 | Unchanged |
| Module 2 cache | 43,365 | 2,378,359,184 | Unchanged; 8,673 logical directories |
| Modules 0–4 source/config/manifest/artifacts, including Modules 3 and 4 | 247 protected files | Per-file evidence retained | All unchanged |

Tree checks compare path/size/mtime fingerprints. The 247 protected files additionally
match content SHA-256. All 8,673 selected primary waveform bytes were freshly hashed.
This is not a new bytewise hash of every file in the three large trees.

## Phase 5B design and frozen-code boundary

Run **12 C2 inner runs**, sequentially: Fold 1, then Fold 2, then Fold 3, each with
`baseline`, `lower_lr`, `stronger_decay`, `stronger_dropout`.
The existing `search` command executes the exact panel and automatically selects its winner.
There is no separate `select` subcommand and no candidate argument to this CLI.

Each run uses only that fold's inner-training speakers for fitting and pos_weight,
only its inner-validation speakers for early stopping/comparison, seed 42,
threshold 0.5, frozen backbone, AdamW, batch 16, maximum 30 epochs, patience 5,
minimum macro-F1 improvement 0.0001, and no augmentation. Candidate values and
split identities remain those in the frozen protocol; do not change them between folds.
Dataset/batch guards and `run_epoch` reject outer-test use. Search neither calls
final retraining nor obtains an outer permit.

Selection maximizes macro-F1, balanced accuracy and valid ROC-AUC, then minimizes
validation loss and trainable parameter count; exact ties use ascending candidate ID.
Early-stopping patience uses macro-F1, independently of these best-epoch tie-breakers.

The frozen hash covers only `src/vaaniswitch/m[012345]*/*.py`.
The new audit/reporter scripts, tests and run-state documentation are outside that
set. No package, CLI, candidate, protocol or training behavior changed. Current code
hash was checked against the original frozen value after these additions.

`scripts/report_module5_phase5b.py` is an external retained-evidence reader:
read-only SQLite access, checksummed restricted checkpoint load, exact identity,
train-only weight and epoch-role checks, recomputed four-candidate selection, and
tie-break explanations. It never instantiates a model or calls training/search.
`--save` creates only a separate immutable report; an identical rerun is a no-op,
and conflicting existing reports are preserved and rejected.

## Exact manual commands

Run from the VaaniSwitch project root. Execute these **sequentially**, stopping on
any nonzero exit code. Concurrent fold execution against the shared event export
has not been validated and is not recommended.

```powershell
Set-Location 'C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_FINAL_Hybrid_VMD_TORGO_Project'

# Fold 1: four candidates, automatic winner selection, then report.
uv run --python 3.12 --extra dev vaaniswitch-m5 search --outer-fold 1 --resume
if ($LASTEXITCODE -ne 0) { throw 'Fold 1 search failed; stop and review.' }
uv run --python 3.12 --extra dev python scripts/report_module5_phase5b.py --outer-fold 1 --save
if ($LASTEXITCODE -ne 0) { throw 'Fold 1 evidence report failed; stop and review.' }

# Fold 2
uv run --python 3.12 --extra dev vaaniswitch-m5 search --outer-fold 2 --resume
if ($LASTEXITCODE -ne 0) { throw 'Fold 2 search failed; stop and review.' }
uv run --python 3.12 --extra dev python scripts/report_module5_phase5b.py --outer-fold 2 --save
if ($LASTEXITCODE -ne 0) { throw 'Fold 2 evidence report failed; stop and review.' }

# Fold 3
uv run --python 3.12 --extra dev vaaniswitch-m5 search --outer-fold 3 --resume
if ($LASTEXITCODE -ne 0) { throw 'Fold 3 search failed; stop and review.' }
uv run --python 3.12 --extra dev python scripts/report_module5_phase5b.py --outer-fold 3 --save
if ($LASTEXITCODE -ne 0) { throw 'Fold 3 evidence report failed; stop and review.' }

uv run --python 3.12 --extra dev vaaniswitch-m5 ledger
```

### Resume and explicit failure recovery

The same `search --outer-fold N --resume` command starts an untouched fold or resumes
its last completed epoch with model, AdamW and RNG state. An interrupted unsaved
epoch is replayed; a run interrupted before its first checkpoint restarts from seed 42.
Completed candidates are reused and completed selection is checked against all four
retained results. The experiment identity includes fold, C2, candidate, seed, protocol,
code, split/records, predecessors and runtime. Drift is rejected.

If the ledger explicitly records FAILED, inspect and resolve its recorded cause,
then supply a factual reason. For example, replacing N with the affected fold:

```powershell
uv run --python 3.12 --extra dev vaaniswitch-m5 search --outer-fold 1 --resume --retry-failed "Transient interruption resolved; code/config/data unchanged"
```

Do not change code or scientific settings to force recovery. OS-managed SQLite locks
release after termination; no manual lock deletion is required.

## Expected evidence and what to return

Under `artifacts/module5/<protocol-hash>/experiments/`:

- `runs/foldN_C2_<candidate>_seed42_677ad640faf5/inner/latest.json` and its checksummed
  `<sha256>.pt`: identity, epoch history, train-only weight, best inner metrics/epoch,
  model, AdamW and RNG. The retained weights are the latest resumable state; the best
  inner epoch is metadata, since later final retraining starts fresh.
- `ledger.sqlite3`, `events.csv`: exact run identities, states and selection/recovery events.
  After each fold, its winner is SELECTED and its other three candidates INNER_COMPLETE.
- `selection/foldN_c2.json`: automatically frozen winner and selected epoch/metrics.
- `reports/foldN_phase5b.json`: all four experiment identities, split/speakers, seed,
  pos_weight, completed/best epochs, requested validation metrics including TN/FP/FN/TP,
  checkpoint paths, ledger states, measured train/validation stage times and tie-break reasoning.

The reporter's outer-forward count is an explicit derived audit of the frozen inner-only
execution path, matching epoch roles/record hashes and absence of outer claims/events;
the existing training code has no separate runtime counter. Recorded wall time sums
epoch training/validation stages, excluding unmeasured setup and interrupted unsaved work.

Return the three search console outputs, all three selection JSONs, all three
Phase 5B report JSONs, the final ledger console output and any failure/recovery messages.
Keep checkpoint files and the SQLite ledger locally for verification. Expected total:
12 completed C2 inner runs, three selected winners, zero final retrains/outer claims/events.

**Stop after Phase 5B selection.** No C2 final retrain, A0/A1/B1/B2/B3/C1 training,
outer evaluation, robustness or Module 6 is authorized in this handoff.
