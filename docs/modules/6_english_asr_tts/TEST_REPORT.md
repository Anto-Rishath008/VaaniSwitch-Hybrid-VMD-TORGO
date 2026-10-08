# Module 6 Test Report — final validation and sign-off

**MODULE 6: COMPLETE / PASSED / SIGNED OFF. FINAL EVALUATION: COMPLETE / PASS.
HUMAN LISTENING: PASS — USER CONFIRMED.**
Frozen 130-record final panel: 130 attempted/transcribed/scored; 0 failures/exclusions.
Raw micro WER 26.11570248%, CER 18.67658575%. Five frozen TORGO sources accepted under
the explicit verified +8 outer-RIFF exception; 125 pass strictly. Original audio unchanged.
93 tests plus retained-runtime validation pass; Ruff, strict mypy (37 files), compileall,
whitespace, dependencies, preflight and pre/post preservation pass. Development ASR/TTS
reused without new execution. Correction disabled. Module 5 not executed; Module 7 not started.


| Gate | Result |
|---|---|
| Full M6 pytest | 93 passed, 1 skipped; 7.03 s |
| Retained Phase-1 evidence opt-in | 1 passed; 0.35 s; no inference |
| Exception/runner/gate synthetic and source subset | 24 passed; 1.41 s; no final model calls |
| Ruff | PASS |
| Plain strict mypy and explicit Python 3.12 strict mypy | PASS; 37 files each |
| Compileall / whitespace / dependencies | PASS |
| New preflight / freeze | PASS; 125 strict + 5 authorized exceptions; 0 forwards |
| Actual frozen final run | PASS; 130 successful / 0 failed; exactly one accepted run |
| Post-run read-only verification | PASS; 130 source/raw/reference/metric bindings and all aggregates |
| Pre/post upstream preservation | PASS; zero protected/environment/ledger/selected-WAV changes |
| Retained computational artifact verification at sign-off | PASS; SHA-256 and retained JSON consistency; no metric recalculation |
| Fresh upstream preservation at sign-off | PASS; protected files, scientific code, prior environment, ledgers, selected WAVs and checkpoints unchanged |
| Human listening | PASS — USER CONFIRMED for M03/Session2/0013 and MC02/Session1/0006 |
| Module 6 final sign-off | COMPLETE / PASSED / SIGNED OFF |

Logs and exact code identities: `artifacts/module6/exception_evaluation_20261008/QUALITY_GATES.json`.
`job_binding_before.log` and `retained_binding_before.log` preserve reproduced failures;
the final regressions pass. The user's `mypy.log` records the untyped-import failure;
`mypy_gate.log` and `mypy_py312.log` record corrected passes. `typing_patch_validation.json`
proves unchanged AST for all five typing-edited files. Junction/overwrite/immutable-raw
tests continue to pass. No coverage percentage beyond the historical report is claimed.

[Final report](FINAL_EVALUATION_REPORT.md), [user listening confirmation](HUMAN_LISTENING_REPORT.md)
and [final sign-off](FINAL_SIGNOFF_REPORT.md). The test/lint/runtime results above are
retained computational gates; they were not rerun for this documentation-only closure.
The agent did not personally listen. No code defect was introduced or fixed during sign-off.

---

Historical recovery report follows; earlier blocked statements describe that prior stage.

# Module 6 Test Report — recovery and contracts

**2026-10-08 recovery: Phase 2 PASS; recovered development Phase 3 PASS;
TTS mechanical PASS; Phase 4 / final preflight FAIL on five overdeclared RIFF headers.**
Final evaluation NOT STARTED (0 forwards). Computational Module 6 INCOMPLETE;
human listening PENDING; sign-off PENDING. All four development ASR outputs and
both TTS demos reused without new runtime execution. Correction DISABLED.
69 tests plus the separately opted-in retained-runtime test pass; Ruff, strict mypy,
compileall, whitespace, dependencies and preservation pass.


| Check | Current result |
|---|---|
| Module 6 tests | 69 passed, 1 explicit runtime test skipped; 6.01 s |
| Retained Phase-1 evidence opt-in | 1 passed; 0.43 s; no model/synthesis execution |
| Ruff | PASS; M6 source, tests and four Python scripts |
| Strict mypy | PASS; 30 files, Python 3.12 |
| Compileall / whitespace | PASS |
| Dependencies | PASS; 44 distributions in isolated .venv-m6 |
| Recovery | 4 raw transcripts, references, metrics/CSV, 2 TTS WAVs verified |
| Preservation | 124 protected / 20366 old-env / 39 ledger / 134 selected WAVs unchanged |
| Final ordered panel | Membership/counts/hash PASS; source formats FAIL on five files |
| Final preflight / freeze | FAIL / BLOCKED; no evaluation forwards |

Logs: `artifacts/module6/recovery_20261008/` (`pytest_delivery.log`,
`retained_runtime_test.log`, `ruff_delivery.log`, `mypy_delivery.log`,
`dependency_check.log`, `compileall.log`, `whitespace_delivery.log`).
No new coverage percentage is claimed. Only Module 6 checks ran; no upstream tests,
scientific execution or accepted-audit reruns occurred.

Regression evidence: `riff_regression_before.log` (1 failed / 5 passed before fix),
`contracts_regression_before.log` (ordered digest and signed CORE failures), and
the passing final suite. New tests cover RIFF bounds, generated strictness, complete
frames, panel order, CORE source, audio-only jobs, reference joins/exclusions,
normalization, failure semantics, weighted metrics/sample SD and blocked freeze rejection.

M6-RIFF-001 parser distinction CLOSED; M6-RIFF-002 source-data blocker OPEN.
M6-PANEL-001 and M6-CORE-001 CLOSED with regressions. M6-REF-001 CLOSED for the
implemented reference-eligibility scope. Existing human correction workflow remains
DISABLED / NOT EVALUATED. The final runner and interrupted-run resume are unvalidated.

[Recovery handoff](RECOVERY_REPORT.md) and [RECOVERY_VALIDATION.json](../../../artifacts/module6/recovery_20261008/recovered_20261008T111946461345Z/RECOVERY_VALIDATION.json).

---

Historical Phase-1 report below is preserved as prior-stage evidence. Its stop boundary
and future-work statements describe that earlier stage, not current authorization.

# Module 6 Test Report — Phase 1 runtime foundation

**2026-10-08: Phase 0 PASS; Phase 1 PASS (CUDA). Implementation IN PROGRESS; NOT SIGNED OFF.**

See [runtime report](PHASE1_RUNTIME_REPORT.md), [machine-readable evidence](PHASE1_RUNTIME_EVIDENCE.json)
and the preserved [Phase-0 plan](PHASE0_AUDIT_AND_PLAN.md).

## Executed checks

| Check | Result |
|---|---|
| Ordinary synthetic M6 pytest | 45 passed, 1 explicit runtime test skipped; 6.27 s |
| Opt-in retained actual runtime evidence | 1 passed; 0.29 s; no new inference |
| Ruff | PASS |
| Strict mypy, explicit Python 3.12 | PASS; 16 files |
| Isolated dependency compatibility | PASS; 44 installed distributions |
| Actual CLI help | PASS |
| Checksum before model loading | Official small.en digest exactly matches; mismatch rejection tested |
| Actual CUDA model smoke | RTX 4070 tensor allocation, model loading and full development transcription PASS |
| Actual TTS smoke | Exact Zira token; valid RIFF/WAVE PCM16 mono 22050 Hz, 80675 frames, 3.658730 s |
| Raw output preservation | Exact leading-space transcript and backend JSON retained; reused on TTS-only retry |
| Protected paths | Traversal, predecessor paths, Windows junctions, hardlinks and overwrite rejection PASS |
| Upstream preservation | 124 Phase-0 files, 20366 prior environment files, 39 ledger files, 134 planned WAVs unchanged |

Synthetic coverage is 54.592721% of statements plus branches (257/437 lines, 58/140 branches).
The real model/acquisition/SAPI subprocess smoke ran separately and is not in that percentage.
The Phase-1 configuration explicitly scopes coverage to M6; no predecessor coverage threshold
or full-module closure is asserted. No full M0-M5 test/audit was rerun.

## Regressions and retained failures

- M6-CONFIG-001: 3 malformed-execution cases failed, then passed after mapping validation.
- M6-TTS-001: original real save timed out at 60 seconds; synthetic idle-proxy lifecycle
  test failed, then passed. Public external-loop settings flush repairs queue ordering.
  Real retry passed; original failed run/partial WAV retained; original ASR not repeated.
- M6-IO-001: first recovery failed on Windows CP1252 JSON decoding before any worker.
  Forced-CP1252 fixture failed, then byte-based JSON reading passed. Real recovery passed.
- M6-REF-001 remains OPEN for future reference-eligibility implementation.

## Commands

```powershell
.venv-m6/Scripts/python.exe -B -m pytest tests/m6_reconstruction -q --cov=vaaniswitch.m6_reconstruction --cov-config=configs/module6/environment/coverage.ini --cov-report=term-missing --cov-report=json:artifacts/module6/setup/coverage.json
# Explicit retained-runtime validation sets M6_RUNTIME_EVIDENCE to the successful run path.
.venv-m6/Scripts/python.exe -B -m pytest tests/m6_reconstruction/test_runtime_smoke.py -q
.venv-m6/Scripts/ruff.exe check src/vaaniswitch/m6_reconstruction tests/m6_reconstruction scripts/module6_runtime.py scripts/module6_preservation.py
.venv-m6/Scripts/python.exe -B -m mypy --strict --python-version 3.12 --follow-imports silent --ignore-missing-imports src/vaaniswitch/m6_reconstruction tests/m6_reconstruction scripts/module6_runtime.py scripts/module6_preservation.py
uv pip check --python .venv-m6/Scripts/python.exe
python -B scripts/module6_preservation.py
```

The separate actual runtime commands and retained attempt paths are in the runtime report.
Logs are in `artifacts/module6/setup/`. Ordinary pytest does not download a model or invoke
real Whisper/SAPI; the explicit runtime test validates sealed smoke evidence only.

## Gate

Phase-1 gates A-P PASS. No evaluation-panel inference, WER/CER, human correction evaluation,
Module 5 execution or Module 7 work. Human listening and later M6 functionality remain pending.
**Module 6 NOT SIGNED OFF. STOP after Phase 1.**
