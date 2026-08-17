# Module 0 — TORGO Manifest, Quality Audit & Speaker Evaluation Protocol

**Target:** Day 1
**Code:** `src/vaaniswitch/m0_data/`
**Tests:** `tests/m0_data/`

## Goal
Convert TORGO into a clean logical manifest and leakage-safe speaker evaluation protocol.

## Detailed tasks
- Parse speaker/condition/session/prompt/transcript/head/array paths.
- One sample per speaker+session+prompt.
- Audit both microphones; use a structurally valid head first, otherwise a structurally valid array with explicit fallback evidence.
- Audit WAV quality.
- Build deterministic speaker-disjoint folds.
- Block microphone duplicate leakage.

## Outputs
- manifest
- speaker folds
- quality report
- leakage report

## Acceptance
- [x] Required outputs exist and pass cross-artifact validation.
- [x] Tests pass: 61 tests; Ruff and strict mypy pass.
- [x] No scientific leakage introduced: hard-gate status PASS with zero violations.
- [x] Test report completed and signed off on 2026-08-16.

## Completion summary

Module 0 produced 9,494 prompt-driven logical records from 15 speakers and 30 sessions. Four records with neither microphone remain excluded. Two zero-byte heads now use valid array primaries with explicit structural fallback evidence; the invalid heads remain audited as alternates. Deterministic three-fold speaker-group cross-validation was selected because three is the maximum fold count that keeps all four condition×gender strata in every evaluation fold.

The broad manifest exposes two later-module subset policies without starting preprocessing: CORE/CLEAN requires normal speech plus `usable`; EXTENDED requires normal speech plus `usable` or `review`. Articulation, instruction/nonstandard, unknown, and excluded records are outside the default core classification experiment. See `TEST_REPORT.md` and `artifacts/metrics/module0_dataset_summary.json` for the audited counts.

Do not start the next module automatically.
