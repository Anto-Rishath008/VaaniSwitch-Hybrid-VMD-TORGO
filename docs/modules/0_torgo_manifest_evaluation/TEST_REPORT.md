# Module 0 Test Report — TORGO Manifest, Quality Audit & Speaker Evaluation Protocol

**Status:** PASSED  
**Completed:** 2026-08-16  
**Scope:** Module 0 only; no resampling, VMD, feature generation, ASR, or model training was run.

## Requirement coverage

| Requirement | Evidence | Result |
|---|---|---|
| Prompt-driven canonical record | 9,494 unique `speaker_id/session_id/prompt_id` rows; zero duplicate logical IDs | PASS |
| Canonical labels | F/M=dysarthric, FC/MC=control; group/prefix/condition/gender cross-validated in config and schema | PASS |
| Real transcripts and conservative prompt types | All 9,494 prompt files decoded from disk; 6,938 word, 2,287 sentence, 118 articulation, 92 instruction/nonstandard, 59 unknown | PASS |
| Validity-aware head-preferred microphone policy | 8,212 selected head, 1,278 selected array, 4 with no structurally usable audio; audit evidence and alternates remain bound to their logical record | PASS |
| WAV quality audit | All 17,635 direct microphone WAVs header-, signal-, size-, and SHA-256-audited | PASS |
| Raw immutability | Before/after metadata snapshots identical; raw metadata SHA-256 `206041a4be1fad47975d25ba38c73bb49fe68e3957ebfb046d0db15c0b14c76a` | PASS |
| Speaker-disjoint evaluation | Deterministic three-fold grouped CV, seed 42; every condition×gender stratum in every evaluation fold | PASS |
| Leakage hard gate | Zero speaker, logical-ID, audio-path, microphone-copy, cross-speaker-hash, or cross-fold-hash violations | PASS |
| Reproducible artifacts | JSONL/CSV parity, cross-artifact fingerprints, headers, row totals, relative paths, and required output presence validated | PASS |

## Verified TORGO inventory

| Metric | Result |
|---|---:|
| Raw extracted files | 82,772 |
| Direct microphone WAVs | 17,635 |
| Readable / unreadable WAVs | 17,633 / 2 |
| Speakers | 15: 8 dysarthric, 7 control |
| Condition×gender speakers | DF 3, DM 5, CF 3, CM 4 |
| Sessions | 30 |
| Logical utterances | 9,494 |
| Usable / review / excluded | 8,863 / 627 / 4 |
| Normal-speech eligible | 9,222 |
| Orphan microphone paths | 381 paths / 379 logical stems |
| Discovery warnings | 0 |

Microphone availability is 7,764 both, 450 head-only, 1,276 array-only, and 4 neither. Validity-aware primary selection produces 8,212 head primaries, 1,278 array primaries, and 4 records without structurally usable audio. The array total comprises 1,276 head-missing fallbacks and 2 invalid-head fallbacks. Direct folders contain 8,218 head WAVs and 9,417 array WAVs. The 381 orphan paths comprise 4 head and 377 array files; 377 lack prompt metadata and 4 have malformed audio filenames or prompt IDs.

## Audio-property audit

All 17,633 readable WAVs are uncompressed PCM, mono, 16 kHz, and 16-bit. Selected-primary results are:

| Property | Result |
|---|---:|
| Selected primary paths | 9,490 |
| Readable / structurally usable | 9,490 / 9,490 |
| Zero-byte selected primaries | 0 |
| Total duration | 29,438.7513125 s / 8.17743092 h |
| Duration min / median / mean / max | 0.00825 / 2.495 / 3.102081276 / 193.828625 s |
| Under 0.1 / 0.25 / 0.5 / 1.0 s | 32 / 33 / 55 / 140 |
| Over 30 s | 6 |
| Near-silent at or below -60 dBFS | 78 |
| At or below -50 / -40 dBFS | 383 / 465 |
| Full-scale-sample indicator | 3,314 |
| Possible clipping at configured ≥1% fraction | 462 |
| Clipping fraction ≥5% / ≥10% | 26 / 2 |

Shortness, low level, and clipping are review indicators, not automatic deletions. The zero-byte heads for `F01/Session1/0067` and `F01/Session1/0068` remain in the full raw-audio audit as structurally unusable alternate paths. Their valid array recordings are now the primaries, with `head_unusable_array_fallback` provenance and normalized head rejection issues `unreadable|zero_byte`.

## Exclusions

- `F03/Session2/0117`, `M04/Session1/0066`, `MC04/Session1/0655`, `MC04/Session1/0656`: `missing_primary_audio`.

No prompt-derived logical record was synthesized from an orphan microphone file.

## Speaker-fold allocation

Five folds were not forced. The smallest condition×gender speaker strata contain three speakers, so three is the largest fold count that preserves dysarthric female, dysarthric male, control female, and control male representation in every evaluation fold.

| Fold | Evaluation speakers | Logical records | Non-excluded balance records |
|---:|---|---:|---:|
| 1 | F03, FC01, M04, MC03, MC04 | 3,096 | 3,092 |
| 2 | F04, FC02, M01, M02, MC02 | 3,211 | 3,211 |
| 3 | F01, FC03, M03, M05, MC01 | 3,187 | 3,187 |

Every speaker is evaluated exactly once. For every fold, train/evaluation speaker, logical-ID, and associated primary/alternate audio-path overlap counts are zero.

## Duplicate and leakage audit

- Duplicate logical IDs: 0.
- Duplicate primary paths: 0.
- Audio paths assigned to multiple logical records: 0.
- Alternate paths used as another record's primary: 0.
- Duplicate SHA-256 groups: 3, all confined to one speaker.
- Cross-speaker duplicate SHA-256 groups: 0.
- Cross-fold duplicate-content violations: 0.
- Leakage status: **PASS**, violation count 0.

The allowed same-speaker duplicate groups are the two zero-byte F01 head alternates, F03 Session3 array 0061/0062, and FC02 Session3 array 0765/0767. Speaker grouping keeps each group on one side of every train/evaluation split.

## Downstream subset policy

The broad Module 0 manifest retains all review records. Later classification modules must select records as follows without preprocessing them in Module 0:

| Subset | Criteria | Recomputed records | Role |
|---|---|---:|---|
| CORE/CLEAN (default) | `normal_speech_eligible == true` and `usability_status == usable` | 8,673 | Default clean classification experiment |
| EXTENDED | `normal_speech_eligible == true` and `usability_status in {usable, review}` | 9,222 | Quality-flag robustness ablation |

Both subsets are restricted to word and sentence prompts. Excluded records, articulation tasks, instruction/nonstandard prompts, and unknown prompts are outside the default core experiment.

## Commands and results

```text
uv run --python 3.12 --extra dev vaaniswitch-m0 build --config configs/data/torgo.yaml
# PASS: 9,494 records; 4 exclusions; leakage PASS (154.0 s final correction run)

uv run --python 3.12 --extra dev vaaniswitch-m0 validate --config configs/data/torgo.yaml
# PASS: manifest SHA-256 91c3327ea1f6a6b2057ffa02198f86d682668d22f247beef27f1466e959a9061

uv run --python 3.12 --extra dev pytest
# PASS: 61 tests

uv run --python 3.12 --extra dev pytest --cov=vaaniswitch.m0_data --cov-branch --cov-report=term-missing
# PASS: 61 tests; 81.70% total branch-aware coverage

uv run --python 3.12 --extra dev ruff check .
# PASS

uv run --python 3.12 --extra dev mypy src/vaaniswitch/m0_data tests/m0_data
# PASS: no issues

uv run --python 3.12 python -m compileall -q src tests
# PASS
```

## Limitations

- TORGO has only 15 independent speakers; fold metrics must be reported per fold and aggregated with uncertainty.
- Raw immutability verification fingerprints file paths, sizes, and modification times. It is metadata-level verification, not a 15.8 GB full-content snapshot.
- The 627 review records remain in the broad manifest; they enter only the documented EXTENDED robustness ablation, not the default CORE/CLEAN subset.
- Repeated prompt text across different speakers is expected and allowed under the speaker-disjoint policy.
- Condition and gender use documented TORGO directory grouping. No medical severity labels were inferred.

## Gate

✅ SIGNED OFF — Module 0 is complete. Module 1 has not been started.
