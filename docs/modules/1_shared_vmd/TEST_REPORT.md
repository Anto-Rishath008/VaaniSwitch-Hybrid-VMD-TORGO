# Module 1 Test Report — Shared VMD Decomposition & Safe Reconstruction

**Status:** ✅ COMPLETE / PASSED / SIGNED OFF  
**Full CORE gate:** PASSED — 8,673/8,673 valid cache entries, zero failures  
**Module 2:** NOT STARTED

## Production contract

- Input: only Module 0 `primary_audio_path` from the CORE subset.
- Working audio: finite float64 mono at 16 kHz with conservative RMS normalization
  (target 0.1, maximum gain 10×, peak limit 0.99).
- VMD: project-owned pure-NumPy `vaaniswitch_numpy_vmd_v1`, NumPy 1.26.4.
- Parameters: K=5, alpha=2000, tau=0, DC=false, init=1, tolerance=1e-7,
  `max_iterations=500`.
- Reconstruction: sum all five modes. Selected-mode ablation is disabled.
- Production config hash:
  `bfd7aec36d99a88cfacc72df85e124d3f4b99b76ed47d71c3ced9a153fa92685`.
- Source manifest fingerprint:
  `91c3327ea1f6a6b2057ffa02198f86d682668d22f247beef27f1466e959a9061`.
- Raw TORGO metadata fingerprint:
  `206041a4be1fad47975d25ba38c73bb49fe68e3957ebfb046d0db15c0b14c76a`.

## Full CORE completion

| Item | Result |
|---|---:|
| Manifest-selected CORE records | 8,673 |
| Successful cache records | 8,673 |
| Computed during full run | 8,667 |
| Valid prior smoke-cache hits | 6 |
| Failed records | 0 |
| Solver-update converged | 8,625 |
| Reached 500-iteration ceiling | 48 |
| Full validation | PASS, 8,673/8,673 |

The header-only failure report confirms zero failed records. The independent final audit
matched every cache logical ID to the CORE manifest, Module 0 primary path and SHA-256,
production config hash, source manifest fingerprint, condition, speaker, session,
prompt, and primary microphone. The full project validator then reloaded and hashed all
arrays and source primaries and recomputed scientific semantics.

## Full reconstruction distributions

These are measured over all 8,673 production records, not smoke estimates.

| Metric | Minimum | Mean | Median | Maximum |
|---|---:|---:|---:|---:|
| Normalized reconstruction error | 0.0136078 | 0.1206379 | 0.1135749 | 0.3874861 |
| Correlation coefficient | 0.9363745 | 0.9930800 | 0.9946054 | 0.9999077 |
| Energy ratio | 0.5910814 | 0.8949182 | 0.9089716 | 0.9982708 |
| RMSE | 0.000926585 | 0.010794245 | 0.010286168 | 0.031356758 |
| MSE | 8.58560e-7 | 1.37316e-4 | 1.05805e-4 | 9.83246e-4 |
| Iterations | 7 | 93.4610 | 69 | 500 |
| VMD seconds per record | 0.015981 | 1.307851 | 0.547787 | 306.630503 |
| Total seconds per record | 0.062410 | 1.406171 | 0.640392 | 315.546370 |

The cached reconstruction is the exact float64 numerical sum of all five cached modes.
These metrics do not demonstrate perfect or lossless reconstruction, denoising, noise
removal, or guaranteed preservation of every dysarthria cue.

## Iteration-ceiling audit

Exactly 48 records did not reach the strict update tolerance before iteration 500. All
48 stopped at iteration 500. Distribution by speaker is F03=2, FC02=3, M02=8, M03=5,
MC01=15, MC02=9, MC03=5, and MC04=1; by condition it is dysarthric=15 and control=33.

Logical IDs:

- F03: `F03/Session1/0179`, `F03/Session2/0063`
- FC02: `FC02/Session3/0233`, `FC02/Session3/0234`, `FC02/Session3/0607`
- M02: `M02/Session1/0036`, `M02/Session1/0062`, `M02/Session1/0076`,
  `M02/Session1/0114`, `M02/Session1/0131`, `M02/Session1/0134`,
  `M02/Session1/0159`, `M02/Session1/0182`
- M03: `M03/Session2/0024`, `M03/Session2/0027`, `M03/Session2/0088`,
  `M03/Session2/0238`, `M03/Session2/0252`
- MC01: `MC01/Session1/0006`, `MC01/Session1/0140`, `MC01/Session1/0169`,
  `MC01/Session2/0109`, `MC01/Session2/0112`, `MC01/Session2/0117`,
  `MC01/Session2/0360`, `MC01/Session3/0004`, `MC01/Session3/0014`,
  `MC01/Session3/0082`, `MC01/Session3/0094`, `MC01/Session3/0128`,
  `MC01/Session3/0334`, `MC01/Session3/0358`, `MC01/Session3/0364`
- MC02: `MC02/Session1/0022`, `MC02/Session1/0076`, `MC02/Session1/0111`,
  `MC02/Session1/0141`, `MC02/Session1/0324`, `MC02/Session1/0367`,
  `MC02/Session2/0028`, `MC02/Session2/0040`, `MC02/Session2/0237`
- MC03: `MC03/Session1/0010`, `MC03/Session1/0120`, `MC03/Session1/0232`,
  `MC03/Session1/0581`, `MC03/Session2/0038`
- MC04: `MC04/Session2/0954`

| Metric for 48 records | Minimum | Mean | Median | Maximum |
|---|---:|---:|---:|---:|
| Final delta | 1.01568e-7 | 3.72832e-5 | 1.07890e-6 | 9.67153e-4 |
| Normalized reconstruction error | 0.0301693 | 0.0791779 | 0.0779644 | 0.1425047 |
| Correlation coefficient | 0.9906044 | 0.9967889 | 0.9971747 | 0.9996033 |
| Energy ratio | 0.8961149 | 0.9512892 | 0.9565827 | 0.9858543 |
| RMSE | 0.000926585 | 0.007610607 | 0.007605593 | 0.012646300 |
| MSE | 8.58560e-7 | 6.55950e-5 | 5.78460e-5 | 1.59929e-4 |
| Iterations | 500 | 500 | 500 | 500 |

For all 48, the audit found no NaN/Inf, malformed array, shape error, invalid center
frequency, cache corruption, source-primary mismatch, or reconstruction inconsistency.
Every source primary SHA-256 was rechecked, every reconstruction metric was recomputed,
and every reconstruction exactly equalled the float64 sum of its modes.

Accepted scientific limitation: **VMD update criterion did not reach tolerance before
the configured 500-iteration ceiling, but output remained finite, structurally valid,
cache-valid, and reconstruction-valid.** Solver-update convergence is not equivalent to
reconstruction correctness. No record was removed, recomputed under a different tau, or
excluded for this reason.

## Measured runtime

- Full pipeline wall time: 12,219.2642315 s = 3.3942401 h.
- Aggregate per-record VMD time: 11,342.9938652 s = 3.1508316 h.
- Aggregate per-record processing time: 12,195.7190227 s.
- Mean/median VMD time: 1.3078512 / 0.5477874 s per record.
- Maximum VMD time: 306.6305027 s for `F03/Session1/0118` (179 iterations,
  3,101,258 samples); its total record time was 315.5463702 s.

Earlier 1.44 h, 1.80 h, and 3.49 h figures were pre-build estimates. They are retained
only as historical planning evidence in the smoke/tau reports; the values above are the
measured full-corpus result.

## Cache inventory

The active production directory is:

`data/processed/vmd/bfd7aec36d99a88cfacc72df85e124d3f4b99b76ed47d71c3ced9a153fa92685/`

| Item | Count/value |
|---|---:|
| Logical cache directories | 8,673 |
| `modes.npy` | 8,673 |
| `reconstruction.npy` | 8,673 |
| `metadata.json` | 8,673 |
| Total files | 26,019 |
| Total bytes | 19,981,366,416 |
| Total GiB | 18.6090976 |
| Temporary/partial files | 0 |
| Unexpected files | 0 |

The nominal array-only estimate was 18.5693150 GiB. Four separate 12-record tau-sweep
namespaces remain under `data/processed/vmd_fidelity_sweep/` and total 117,019,024 bytes
(0.1089825 GiB). They are experimental evidence, not production caches, and were not
deleted.

## Historical smoke and tau evidence

Before the full build, a deterministic six-record smoke cache (three dysarthric, three
control; four head and two array primaries) passed strict validation with zero failures.
Its normalized error range was 0.081341–0.177075, correlation 0.987492–0.997057, and
energy ratio 0.810321–0.940647. Those values were smoke observations, not full-corpus
statistics.

A fixed 12-record panel then compared tau 0, 0.1, 0.25, and 0.5. Tau=0.1 improved mean
numerical fidelity but converged for only 4/12 records and was 2.67× slower than tau=0;
tau=0.25 and 0.5 converged for 2/12 and 1/12. The production decision remained tau=0.
Full historical evidence is in `TAU_FIDELITY_REPORT.md` and the 48-row sweep CSV.

## Final commands and results

```text
uv run --python 3.12 --extra dev vaaniswitch-m1 validate --config configs/vmd/default.yaml --subset core
PASS; 8,673 selected; 8,673 valid caches; 8,625 converged

uv run --python 3.12 --extra dev python scripts/audit_module1_full_build.py
PASS; 8,673 source-bound cache entries; 48/48 ceiling records numerically valid

uv run --python 3.12 --extra dev pytest
144 passed in 5.02s

uv run --python 3.12 --extra dev pytest --cov=vaaniswitch.m0_data --cov=vaaniswitch.m1_vmd --cov-branch --cov-report=term-missing
144 passed; 81.70% branch-aware coverage; 80% gate passed

uv run --python 3.12 --extra dev ruff check .
All checks passed!

uv run --python 3.12 --extra dev mypy src/vaaniswitch/m0_data src/vaaniswitch/m1_vmd tests/m0_data tests/m1_vmd
Success: no issues found in 43 source files

uv run --python 3.12 python -m compileall -q src tests
PASS
```

The full suite includes all 61 Module 0 tests, which remain green. The raw TORGO
fingerprint and signed Module 0 manifest fingerprint exactly match their pre-build
values. No production output resides under `data/raw`, and the Module 0 reports and
speaker folds were not overwritten.

## Final artifacts

- `artifacts/metrics/module1_vmd_summary.json`
- `artifacts/metrics/module1_full_signoff_audit.json`
- `artifacts/logs/module1_vmd_failures.csv`
- `artifacts/logs/module1_nonconverged_records.csv`
- `artifacts/logs/module1_cache_inventory.json`
- `artifacts/logs/module1_vmd_pipeline.log`
- `artifacts/metrics/module1_tau_fidelity_sweep.json`
- `artifacts/logs/module1_tau_fidelity_per_record.csv`
- `artifacts/plots/module1_vmd/`
- `artifacts/plots/module1_tau_fidelity/`

## Sign-off

Module 1 is **COMPLETE / PASSED / SIGNED OFF**. Module 2 is **NOT STARTED** and requires
explicit authorization.
