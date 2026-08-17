# Module 2 Test Report — Mel, Visibility-Graph & Recurrence Representation Generation

**Status:** PASSED — IMPLEMENTATION READY / AWAITING FULL MANUAL CORE BUILD  
**Scope:** implementation, 12-record production smoke, and six-record real production
crop gate only; no full CORE build, encoder, GCN, attention, training, ASR, or TTS was
run.

## Signed input contract

All 12 smoke records loaded the production all-mode `reconstruction.npy` under VMD hash
`bfd7aec36d99a88cfacc72df85e124d3f4b99b76ed47d71c3ced9a153fa92685`.
The actual manifest and signed Module 0/1 artifacts independently agreed on 8,673 CORE
records and manifest SHA-256
`91c3327ea1f6a6b2057ffa02198f86d682668d22f247beef27f1466e959a9061`.

## Requirement coverage

| Area | Evidence | Result |
|---|---|---|
| Production source only | Loader constructs the signed production cache path and reads only `reconstruction.npy`; raw-WAV sentinel test | PASS |
| Experimental/selected-mode rejection | Tau-sweep root, wrong VMD hash, and non-`all_modes` policy regression tests | PASS |
| Source integrity | Logical ID, manifest, primary binding, 16 kHz mono semantics, shape, dtype, finiteness, byte count, SHA-256 | PASS |
| CORE boundary | Actual signed count 8,673; excluded and non-normal synthetic records absent | PASS |
| Iteration ceiling | `F03/Session1/0179` had `module1_converged=false` and produced/validated all outputs | PASS |
| Mel | Deterministic finite float32 224x224; 25 ms/10 ms STFT; fixed 8 s padding/crop tests | PASS |
| Real crop path | Six signed >8 s reconstructions matched explicit centered 128,000-sample transforms | PASS |
| Full-signal RP/NVG | All six differed from 8 s crop references and recorded complete input sample counts | PASS |
| Recurrence | Deterministic symmetric binary uint8 224x224 with unit diagonal and bounded states | PASS |
| Natural visibility | Peak, valley, and collinear hand checks; deterministic <=256-node compact graph | PASS |
| Graph integrity | Valid unique bidirectional int16 indices, no self-loops, connected smoke graphs | PASS |
| Label independence | Same waveform produces bit-identical outputs without label/speaker inputs | PASS |
| Cache | Atomic round trip, completion marker, config namespace, corruption/hash rejection, resume skip | PASS |
| CLI | Real smoke build and validation; synthetic CLI regression | PASS |
| Protected inputs | Raw metadata fingerprint and Module 1 metadata-tree fingerprint unchanged | PASS |

The final raw snapshot remained 82,772 files / 15,845,129,970 bytes with metadata
SHA-256 `206041a4be1fad47975d25ba38c73bb49fe68e3957ebfb046d0db15c0b14c76a`.
The production Module 1 namespace remained 26,019 files / 19,981,366,416 bytes with the
same path-size-mtime fingerprint before and after Module 2:
`b4944b720c6664d55e57180c94e1c0c6149ac080b49705035ac12ca16be54baa`.

## CORE duration audit and Mel policy

| Statistic | Seconds |
|---|---:|
| Minimum | 0.150000 |
| Median | 2.438313 |
| Mean | 2.993403 |
| 90th percentile | 4.760000 |
| 95th percentile | 6.300000 |
| 99th percentile | 11.538818 |
| Maximum | 193.828625 |

The fixed 8-second policy predicts 8,438/8,673 (97.2904%) padded records and
235/8,673 (2.7096%) centered-cropped records. The maximum discarded duration is
185.828625 seconds, driven by the 193.828625-second CORE outlier. No utterance-specific
time stretching is used.

### Records longer than eight seconds

The signed CORE contains exactly 235/8,673 (2.709558%) records longer than eight seconds:
205 dysarthric and 30 control; 34 female and 201 male. Their duration distribution is
8.005063 s minimum, 10.225000 s median, 13.376457 s mean, 17.300038 s p90,
21.345000 s p95, 42.697762 s p99, and 193.828625 s maximum. Threshold counts are 235
over 8 s, 128 over 10 s, 72 over 12 s, 13 over 20 s, and six over 30 s.

| Speaker | Count | Speaker | Count | Speaker | Count |
|---|---:|---|---:|---|---:|
| F01 | 4 | F03 | 17 | F04 | 5 |
| FC01 | 2 | FC02 | 3 | FC03 | 3 |
| M01 | 14 | M02 | 18 | M03 | 4 |
| M04 | 67 | M05 | 76 | MC01 | 8 |
| MC02 | 6 | MC03 | 1 | MC04 | 7 |

### Real production crop panel

The fixed panel was chosen by descriptive duration/condition/gender strata before any
transform was generated. Every Mel was deterministic, finite float32 `224x224`, used
exactly 128,000 centered samples and 798 fixed STFT frames, and matched a transform of
the explicit source slice bit for bit. Every recurrence was symmetric binary uint8
`224x224`. Every NVG had 256 nodes, valid unique bidirectional int16 edges, no self-loops,
zero isolated nodes, and one connected component.

| Logical ID | Condition/gender | Duration (s) | Crop interval (s) | Cropped (s) | RP full input | NVG full input |
|---|---|---:|---:|---:|---|---|
| F03/Session1/0024 | dysarthric/female | 9.973250 | 0.986625-8.986625 | 1.973250 | yes | yes |
| FC02/Session3/0317 | control/female | 9.555000 | 0.777500-8.777500 | 1.555000 | yes | yes |
| MC04/Session1/0210 | control/male | 10.015000 | 1.007500-9.007500 | 2.015000 | yes | yes |
| M05/Session2/0242 | dysarthric/male | 19.630000 | 5.815000-13.815000 | 11.630000 | yes | yes |
| M03/Session2/0027 | dysarthric/male | 43.155063 | 17.577500-25.577500 | 35.155063 | yes | yes |
| F03/Session1/0118 | dysarthric/female | 193.828625 | 92.914313-100.914313 | 185.828625 | yes | yes |

The longest record is usable dysarthric female speech from speaker F03, prompt type
`word`, transcript `no`, and head primary microphone. It remains in CORE unchanged.
Its combined diagnostic shows the full-duration timeline, centered Mel crop, complete-
signal recurrence plot, and complete-signal natural visibility graph.

The 8-second policy is retained as the initial production compromise: 97.2904% of CORE
does not crop, and 224 pixels across eight seconds retain about 35.7 ms per final time
pixel. Expanding the same 224-pixel canvas would reduce cropping but lower temporal
resolution. The 235-record centered-crop limitation is explicitly reserved for Module 5
robustness analysis; it is not grounds for record exclusion.

## Smoke panel

| Logical ID | Condition | Gender | Duration (s) | Primary | Module 1 converged |
|---|---|---|---:|---|---|
| F01/Session1/0082 | dysarthric | female | 1.014625 | head | yes |
| FC03/Session3/0151 | control | female | 0.997625 | head | yes |
| M04/Session1/0125 | dysarthric | male | 1.050000 | array | yes |
| MC03/Session2/0232 | control | male | 1.200000 | array | yes |
| F03/Session1/0179 | dysarthric | female | 2.494500 | head | no; valid ceiling record |
| FC01/Session1/0007 | control | female | 2.420000 | head | yes |
| M02/Session1/0133 | dysarthric | male | 2.497313 | head | yes |
| MC04/Session1/0022 | control | male | 2.500000 | head | yes |
| F04/Session1/0038 | dysarthric | female | 5.850000 | array | yes |
| FC02/Session2/0151 | control | female | 6.000000 | array | yes |
| M03/Session2/0096 | dysarthric | male | 6.005000 | head | yes |
| MC02/Session2/0142 | control | male | 6.000000 | array | yes |

## Scientific smoke checks

Mel outputs were all float32 224x224 and finite. Their aggregate range was -100.0 to
-1.416121 dB, native frame counts ranged from 98 to 599 (mean 314.917, median 247.5),
and all 12 smoke records padded because none exceeded 8 seconds.

Recurrence outputs were all uint8 224x224, symmetric, binary, and unit-diagonal. Each
used 228 scalar RMS bins to create 224 embedded states. Smoke density was 0.153858 for
every record; this includes the diagonal in addition to the 15% off-diagonal target.

Visibility graphs all had 256 nodes, zero isolated nodes, and one connected component.
Undirected edge count ranged 1,366–3,804 (mean 2,205.5, median 2,034.5); density ranged
0.041850–0.116544 (mean 0.067570, median 0.062331). Stored edge counts are twice the
undirected counts because `edge_index` contains both directions.

## Performance and storage

| Stage | Mean (s) | Median (s) | Maximum (s) |
|---|---:|---:|---:|
| Mel | 0.010253 | 0.010176 | 0.012360 |
| Recurrence | 0.005555 | 0.004982 | 0.009664 |
| Visibility graph | 0.021157 | 0.021651 | 0.030675 |
| Three transforms total | 0.036964 | 0.038365 | 0.048729 |

The final smoke pipeline wall time was 2.362799 seconds. Linear estimates are 320.59
seconds (5.34 minutes) for transforms alone and 1,707.71 seconds (28.46 minutes) using
the complete smoke pipeline wall rate. These are estimates, not measured full-corpus
runtimes. The 12 complete cache entries occupy 3,311,752 bytes; scaling their measured
mean gives 2,393,568,758 bytes (2.22918 GiB) for full CORE.

## Commands and gates

```text
uv run --python 3.12 --extra dev pytest tests/m2_representations -q
# 40 passed

uv run --python 3.12 --extra dev vaaniswitch-m2 build --config configs/representations/default.yaml --subset core --limit 12 --force
# 12 computed, 0 failed

uv run --python 3.12 --extra dev vaaniswitch-m2 validate --config configs/representations/default.yaml --subset core --limit 12
# PASS, 12/12 valid

uv run --python 3.12 --extra dev pytest
# 184 passed

uv run --python 3.12 --extra dev pytest --cov=vaaniswitch.m0_data --cov=vaaniswitch.m1_vmd --cov=vaaniswitch.m2_representations --cov-branch --cov-report=term-missing
# 184 passed; 82.07% branch-aware coverage; 80% gate passed

uv run --python 3.12 --extra dev ruff check .
# All checks passed

uv run --python 3.12 --extra dev mypy src/vaaniswitch/m0_data src/vaaniswitch/m1_vmd src/vaaniswitch/m2_representations tests/m0_data tests/m1_vmd tests/m2_representations
# Success: no issues found in 62 source files

uv run --python 3.12 python -m compileall -q src tests
# PASS
```

## Artifacts

- `artifacts/metrics/module2_representation_summary.json`
- `artifacts/logs/module2_representation_failures.csv` (header only; zero failures)
- `artifacts/logs/module2_representation_pipeline.log`
- `artifacts/plots/module2_representations/F01__Session1__0082.svg`
- `artifacts/plots/module2_representations/FC03__Session3__0151.svg`
- `artifacts/metrics/module2_real_crop_audit.json`
- `artifacts/logs/module2_real_crop_audit_console.log`
- `artifacts/plots/module2_crop_audit/F03__Session1__0118_real_crop_diagnostic.svg`
- `data/processed/representations/d8a36c6ab8c48b15e61ecd63194ffa30ede4e7e7e7ab0a30c197002652f9d65d/`

## Defect found and fixed

The initial 128-band HTK Mel proposal produced one empty low-frequency triangle at a
512-point FFT resolution. A failing non-empty-filter regression test reproduced it. The
native bank was changed to 80 bands, all filters were verified non-empty, and generation
now explicitly rejects any configuration that creates an empty band. Final 224-row
frequency conversion remains deterministic linear interpolation.

## Limitations and gate

- The 8-second choice trades 35.7 ms final time pixels against centered cropping for
  2.7096% of CORE; the six extreme >30-second records dominate maximum discarded time.
  This is a required Module 5 robustness-analysis target.
- The 80-to-224 frequency interpolation adds spatial sampling, not new spectral detail.
- Recurrence-rate ties can produce density above the nominal target; this did not occur
  beyond the expected diagonal contribution in smoke.
- Runtime and storage are smoke-scaled estimates until the manual full build finishes.
- No representation parameter was tuned using classification accuracy.

✅ Implementation and smoke gate passed. Module 2 remains **not complete** and **not
signed off** until the manual 8,673-record build and subsequent full validation pass.
Module 3 remains not started.

Eventual manual commands:

```text
uv run --python 3.12 --extra dev vaaniswitch-m2 build --config configs/representations/default.yaml --subset core
uv run --python 3.12 --extra dev vaaniswitch-m2 validate --config configs/representations/default.yaml --subset core
```
