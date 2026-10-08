# Module 2 Test Report — Mel, Visibility-Graph & Recurrence Representation Generation

**Status:** COMPLETE / PASSED / SIGNED OFF
**Scope:** implementation, historical smoke and real-crop gates, full 8,673-record CORE
build and validation, complete numerical/provenance/cache audit, protected-input audit,
and final regression gates. No encoder, GCN, attention, classifier training, ASR, TTS, or
UI work was run.

## Authoritative full CORE sign-off — 2026-09-01

The manual production build and the independently re-run strict validator both pass. The
final build computed 8,661 entries and resumed 12 previously valid smoke entries:

| Completion item | Count |
|---|---:|
| Expected CORE | 8,673 |
| Selected | 8,673 |
| Newly computed | 8,661 |
| Cache hits | 12 |
| Successful | 8,673 |
| Failed | 0 |
| Strictly validated | 8,673 |

The validator returned `status=PASS` with representation hash
`d8a36c6ab8c48b15e61ecd63194ffa30ede4e7e7e7ab0a30c197002652f9d65d`, Module 1 hash
`bfd7aec36d99a88cfacc72df85e124d3f4b99b76ed47d71c3ced9a153fa92685`, and manifest
fingerprint `91c3327ea1f6a6b2057ffa02198f86d682668d22f247beef27f1466e959a9061`.

### Production cache inventory and storage

| Item | Count / bytes |
|---|---:|
| Logical cache directories | 8,673 |
| `mel.npy` | 8,673 |
| `recurrence.npy` | 8,673 |
| `graph_nodes.npy` | 8,673 |
| `graph_edges.npy` | 8,673 |
| `metadata.json` | 8,673 |
| Total production files | 43,365 |
| Partial / temporary / unexpected / zero-byte files | 0 / 0 / 0 / 0 |
| Duplicate logical IDs | 0 |
| Total bytes | 2,378,359,184 |
| Total GiB | 2.215019599 |
| Mean bytes per logical record | 274,225.664 |

Artifact contributions are 1,741,815,936 bytes for Mel, 436,286,592 for recurrence,
18,872,448 for graph nodes, 138,923,656 for graph edges, and 42,460,552 for metadata.
These are measured full-namespace values, not the historical smoke-scaled estimate.

### Full-corpus Mel audit

All 8,673 arrays are finite `float32[224,224]`; there are no NaN, Inf, wrong-shape, or
wrong-dtype outputs. Metadata reports 8,438 padded, 235 cropped, and zero exact records.

| Statistic | Minimum | Mean | Median | Maximum |
|---|---:|---:|---:|---:|
| Per-record Mel minimum (dB) | -100.000000 | -99.991930 | -100.000000 | -84.354958 |
| Per-record Mel maximum (dB) | -37.631699 | -8.126929 | -7.766419 | 2.315351 |
| Native frame count | 13 | 297.506630 | 242 | 19,381 |
| Padded seconds, all records | 0 | 5.152275 | 5.561688 | 7.850000 |
| Cropped seconds, all records | 0 | 0.145678 | 0 | 185.828625 |
| Positive padded seconds (8,438) | 0.009313 | 5.295767 | 5.600000 | 7.850000 |
| Positive cropped seconds (235) | 0.005063 | 5.376457 | 2.225000 | 185.828625 |

### Full-corpus recurrence audit

All 8,673 arrays are binary `uint8[224,224]`, finite, symmetric, and have a unit
diagonal. The total density is 0.153858418 for every record. Off-diagonal density is
0.150064061 for every record, only 0.000064061 above the 0.15 target because the finite
symmetric matrix admits discrete recurrence counts. No record exceeds 0.16, so no
substantial tie-driven excess occurs.

| Statistic | Minimum | Mean | Median | Maximum |
|---|---:|---:|---:|---:|
| Total recurrence density | 0.153858 | 0.153858 | 0.153858 | 0.153858 |
| Off-diagonal density | 0.150064 | 0.150064 | 0.150064 | 0.150064 |
| Distance threshold | 0.002011 | 0.109399 | 0.055559 | 1.276016 |

### Full-corpus visibility-graph audit

All 8,673 graphs have finite `float32[N,2]` nodes and `int16[2,E]` edges with valid
indices, no stored self-loops, no duplicate directed edges, and both directions for every
undirected edge. No graph has an isolated node or more than one connected component.

| Statistic | Minimum | Mean | Median | Maximum |
|---|---:|---:|---:|---:|
| Node count | 256 | 256 | 256 | 256 |
| Undirected edge count | 604 | 1,986.243399 | 1,822 | 5,640 |
| Graph density | 0.018505 | 0.060853 | 0.055821 | 0.172794 |
| Isolated node count | 0 | 0 | 0 | 0 |
| Connected component count | 1 | 1 | 1 | 1 |

Invalid graphs: zero. Graphs requiring isolated/disconnected follow-up: zero.

### Full 235-record long-signal regeneration

Every CORE record longer than eight seconds was regenerated from its complete signed
Module 1 reconstruction and compared bit for bit with its production cache. All 235 Mel
arrays match both full-input generation and the explicit centered 128,000-sample slice.
All 235 recurrence arrays and all 235 visibility graphs match generation from the complete
reconstruction; zero regeneration mismatches occurred.

For `F03/Session1/0118`, the signed source is 3,101,258 samples / 193.828625 seconds. Its
Mel crop spans samples 1,486,629 through 1,614,628, or 92.9143125–100.9143125 seconds.
The cached Mel equals the explicit center-crop transform. The cached recurrence and graph
equal full-signal regeneration and each differs from the 8-second crop reference. Thus the
8-second policy affects Mel only.

### Source integrity and protected inputs

Every cache entry has a unique production source path and is bound to its logical ID,
manifest fingerprint, Module 1 production hash, reconstruction path, reconstruction
SHA-256, and Module 2 configuration hash. All paths are under
`data/processed/vmd/bfd7aec36d99a88cfacc72df85e124d3f4b99b76ed47d71c3ced9a153fa92685/<speaker>/<session>/<prompt>/reconstruction.npy`. No source
uses `data/raw`, `vmd_fidelity_sweep`, selected-mode reconstruction, a tau namespace, or
another VMD configuration.

The signed raw TORGO subtree remains exactly 82,772 files / 15,845,129,970 bytes with
path-size-mtime SHA-256
`206041a4be1fad47975d25ba38c73bb49fe68e3957ebfb046d0db15c0b14c76a`.
The Module 1 production namespace remains exactly 26,019 files / 19,981,366,416 bytes
with path-size-mtime SHA-256
`b4944b720c6664d55e57180c94e1c0c6149ac080b49705035ac12ca16be54baa`.

The failure CSV contains only its header and zero failure rows. No second root-level
`DEFECT_LOG.md` exists.

### Measured full-corpus runtime

The full production pipeline measured 1,062.363907 seconds (17.7061 minutes), including
source validation, cache writes, reporting, and plots. Per-record times below are measured
metadata from all production entries; source validation is reported separately.

| Stage | Mean (s) | Median (s) | Maximum (s) |
|---|---:|---:|---:|
| Source validation | 0.021107 | 0.019087 | 0.614292 |
| Mel | 0.010351 | 0.010119 | 0.021938 |
| Recurrence | 0.004260 | 0.005259 | 0.019240 |
| Visibility graph | 0.016520 | 0.017668 | 0.081800 |
| Three transforms total | 0.031131 | 0.033787 | 0.099927 |

The slowest combined transform was `M05/Session2/0236` at 0.099927 seconds, dominated by
0.081800 seconds for the visibility graph. Historical smoke-scaled runtime values below
remain provenance only and are not the final runtime.

### Final regression gates

```text
uv run --python 3.12 --extra dev pytest tests/m2_representations
# 40 passed in 4.54 s

uv run --python 3.12 --extra dev pytest
# 184 passed in 6.82 s

uv run --python 3.12 --extra dev pytest --cov=vaaniswitch.m0_data --cov=vaaniswitch.m1_vmd --cov=vaaniswitch.m2_representations --cov-branch --cov-report=term-missing
# 184 passed; 82.07% branch-aware coverage; 80% gate passed

uv run --python 3.12 --extra dev ruff check .
# All checks passed

uv run --python 3.12 --extra dev mypy src/vaaniswitch/m0_data src/vaaniswitch/m1_vmd src/vaaniswitch/m2_representations tests/m0_data tests/m1_vmd tests/m2_representations
# Success: no issues found in 62 source files

uv run --python 3.12 python -m compileall -q src tests
# PASS
```

No new implementation defect was found during final audit. The centralized defect log
records the retained fixed-duration Mel crop as `M2-SCI-001` `ACCEPTED_LIMITATION`.
No classifier has been trained, and these deterministic representation checks establish no
classification or diagnostic performance.

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

## Historical smoke performance and storage

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

- `artifacts/metrics/module2_full_signoff_audit.json`
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
- Recurrence-rate ties can produce density above the nominal target. Across full CORE,
  off-diagonal density was uniformly 0.150064061; no substantial tie excess occurred.
- The smoke runtime and storage values in the historical section above are superseded by
  the measured full-corpus values in the authoritative sign-off section.
- No representation parameter was tuned using classification accuracy.

✅ Module 2 is **COMPLETE / PASSED / SIGNED OFF**. Module 3 remains not started.

Executed full-build commands retained for provenance:

```text
uv run --python 3.12 --extra dev vaaniswitch-m2 build --config configs/representations/default.yaml --subset core
uv run --python 3.12 --extra dev vaaniswitch-m2 validate --config configs/representations/default.yaml --subset core
```
