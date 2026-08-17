# Module 1 Pre-Full-Build Tau Fidelity and Storage Review

**Status:** PASSED — DEFAULT `tau=0` RETAINED  
**Scope:** Numerical preprocessing fidelity only; no classification metric or fold outcome
was used.  
**Full CORE build:** Not run during this pre-build experiment; subsequently completed
and signed off with the retained `tau=0` configuration.

## Fixed experiment design

The experiment held K=5, alpha=2000, DC=false, init=1, tolerance=1e-7, and
`max_iterations=500` fixed. Only `tau` varied over 0, 0.1, 0.25, and 0.5.

The 12 CORE records were fixed before inspecting tau results. They contain 12 distinct
speakers, three records from every condition-by-gender stratum, eight head and four array
primaries overall, and short/medium/long utterances in every stratum.

| Logical ID | Condition | Gender | Duration stratum | Seconds | Primary mic |
|---|---|---|---|---:|---|
| F01/Session1/0082 | dysarthric | female | short | 1.014625 | head |
| F03/Session1/0179 | dysarthric | female | medium | 2.494500 | head |
| F04/Session1/0038 | dysarthric | female | long | 5.850000 | array |
| FC03/Session3/0151 | control | female | short | 0.997625 | head |
| FC01/Session1/0007 | control | female | medium | 2.420000 | head |
| FC02/Session2/0151 | control | female | long | 6.000000 | array |
| M04/Session1/0125 | dysarthric | male | short | 1.050000 | array |
| M02/Session1/0133 | dysarthric | male | medium | 2.497313 | head |
| M03/Session2/0096 | dysarthric | male | long | 6.005000 | head |
| MC03/Session2/0232 | control | male | short | 1.200000 | array |
| MC04/Session1/0022 | control | male | medium | 2.500000 | head |
| MC02/Session2/0142 | control | male | long | 6.000000 | array |

Total selected audio duration was 38.0290625 seconds.

## Aggregate fidelity comparison

Each cell below is `minimum / mean / median / maximum` across the same 12 records.

| Tau | Normalized error | MSE | RMSE |
|---:|---|---|---|
| 0 | 0.044183 / 0.144419 / 0.154687 / 0.196549 | 1.9522e-5 / 1.8599e-4 / 2.1039e-4 / 3.8631e-4 | 0.004418 / 0.012817 / 0.014490 / 0.019655 |
| 0.1 | 0.012582 / 0.076264 / 0.058671 / 0.174867 | 1.5831e-6 / 6.7797e-5 / 3.0699e-5 / 2.8131e-4 | 0.001258 / 0.006614 / 0.005461 / 0.016772 |
| 0.25 | 0.012882 / 0.087137 / 0.053385 / 0.198028 | 1.6594e-6 / 9.5885e-5 / 2.8510e-5 / 3.9215e-4 | 0.001288 / 0.007702 / 0.005339 / 0.019803 |
| 0.5 | 0.006135 / 0.094533 / 0.084470 / 0.303989 | 3.7638e-7 / 8.3772e-5 / 4.2336e-5 / 2.2784e-4 | 0.000613 / 0.007550 / 0.006506 / 0.015095 |

| Tau | Correlation | Energy ratio |
|---:|---|---|
| 0 | 0.983599 / 0.990718 / 0.990826 / 0.999482 | 0.775356 / 0.866813 / 0.856829 / 0.964639 |
| 0.1 | 0.984849 / 0.995607 / 0.998235 / 0.999921 | 0.991981 / 1.004219 / 1.001887 / 1.015155 |
| 0.25 | 0.981076 / 0.994247 / 0.998578 / 0.999917 | 0.993115 / 1.008808 / 1.003978 / 1.046169 |
| 0.5 | 0.956988 / 0.992676 / 0.996276 / 0.999981 | 0.993593 / 1.015665 / 1.003476 / 1.106129 |

## Convergence and performance

| Tau | Converged | Iterations min/mean/median/max | VMD seconds min/mean/median/max | Runtime vs tau=0 | Duration-scaled CORE VMD estimate |
|---:|---:|---|---|---:|---:|
| 0 | 11/12 | 12 / 145.42 / 102.5 / 500 | 0.032 / 1.536 / 0.545 / 5.723 | 1.00× | 3.49 h |
| 0.1 | 4/12 | 127 / 392.33 / 500 / 500 | 0.761 / 4.106 / 1.269 / 12.094 | 2.67× | 9.34 h |
| 0.25 | 2/12 | 134 / 439.42 / 500 / 500 | 0.750 / 4.714 / 1.761 / 12.774 | 3.07× | 10.73 h |
| 0.5 | 1/12 | 182 / 473.50 / 500 / 500 | 0.734 / 4.814 / 2.124 / 13.209 | 3.13× | 10.96 h |

The lone tau=0 ceiling case, `F03/Session1/0179`, ended at delta 2.25e-7,
close to the 1e-7 tolerance. Nonzero-tau ceiling cases were widespread and had maximum
final deltas from 0.00231 to 0.00663. All mode arrays remained finite, center frequencies
were bounded and ordered, relative energies were normalized, and no mode peak exceeded
1.128 times the standardized input peak. Non-convergence—not obviously pathological
stored modes—was the decisive problem.

## Compact per-record result

Cells contain `normalized error / iterations / convergence`, where `Y` means the stated
tolerance was met. The authoritative CSV contains MSE, RMSE, correlation, energy ratio,
runtime, final delta, center frequencies, relative mode energies, and diagnostic flags for
all 48 record-by-tau runs.

| Logical ID | tau=0 | tau=0.1 | tau=0.25 | tau=0.5 |
|---|---|---|---|---|
| F01/Session1/0082 | .172021 / 18 / Y | .022862 / 500 / N | .018232 / 500 / N | .014391 / 500 / N |
| F03/Session1/0179 | .070583 / 500 / N | .019070 / 147 / Y | .012882 / 134 / Y | .006135 / 182 / Y |
| F04/Session1/0038 | .151572 / 226 / Y | .045217 / 264 / Y | .054374 / 500 / N | .125487 / 500 / N |
| FC03/Session3/0151 | .196549 / 112 / Y | .167722 / 500 / N | .157523 / 500 / N | .150945 / 500 / N |
| FC01/Session1/0007 | .141170 / 138 / Y | .047530 / 170 / Y | .035461 / 139 / Y | .303989 / 500 / N |
| FC02/Session2/0151 | .154032 / 76 / Y | .069812 / 500 / N | .157677 / 500 / N | .065582 / 500 / N |
| M04/Session1/0125 | .138219 / 193 / Y | .070111 / 500 / N | .034332 / 500 / N | .103357 / 500 / N |
| M02/Session1/0133 | .044183 / 93 / Y | .012582 / 127 / Y | .037703 / 500 / N | .022878 / 500 / N |
| M03/Session2/0096 | .172707 / 243 / Y | .138642 / 500 / N | .198028 / 500 / N | .142258 / 500 / N |
| MC03/Session2/0232 | .165186 / 12 / Y | .107194 / 500 / N | .155093 / 500 / N | .108118 / 500 / N |
| MC04/Session1/0022 | .155342 / 48 / Y | .039558 / 500 / N | .052397 / 500 / N | .027720 / 500 / N |
| MC02/Session2/0142 | .171463 / 86 / Y | .174867 / 500 / N | .131941 / 500 / N | .063540 / 500 / N |

## Decision

Tau=0.1 was the best nonzero candidate by mean reconstruction error: it reduced mean
normalized error by 47.19%, improved mean correlation from 0.990718 to 0.995607, and moved
mean energy ratio from 0.866813 to 1.004219. It was not adopted because only 4/12 records
converged, mean VMD time increased 2.67×, eight records hit the iteration ceiling, and
`MC02/Session2/0142` slightly worsened in both normalized error and correlation.

Tau=0 therefore remains the best practical setting among the requested candidates. The
default YAML and production config hash remain unchanged:

`bfd7aec36d99a88cfacc72df85e124d3f4b99b76ed47d71c3ced9a153fa92685`

## Storage review

Module 2 requires only the shared all-mode reconstruction. Individual modes are not a
Module 2 input. The measured nominal options for 415,388,545 CORE samples are:

| Policy | Nominal storage |
|---|---:|
| A: five float64 modes plus float64 reconstruction | 18.5693 GiB |
| B: float64 reconstruction only | 3.0949 GiB |
| B plus this fixed 12-record float64 mode diagnostic panel | 3.1176 GiB |
| C: five float32 modes plus float32 reconstruction | 9.2847 GiB |
| C: float32 reconstruction only | 1.5474 GiB |

Float32 casting was numerically tiny on the 48 runs: direct reconstruction-storage
relative error was at most 2.66e-8, and resumming float32 modes differed from the float64
sum by at most 5.93e-8. Nevertheless, lower precision is not needed to make the current
build feasible and would require a new cache schema and validation contract.

Policy A is retained for the upcoming full build. It is not required by Module 2, but it
preserves the current strict cache validation, mode-level audit evidence, and later
selected-mode ablations without re-running VMD. The host had approximately 122.3 GiB free
during this review, so 18.57 GiB is operationally acceptable. Policy B is a valid future
optimization only if storage pressure justifies a separately tested optional-mode cache
schema; it was not implemented in this narrow check.

## Artifacts

- `artifacts/metrics/module1_tau_fidelity_sweep.json`
- `artifacts/logs/module1_tau_fidelity_per_record.csv`
- `artifacts/plots/module1_tau_fidelity/F03__Session1__0179.svg`
- `artifacts/plots/module1_tau_fidelity/MC02__Session2__0142.svg`
- `data/processed/vmd_fidelity_sweep/<tau-specific-config-hash>/`

The two SVGs compare standardized input, all-mode tau=0 reconstruction, tau=0.1
reconstruction, and both residuals using the same input-derived vertical scale. Raw TORGO
metadata retained fingerprint
`206041a4be1fad47975d25ba38c73bb49fe68e3957ebfb046d0db15c0b14c76a`.
