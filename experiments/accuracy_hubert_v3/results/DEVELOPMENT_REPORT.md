# V3 development-only report

Grouped CV and selection evidence; these are not outer-test results or independent estimates.

| Outer-fold context | Recipe | CV macro-F1 | CV balanced accuracy |
|---|---|---:|---:|
| 1 | ref_L09_C0.01_B0_N0_A0 | 0.549079 | 0.660093 |
| 1 | weight_L09_C0.01_B2_N0_A0 | 0.553772 | 0.664752 |
| 1 | robust_L09_C0.01_B0_N1_A0 | 0.547055 | 0.658139 |
| 1 | l2_L09_C0.01_B0_N2_A0 | 0.550605 | 0.661011 |
| 1 | channel_L09_C0.01_B0_N0_A1 | 0.543865 | 0.657729 |
| 2 | ref_L09_C0.01_B0_N0_A0 | 0.643698 | 0.677850 |
| 2 | weight_L09_C0.01_B2_N0_A0 | 0.625351 | 0.662432 |
| 2 | robust_L09_C0.01_B0_N1_A0 | 0.655612 | 0.687478 |
| 2 | l2_L09_C0.01_B0_N2_A0 | 0.642946 | 0.677352 |
| 2 | channel_L09_C0.01_B0_N0_A1 | 0.641683 | 0.676626 |
| 3 | ref_L09_C1_B0_N0_A0 | 0.702053 | 0.739774 |
| 3 | weight_L09_C1_B2_N0_A0 | 0.695447 | 0.738049 |
| 3 | robust_L09_C1_B0_N1_A0 | 0.691858 | 0.734239 |
| 3 | l2_L09_C1_B0_N2_A0 | 0.700504 | 0.738765 |
| 3 | channel_L09_C1_B0_N0_A1 | 0.698153 | 0.737799 |

This document contains development selection evidence only.
Real outer status is recorded separately in FROZEN_DECISIONS.json, outer seals and FINAL_EXPERIMENT_REPORT.md.
Development metrics are not final V3 outer accuracy. All decisions must freeze before any outer evaluation.
