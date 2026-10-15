# V3 test report — 2026-10-13

PASS: 54 automated tests, zero failures/errors/skips, 13.817 seconds.
Final log: validation/tests_final2.log. Python AST parsing passed for 15 Python
files, and PowerShell AST parsing passed. Existing .venv-accuracy was used without
installation/reinstallation. Tests did not execute the real full V3 experiment.

Coverage includes live original CORE/split and V2 source/revision contracts;
speaker/hash/record/role isolation; B0/B1 equality and V2 weight parity; B2 cap and
class totals; parent/speaker/class/objective mass after augmentation; V2 scaler
parity, robust zero-IQR behavior, L2 zero vectors, train-only statistics; waveform
determinism/immutability/SNR/bandwidth/peak/silence/budget; augmented metadata/payload/
identity corruption; read-only original features; finite LR/normalization/gradient
state and convergence failure; OOM halving, batch-one stop, exact-length batching
and malformed extractor results; checkpoint corruption, interrupted solve restart,
completed checkpoint/publication recovery, RNG restore; metric hand calculations,
undefined class rates, all-record speaker averaging, fixed 0.5 threshold, tie and
combination gates; separate calibration roles; explicit outer authorization,
once-only seals, ambiguous claim rejection, decision tampering; full synthetic
selection/freeze/terminal-resume/report generation; and canonical CSV serialization.

The synthetic end-to-end fixture runs the declared orchestration using small
synthetic features and fake augmentation, with every prediction/report labeled
SYNTHETIC_TEST_FIXTURE. It verifies all three decisions precede outer access,
all expected reports/five figures exist, and sealed results resume without a
second forward. Temporary synthetic files are removed by the test fixture.
No synthetic numbers appear as V3 experimental accuracy.

## Bounded real RTX 4070 validation

Latest evidence:
validation/cuda-26a8ab67c5bf471e821fd2a4a901a0c3/REPORT.json,
referenced with SHA-256 by validation/latest_cuda.json. Its source identity equals
the finalized src/config/protocol/launcher/lineage hashes. The launcher was invoked
only with `-Mode Validate`, never its full Develop or real Outer mode.

| Check | Measured result |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| Original cache verification | 8,673 entries PASS; zero re-extractions |
| Actual training panel | 16 originals, all eight Fold 1 inner-training speakers |
| Actual validation panel | Four unaugmented originals, M03 and MC02 |
| New waveform-derived features per bounded run | Eight training variants, [8,3,1536] |
| Profiles exercised | band_noise and tilt_echo |
| Actual original audio hashes before/after | 20/20 unchanged |
| Faithful V2 E1 parity | Coefficients and intercept exactly equal |
| Other small panels | B2, N1, N2 and augmented LR completed with finite state |
| Checkpoint roundtrip/reuse | PASS |
| Interrupted-start restart | Identical validation probabilities |
| Completed-checkpoint interrupted publication | Reused model; identical probabilities |
| RNG roundtrip | Python, NumPy, Torch CPU and CUDA PASS |
| Peak allocated | 943,448,064 bytes = 0.878654 GiB |
| Peak reserved | 1,270,874,112 bytes = 1.183594 GiB |
| Eight augmentation inference calls | 0.261748 seconds, excluding setup/verification |
| Bounded panel duration | 3.202869 seconds, excluding environment/cache verification |
| Worst-duration benchmark | Augmented 128,000-sample development segment; batches 1/2/4 passed |
| Real outer forwards | Zero |
| Neural classifier/FP16 GradScaler | Not implemented/used; LR is CPU sklearn |

Three small CUDA validation namespaces were retained as source/report safeguards
were finalized: 24 variant entries total across three isolated validation caches,
eight per run, on the same bounded development panel. Earlier source identities
remain evidence of those earlier checks; only the final source-bound report is
the handoff validation. This was not production augmentation or full training.

The first test attempt had one fixture-path error (V3-D001). A later expanded
terminal-resume test caught CSV ordering drift (V3-D008). Both failing logs are
retained; fixes and focused regressions are recorded in DEFECT_LOG.md. The final
54-test run passes. Deliberate corruption/OOM tests are synthetic fault injection.

Full-size convergence, full candidate runtime, production augmentation cache and
real outer reporting remain unexecuted. The scientific effect of any modification
and the final V3 accuracy are unknown. No statistical significance is claimed.
