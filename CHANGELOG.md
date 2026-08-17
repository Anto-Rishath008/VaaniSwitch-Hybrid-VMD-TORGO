# Changelog

## 2026-08-17 — Module 2 real center-crop pre-full-build gate passed

- Audited all 235/8,673 CORE records longer than eight seconds: 205 dysarthric and 30
  control across 15 speakers; six records exceed 30 seconds.
- Exercised a fixed six-record real production crop panel spanning both conditions and
  genders, 9.555-193.828625 seconds, using only signed all-mode Module 1
  `reconstruction.npy` artifacts.
- Verified each Mel result is a deterministic finite float32 224x224 transform of the
  exact centered 128,000 samples with 798 fixed STFT frames.
- Verified recurrence and natural visibility continue to summarize the complete
  reconstruction, not the Mel crop, for every panel record.
- Added a combined full-duration/crop/Mel/RP/NVG diagnostic for the longest record,
  `F03/Session1/0118`, and retained it in CORE without reclassification.
- Retained the 8-second initial Mel policy and unchanged representation config hash;
  documented the 235 cropped records as a Module 5 robustness-analysis limitation.
- Consolidated M2-001 into `docs/defects/DEFECT_LOG.md` and removed the redundant root
  defect log; no duplicate YAML keys or coverage source settings existed.
- Hardened production 80-to-224 resize and >8-second crop regression coverage. All 40
  Module 2 tests and all 184 project tests pass with 82.07% branch coverage; Ruff,
  strict mypy, and compileall pass. Module 2 remains awaiting the manual full build.

## 2026-08-17 — Module 2 implementation and production smoke validation ready

- Added a strict read-only Module 1 source boundary that consumes only signed production
  all-mode `reconstruction.npy` files and rejects tau-sweep, selected-mode, stale,
  non-finite, or hash-invalid sources.
- Added project-owned NumPy log-Mel generation using a fixed physical 8-second time axis,
  25 ms/10 ms STFT, 80 non-empty HTK/Slaney-area Mel bands, power dB, and deterministic
  224x224 conversion without utterance-specific time stretching.
- Added a 224-state binary Euclidean recurrence plot from block-RMS scalar reduction,
  dimension-3 delay-2 embedding, and a label-independent 15% off-diagonal recurrence rule.
- Added strict natural visibility graphs from a 256-bin block-RMS envelope with two
  deterministic node features and compact bidirectional int16 edge-index storage.
- Added content-addressed atomic caches with completion metadata, source/config binding,
  artifact hashes, semantic validation, corruption recovery, and resumability.
- Built and validated the fixed 12-record production smoke panel with zero failures,
  including `F03/Session1/0179`, a valid Module 1 iteration-ceiling record.
- Measured a 0.036964-second mean three-transform time and 275,979.3 bytes per complete
  smoke cache; estimated 5.34 transform-only minutes, 28.46 pipeline minutes, and
  2.22918 GiB for full CORE. These remain smoke-scaled estimates.
- Added 39 Module 2 tests. The full 183-test suite, 82.01% branch coverage, Ruff, strict
  mypy, and compileall gates pass; raw TORGO and Module 1 production cache are unchanged.
- Left Module 2 implementation ready but not complete pending the explicit manual full
  CORE build and validation. Module 3 remains not started.

## 2026-08-17 — Module 1 full CORE completion and sign-off

- Verified the completed CORE build against the signed Module 0 manifest: 8,673/8,673
  valid cache entries, zero failed records, and unchanged manifest/raw fingerprints.
- Confirmed the production identity remains K=5, alpha=2000, tau=0, DC=false, init=1,
  tolerance=1e-7, 500 iterations, all-mode reconstruction, and disabled selected-mode
  ablation.
- Audited all 48 records that reached the iteration ceiling. Every result is finite,
  correctly shaped, frequency-bounded, cache-valid, source-primary-bound, an exact sum
  of its five modes, and reconstruction-metric-valid; recorded this as an accepted
  scientific limitation rather than a failure or exclusion.
- Measured 12,219.264 s (3.394 h) full pipeline time and 11,342.994 aggregate VMD seconds.
- Inventoried 8,673 complete cache directories and 26,019 expected files occupying
  19,981,366,416 bytes (18.6091 GiB), with no partial, temporary, or unexpected files.
- Passed the full 144-test Module 0+1 suite, 81.70% branch-aware coverage gate, Ruff,
  strict mypy, compileall, and full `components-m1 validate` gate.
- Marked Module 1 complete and signed off. Module 2 remains not started.

## 2026-08-16 — Module 1 pre-full-build tau fidelity check

- Compared tau values 0, 0.1, 0.25, and 0.5 on a fixed 12-record CORE panel spanning 12
  speakers, all condition-by-gender strata, three duration bands, and both microphones.
- Kept K=5, alpha=2000, DC=false, init=1, tolerance=1e-7, and 500 iterations fixed; no
  classification result or evaluation-fold accuracy informed the decision.
- Retained tau=0: tau=0.1 improved mean reconstruction error by 47.19% but converged for
  only 4/12 records and increased average runtime 2.67×; larger tau values were less stable.
- Preserved the existing default config and production cache hash; experimental caches
  use separate tau-specific namespaces and did not overwrite the six-record smoke cache.
- Confirmed Module 2 requires reconstruction only, but retained full float64 modes plus
  reconstruction for strict validation, mode auditability, and selected-mode ablation.
- Quantified alternatives: 18.57 GiB current, 3.09 GiB float64 reconstruction-only, and
  9.28 GiB float32 full arrays; no cache schema or precision change was made.
- Added same-scale input/reconstruction/residual SVGs and complete 48-row JSON/CSV reports.

## 2026-08-16 — Module 1 implementation and smoke validation ready

- Added float64 PCM loading, mono conversion, 16 kHz resampling fallback, conservative RMS
  normalization, and strict Module 0 header/hash validation without rewriting raw audio.
- Added a project-owned pure-NumPy VMD implementation with K=5, alpha=2000, tau=0,
  DC=false, init=1, tolerance=1e-7, and a 500-iteration limit.
- Added all-mode reconstruction as the default and retained selected-mode reconstruction
  only as a disabled explicit ablation interface.
- Added per-mode frequency/energy/RMS/centroid metadata and descriptive reconstruction
  MSE, RMSE, normalized-error, correlation, and energy metrics.
- Added atomic, content-addressed caches with source/config/manifest identity, semantic
  validation, corruption recovery, resumability, and strict portable output validation.
- Added fail-fast guards preventing Module 1 outputs from entering `data/raw` or
  overwriting signed Module 0 inputs.
- Recomputed and validated a six-record real TORGO CORE smoke cache with zero failures;
  raw data remained unchanged.
- Recorded the `tau=0` smoke fidelity limitation (normalized error 0.0813–0.1771) without
  denoising, losslessness, or classification claims.
- Added 83 Module 1 tests; the combined 144-test suite, 81.70% branch-aware coverage,
  Ruff, strict mypy, and compileall gates pass.
- Marked Module 1 implementation ready while leaving the full CORE build and Module 2
  explicitly pending.

## 2026-08-16 — Module 0 validity-aware microphone correction

- Changed canonical microphone selection to audit both available recordings before choosing the primary.
- Kept structurally valid head audio mandatory; permitted array fallback only when the head is missing or has explicit structural-audit failure evidence.
- Recovered F01/Session1/0067 and F01/Session1/0068 with valid array primaries while retaining their zero-byte heads in the full raw-audio audit.
- Rebuilt 9,494 logical records with 8,863 usable, 627 review, and four excluded records; the speaker folds are unchanged and the leakage hard gate passes.
- Exposed the documented CORE/CLEAN (8,673 records) and EXTENDED (9,222 records) downstream subset policies without starting preprocessing.
- Expanded the Module 0 regression suite to 61 tests; Ruff, strict mypy, compileall, and 81.70% branch-aware coverage pass.

## 2026-08-16 — Module 0 completed

- Added the prompt-driven TORGO discovery and canonical one-record-per-speaker/session/prompt manifest pipeline.
- Enforced TORGO F/M dysarthric and FC/MC control labels without medical-severity inference.
- Enforced validity-aware head-microphone priority, evidence-backed array fallback, and logical binding of microphone copies.
- Added full direct-WAV header, signal-quality, file-size, and SHA-256 auditing without altering raw data.
- Added conservative word, sentence, articulation, instruction/nonstandard, and unknown prompt classification.
- Added deterministic three-fold condition×gender-stratified speaker-group evaluation with seed 42.
- Added hard leakage, duplicate, schema, artifact-parity, raw-fingerprint, and relative-path validation gates.
- Generated and validated 9,494 manifest records; documented four exclusions and 381 orphan microphone paths.
- Added 61 synthetic Module 0 tests; Ruff, strict mypy, and 81.70% branch-aware coverage pass.
- Signed off Module 0; Module 1 remains unstarted.

## 2026-08-16 — Final revision
- VMD only.
- All-mode reconstruction is default.
- Mode deletion is optional ablation.
- 16 Mel tokens.
- 1 visibility-graph token.
- 16 recurrence tokens.
- 17 nonlinear tokens.
- token-level Mel→nonlinear cross-attention.
- pure-PyTorch GCN.
- no GMM-HMM.
- frozen English ASR separate from classifier.
- standard English TTS.
- stronger speaker-disjoint fold reporting.
