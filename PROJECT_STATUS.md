# Project Status

**Current state:** Modules 0 and 1 complete and signed off; Module 2 implementation ready.  
**Active module:** Module 2 — awaiting full manual CORE build and validation.  
**Next module:** Finish Module 2 full build; Module 3 remains not started.  

Module 0 remains complete and signed off. Its 9,494-record manifest, validity-aware
primary-microphone decisions, four exclusions, deterministic speaker folds, and leakage
PASS are unchanged.

Module 1 provides float64 16 kHz mono working audio, conservative RMS normalization,
project-owned pure-NumPy VMD, and all-mode reconstruction. The full CORE build completed
with 8,673/8,673 valid cache entries and zero failures. The production run measured
12,219.264 seconds (3.394 hours) end to end and 11,342.994 aggregate VMD seconds. Its
cache occupies 19,981,366,416 bytes (18.6091 GiB).

The strict VMD update criterion converged for 8,625 records. The remaining 48 reached
the configured 500-iteration ceiling, but an independent audit and the full validator
confirmed finite arrays, correct dimensions and frequency bounds, exact all-mode sums,
valid source/cache identities, and reproducible reconstruction metrics. This is an
accepted scientific limitation, not a record failure. Reconstruction metrics remain
descriptive; Module 1 does not claim denoising, losslessness, or guaranteed preservation
of every dysarthria cue. Raw TORGO and the signed Module 0 interface remain unchanged.

Authoritative Module 1 evidence:

- `docs/modules/1_shared_vmd/TEST_REPORT.md`
- `artifacts/metrics/module1_vmd_summary.json`
- `artifacts/metrics/module1_full_signoff_audit.json`
- `artifacts/logs/module1_vmd_failures.csv`
- `artifacts/logs/module1_nonconverged_records.csv`
- `artifacts/logs/module1_cache_inventory.json`
- `artifacts/logs/module1_vmd_pipeline.log`
- `artifacts/plots/module1_vmd/`
- `docs/modules/1_shared_vmd/TAU_FIDELITY_REPORT.md`
- `artifacts/metrics/module1_tau_fidelity_sweep.json`

Module 2 now provides deterministic log-Mel, binary recurrence, and strict natural
visibility representations from the same signed all-mode VMD reconstruction. Its
12-record production smoke panel passed build and validation with zero failures, including
the valid iteration-ceiling record `F03/Session1/0179`. The representation config hash is
`d8a36c6ab8c48b15e61ecd63194ffa30ede4e7e7e7ab0a30c197002652f9d65d`.

The pre-full-build crop gate audited all 235 CORE records longer than eight seconds and
generated a fixed six-record real crop panel. Every Mel output matched the explicit
centered 128,000-sample reference, while recurrence and visibility continued to summarize
the complete reconstruction. The longest record, `F03/Session1/0118` (193.828625 s),
passed with a Mel interval of 92.9143125-100.9143125 s. The 8-second initial policy and
representation config hash remain unchanged; the 235 cropped records are explicitly
deferred to Module 5 robustness analysis.

The 40 Module 2 tests and full 184-test project suite pass with 82.07% branch-aware
coverage; Ruff, strict mypy, and compileall pass. Raw TORGO and the production Module 1
cache remain unchanged. Do not mark Module 2 complete or start Module 3 until the manual
8,673-record build and full Module 2 validation pass.
