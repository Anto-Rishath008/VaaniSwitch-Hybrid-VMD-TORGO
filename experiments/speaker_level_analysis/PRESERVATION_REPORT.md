# POST-HOC EXPLORATORY SPEAKER-LEVEL ANALYSIS — preservation

**PASS.** All new files are confined to `C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_Accuracy_Study\experiments\speaker_level_analysis`.

- Original project: 191,833 inventoried files unchanged before/after.
- 22,175 project/evidence files: SHA-256, size and modification time unchanged. This includes all inventoried source, tests, configs, documents, checkpoints, ledgers, predictions and sign-off artifacts.
- 169,658 raw/processed files: path, size and modification time unchanged. Contents were not opened or rehashed; this does not assert a new full corpus integrity audit.
- All 636 signed Module5 closure input hashes and 8 package files match their pinned hashes; the manifest itself matches `f9d4e56917b2bfc63abbf312e2b389457aa2ad29734424f01da5ee03109a8a6d`.
- All 417 tracked files in the experimental checkout remain byte-for-byte unchanged.
- Both HEADs remain `2dee81f7656e5b5cae800d9a3c8692e5e8107bf9`; original release and experimental branch refs unchanged.
- Original pre-existing modified/untracked status preserved, including `artifacts/metrics/module2_representation_summary.json`. No commit or push.
- No original project Python imports, model loading/deserialization, forwards, training, fine-tuning, original metric replacement or Modules0-6 edits. Checkpoints were read only as bytes for hashing. No raw WAV or processed representation loaded.
- Exclusions: virtual environments, Git internals, Python bytecode and tool caches. Git branch refs/commit/status checked separately using GIT_OPTIONAL_LOCKS=0. Matplotlib cache is under this analysis directory.

Evidence: `preservation_before.json`, `preservation_after.json`, `preservation_summary.json`, `source_provenance.json`, `output_manifest.json`. The before/after original-file mappings are identical.

No existing module defect was fixed, so original module tracker, changelog, defect logs and test reports were preserved. An analysis-only fold fingerprint check was corrected with a regression test; TEST_REPORT.md, ANALYSIS_TRACKER.md, CHANGELOG.md and DEFECT_LOG.md record that work inside this analysis directory.

**STOP. Further experiments require user review.**
