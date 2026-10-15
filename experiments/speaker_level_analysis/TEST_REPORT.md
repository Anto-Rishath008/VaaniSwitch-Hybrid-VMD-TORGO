# Exploratory analysis validation report

SLA-001 was reproduced before the fix: the retained native predictions were
incorrectly rejected with `ValueError: Manifest/split hash differs`. See
TEST_FAILURE_BEFORE_FIX.log. The original code defines the fold fingerprint over
canonical JSON, not serialized file bytes.

Targeted regression coverage:

- Accept all six signed native B2/C2 outer cells with their original canonical fold identities.
- Verify that serialization whitespace does not change the canonical fingerprint,
  and that the split file byte hash is correctly treated as a separate identity.
- Detect a changed speaker assignment in an in-memory test document.

Command from the experimental worktree:

```powershell
python -B .\experiments\speaker_level_analysis\test_analysis_regression.py
```

Result: **3 tests PASS**, recorded in REGRESSION_TESTS.log. These are read-only
analysis regressions; no model construction, forwards, fitting or original data
mutation occurs. The original module test reports remain untouched.

Publication additionally verifies exact CORE record coverage, finite saved
probabilities, labels and speaker roles against the signed manifest, six native
result/checkpoint hashes, all original utterance metrics against scikit-learn,
independent arithmetic metrics/AUC, 15 unique speakers per model, CSV/JSON metric
agreement, output checksums, and before/after preservation. Re-run retained output
verification with `analyze_speakers.py --verify`.
