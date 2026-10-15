# Exploratory analysis defect log

| ID | Defect | Resolution | Status |
|---|---|---|---|
| SLA-001 | The new analysis compared the stored canonical JSON fold fingerprint with the byte hash of speaker_folds.json, incorrectly rejecting unchanged evaluation evidence. | Match src/vaaniswitch/m0_data/leakage.py speaker_folds_fingerprint; retain the distinct file byte hash in provenance. Regression reproduces the original failure, accepts six frozen native cells, ignores formatting and detects a changed speaker assignment. | CLOSED after regression PASS |

This defect was confined to the new exploratory script. No original data, module,
prediction, split, checkpoint or scientific sign-off was changed. Original
module logs remain protected. See TEST_FAILURE_BEFORE_FIX.log and REGRESSION_TESTS.log.
