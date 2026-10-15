# Exploratory analysis tracker

| Phase | Status |
|---|---|
| Locate B2/C2 native frozen outer predictions | PASS: six cells, three folds |
| Verify IDs, probabilities, true labels and speaker roles | PASS: 8,673 records and 15 speakers per model |
| Fixed mean-probability aggregation at 0.5 | PASS: exactly one decision per speaker per model |
| Metrics, uncertainty and comparison | COMPLETE: stored in speaker_metrics.json and ANALYSIS_REPORT.md |
| SLA-001 provenance check regression | PASS: three regression checks; see REGRESSION_TESTS.log |
| Preservation and output integrity | PASS upon publication: see PRESERVATION_REPORT.md and output_manifest.json |
| Further experiments | STOP: await user review |

This file tracks only this exploratory analysis. Original MODULE_TRACKER.md and
all Modules 0–6 remain unchanged.
