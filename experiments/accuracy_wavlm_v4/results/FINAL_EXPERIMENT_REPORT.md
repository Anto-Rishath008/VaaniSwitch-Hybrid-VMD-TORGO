# V4 POST-HOC EXPLORATORY

Recording endpoint; equal-fold mean and sample SD. Positive class: dysarthric.
Reused TORGO folds: POST-HOC EXPLORATORY. Independent external validation has not occurred.

| Model | Accuracy mean +/- SD | Macro-F1 mean +/- SD | Speaker accuracy |
|---|---:|---:|---:|
| frozen_reference | 0.8018 +/- 0.0899 | 0.7793 +/- 0.0886 | 0.8000 |
| selected | 0.7922 +/- 0.0992 | 0.7680 +/- 0.1010 | 0.7333 |

Historical comparison (same endpoint, exploratory):
- Original B2: accuracy 0.6972 +/- 0.1607.
- Original C2: accuracy 0.6502 +/- 0.0903.
- V2 E1: accuracy 0.7479 +/- 0.0817.
- V3 Selected: accuracy 0.7670 +/- 0.0595.

15 speakers and previously exposed outer results; no untouched external test set.
No clinical diagnostic validity, significance, SOTA or guaranteed 90 percent claim.
Equal-fold mean and sample SD; speaker aggregation is descriptive.
Development search, early stopping, overlapping development contexts and one seed limit inference.
