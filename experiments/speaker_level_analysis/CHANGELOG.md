# Exploratory analysis changelog

2026-10-11 (Asia/Calcutta): Add fixed-rule speaker aggregation of existing B2/C2
outer predictions, metrics, uncertainty, charts and preservation evidence. No
project/model imports, retraining, inference or original module changes.

Correct SLA-001, an analysis-only provenance check: canonical JSON fold identity
is distinct from the split file byte hash. Retain the original before-analysis
snapshot and resume without replacing it. Add three targeted regression checks.
