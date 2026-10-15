# Accuracy HuBERT V2 changelog

2026-10-13 (Asia/Calcutta):

- Forked the study source into V2; retained all V1 artifacts and identities.
- Reproduced V1's FP16 scaled-backward overflow from its last complete checkpoint.
- Switched E2/E3 training, benchmark and prediction to FP32; kept extraction unchanged.
- Added fail-closed finite checks, record-level numerical diagnostics and explicit
  checkpoint precision validation; retained bounded epoch rollback for OOM.
- Added verified read-only feature reuse with complete identity and crop checks.
- Retained original model architectures, candidate grid, folds, labels and metrics.
- Corrected the V2 report's reproduction command to point to the V2 launcher.
- Added numerical and launcher regression tests and bounded real-development CUDA
  validation with interruption/resume checks. Full experiment awaits user review.
