# V2 defect log

## AHV2-001: FP16 loss scaling overflow aborts before scaler recovery

Reproduced from the V1 epoch-3 checkpoint at epoch 4, step 540, loss scale 131,072.
The final head weight gradient is non-finite despite finite inputs, parameters,
AdamW state, logits and loss. The same batch has finite gradients at FP16 scale one
and FP32. Clipping with `error_if_nonfinite=True` raises before GradScaler can
skip the update and reduce its scale. The clip/unscale order itself is correct;
the unsafe assumption was that every unscaled AMP gradient would be finite.

Fix: V2 E2/E3 use FP32 throughout and no GradScaler. Explicit checks cover inputs,
state, forward, loss, individual gradients, norm and post-step state. No silent
gradient skipping or numerical retry. Identity-bound epoch checkpoint/RNG recovery
is retained for interruption/OOM only.

Regression evidence: `test_numerical_stability.py`, inherited recovery tests,
`logs/v1_diagnosis.json` and the bounded CUDA validation report.

## AHV2-002: A copied V1 report would advertise the old launcher

Fix: V2 report reproduction instructions target `accuracy_hubert_v2/run_all.ps1`.
Regression: synthetic report-generation assertions and V2 launcher orchestration
test. Original V1 reporting source is unchanged.

No scientific settings or original module defects were changed.
