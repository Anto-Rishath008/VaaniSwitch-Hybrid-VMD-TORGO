# V2 numerical-stability repair handoff

Date: 2026-10-13, Asia/Calcutta. Implementation and bounded validation PASS.
The complete experiment has not been run. No real outer-test inference or final
accuracy is reported. Stop for user review.

## Diagnosed cause

The V1 log and three epoch checkpoints correct the initial failure description:
epoch 3 completed. Replaying the last saved state, AdamW, GradScaler and RNG on
the original 4,760 Fold 1 inner-training records reproduces failure at epoch 4,
step 540. The loss scale starts at 65,536, grows to 131,072, and the gradient of
`head.3.weight` becomes non-finite. Model parameters, optimizer state, inputs,
forward logits and loss remain finite. The failing loss is 0.5813971758.

On the same batch with restored pre-forward RNG, FP16 without amplification
(scale one) gives finite gradients with norm 10.4059725; FP32 gives finite
gradients with norm 12.1264553. Autocast and FP32 can consume CUDA dropout RNG
differently, so these norms/losses are diagnostic checks, not numerical-equivalence
or accuracy comparisons. This identifies FP16 scaled-backward overflow, not
corrupt input or checkpoint parameters.

V1 correctly unscaled before clipping, but `error_if_nonfinite=True` raised before
GradScaler could skip the overflowed update and back off its scale. The native
adaptive-pooling deterministic warning is separate from this E2 failure.
Evidence: `logs/v1_diagnosis.json`, `diagnose_v1.py`, original immutable run log.

## Repair and numerical policy

- E2/E3 training, benchmarking, validation and future outer prediction: CUDA FP32;
  autocast explicitly disabled, no GradScaler, no BF16, TF32 disabled for neural training.
- Frozen HuBERT extraction: unchanged FP16 autocast plus FP32 masked pooling.
  `src/features.py` is byte-identical to V1; no corpus extraction was repeated.
- Check finite raw/standardized inputs, scaler, RP, parameters, AdamW state,
  logits, weighted loss, every gradient, total norm, and post-update state.
- Keep clipping at norm 5 with `error_if_nonfinite=True`. Numerical faults stop,
  identify records/model/layer/candidate/epoch/batch, and cannot publish a completed
  checkpoint or silently skip a batch.
- OOM alone permits whole-epoch rollback with a halved batch, stopping at one.
  Parameters, AdamW and Python/NumPy/CPU/CUDA RNG restore together. Clipping OOM
  preserves its exception type for this recovery.
- Checkpoints bind the V2 identity, FP32 policy, scaler provenance, optimizer, RNG,
  best state, history, patience and epoch. V1 AMP checkpoints cannot resume V2.

Architecture, labels, logical records, speaker folds, weights, threshold .5,
selection macro-F1, candidate grid and optimization hyperparameters are unchanged.

## V1 preservation and feature reuse

All 17,540 V1 files match pre-repair SHA-256, size and mtime, including frozen_run,
stages, all features, E1/E2 checkpoints, failure logs and implementation evidence.
No V1 results/outer artifacts exist. No V1 identity was reset.

All 8,673 V1 entries passed complete payload/per-layer checksums, exact logical and
speaker identity, primary-source bindings, pinned model revision/file hashes,
feature/preprocessing/dependency identity and crop-interval validation. All 235
cropped intervals agree. V2 uses a read-only cache class whose save operation
always raises; missing/corrupt entries fail and are never repaired. The full
launcher does not construct an extractor. A verified local copy of the model files
supports the unchanged synthetic-wave extraction regression test only.

V2 is `experiments/accuracy_hubert_v2/`, with its own source and protocol identities,
checkpoints, stages, results, plots, logs and launcher. V1 E1 fits remain preserved;
V2 will fit its own E1 stages under the new identity. `V1_LINEAGE.json` binds the
incomplete predecessor and documents the repair. The first feature-only V2
preparation was retained under `logs/preparation_revision_1/` before the final
clipping-OOM exception correction; no study training/outer inference occurred
under that provisional identity.

## Validation

56 tests PASS, zero failures/errors/skips, 13.755 seconds. This includes all 38
inherited tests plus 18 new numerical/recovery/launcher tests. The opt-in real
HuBERT CUDA extraction test used synthetic waveforms, not new corpus extraction.

Actual development-only CUDA results:

| Check | E2 | E3 |
|---|---|---|
| Main layer-6 optimization | 4 epochs, 2,380 steps, PASS | 4 epochs, 2,380 steps, PASS |
| Records per epoch | 4,760, eight Fold 1 inner-training speakers | same |
| Interruption/resume | save at epoch 2, resume to 4, PASS | same |
| RNG roundtrip and completed-fit reuse | PASS | PASS |
| Repeated fixed-checkpoint predictions | exact equality | exact equality |
| Uninterrupted vs resumed training | model/AdamW/RNG/best/history/scaler exact | not claimed |
| Layer 9/12 regularized panels | 16 steps each, PASS | 16 steps each, PASS |

E2 also ran 2,380 uninterrupted comparison steps. The main suite took 286.45 seconds,
excluding initial input/cache loading. No non-finite state occurred. A small final
source-bound check and FP32 benchmarks exercise the finalized code after the narrow
OOM-handler correction; see `logs/final_runtime_check.json`.

Evidence: `logs/tests_final.log`, `logs/cuda_validation.json` pointing to the
retained detailed report, `logs/cuda_validation.log`, `logs/feature_reuse.json`,
`logs/final_runtime_check.json`. Training losses/inner panel metrics are
diagnostic development evidence, not outer accuracy.

The synthetic launcher test mocks fitting/evaluation and verifies all nine
model/fold decisions precede any evaluation. Its test log contains the launcher's
normal progress strings, including “FULL EXPLORATORY STUDY COMPLETE”; that string
belongs only to the synthetic orchestration test and is not a real experiment.
The original five reports and three figures were generated with synthetic fixtures.

## One-command handoff

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_Accuracy_Study\experiments\accuracy_hubert_v2\run_all.ps1"
```

Uses the existing .venv-accuracy and CUDA, verifies/reuses embeddings, executes
E1/E2/E3 across all three folds, safely resumes unfinished work, freezes all final
fits before outer evaluation, and publishes the original required outputs.
No interactive input. The command was parsed and its orchestration tested, but
the complete launcher was deliberately not executed.

## Limits and stop boundary

No known blocker to the manual V2 launch remains. The 63-candidate search, nine final
fits, actual outer evaluation, full runtime and achieved accuracy remain untested.
E3's native adaptive-pooling backward remains nondeterministic; bitwise resumed
training equality is not promised. FP32 may take longer than FP16.

Original Modules 0–6, the release branch and prior speaker analysis are unchanged.
Raw/processed original corpus preservation uses metadata inventories; project
evidence and V1 use hashes. See PRESERVATION_REPORT.md for exact scope.
No commit, push, cleanup, reset or original scientific-data write occurred.

STOP: repair/testing complete; wait for user review.
