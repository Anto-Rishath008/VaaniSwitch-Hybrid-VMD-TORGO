# VaaniSwitch accuracy-study implementation handoff

Date: 2026-10-13, Asia/Calcutta.

**IMPLEMENTATION STATUS: PASS. CUDA SMOKE TEST: PASS.**

This is implementation readiness backed by focused and bounded runtime tests.
The complete experiment has not run. There are no observed E1/E2/E3 outer metrics,
final accuracy claims, full-corpus extraction results or claims of 80–90% accuracy.

## 1. Scope and source audit

All explicit source/output writes are under the experimental worktree's
`experiments/accuracy_hubert/`. Branch:
`experiment/dysarthria-accuracy-improvement`. HEAD:
`2dee81f7656e5b5cae800d9a3c8692e5e8107bf9`.

Read PROJECT_SCOPE_LOCK.md first, then AGENTS.md, FINAL_WORKFLOW.md,
MODULE_TRACKER.md and PROJECT_STATUS.md. Inspected Module 0 manifest/schema/selection,
Module 1 audio and VMD provenance, Module 2 cache/config, Module 3 recurrence code
and preprocessing, Module 4 splits/fusion, Module 5 ablations/search/training contracts,
their signed reports, Module 6 current sign-off, and the existing speaker analysis.
The new study does not introduce Module 7 or replace the locked project architecture.

The original manifest contains 9,494 logical records. The unchanged CORE selection
contains 8,673, with 7,503 head and 1,170 array primaries. Its three original outer
folds contain 2,819, 2,873 and 2,981 records. Exact original inner-validation speakers
are reused. All speaker/record/path/waveform-hash role intersections are checked.
Existing full-signal VMD recurrence plots are consumed, never regenerated.

Original Module 5/6 signed outputs and the earlier 11/15-speaker B2/C2 assessments
remain unchanged. The historical Fold1/A1 RECOVERED_PROTOCOL_EXCEPTION and unknown
prior native-forward count are not relabelled or repaired by this study.

## 2. Source files created

- `src/common.py`: identities, atomic publication, exclusive process lock, stage events, preservation.
- `src/data.py`: frozen CORE/microphone/speaker roles and read-only RP checks.
- `src/features.py`: pinned checkpoint, CUDA extraction, exact-length masks/pooling, immutable cache.
- `src/models.py`: compact E2 and fresh B2/HuBERT fusion E3.
- `src/training.py`: E1, scalers/weights, CUDA fits, RNG/checkpoint recovery and outer claims.
- `src/pipeline.py`: audit, dedicated smoke and one-command full-run orchestration.
- `src/reporting.py`: metrics, speaker descriptions, comparisons, five output files and three plots.
- `src/native_b2_contract.py`, `native_b2_encoder.py`, `native_b2_preprocessing.py`: isolated original recurrence adapter.
- `src/__init__.py`; `tests/test_contracts.py`, `test_recovery.py`, `test_pipeline.py`.
- `run_all.ps1`, `TRAINING_CONFIG.yaml`, `.gitignore`, README, architecture/protocol/test reports,
  this handoff, experiment tracker, changelog and defect log.

The adapter's recurrence implementation matches the original AST after two import
redirections. Preprocessing matches the original bytes. The production implementation
and environments are not edited; torchvision/pytest installation is unnecessary.

## 3. Models

| Model | Architecture | Trainable classifier parameters |
|---|---|---:|
| E1 | Frozen HuBERT → training-only StandardScaler → regularized logistic regression | 1,537 |
| E2 | Frozen HuBERT → 128-wide projection → compact MLP → binary logit | 213,633 |
| E3 | Original RP CNN → 128-wide mean-pooled RP embedding + 128-wide projected HuBERT → concatenation → fusion MLP → binary logit | 310,833 |

HuBERT has 94,371,712 frozen parameters, outside every classifier optimizer.
The E3 CNN and head start fresh in every candidate/final fit; no original checkpoint
exposed to test speakers is loaded. All new neural components receive gradients.

## 4. Exact pretrained model and features

[Official facebook/hubert-base-ls960](https://huggingface.co/facebook/hubert-base-ls960),
revision `dba3bb02fda4248b6e082697eee756de8fe8aa8a`, loaded using Transformers 4.57.6.
The official weight file downloaded and verified successfully:
`062249fffb353eab67547a2fbc129f7c31a2f459faf641b19e8fb007cc5c48ad`.
Config and preprocessor hashes are also pinned in TRAINING_CONFIG.yaml and checked
on every acquisition/reuse. No external download blocker remains.

Use original primary 16 kHz mono audio, centered maximum-eight-second segments,
checkpoint per-waveform normalization, eval/frozen/inference_mode, CUDA FP16 autocast
and float32 masked mean+population-std pooling. Candidate hidden states 6/9/12 each
produce 1,536 features; each cache entry is [3,1536]. Layer selection is development-only.
Record every crop interval; the frozen corpus has 235 recordings over eight seconds.

GroupNorm requires an explicit padding policy: short signals remain unpadded, and
batches contain exact equal lengths. This avoids padding-dependent valid features.
Sample masks and convolution frame lengths are verified; mixed-length batches fail.
Every entry binds record/speaker/source/model/layer/shape/checksum/completion identities.

## 5. Fold protocol

Original three outer folds; 8 inner-training/2 inner-validation/5 test speakers per
fold. Fit weighted scalers and speaker weights using only training speakers. No
additional class weighting or weighted sampling. Fixed seed 42 and threshold .5.
E1 searches three layers × C {.01,.1,1}. E2/E3 each search three layers × two regularization
settings, AdamW at .0003, maximum 20 epochs, patience 4. Primary selection: inner macro-F1.

Fresh final fits use all ten development speakers for the selected inner epoch count.
All nine model/fold decisions and final fits are frozen before any real outer inference.
The study will run 63 candidate fits plus nine final fits and serial outer evaluation.

## 6. Actual CUDA smoke and performance

Existing environment verified: Python 3.12.2, PyTorch 2.13.0+cu126, NumPy 1.26.4,
Transformers 4.57.6, NVIDIA GeForce RTX 4070 Laptop GPU. No rebuild/reinstallation.
The final smoke used six Fold 1 development records only: four training records
from FC02/F01 and one validation record each from M03/MC02. Two were center-cropped.
One neural epoch per E2/E3; actual feature extraction, finite optimization, checkpoint
roundtrips, cache reuse and stage reuse pass. No real outer forward occurred.

Evidence: `smoke/cuda-1abefcda4275498b9684a48562d589c1/SMOKE_REPORT.json` and
`logs/latest_smoke.json`. A previous interrupted smoke resumed its already completed
six feature entries; historical namespaces and failure logs are retained.

- Peak PyTorch allocated: **943,448,064 bytes = 0.879 GiB**.
- Peak PyTorch reserved: **1,270,874,112 bytes = 1.184 GiB**.
- These include disposable extraction/training benchmarks, and exclude whole-driver/context accounting.
- Eight-second HuBERT batch timings: 1 = .0359 s, 2 = .0630 s, 4 = .0657 s; selected conservative cap two.
- Batch-eight measured neural steps: E2 .00773 s, E3 .03253 s; conservative neural cap eight.
- Actual LR BLAS/OpenMP fitting threads are bounded at two and covered by a runtime test.

**Full runtime is unmeasured.** Tiny-batch extrapolation suggests approximately
**1–5 hours for planning**, depending on early stopping, validation, checkpoint I/O,
CPU logistic fitting, thermals and OneDrive. The measured-step neural compute estimate
alone is .59–2.75 hours under 5–20 inner epochs and 1–20 final epochs; it excludes
common extraction and other overhead. See `logs/runtime_estimate.json`.

The native B2 adaptive-pooling CUDA backward has no deterministic implementation.
The exact B2 architecture is preserved; seeded warn-only training records the warning.
Fresh/resumed E3 training is not promised bitwise equality. Repeated fixed-checkpoint
evaluation passes exact-equality tests. E2 interrupted and OOM-recovered parameter
equality is separately verified.

## 7. Single full-run command

Run manually after review, from any location:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_Accuracy_Study\experiments\accuracy_hubert\run_all.ps1"
```

No separate fold commands, interactive choices, epoch edits or checkpoint selection.
The launcher verifies environment/GPU/free disk, inputs and leakage, verifies/acquires
the pinned model, extracts/validates features, trains/selects/final-fits E1/E2/E3,
executes folds 1/2/3, publishes recording/speaker reports and figures, then checks integrity.

## 8. Recovery

Rerun the same command. Verified completed features and fits are reused. Training
checkpoints preserve model, AdamW, AMP, Python/NumPy/CPU/CUDA RNG, best epoch, patience
and history. An interrupted partial epoch restarts from the last complete boundary.
OOM restarts only unfinished pre-evaluation work with a smaller batch and rollback.
Historical checkpoint generations and failure events remain available.

A completed outer result is reused without test inference. A consumed claim lacking
a valid result+seal, or a changed/corrupt claim/result, stops for explicit review.
There is no automatic outer retry/reset. Config/source/dependency/input drift also stops.

## 9. Tests and preservation

**38 tests PASS, zero failures/skips, 16.208 s**; see TEST_REPORT.md and
`logs/handoff_tests.log`. These cover every requested contract plus failure recovery,
synthetic report/figure generation and bounded CPU pools. Python source/test parsing,
PowerShell parsing, tracked whitespace and cache ignore checks pass.

**Preservation PASS:** 191,833 original inventoried files unchanged: 22,175
project/evidence files by SHA-256/size/mtime, and 169,658 raw/processed corpus files
by path/size/mtime. All 417 pre-existing tracked experimental files and 23 existing
speaker-analysis files match. Original dirty/untracked state, both branch names and
baseline HEADs are unchanged. The original Module 5 closure manifest and eight
package hashes also match their signed pins.

Corpus inventories are metadata checks, not a new full content hash audit. The six
selected smoke primary WAVs and RP artifacts were separately SHA-256 verified. The
full launcher will check every selected primary/RP input before full extraction.
Environments, Git internals and tool caches are excluded from preservation inventories;
no installation/write was issued against either original environment. No original
checkpoint was deserialized, no original model was forwarded, and no commit/push occurred.

Retained machine evidence: `logs/implementation_preservation_before.json`,
`implementation_preservation_after.json`, `implementation_preservation_summary.json`,
`handoff_validation.json`, and `HANDOFF_MANIFEST.json`. Earlier validation snapshots
remain retained under their original filenames.

## 10. Remaining boundary

No unresolved external or implementation blocker is known. Full-corpus extraction,
complete nested training, real outer evaluation and observed final result generation
remain deliberately unexecuted. Their runtime success and achieved accuracy remain
unverified until the user's manual run. The reporting generator and all three figures
were tested using explicitly synthetic fixtures only.

**STOP: implementation and bounded validation complete; wait for user review.**
