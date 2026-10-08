# Module 4 Test Report — Token-Level Cross-Attention Fusion & Dysarthria Classification

**Status:** COMPLETE / PASSED / SIGNED OFF
**Closure date:** 2026-09-13
**Scope:** Completed Module 4 baseline audit and final documentation closure.
**Active module:** None. Module 5 is not started and awaits explicit authorization.

## Current baseline result

The manual real baseline completed **six epochs** and stopped after **five consecutive
non-improvements** under the locked patience-5, min-delta-1e-4 validation macro-F1 rule.
**Epoch 1** remains selected: it has the highest validation macro-F1. Later lower training
loss, and epoch 4's lower validation loss, do not change the frozen selection criterion.
No training, threshold tuning, hyperparameter change or outer-fold evaluation was performed
as part of final closure.

These values are a **SPEAKER-DISJOINT DEVELOPMENT-VALIDATION BASELINE ONLY**.
They are not final test accuracy, final TORGO performance, final generalization performance
or the final project result. Reserved outer Fold 1 remains untouched during Module 4.

| Partition | Speakers | Control | Dysarthric | Records |
|---|---|---:|---:|---:|
| Train | F01, F04, FC02, FC03, M01, M02, M05, MC01 | 3,306 | 1,454 | 4,760 |
| Validation | M03, MC02 | 684 | 410 | 1,094 |
| Reserved Fold 1 | F03, FC01, M04, MC03, MC04 | Metadata only | Metadata only | 2,819 |

All pairwise speaker overlaps are zero. All 8,673 CORE logical speaker/session/prompt keys
are unique. Cross-partition intersections of logical IDs, primary audio paths and signed
primary-audio SHA-256 values are zero; microphone alternatives do not create dataset rows.
Independent train-only `pos_weight=3306/1454=2.2737276478679505` equals the stored weight.
The loss uses its normal float32 representation, recorded precisely in the audit artifact.
Validation and reserved speakers contribute no weight counts.

## Saved and independently reproduced validation

The user's standalone validation artifact is PASS. A separate audit reloaded `best.pt`,
ran the production validation loop on all 1,094 M03/MC02 records, and independently
recomputed confusion-based metrics, pairwise ROC-AUC with half-credit for ties, and stable
double-precision weighted BCE. Per-record predictions were kept in memory only.

| Metric | Saved epoch 1 | Independent calculation |
|---|---:|---:|
| Weighted BCE loss | 0.4966598420354522 | 0.4966598356960653 |
| Accuracy | 0.896709323583181 | 0.896709323583181 |
| Balanced accuracy | 0.8744080730280988 | 0.8744080730280988 |
| Macro-F1 | 0.8858803969843076 | 0.8858803969843076 |
| Positive precision | 0.9279538904899135 | 0.9279538904899135 |
| Sensitivity | 0.7853658536585366 | 0.7853658536585366 |
| Specificity | 0.9634502923976608 | 0.9634502923976608 |
| ROC-AUC | 0.9620774497218656 | 0.9620774497218656 |
| TN / FP / FN / TP | 659 / 25 / 88 / 322 | 659 / 25 / 88 / 322 |

The reproduced production metrics equal the saved and user-validation metrics. The maximum
saved-versus-independent absolute difference is **6.339386904308242e-09**, solely in BCE;
it is below the 1e-6 tolerance. Classification metrics and counts agree exactly.
The matrix `[[659,25],[88,322]]` contains exactly 684 controls and 410 dysarthric records.

## Checkpoint, freeze, numerical health and determinism

Best: `artifacts/checkpoints/module4/fold1_seed42_6fc69cf0ad55/best.pt`, epoch 1,
**48,367,037 bytes / 46.1264009475708 MiB**; SHA-256
`07bae2a14fd3188c3a021f05a3a11c4dbb94cef3fc35cccd35f79cc691844051`.
Last: the same directory's `last.pt`, epoch 6, **48,368,829 bytes / 46.1281099319458 MiB**;
SHA-256 `df5775291a0f5a36d5d98c5fd9978c5ab8daf70c3753a30d467b179fa3d25c8c`.
Both checksums match their sidecars; both retain best epoch 1.

The exact checkpoint field list is: `schema_version`, `module`, `mode`, `model_state`,
`optimizer_state`, `epoch`, `best_metric`, `best_epoch`, `bad_epochs`, `pos_weight`,
`hashes`, `processing_identity`, `split`, `split_hash`, `selection`, `seed`, `rng`,
`history`, `pretrained_weights`, `runtime`, `outer_test_forward_count`. Required model,
AdamW, RNG, epoch/history, seed-42, config/source hashes, split and selection identities
all passed restricted loading and schema checks. There are no embedded dataset arrays.
Best/last AdamW steps are 298/1,788, matching 298 batches per epoch and epochs 1/6.

All **120 ResNet backbone state tensors**, including BatchNorm buffers, in both checkpoints
are bit-exact to the original locally cached, hash-verified ImageNet-1K V1 ResNet18 weights.
All **11,176,512** backbone parameters have `requires_grad=False`, no gradients, and no
optimizer membership. The backbone remains in eval mode through model training mode.
The full 170-tensor model state and optimizer tensors are finite in both checkpoints.
Every retained epoch loss/metric is finite and each confusion matrix is consistent.
Reproduced logits/probabilities are finite; probabilities range from
0.01134691946208477 to 0.99433434009552.

After reload, two eval forwards of the same four-record validation batch produced
**bit-exact logits and probabilities**. The batch contains M03/Session2/0002,
MC02/Session1/0006, M03/Session2/0003 and M03/Session2/0004. It contains no Fold 1 record.

## Evidence for zero reserved-test peeking

`outer_test_forward_count = 0`. Actual source-load and model pre-forward hooks in the
audit permitted only M03 and MC02: 1,094 unique validation records over 69 batches,
plus four loads for the fixed batch and two four-record determinism forwards. Total audit
loads were 1,098 and record-forwards 1,102; reserved-fold loads and forwards were zero.
The validation logical-ID hash exactly matches the frozen checkpoint selection.

The retained baseline proof combines the metadata-only speaker split, source selection,
pre-forward guards, one baseline run-start and six matching epoch rows, best/last history,
training-only weighting and locked threshold/selection policy. Prediction, attention-map,
representation-output and demo-output directories contain no generated files. The Module 4
artifact/text/temporary-output inventory contains no reserved logical record IDs and no
unexpected checkpoint collection. Historical upstream Module 1–3 diagnostics are preserved
and are not Module 4 baseline evaluations.

The original training counter is a policy assertion, not a historical instrumented trace.
Its auditable equivalent is the consistent selection/log/checkpoint/code evidence above;
actual source and forward counters apply to the reproduced validation audit. This audit
establishes what the retained evidence supports without claiming a retroactive trace.

## Measured runtime and development behavior

| Measurement | Seconds |
|---|---:|
| Total epoch training stages | 3,849.341423500002 |
| Total epoch validation stages | 609.903653399997 |
| Total measured train + validation stages | 4,459.245076899999 |
| Mean / median epoch training | 641.556903916667 / 639.0533179000013 |
| Minimum / maximum epoch training | 523.4901983999953 / 753.377751 |
| Mean epoch validation | 101.6506088999995 |

The measured stage total is **74.320751 minutes**. It excludes unmeasured setup, checkpoint
writes and other command overhead; it is not the earlier estimated maximum 30-epoch runtime.
Training loss decreased substantially after epoch 1 while validation macro-F1 never
exceeded its epoch-1 maximum. This is compatible with early development-set divergence or
possible overfitting; stronger conclusions and optimization decisions belong to Module 5.

One compact curve is retained at
`artifacts/plots/module4_baseline/development_curve.png` (train loss, validation loss and
validation macro-F1 over six epochs; development only). No duplicate curve was generated.

## Final gates, failures and integrity

| Gate | Final result | Evidence provenance |
|---|---|---|
| Module 4 tests | 96 passed | Fresh `module4_signoff_targeted_tests.log`, 90.08 s; user also confirmed |
| Full project tests | 328 passed | User-confirmed final manual run; retained implementation log agrees |
| Branch-aware pytest-cov gate | 84.3782837127846%, required >=80%: PASS | User-confirmed final manual run; retained coverage JSON/log agrees |
| Ruff | PASS | User-confirmed final manual run; prior implementation gate agrees |
| Strict mypy | PASS, 109 source/test files | User-confirmed final manual run; prior implementation gate agrees |
| compileall | PASS | User-confirmed final manual run; prior implementation gate agrees |

The final full-suite/static commands were not rerun during documentation closure. Their
latest successful manual completion is the user's explicit attestation. Retained full-suite
and coverage logs predate baseline training; they are supporting implementation evidence,
not relabelled as newly captured final logs. The configured coverage gate combines statements
and branches: 7,227/8,565 = 84.38% overall and 1,082/1,178 = 91.85% for Module 4.
Branches alone are 1,583/2,190 = 72.28% overall and 238/286 = 83.22% for Module 4.
No exclusions, threshold reductions or failure suppressions were introduced.

Production failure CSV: zero rows. The completed baseline has no recorded failed record,
failed batch, checkpoint failure or resume/recovery event. Both checkpoints load successfully;
the audit has zero failed records/batches. Earlier intentional defect reproductions and
synthetic rejection tests are historical test evidence, not production failures.

The recorded sign-off-input scan equals both the original implementation snapshot and signed
predecessor values: raw 82,772 files / 15,845,129,970 bytes; Module 1 26,019 files /
19,981,366,416 bytes; Module 2 8,673 logical directories / 43,365 files / 2,378,359,184 bytes.
The exact fingerprints are preserved in the protected-input table in the historical evidence
below and in the final audit JSON. All 58 captured signed predecessor files match content
SHA-256, size and mtime. Tree inventories establish path/size/mtime equality; they do not
rehash every raw byte. Final closure uses the completed scan and does not rebuild caches.
Module 3 remains D=128, frozen ImageNet ResNet18, RP channels [16,32,64], GCN 2->96->128.

No new production defect was found. M4-CODE-001/002 remain CLOSED. M4-SCI-001 remains an
accepted limitation: both validation speakers are male. Upstream VMD fidelity/iteration
ceiling and Mel crop limitations persist. Attention weights are diagnostic associations.

## Authoritative closure and next action

`artifacts/metrics/module4_full_signoff_audit.json` records current sign-off, evidence
provenance and integrity. Original `module4_baseline_training.json`,
`module4_baseline_validation.json`, `module4_baseline_audit.json` and implementation/smoke
artifacts remain immutable historical run reports; their pending or false sign-off fields
describe the time they were produced and are superseded by this closure.

All acceptance items in MODULE.md are checked. Reserved Fold 1 remains untouched.
Module 5 is NOT STARTED. The exact next action is explicit user authorization for
Module 5 — Accuracy Optimization, Ablation & Robustness. No next-module work was performed.

## Historical implementation-smoke report (superseded)

Everything below records the earlier implementation stage before the manual baseline.
Its pending-training statements, commands, estimates and gates are historical, not the
current state. The one authoritative current status appears at the top of this report.

Historical status: IMPLEMENTATION READY / AWAITING MANUAL BASELINE TRAINING;
the implementation stage alone did not sign off the module.

## Signed contracts and semantic identity

| Identity | Verified value |
|---|---|
| Module 3 encoder | `7c8e44eddb11704d08e45a1b63bc9cc54b28d679f5e3a3ae1a474b61e3a40383` |
| Module 2 representation | `d8a36c6ab8c48b15e61ecd63194ffa30ede4e7e7e7ab0a30c197002652f9d65d` |
| Module 1 VMD | `bfd7aec36d99a88cfacc72df85e124d3f4b99b76ed47d71c3ced9a153fa92685` |
| Manifest | `91c3327ea1f6a6b2057ffa02198f86d682668d22f247beef27f1466e959a9061` |
| Module 4 fusion | `6fc69cf0ad558258a4b51196e82fee8fc9f2bc6cd25349babd6a99144e68f6de` |
| Speaker-fold document | `5ab4e1af6c54f97af940c3e5ba5d447c77d827102b88656bf19cd5227b5fee27` |
| Inner split | `c911dfcfc4a4a063dfa9dfa05d2c123bccb471ba8d89349d3df171dde3899a45` |
| Signed CORE | 8,673 records |

The actual configs, sign-offs, manifest and full fold document were independently read
and verified. Module 4 consumes the signed Module 3 implementation and its existing
dataset/collate path on the fly. No representation rebuild, learned-token cache, or
Module 0-3 implementation/config change was made. No active D=192 config remains.

## Exact architecture and loss

Q=Mel `[B,16,128]`; K=V=nonlinear `[B,17,128]` (graph then recurrence). One PyTorch
MultiheadAttention with four heads, head width 32, dropout 0.10 and batch-first tensors.
Apply `LayerNorm(original_mel + attention_output)`, mean-pool 16 attended Mel tokens
and all 17 original nonlinear tokens, concatenate two 128-vectors, then
`Linear(256,128) -> GELU -> Dropout(0.30) -> Linear(128,1)` with public logits `[B]`.

Control=0 and dysarthric=1 agree with the actual Module 0 enum/schema. Training uses
BCEWithLogitsLoss without sigmoid in the loss path; the inference convenience property
computes sigmoid. Normal forwards return no attention weights; explicit
`return_attention=True` returns `[B,4,16,17]`. Weights are diagnostic, not causal.

## Development split established before training

| Role | Speakers | Control | Dysarthric | Total |
|---|---|---:|---:|---:|
| Train | F01, F04, FC02, FC03, M01, M02, M05, MC01 | 3,306 | 1,454 | 4,760 |
| Validation | M03, MC02 | 684 | 410 | 1,094 |
| Untouched test Fold 1 | F03, FC01, M04, MC03, MC04 | Metadata only | Metadata only | 2,819 |

All pairwise speaker overlaps are zero. No utterance-level partition is implemented.
One speaker per class is chosen deterministically from the ten development speakers by
seed/fold/class/speaker SHA-256 ranking; settings and split are stored before model
initialization. Train-only baseline `pos_weight=3306/1454=2.2737276478679505`.
No validation/test records contribute. The smoke-panel weight is separately reported
as 1.0 from its two training controls and two training positives.

Both validation speakers are male; this is a development limitation, not a final
generalization result. All three signed outer folds are supported by metadata splitting
and synthetic regression tests; only Fold 1 reservation was used for real Module 4 work.
The suggested F03 smoke records were omitted because the strict untouched-test rule
takes precedence. Existing upstream F03 evidence remains preserved.

## Real forward, gradients and tiny-batch checks

The fixed four-record panel comes entirely from training speakers and contains both
conditions and genders: `F01/Session1/0006`, `M01/Session1/0007`,
`FC02/Session2/0005`, `MC01/Session1/0005`. Its deterministic loader order is stored
in `module4_fusion_summary.json`; no diagnostic prediction is claimed scientifically.

| Output | Shape | Result |
|---|---|---|
| Mel | `[4,16,128]` | finite |
| Graph | `[4,1,128]` | finite |
| Recurrence | `[4,16,128]` | finite |
| Nonlinear | `[4,17,128]` | finite |
| Attention / residual+LN | `[4,16,128]` | finite |
| Mel / nonlinear pools | `[4,128]` each | finite |
| Fusion vector | `[4,256]` | finite |
| Logits / probabilities | `[4]` each | finite; probabilities in [0,1] |
| Optional attention weights | `[4,4,16,17]` | finite |

The BCE backward audit found finite gradients and nonzero aggregate gradient magnitude
in Mel projection, Mel LayerNorm, RP-CNN, GCN, attention/LayerNorm and classifier.
Every frozen ResNet parameter retained requires_grad=false and grad=None. All frozen
weights and BatchNorm buffers remained bit-exact after optimization. Repeated eval
logits and checkpoint reload logits were bit-exact.

Exactly 25 AdamW updates on the same four training records reduced eval BCE from
**0.7021844387 to 0.2713173032**, a **61.36096%** reduction. This is solely a wiring and
optimization sanity check; its accuracy is not a scientific result.

## Checkpoint and training CLI smoke

A separate one-epoch smoke uses four training records and four validation records.
It exercises the actual training loop, weighted BCE, validation metrics, best/last save,
and standalone checkpoint validation. Recomputed saved-epoch validation metrics match
within 1e-6; discrete metrics and confusion counts match exactly. The namespace is
`artifacts/checkpoints/module4/fold1_seed42_6fc69cf0ad55_smoke/` and cannot be accepted
as a full baseline checkpoint. Checkpoints are ignored by Git.

Checkpoint state includes all Module 3/4 weights, optimizer moments, config/split/record
identities, epoch/selection/history, train-only class weight and Python/NumPy/PyTorch
RNG state. Restricted loading, checksum matching, exact tensor/optimizer schemas,
shape/dtype/finiteness and provenance rejection tests pass. Tests also demonstrate
bit-exact interrupted/resumed training using synthetic fixtures.

Long baseline training was not launched. Its intended fixed policy is batch16 AdamW,
lr3e-4, weight decay1e-4, <=30 epochs, patience5, min-delta1e-4 on validation macro-F1,
threshold0.5, seed42, no scheduler/resampling/augmentation/backbone unfreezing.

## Parameter audit

| Component | Total | Trainable | Frozen |
|---|---:|---:|---:|
| Module 3 | 11,369,104 | 192,592 | 11,176,512 |
| Cross-attention | 66,048 | 66,048 | 0 |
| Residual LayerNorm | 256 | 256 | 0 |
| Classifier | 33,025 | 33,025 | 0 |
| Combined | 11,468,433 | 291,921 | 11,176,512 |

All frozen parameters belong to the unchanged ResNet18 feature trunk. RP channels remain
`[16,32,64]`; GCN remains `2->96->128`; token dimension remains 128.

## CPU timing and manual runtime estimate

CPU, two PyTorch threads, real batch size four, one warm-up and five measured eval
forwards per stage. Separate measurements have normal timing variability and are not
additive; they are not full training runtimes.

| Stage | Mean seconds | Median seconds | Max seconds |
|---|---:|---:|---:|
| Module 3 encoding | 0.377280 | 0.382899 | 0.399768 |
| Module 4 fusion/classifier | 0.001006 | 0.001000 | 0.001038 |
| Complete forward | 0.346372 | 0.342416 | 0.371543 |

The tiny training updates measured mean 0.528232 s per four-record update; one strict
load/collate measured 0.258027 s. Linear scaling gives an **ESTIMATE** of 1,100.95 s
(18.35 minutes) per full development epoch, or **9.17 hours for 30 epochs**. Batch16
throughput, filesystem caching, system load and early stopping can substantially change
this estimate. No full-epoch duration has been measured.

## Metric validation and requirement coverage

Hand example: labels `[0,0,0,0,1,1]`, probabilities `[.1,.2,.6,.8,.3,.9]` gives
TN=2, FP=2, FN=1, TP=1, accuracy=.5, balanced accuracy=.5, macro-F1=17/35,
precision=1/3, sensitivity=.5, specificity=.5, ROC-AUC=.75. Ties receive average ranks;
single-class AUC is null with a reason. Threshold .5 is inclusive for positives.

| Requirement group | Evidence | Result |
|---|---|---|
| Signed sources and fixed architecture | config/source rejection tests and live hashes | PASS |
| Q/K/V direction, nonlinear sensitivity, residual and mean pooling | attention hooks and numerical assertions | PASS |
| One-logit direct BCE and canonical labels | extreme-logit and train-only weighting tests | PASS |
| Speaker-disjoint protocol/no outer test forward | all-fold metadata tests; source/batch sentinels; stored real split | PASS |
| Metrics, ties, single-class input | independent hand-computable examples | PASS |
| Gradients, frozen backbone and tiny loss reduction | synthetic and real audits | PASS |
| Checkpoint, corruption, RNG resume and CLI | integration/reproduction tests and real smoke reload | PASS |
| Raw/Module 1/Module 2 immutability | synthetic sentinels plus live full-tree pre/post audit | PASS |
| Prior module regression | all 232 Module 0-3 tests remain green | PASS |
| Final branch-aware coverage | 84.38% overall; 91.85% Module 4; >=80% gate passed | PASS |

## Commands
```powershell
uv run --python 3.12 --extra dev pytest tests/m4_fusion
# 96 passed
uv run --python 3.12 --extra dev pytest
# 328 passed
uv run --python 3.12 --extra dev pytest --cov=vaaniswitch.m0_data --cov=vaaniswitch.m1_vmd --cov=vaaniswitch.m2_representations --cov=vaaniswitch.m3_encoders --cov=vaaniswitch.m4_fusion --cov-branch --cov-report=term-missing
# 328 passed; 84.38% overall branch-aware coverage; 91.85% Module 4
uv run --python 3.12 --extra dev ruff check .
# PASS
uv run --python 3.12 --extra dev mypy
# PASS: 109 source/test files, strict mode
uv run --python 3.12 python -m compileall -q src tests
# PASS
git diff --check
# PASS
```

## Protected-input audit

| Input | Files | Bytes | Exact path/size/mtime SHA-256 |
|---|---:|---:|---|
| Raw `data/raw/torgo/extracted` | 82,772 | 15,845,129,970 | `206041a4be1fad47975d25ba38c73bb49fe68e3957ebfb046d0db15c0b14c76a` |
| Module 1 production | 26,019 | 19,981,366,416 | `b4944b720c6664d55e57180c94e1c0c6149ac080b49705035ac12ca16be54baa` |
| Module 2 production | 43,365 | 2,378,359,184 | `c951a57640c0bbada8145ef5eeeab6cc534e03adcfef0e7f00fb6307f482ab9f` |

All three match both pre-implementation snapshots and signed Module 3 values. Module 2
still has 8,673 logical directories. Tree fingerprints verify metadata, not every raw
content byte. Separately, all 58 captured signed configs, manifest/folds, Module 3
source/tests/docs and Module 0-3 metric/log artifacts have identical content SHA-256,
size and mtime. No prior signed artifact changed.

## Defects and limitations

- M4-CODE-001 CLOSED: reproduced malformed restricted-pickle handling and unexpected
  optimizer state entries; added controlled rejection and exact optimizer index checks.
  Targeted, full project and branch-aware coverage gates pass.
- M4-CODE-002 CLOSED: reproduced an artifact-category redirect into raw data; enforce
  canonical output category and basename. Targeted, full and coverage gates pass.
- M4-SCI-001 ACCEPTED_LIMITATION: validation includes two male speakers; it cannot
  establish female generalization.
- Upstream VMD fidelity/iteration-ceiling and eight-second Mel crop limitations persist.
- Attention weights are not causal explanations. CPU timing is a small-batch estimate.
- Manual baseline training/validation and Module 4 scientific sign-off remain pending.

## Artifacts and manual next action

`module4_fusion_summary.json`, `module4_split_audit.json`, `module4_training_smoke.json`,
`module4_checkpoint_smoke_validation.json`, and `module4_protected_inputs_audit.json`
under `artifacts/metrics/` contain the evidence. The compact closure record is
`module4_implementation_audit.json`. `artifacts/logs/module4_training.log`
records the tiny smoke epoch; `module4_fusion_failures.csv` contains zero real-run
failure rows. Defect reproduction, test and CLI console logs are retained separately.

```powershell
uv run --python 3.12 --extra dev vaaniswitch-m4 train --config configs/models/fusion.yaml --outer-test-fold 1 --seed 42
uv run --python 3.12 --extra dev vaaniswitch-m4 validate --config configs/models/fusion.yaml --checkpoint artifacts/checkpoints/module4/fold1_seed42_6fc69cf0ad55/best.pt --outer-test-fold 1
```

These commands are for manual execution after implementation readiness. They report
train/validation only and cannot evaluate the reserved Fold 1. Do not run Module 5.

## Historical implementation gate
At implementation handoff Module 4 was not signed off and long baseline training was
pending. The current final closure above supersedes that historical state.

Coverage terminology: the configured pytest-cov gate combines statements and branches:
overall 7,227/8,565 = 84.38%; Module 4 1,082/1,178 = 91.85%. Branches alone are
1,583/2,190 = 72.28% overall and 238/286 = 83.22% for Module 4. No exclusions or
failure suppressions were added to meet the gate.
