# Module 4 — Token-Level Cross-Attention Fusion & Dysarthria Classification

**Target:** Days 9–10
**Code:** `src/vaaniswitch/m4_fusion/`
**Tests:** `tests/m4_fusion/`

**Status:** COMPLETE / PASSED / SIGNED OFF.

The manual baseline completed six epochs, with early stopping and epoch 1 selected by
validation macro-F1. Independent validation, checkpoint/freeze/protected-input audits,
and final regression gates passed. See [TEST_REPORT.md](TEST_REPORT.md) for the current
evidence, timing, metric agreement and gate provenance.

## Acceptance

- [x] implementation
- [x] forward smoke
- [x] gradient sanity
- [x] tiny-batch training sanity
- [x] speaker-safe split
- [x] manual real baseline training
- [x] early stopping
- [x] best-checkpoint selection
- [x] independent validation reproduction
- [x] zero reserved-test peeking
- [x] frozen ResNet verification
- [x] checkpoint integrity
- [x] protected-input integrity
- [x] Module 4 regression gates
- [x] full-project regression gates
- [x] signed off

## Architecture and interface

Consume signed Module 3 encoders on the fly, with no learned-token cache. The ImageNet
ResNet18 feature trunk stays frozen and in evaluation mode, including BatchNorm buffers.
Mel projection/LayerNorm, RP-CNN, GCN, fusion and classifier train together.

One `MultiheadAttention(128,4,dropout=0.10,batch_first=True)` uses Q=Mel `[B,16,128]`
and K=V=nonlinear `[B,17,128]`, ordered graph first then 16 recurrence tokens. Apply
`LayerNorm(mel + attention_output)`, mean-pool its 16 positions and all 17 original
nonlinear positions separately, then concatenate into `[B,256]`.

The head is `Linear(256,128) -> GELU -> Dropout(0.30) -> Linear(128,1)`, returning
`FusionOutput.logits[B]`. Its `probabilities` property computes sigmoid explicitly;
training passes logits directly to `BCEWithLogitsLoss`. Control=0, dysarthric=1 agrees
with Module 0 `Condition`. The typed result also exposes attended Mel tokens, both
pools, the fusion vector and raw attention output. `return_attention=True` adds
per-head `[B,4,16,17]` weights; the default is `None`. Weights describe associations,
not causal explanations. Training dropout means weight rows need not sum to one in
train mode; evaluation rows do.

## Development protocol frozen before training

`configs/models/fusion.yaml` is the sole fusion config. Its canonical sorted JSON
semantic hash covers architecture, label mapping, source hashes/count, optimizer,
batch size, seed, split algorithm, threshold and stopping policy. Paths and CPU thread
count are operational; actual threads/version are recorded. CLI batch-size overrides
change the semantic hash and checkpoint namespace.

Fold 1 remains untouched. From the other ten speakers, choose one speaker per class
by minimum SHA-256 of `42|outer_fold|condition|speaker_id` (speaker ID breaks a tie).
The remaining eight speakers train. Selection uses metadata only and is invariant to
manifest order. No authoritative inner split existed at implementation time.

| Role | Speakers | CORE records |
|---|---|---:|
| Train | F01, F04, FC02, FC03, M01, M02, M05, MC01 | 4,760 |
| Validation | M03, MC02 | 1,094 |
| Reserved outer test | F03, FC01, M04, MC03, MC04 | 2,819 |

Both training and validation contain both classes; all speaker overlaps are zero.
The two validation speakers are male. This deterministic development demonstration
does not measure female validation generalization or final grouped-fold performance.
All three signed outer folds are supported by the split infrastructure. No test
dataset/evaluation command exists. The source and per-batch guards reject reserved
speakers before forward. F03/Session1/0179 and F03/Session1/0118 remain protected
upstream records and are deliberately absent from Module 4 smoke because F03 is reserved.

## Fixed initial training

CPU; batch size 16; AdamW lr=3e-4 and weight decay=1e-4; no scheduler, resampling,
augmentation, threshold tuning, or backbone unfreezing. `pos_weight=3306/1454` uses
training records only. At most 30 epochs; stop after five epochs without a validation
macro-F1 improvement greater than 1e-4. Threshold stays at 0.5. Save best and last only.

Epoch logs include weighted train/validation BCE, validation accuracy, balanced accuracy,
macro-F1, positive precision/sensitivity, control specificity, rank-based ROC-AUC,
TN/FP/FN/TP, learning rate and best-so-far flag. Loss is averaged per record, including
the final partial batch. Undefined AUC is JSON null with a reason. Absent-class recall
is null, balanced accuracy averages present classes, and macro-F1 always averages the
two fixed labels with zero division set to zero.

Checkpoints include the complete model state (including the frozen ImageNet trunk),
optimizer moments, epoch/history/stopping state, train-only weight, all config hashes,
split identity, train/validation speaker IDs, record-selection hashes, seed and Python/
NumPy/PyTorch RNG states. No data or learned-token arrays are included. Loading requires
a matching SHA-256 sidecar, restricted `weights_only=True`, exact state keys/shapes/
dtypes/finiteness, and matching config/split/source/mode. Full frozen weights permit
offline reload without downloading ImageNet weights. Smoke checkpoints have a separate
namespace and cannot validate or resume as baseline checkpoints.

## Completed manual commands and stop boundary

These commands document the completed run; another training run is not the next action.

```powershell
uv run --python 3.12 --extra dev vaaniswitch-m4 train --config configs/models/fusion.yaml --outer-test-fold 1 --seed 42
uv run --python 3.12 --extra dev vaaniswitch-m4 validate --config configs/models/fusion.yaml --checkpoint artifacts/checkpoints/module4/fold1_seed42_6fc69cf0ad55/best.pt --outer-test-fold 1
```

Resume an interrupted identical run with the training command plus
`--resume artifacts/checkpoints/module4/fold1_seed42_6fc69cf0ad55/last.pt`.
Do not restart an existing namespace without resume. Keep its `.pt.sha256` files.
Validation recomputes validation metrics and compares them to the saved epoch metrics;
it does not sign the module off automatically.

Implementation diagnostics: `split-audit`, `train --dry-run`, `smoke`, `train --smoke`,
and `validate --smoke`. All require an explicit `--outer-test-fold`. Tiny updates and
smoke metrics are sanity checks only. Python, NumPy and PyTorch are seeded with 42 and
deterministic PyTorch algorithms are enabled; cross-hardware bitwise identity is not
promised. The 25-update sanity and one tiny smoke epoch are bounded.

Module 4 closure is complete. The reported metrics are a SPEAKER-DISJOINT
DEVELOPMENT-VALIDATION BASELINE ONLY. Reserved Fold 1 remains untouched. Training loss
decreased substantially after epoch 1 while validation macro-F1 never exceeded its
epoch-1 maximum, compatible with early development-set divergence or possible overfitting.
No hyperparameters were changed during the audit. Module 5 owns ablations, optimization, robustness,
and the final frozen grouped-fold evaluation with fold mean +/- standard deviation.
ASR, TTS, UI and all Module 5 work remain outside this session. No module is active.
The next action is explicit authorization for Module 5; it has not started.

API references: [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MultiheadAttention.html),
[restricted checkpoint loading](https://docs.pytorch.org/docs/2.14/generated/torch.load.html),
[reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html).
