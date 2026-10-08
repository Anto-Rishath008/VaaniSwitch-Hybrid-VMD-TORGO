# Module 5 — Accuracy Optimization, Ablation & Robustness

**Module 5: COMPLETE / PASSED / SIGNED OFF (exception-aware).**
**Module 6: NOT STARTED; handoff prepared only.**

All 21 frozen outer cells are independently verified: 20 NATIVE and Fold1/A1
RECOVERED_PROTOCOL_EXCEPTION. Its original prior native-forward count remains UNKNOWN;
its original consumed claim, native state and absent native result remain intact.
The eight-entry remaining sequence is COMPLETE with no next target.
Frozen foldwise mean/sample-SD aggregation and C2 descriptive robustness are complete.
No integrity blocker remains. This sign-off does not assert uninterrupted native
once-only compliance for Fold1/A1.

[Final closure report and handoff](FINAL_CLOSURE_MODULE6_HANDOFF.md).
The scientific protocol below is unchanged. All phases, final retraining, outer
verification, frozen aggregation and C2 descriptive robustness are complete.

Code: `src/vaaniswitch/m5_experiments/`; tests: `tests/m5_experiments/`.
The empty `m5_optimization` scaffold is retired. The sole experiment configuration
is `configs/experiments/module5/protocol.yaml`. The unused provisional 40-epoch,
patience-7 augmentation configuration `configs/experiments/training.yaml` was removed.

## Nested scientific protocol

Use the signed 8,673 CORE records and three speaker folds. One logical
speaker/session/prompt record enters once; alternate microphones never create rows.
Reserve five outer speakers and use the other ten for development. The unchanged
Module 4 method chooses one validation speaker per class by minimum SHA-256 of
`42|outer_fold|condition|speaker_id`, with speaker ID as a deterministic fallback.
Selection uses metadata only. No utterance-level partition is implemented;
training-batch shuffling does not change speaker assignments.

| Fold | Inner training speakers | Inner validation | Outer speakers | Inner train / validation / outer CORE |
|---:|---|---|---|---|
| 1 | F01, F04, FC02, FC03, M01, M02, M05, MC01 | M03, MC02 | F03, FC01, M04, MC03, MC04 | 4,760 / 1,094 / 2,819 |
| 2 | F01, FC01, M03, M04, M05, MC01, MC03, MC04 | F03, FC03 | F04, FC02, M01, M02, MC02 | 4,273 / 1,527 / 2,873 |
| 3 | F04, FC01, FC02, M01, M02, M04, MC03, MC04 | F03, MC02 | F01, FC03, M03, M05, MC01 | 4,458 / 1,234 / 2,981 |

All roles retain both classes. Inner train, validation and outer speakers, logical
IDs, primary paths and source hashes are disjoint. Store complete split identities
and record hashes before fitting. Fold 1 reuses Module 4's exact split; none of its
outer speakers were loaded or forwarded during Module 5 implementation.

Module 4 already developed on Fold 2/3 speakers. Nested evaluation prevents
within-fold selection leakage but cannot undo historical design exposure: these
folds are not pristine external validation. In manual nested evaluation, a speaker
may be development data for another fold. Freeze protocol/code globally first and
never retune from cross-fold results. Select Fold 1's C2 settings first in Phase 5B.

## Identities and namespaces

Protocol hash:
`677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9`.

A0 preprocessing hash:
`9b57afd31f1b5d7d7c03c4d82828f3479bf0e115a0a35d3a262c0d190337f8d4`.

Experiments live under `artifacts/module5/<protocol-hash>/`. IDs contain outer
fold, ablation, candidate, seed and protocol hash; smoke has a separate ledger and
run directory. Protocol freezing additionally binds Modules 0–5 Python source
content. Checkpoints bind all five signed predecessor hashes, complete split and
record identities, runtime, optimizer and selected configuration. Changed identities
cannot silently resume. Actual predecessor hashes and all split hashes are retained
in `artifacts/module5/implementation/protocol_audit.json`.

The implementation produced a metadata-only protocol audit. The user subsequently
froze the production protocol in Phase 5A; its exact code/protocol identities pass
the current read-only closure audit.

## Required ablations

Reuse signed encoders at D=128: frozen ImageNet-1K V1 ResNet18 produces 16 Mel
tokens; pure-PyTorch GCN 2->96->128 with mean+max pooling produces one graph token;
RP channels [16,32,64] produce 16 tokens. Unused branches are not instantiated or
forwarded. Signed entry validation may read all Module 2 arrays; unused arrays do
not enter the ablation computation. A0 never calls the Module 2 source loader.

| ID | Computation | Trainable parameters | Head parameters |
|---|---|---:|---:|
| A0 | Conditioned primary without VMD -> Mel -> mean pool -> head | 82,561 | 16,641 |
| A1 | Signed VMD Mel -> mean pool -> head | 82,561 | 16,641 |
| B1 | Signed graph -> GCN token -> head | 62,497 | 16,641 |
| B2 | Signed RP -> CNN tokens -> mean pool -> head | 97,457 | 16,641 |
| B3 | Graph token concatenated with mean RP tokens -> head | 159,697 | 33,025 |
| C1 | Mean Mel concatenated with mean 17 nonlinear tokens -> head | 225,617 | 33,025 |
| C2 | Signed Mel-query token cross-attention fusion -> head | 291,921 | 33,025 |

Heads are `Linear(input,128) -> GELU -> Dropout -> Linear(128,1)` with logits `[B]`.
Inputs are 128 for A0/A1/B1/B2 and 256 for B3/C1/C2; this required dimension explains
head capacity differences. C2 reuses signed Module 4 fusion: Q=16 Mel tokens,
K=V=17 graph-then-RP tokens, four heads, attention dropout 0.10, Mel residual plus
LayerNorm, then mean attended Mel and mean nonlinear pooling. C1/B3 have no attention.
Overlapping branches have identical initial weights using deterministic substreams
of the seed-42 experiment: Mel 42, graph 43, RP 44, head construction 45. These are
initialization substreams, not additional experimental seeds.

C3 is **disabled / design only**. Later selected-mode experiments would need new
experimental VMD, representation and protocol hashes outside production. No C3
builder, dispatch or execution is enabled, and it does not block A0–C2.

## A0 exact signal and cache contract

Require exact equality with the signed manifest record, covering logical ID,
speaker/session/prompt, microphone selection and alternate provenance, primary path,
primary SHA-256 and source/structural metadata. Bind the manifest fingerprint;
resolve only the selected primary within raw TORGO and verify its SHA-256.

Call signed Module 1 audio conditioning: float64 PCM decode, arithmetic-mean mono
downmix, deterministic linear-interpolation fallback to 16 kHz, and conservative
RMS normalization (target 0.1, gain cap 10, peak limit 0.99, silence epsilon 1e-12).
TORGO already uses mono 16 kHz. **Bypass VMD** and call the unchanged Module 2 Mel:
FFT 512, periodic Hann 400, hop 160, 80 HTK bands 20–8000 Hz, Slaney area normalization,
window-energy-normalized power, power dB/epsilon 1e-10, eight-second canvas, center
crop/right-zero pad, area-average time resize and linear frequency resize to finite
float32 [224,224]. A0/A1 differ only in the pre-Mel signal: conditioned primary versus
its all-mode VMD reconstruction.

Cache path: `data/processed/module5_ablation/no_vmd_mel/<A0-hash>/<logical-id>/`.
Each complete entry has `mel.npy` and `metadata.json`. Atomic directory publication,
exact binding, completion marker, array SHA-256, file-set, dtype, shape and finite
checks protect resume. Full validation also checks source hashes. Valid entries
are reused; invalid existing entries fail explicitly and remain for review. There
is no force/overwrite option. Interrupted staging directories remain traceable.
Canonical path checks reject traversal and redirected ancestors; A0 cannot target
the signed Module 2 representation namespace.

## Four frozen-backbone C2 candidates

| Candidate | Learning rate | Weight decay | Classifier dropout |
|---|---:|---:|---:|
| baseline | 3e-4 | 1e-4 | 0.30 |
| lower_lr | 1e-4 | 1e-4 | 0.30 |
| stronger_decay | 3e-4 | 1e-3 | 0.30 |
| stronger_dropout | 3e-4 | 1e-4 | 0.40 |

Choose by inner-validation macro-F1, then balanced accuracy, valid ROC-AUC, lower
weighted validation loss and fewer trainable parameters. Candidate ID only breaks
a remaining exact tie. Recovery recomputes the winner from all retained inner
results and requires selection-file equality with ledger evidence.

Each outer fold uses its selected C2 **settings** for every matched architecture;
each ablation selects its own epoch from its own inner-validation trajectory.
This is a controlled ablation under common settings, not independent optimization
of every baseline. No outer result changes candidate sets, settings, epochs or seeds.

Use AdamW, batch 16, <=30 inner epochs, seed 42, threshold 0.5, BCEWithLogitsLoss and
controls/positives from the fitting role only. No augmentation, scheduler, resampling,
threshold search or unfreezing is enabled. Patience 5 resets only for macro-F1
improvement >1e-4; best-epoch selection uses the metric/tie-break order independently
of the patience counter. VMD K=5/all modes, graph nodes=256, Mel canvas=8s, D=128 and
four attention heads remain fixed.

## Fresh retraining and one-use outer evaluation

Instantiate a fresh model from the same ImageNet starting point; never load inner
weights into final training. Fit all ten development speakers (5,854 / 5,800 / 5,692
records) for exactly the selected inner epoch count. Recompute development-only
pos_weight; perform no validation or early stopping. Freeze all seven ablation
selections and complete their retraining for a fold before any outer predictions.

Ledger: PLANNED -> RUNNING -> INNER_COMPLETE -> SELECTED -> FINAL_TRAIN_COMPLETE ->
OUTER_EVALUATED. Unselected C2 candidates stay INNER_COMPLETE. Failure records FAILED,
cause and prior state. Explicit pre-outer recovery requires `--resume --retry-failed
"reason"` and an event. INVALIDATED preserves evidence/reason and cannot restart.
Completed outer results cannot transition back to training or failure.

SQLite transactions enforce transitions and per-experiment locks prevent concurrent
duplicate work; process termination releases locks. Checkpoints contain model,
AdamW, RNG, all provenance, epoch/history, inner best metrics, class weight and
selection state. Final checkpoints also bind selected configuration/epoch and all
development/test speakers. Restricted loading validates schema, hashes, tensors and
optimizer. Checksummed generations plus an atomic latest pointer retain a usable
previous generation during publication; only the superseded resumable generation
is removed. Completed scientific results remain preserved.

An interruption resumes explicitly under the identical identity from the last
completed epoch. An incomplete epoch is replayed from its saved RNG/optimizer
boundary. A pre-checkpoint interruption can explicitly restart the same seeded run.
Recorded FAILED runs additionally require a reason. Completed stages return retained
evidence without fitting. Code changes never silently resume an older smoke.

Outer evaluation checks frozen protocol, selections, final checkpoint provenance
and epoch, then commits a one-use claim before constructing the dataset. Source and
batch guards verify exact records/speakers. No fitting, class-weight calculation,
selection or threshold control exists in the evaluator. Predictions/metrics publish
atomically without replacement, and their SHA-256 enters the ledger. Repeated
completed commands return checksum-verified evidence without new forwards. A failure
after claim cannot retry automatically, even before publication: retain it for
explicit defect/invalidation review.

## Reporting and limitations

Per fold: accuracy, balanced accuracy, macro-F1, precision, sensitivity, specificity,
ROC-AUC and TN/FP/FN/TP. Matrix orientation is `[[TN,FP],[FN,TP]]`; control=0 and
dysarthric=1. Single-class AUC is null with a reason. Require exactly three folds per
required ablation for unweighted means and sample standard deviations (`ddof=1`).
An unavailable fold metric has no three-fold mean/std and retains its count. Pooled
confusion is secondary only. Include selected settings/epochs, parameter counts,
training times and fold-wise C2-minus-baseline macro-F1 differences.

C2 descriptive robustness reads immutable predictions only: gender, primary
microphone, word/sentence, duration<=8s/>8s, padded/cropped/exact Mel and speaker.
Report per-fold and separately labelled descriptive pooled groups, sample/speaker/
class counts and small subgroups. Never feed these results into model selection.

Carry forward: only 15 speakers; male-only Module 4 validation; RMS may suppress
absolute amplitude cues; 235 CORE Mel center crops; NumPy VMD lacks independent-
package cross-validation; 48 valid VMD records reached the iteration ceiling.
Three folds do not establish significance. C2 remains intended, with no guaranteed
accuracy or fabricated superiority. Report comparable/better simpler models honestly.
This is not a medical diagnostic system.

## Ordered manual phases and stop boundary

1. **5A:** review/freeze protocol; build and validate full A0 cache.
2. **5B:** four C2 inner candidates per fold, then frozen selections.
3. **5C (complete):** matched ablation-specific inner epochs; three selected C2
   runs reused and 18 new A0/A1/B1/B2/B3/C1 runs completed manually and audited.
4. **Final retraining (complete):** all 21 fresh CPU final stages at frozen epoch
   counts on ten development speakers passed audit; no validation/early stopping.
5. **Outer sequence complete:** all 21 retained cells verified, including the
   explicit Fold1/A1 recovery exception; all eight linked completions intact.
6. **5D/5E complete:** verified predictions, fold mean/sample SD, the frozen C2
   subgroup analysis and conservative scientific review are retained in final_closure.

Module5 is COMPLETE / PASSED / SIGNED OFF (exception-aware). The final closure uses
retained predictions only and performs zero production model forwards, training,
native/scientific writes or retuning. Frozen code/protocol/runtime are unchanged.
Module6 handoff is prepared; Module6 NOT STARTED. Stop before Module6 execution.
