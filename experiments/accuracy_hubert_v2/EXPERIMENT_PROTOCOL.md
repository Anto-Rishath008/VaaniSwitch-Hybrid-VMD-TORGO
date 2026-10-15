# POST-HOC EXPLORATORY IMPROVEMENT STUDY

## V2 numerical-stability amendment, 2026-10-13

This separately identified study repairs incomplete V1 (`accuracy_hubert`).
V1's actual log and epoch checkpoints show three completed E2 epochs; the failure
reproduces at epoch 4, batch 540 when FP16 loss scaling is 131,072. V1 performed
no outer evaluation. Its identities, features, checkpoints, stages, logs and prior
implementation evidence remain immutable. V2 starts fresh classifier fits with
its own source/protocol identity and outputs; V1 E1 fits are retained as V1 evidence.

Only the classifier numerical policy changes: E2/E3 optimization, benchmarking,
validation and future outer prediction use CUDA FP32, with autocast disabled and
no GradScaler. BF16 is not used. Frozen HuBERT extraction remains the original
FP16-autocast/FP32-pooling transform. All 8,673 completed entries are reused through
verified read-only references. No extraction, overwriting or repair is permitted
on this reuse path. All model files, revisions, logical identities, source bindings,
feature/per-layer hashes, dependency/extraction settings and crop intervals are checked.

Model architecture, labels, splits, weights, candidate grid, hyperparameters,
selection metric and threshold remain the V1 protocol below. Non-finite inputs,
parameters, optimizer tensors, logits, losses, individual gradients, gradient norm,
or post-update state stop execution. No invalid batch is skipped. Numerical failure
diagnostics retain the model, layer, candidate, epoch, batch and logical record IDs.
The last complete checkpoint remains authoritative; there is no automatic numerical retry.
OOM alone permits rollback of the entire unfinished epoch and halving the batch down
to one, restoring model, AdamW and Python/NumPy/CPU/CUDA RNG state. At batch one OOM
stops. Checkpoints require the V2 FP32 policy; V1 FP16 checkpoints cannot resume V2.

V2's manual launcher still executes E1/E2/E3 for all three speaker-disjoint folds,
freezes all nine selections/final fits before any outer forward, and produces the
same five result files and three figures. The repair session runs only bounded
development validation. Full training, final evaluation and accuracy are unverified.

Protocol prepared 2026-10-13, Asia/Calcutta. The original outer results already
influenced this research direction. New results are exploratory, not a pristine
independent benchmark. There is no guaranteed target accuracy, clinical claim,
SOTA claim or replacement of original results.

## Frozen dataset and representations

Consume the original JSONL manifest with SHA-256
`91c3327ea1f6a6b2057ffa02198f86d682668d22f247beef27f1466e959a9061`.
CORE requires `normal_speech_eligible`, `usability_status=usable` and prompt type
word/sentence. This gives 8,673 unique speaker/session/prompt records and 15 speakers
(8 dysarthric, 7 control). There are 7,503 head and 1,170 array primaries in CORE.
Prefer structurally usable head audio; otherwise use the frozen valid array fallback.
Alternates never create additional rows. No random recording split is permitted.

The original RP namespace is
`data/processed/representations/d8a36c6ab8c48b15e61ecd63194ffa30ede4e7e7e7ab0a30c197002652f9d65d/`.
Its source VMD namespace is
`data/processed/vmd/bfd7aec36d99a88cfacc72df85e124d3f4b99b76ed47d71c3ced9a153fa92685/`.
RP is the existing binary 224×224 full-reconstruction representation: block RMS,
z-score sequence, delay 2, embedding dimension 3, off-diagonal recurrence rate .15,
diagonal included. No RP or VMD artifact is generated or modified. The loader checks
completion, config, manifest, primary-audio and reconstruction bindings, file SHA-256,
dtype, shape and binary values. VMD tensors are not opened by the study.

## Original speaker roles

Fold file byte SHA-256:
`3d0303d9cb8af96da7b0bd4e4ad26b67ea0350c24a5607a7571d067e3a08aa8c`.
Its distinct canonical JSON fingerprint is
`5ab4e1af6c54f97af940c3e5ba5d447c77d827102b88656bf19cd5227b5fee27`.

| Outer fold | Test speakers | Test records | Inner validation speakers |
|---|---|---:|---|
| 1 | F03, FC01, M04, MC03, MC04 | 2,819 | M03, MC02 |
| 2 | F04, FC02, M01, M02, MC02 | 2,873 | F03, FC03 |
| 3 | F01, FC03, M03, M05, MC01 | 2,981 | F03, MC02 |

For each fold, the remaining eight development speakers form inner training.
Validation is exactly the original seed-42 class-stratified speaker SHA-256 ranking.
Speaker, logical-ID, primary-path and primary-waveform-hash intersections across
roles must all be empty. Source metadata audits may inspect frozen class identities;
selection functions receive only guarded development partitions, never outer labels.

## Frozen speech extraction

Official checkpoint attribution and architecture are described in ARCHITECTURE.md.
Pinned revision: `dba3bb02fda4248b6e082697eee756de8fe8aa8a`.

| File | SHA-256 |
|---|---|
| pytorch_model.bin | 062249fffb353eab67547a2fbc129f7c31a2f459faf641b19e8fb007cc5c48ad |
| config.json | 56be398848bbd9cbc720172b0d45b2f01cac7652f2968f0f31f5faf9f2986acc |
| preprocessor_config.json | 4a93853b74278b7c769d07f5a861e5d12ceb5db2bced5620d335f87238cb9e86 |

Read only frozen primary audio. Decode float32 PCM, downmix by channel mean and
resample with deterministic polyphase conversion if needed; this CORE is already
16 kHz mono. For signals over 128,000 samples, use
`start=(n-128000)//2`, `end=start+128000`. The same centered interval as original
Mel is used. Record source length, interval, valid length and crop status. Frozen
metadata identifies 235 cropped CORE recordings; the full extractor must reproduce
that count. The shortest CORE recording is .15 seconds, above the 400-sample HuBERT
receptive-field minimum.

Short signals remain unpadded. This is an explicit technical difference from Mel's
right-zero padding: the checkpoint's temporal GroupNorm would change valid features
when a short signal is padded. Batches therefore contain exact equal lengths only.
The checkpoint feature extractor applies zero-mean/unit-variance normalization on
each valid signal. The sample attention mask must contain only valid samples in
these batches. The GroupNorm pretrained backbone is called without a padding mask;
convolution frame counts are computed from the exact seven kernels/strides and used
as masks for temporal pooling. Mixed-length batches are rejected.

Use model.eval(), requires_grad_(False), torch.inference_mode(), CUDA FP16 autocast,
then float32 masked mean and population standard deviation. The candidate outputs
are `hidden_states[6]`, `[9]`, `[12]`: index 0 is the pre-transformer embedding state;
index k is the output after transformer block k, for k=1..12. Concatenate mean and
std to obtain 1,536 features per layer; store one float32 [3,1536] entry per record.

Each completion marker binds logical ID, speaker, source path/hash, exact checkpoint
revision and file hashes, extraction settings/library versions, per-layer identifiers,
shapes, checksums and completion status. Corrupt or stale completed entries stop;
they are not silently recomputed. Interrupted unpublished payloads are preserved,
and a fresh payload can complete the same unfinished entry. All outer unlabelled
features may be computed once because this frozen transform fits no corpus statistics.

## Predeclared selection and training

E1 searches 3 layers × C in {.01,.1,1}, L2 logistic regression with lbfgs, maximum
3,000 iterations. Nonconvergence stops instead of accepting an unfinished fit.
E2 and E3 each search 3 layers × two settings: dropout .3/weight decay .001 or
dropout .5/weight decay .01. The learning rate is .0003. No broad search is added.
Across three folds this is 63 candidate fits and nine final development fits.

All StandardScalers fit only the current training role, with training-speaker weights.
Each training recording has weight proportional to `1 / recordings_for_its_speaker`,
normalized to mean one. This gives each training speaker equal total loss weight.
It is used as LR sample_weight or neural per-example BCE weight. There is no
additional class_weight, pos_weight, balanced sampler or class compensation.

Neural training uses AdamW, maximum 20 epochs, patience 4, minimum macro-F1 improvement
1e-4, CUDA FP32 without AMP and gradient clipping at norm 5. Seed 42, threshold .5, no augmentation.
Choose a candidate by inner-validation recording macro-F1, then balanced accuracy,
ROC-AUC, lower BCE, layer and stable candidate ID. Parameters are the same size across
these candidates. Within a candidate, keep the first checkpoint whose macro-F1
improves by the declared minimum; equal macro-F1 does not reset patience.

For final fit, refit the scaler on all ten development speakers, initialize a fresh
classifier/CNN, and train exactly the selected inner epoch count. Final training has
no outer or validation checkpoint/epoch selection. E1 refits the selected C/layer on
all development speakers. Finish and freeze all nine final model/fold decisions
before any real outer inference. Outer metrics never feed back into selection.

Extraction benchmarks batches 1/2/4 on the longest development segment; conservative
batch cap two, exact-length grouping and bounded 32-record decoding lookahead.
Neural training benchmarks disposable development models at 1/2/4/8/16, then uses
a conservative cap eight. The CUDA allocator is limited to 80% physical VRAM.
The historical extraction implementation halves batches on OOM; V2 reuses features.
Training OOM rolls back the entire unfinished epoch,
including parameters, optimizer and RNG, before using a smaller batch. Folds
and classifiers run sequentially; CPU fitting/decoding/metrics use bounded threads.

## Once-only outer inference and reporting

Before an outer forward, consume an exclusive, durable claim bound to the frozen
decisions, selected checkpoint hash, split and run identity. Write immutable results
and a checksum seal after success. A consumed claim without a valid result+seal
is ambiguous and requires explicit review; the launcher never repeats it. A sealed
result is reused even if later aggregation/report publication was interrupted.

Report accuracy, balanced accuracy, macro-F1, dysarthria precision/sensitivity,
control specificity, ROC-AUC, [[TN,FP],[FN,TP]], training time and classifier parameters
for every model/fold. Report equal-weight arithmetic fold means and sample SD
(ddof=1). Training time sums all candidate and final training/validation computation;
checkpoint I/O and common feature extraction are separately logged.

Retain all held-out recording predictions. Descriptive speaker decisions use the
mean of all that speaker's held-out recording probabilities and threshold .5.
Report speaker accuracy, confusion matrix and errors separately. Compare recording
fold mean accuracy/macro-F1 with exact signed B2/C2 values, not speaker accuracy.
The signed Module 5 closure manifest and its eight package files are checked before
baseline import. B2 recording accuracy is 69.7158513% and C2 is approximately 65.02%.
Original results and earlier 11/15-speaker assessments remain unchanged.

There are only 15 speakers, one seed and one two-speaker inner split per fold.
Pretraining/domain/channel effects, overlapping development sets, center-crop/full-RP
coverage differences, and nondeterministic native B2 CUDA backward limit claims.
A future independent speaker cohort is needed for an independent confirmation.
