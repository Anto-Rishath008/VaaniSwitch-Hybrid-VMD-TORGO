# V3 post-hoc exploratory protocol

Prepared 2026-10-13, Asia/Calcutta. This separate HuBERT study does not change the
signed shared-VMD VaaniSwitch architecture or Modules 0-6. TORGO English binary
dysarthric/control labels, 8,673 CORE logical records and the original three
speaker-disjoint outer partitions remain fixed. Original and V1/V2 results have
already influenced the research direction. Any future outer comparison is
post-hoc exploratory, not independent confirmation. Untouched speakers or an
external dataset would be required for independent confirmation; none is invented.

The V2 observations motivating this design are uneven speaker errors and
observational microphone differences. No speaker-specific corrections, removal,
error-based sampling, test-label weighting, threshold tuning, or test-population
normalization is permitted. Microphone/session/prompt/speaker are confounded.

## Inherited mechanisms and faithful reference

V2 E1 uses sklearn LogisticRegression: L2, lbfgs, C in {0.01,0.1,1}, max_iter=3000,
random_state=42, no class_weight; convergence warnings stop. Its StandardScaler
fits original training records with inverse speaker-frequency weights, then
transforms float32 inputs using sklearn semantics. Threshold is 0.5. Each sample
weight is N/(S*n_s); S speakers each receive N/S total weight. Class masses are
N*S_class/S. V2 already balances speakers. Its original two-speaker inner split
and full nine-candidate grid are reproduced development-only in V3; no V2 outer
score serves as a development score. Focused parity tests compare actual fits
with the unchanged V2 implementation.

V3 imports immutable V2 Corpus/Record, source-audio decoding/cropping,
FeatureCache verification, frozen Extractor, RNG serialization, atomic checkpoint
publication and once-only claim semantics. All imported source hashes are pinned
by V2_LINEAGE and checked before execution. No V2 launcher, environment initializer,
report writer or reuse-report writer is called. Original caches physically reside
in V1; V2 uses read-only references, and V3 follows that verified lineage.

Encoder is facebook/hubert-base-ls960, revision
dba3bb02fda4248b6e082697eee756de8fe8aa8a. File hashes, dependency versions and all
8,673 cache payloads/per-layer hashes are checked. V2 preprocessing stays 16 kHz
mono, center crop to 128,000 samples, no padding, exact-length GroupNorm batches,
per-record processor normalization, frozen eval/inference mode, CUDA FP16
extraction with FP32 mean plus population-SD pooling. Layers 6/9/12 each yield
1,536 values. No original recording is re-extracted. No encoder fine-tuning.

## Weighting and normalization

B0 is exactly V2: weight N/(S*n_s), no extra class multiplier.
B1 explicitly repeats equal global total mass per training speaker with the same
class policy. B0=B1 mathematically; B1 is documented and never retrained.
B2 uses u_s=clip(sqrt(median_training_speaker_count/n_s),0.5,2), then normalizes
record weights to mean one. This reduces frequency dependence while bounding the
original per-record max/min ratio at four. Class masses are reported; no hidden
class_weight, pos_weight or sampler doubles compensation. Counts and the median
come from the current training role only. Equal global speaker mass and equal
class mass cannot both hold when class speaker counts differ; we preserve global
speaker balancing in B0/B1 and explicitly expose its induced class prior.

N0 is V2 weighted StandardScaler. N1 uses training-weighted marginal median and
IQR, with stable-sort weighted midpoint-CDF interpolation at 25/50/75 percentiles.
IQR <=1e-12 gets scale one, including constant dimensions. This is a custom
weighted robust transform, not sklearn RobustScaler falsely claiming weights.
N2 deterministically L2-normalizes each original recording vector (norm <=1e-12
gets divisor one), then fits weighted StandardScaler on training originals.
For all policies the scaler fits ORIGINAL training records only; augmentation
does not change its fitted statistics. N2 may discard useful magnitude information;
only matched development ablations can establish its utility. No claim is made now.

Every augmented variant shares its parent speaker/label/record. Original weight
is divided equally over the original and its one variant. The total objective
mass remains N original records, rather than N plus variants; LR C consequently
does not change just because copies were added. Speaker and class totals are logged.

## Evaluation, computation and stop boundary

See MODEL_SELECTION_PROTOCOL.md and AUGMENTATION_PROTOCOL.md for frozen rules.
Primary selection uses development recording macro-F1 in grouped speaker folds,
with equal weighting of the three development folds. Secondary metrics include
balanced accuracy, recording accuracy, ROC-AUC when defined, descriptive speaker
decisions, per-speaker sensitivity/specificity (undefined class rates are null),
Brier/log loss, ten-bin reliability/ECE, and between-speaker variability.

All selected/reference final fits for all three outer folds must finish and be
frozen before any outer access for inference. The full manual launcher defaults
to development only. A separate explicit -AuthorizeOuter switch is necessary
for real outer inference. Exclusive claim publication precedes inference; an
unsealed consumed claim stops without automatic retry. No unfavorable result is
discarded. All held-out predictions are retained at threshold 0.5.

LR is CPU sklearn in the existing environment, two CPU threads; its finite
features/scaler/weights/coefficients/logits/loss/analytical gradients are checked.
There is no neural head, AdamW, GradScaler, or neural clipping in V3. Those V2
mechanisms are studied but not incorrectly claimed as active. GPU is used only
for frozen HuBERT augmentation extraction, one waveform at a time, allocator
limit 80%. The generic exact-length helper supports bounded OOM batch halving;
OOM at batch one stops. Numerical failure never retries or skips records.

Interrupted LBFGS cannot resume internal iterations through sklearn. V3 saves
the start RNG, restarts an interrupted solve deterministically, and reuses a
finished checksum-bound checkpoint if only publication was interrupted. Python,
NumPy, Torch CPU and CUDA RNG are serialized/restored. No optimizer-state resume
claim is made for this non-neural solver. Completed fits remain immutable.

Final fold summaries use equal-weight arithmetic mean and sample SD (ddof=1).
Speaker decisions average all of a speaker's held-out probabilities. They are
descriptive and do not replace recording metrics. No significance, causal channel
effect, medical diagnosis, SOTA, or guaranteed accuracy claim is permitted.

Implementation session: bounded development validation only. Full V3 training
and real V3 outer inference are explicitly NOT authorized by this protocol.

## Technical references

- Pinned model: https://huggingface.co/facebook/hubert-base-ls960/tree/dba3bb02fda4248b6e082697eee756de8fe8aa8a
- sklearn 1.7 LR: https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html
- Train-only robust scaling background: https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.RobustScaler.html
- Separate calibration fitting: https://scikit-learn.org/1.7/modules/calibration.html

Installed source/API and unchanged V2 code are authoritative for parity. Mild
augmentation ranges are predeclared research choices, not proven label-preserving
or optimal for dysarthria. Functional channel checks do not establish efficacy.
