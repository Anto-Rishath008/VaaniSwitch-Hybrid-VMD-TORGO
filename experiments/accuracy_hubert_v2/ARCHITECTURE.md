# Experimental architectures

This separate study does not replace the signed VMD/Mel/GCN/RP cross-attention
network or frozen ASR/TTS utilities. The proposed E3 tests whether pretrained
speech features add useful information to the existing recurrence branch.

| Model | Computation | Trainable parameter count |
|---|---|---:|
| E1 | 1,536 HuBERT features → training-only weighted StandardScaler → L2 logistic regression | 1,537 |
| E2 | Scaled 1,536 → Linear(128), LayerNorm, GELU, dropout → Linear(128), GELU, dropout → Linear(1) | 213,633 |
| E3 | Existing binary RP → original recurrence encoder → mean of 16 × 128 tokens; concatenate with E2's 128-wide speech projection → Linear(256,128), GELU, dropout → Linear(128,1) | 310,833 |

HuBERT is frozen, outside the classifier optimizer and evaluated only once per
logical record in a feature namespace. Each selected layer contributes 768
temporal means followed by 768 population standard deviations. All three candidate
layers are cached together; classifiers use one development-selected layer each.
The pretrained backbone's parameter count is recorded by extraction evidence and
is separate from the classifier counts above.

The RP encoder is copied from the original Module 3 implementation used by B2.
Only its two imports are redirected to isolated study files. AST equality checks
cover that exact implementation; preprocessing is byte-for-byte identical to the
original. The adapter retains channels [16,32,64], two 3×3 convolutions per block,
GroupNorm/GELU, two max pools, Dropout2d(0.2), adaptive pooling to 4×4, Linear(64,128)
and LayerNorm. Binary uint8 input becomes float32 without corpus statistics.
The original B2 implementation is not edited or imported through torchvision.

Every candidate and final E3 model starts fresh from seed 42; no original B2/C2
checkpoint is loaded. Its recurrence CNN, speech projection and fusion head all
receive gradients within that fit's allowed development partition.

The pinned checkpoint is [facebook/hubert-base-ls960](https://huggingface.co/facebook/hubert-base-ls960),
revision `dba3bb02fda4248b6e082697eee756de8fe8aa8a`. It is HuBERT Base: 12 transformer
layers, hidden width 768. This study uses it as a frozen acoustic feature extractor,
not as an ASR decoder. No transcript or prompt text enters its input.

PyTorch reports no deterministic CUDA backward implementation for the original
adaptive average pool. Training seeds every RNG and uses deterministic algorithms
with `warn_only=True`, preserving that architecture and the warning. E3 training
is therefore not claimed bitwise reproducible across fresh runs or resumptions.
Fixed-checkpoint evaluation is separately checked for exact repeated outputs.
