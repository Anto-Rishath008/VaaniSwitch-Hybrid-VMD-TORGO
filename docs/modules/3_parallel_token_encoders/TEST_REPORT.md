# Module 3 Test Report — Parallel Token Encoders: Mel-CNN, VG-GCN & RP-CNN

**Status:** COMPLETE / PASSED / SIGNED OFF
**Production smoke:** PASS — 8/8 signed real Module 2 records
**Module 4:** NOT STARTED

## Architecture resolution and environment

The tracked provisional `D=192` settings in `configs/models/encoders.yaml` and
`configs/models/fusion.yaml` were explicitly superseded before implementation. Both now
use `D=128`; no active architecture setting remains at 192. The lightweight GCN hidden
width remains 96 and recurrence channels remain `[16,32,64]`.

| Item | Result |
|---|---|
| Encoder config | `configs/models/encoders.yaml` |
| Encoder config hash | `7c8e44eddb11704d08e45a1b63bc9cc54b28d679f5e3a3ae1a474b61e3a40383` |
| PyTorch | `2.14.0+cpu` |
| torchvision | `0.29.0+cpu` |
| Mel weights | `ResNet18_Weights.IMAGENET1K_V1`, loaded successfully |
| CUDA | Unavailable; CPU validation is authoritative |

## Signed source contract

Module 3 consumed only
`data/processed/representations/d8a36c6ab8c48b15e61ecd63194ffa30ede4e7e7e7ab0a30c197002652f9d65d/`.
The global and per-record loader checks:

- Module 2 sign-off `PASS`, 8,673 selected/successful/validated, and zero failures;
- representation hash `d8a36c6ab8c48b15e61ecd63194ffa30ede4e7e7e7ab0a30c197002652f9d65d`;
- Module 1 hash `bfd7aec36d99a88cfacc72df85e124d3f4b99b76ed47d71c3ced9a153fa92685`;
- manifest fingerprint `91c3327ea1f6a6b2057ffa02198f86d682668d22f247beef27f1466e959a9061`;
- exact `mel.npy`, `recurrence.npy`, `graph_nodes.npy`, `graph_edges.npy`, and
  `metadata.json` directory contents;
- logical ID, production reconstruction path, all-mode provenance, artifact sizes and
  SHA-256 values, shapes, dtypes, and numerical graph/RP/Mel invariants.

The dataset joins condition, gender, prompt type, primary microphone, and evaluation
fold from the signed manifest and `speaker_folds.json`. Collation stacks image tensors,
pads graph nodes only to the batch maximum, builds a boolean valid-node mask, preserves
per-record edge indices, and carries immutable provenance. Neither labels nor folds enter
any numerical transform. No raw WAV, Module 1 array, random utterance split, or learned
token cache is used.

## Exact encoders

### Mel

Finite Module 2 `float32[224,224]` dB values are clamped to `[-100,+20]`, mapped
linearly to `[0,1]`, replicated from one channel to three, then normalized with fixed
ImageNet mean `[0.485,0.456,0.406]` and standard deviation `[0.229,0.224,0.225]`.
No corpus, class, speaker, or fold statistics are calculated.

The ImageNet-1K V1 ResNet18 feature trunk excludes the adaptive global pool and original
FC classifier. It produces 512 channels, followed by adaptive average pooling to 4x4,
spatial flattening to 16 tokens, `Linear(512,128)`, and LayerNorm. The trunk is frozen;
the projection and LayerNorm remain trainable and controlled unfreezing is supported.

### Recurrence

Binary `uint8[224,224]` values convert exactly to float32 0/1. Three compact blocks use
two 3x3 convolutions per block, GroupNorm(8), GELU, and dropout 0.20, with channels
`1→16→32→64`; the first two blocks end in 2x2 maximum pooling. Adaptive average pooling
to 4x4 yields 16 positions, followed by `Linear(64,128)` and LayerNorm.

### Visibility graph

Stored edges are unique, loop-free, bidirectional int16 `[2,E]`. For each valid graph,
Module 3 constructs the binary adjacency, adds exactly one self-loop per valid node,
computes degree, and applies `D^-1/2(A+I)D^-1/2`. Dense matrices are bounded at 256
nodes and use only PyTorch. The two layers are `2→96→128`, with GELU and dropout after
the first layer. Valid-node mean and maximum pools concatenate to 256 features, followed
by `Linear(256,128)` and LayerNorm. Padded nodes contribute to neither adjacency,
degree, convolution, mean, nor maximum pooling.

## Production smoke panel

| Logical ID | Condition/gender | Primary | Fold | Result |
|---|---|---|---:|---|
| F01/Session1/0082 | dysarthric/female | head | 3 | PASS |
| FC03/Session3/0151 | control/female | head | 3 | PASS |
| M04/Session1/0125 | dysarthric/male | array | 1 | PASS |
| MC03/Session2/0232 | control/male | array | 1 | PASS |
| F03/Session1/0179 | dysarthric/female | head | 1 | PASS; valid Module 1 ceiling record |
| F03/Session1/0118 | dysarthric/female | head | 1 | PASS; 193.828625-second source |
| FC01/Session1/0007 | control/female | head | 1 | PASS |
| M02/Session1/0133 | dysarthric/male | head | 2 | PASS |

All eight source caches passed strict loading. The long original duration has no special
encoder path because Module 2 already supplies fixed-size images and a bounded graph.

## Token and numerical results

| Stream | Shape | Minimum | Mean | Std | Maximum | Token-L2 min / mean / median / max |
|---|---|---:|---:|---:|---:|---|
| Mel | `[8,16,128]` | -3.299641 | -3.04e-10 | 0.999960 | 3.355777 | 11.312163 / 11.313252 / 11.313489 / 11.313666 |
| Graph | `[8,1,128]` | -2.874022 | 7.86e-9 | 0.999690 | 3.315367 | 11.307742 / 11.310200 / 11.310207 / 11.311773 |
| Recurrence | `[8,16,128]` | -3.095664 | -1.63e-9 | 0.999821 | 3.262790 | 11.307582 / 11.311683 / 11.312503 / 11.313494 |
| Nonlinear | `[8,17,128]` | — | — | — | — | exact graph-then-recurrence concatenation |

All output values are finite. Two consecutive evaluation passes are bit-exact for every
branch and the nonlinear concatenation. These values are numerical sanity statistics,
not evidence of diagnosis, classification performance, or class separation.

## Gradient, freeze, state, and parameter audits

The sole backward pass used the sum of mean squared branch tokens. Every trainable Mel
projection/LayerNorm, RP-CNN, and GCN parameter tensor received a finite gradient. Every
ResNet feature-trunk parameter retained `requires_grad=False` and `grad=None`. No
optimizer step occurred. Saving to an in-memory state dict, constructing a new
weights-none architecture, loading the state, and repeating evaluation produced bit-exact
outputs in all three branches.

| Component | Total | Initially trainable | Initially frozen |
|---|---:|---:|---:|
| Mel ResNet18 feature trunk | 11,176,512 | 0 | 11,176,512 |
| Mel projection + LayerNorm | 65,920 | 65,920 | 0 |
| Recurrence CNN | 80,816 | 80,816 | 0 |
| Visibility GCN | 45,856 | 45,856 | 0 |
| **Total** | **11,369,104** | **192,592** | **11,176,512** |

## CPU performance

Batch size was two. Each branch and combined measurement used one warm-up followed by
three measured evaluation forwards.

| Stage | Mean (s) | Median (s) | Maximum (s) |
|---|---:|---:|---:|
| Mel | 0.019554 | 0.019469 | 0.019966 |
| Recurrence | 0.007708 | 0.007826 | 0.007874 |
| Graph | 0.012151 | 0.012082 | 0.012818 |
| Combined | 0.037647 | 0.037261 | 0.040940 |

CUDA was unavailable, so no CUDA timings are reported.

## Requirement coverage

| Requirement | Evidence | Result |
|---|---|---|
| Only signed Module 2 production cache accepted | Config/sign-off/namespace/provenance/hash rejection tests | PASS |
| Array and graph corruption rejected | Shape, dtype, finite, edge index, duplicate, loop, bidirection, artifact SHA tests | PASS |
| Speaker/fold metadata retained; no random split | Dataset order, provenance, and fold tests | PASS |
| Fixed label-independent preprocessing | Endpoint, replication, ImageNet constant, and relabelled-forward tests | PASS |
| Mel `[B,16,128]` and no FC head | Feature-trunk shape/head/freeze tests and real smoke | PASS |
| RP `[B,16,128]` | Binary conversion, shape, determinism, and real smoke | PASS |
| GCN `[B,1,128]` | Hand graphs, 1-node graph, mask/pool tests, and real smoke | PASS |
| Nonlinear `[B,17,128]` | Typed output contract and real smoke | PASS |
| Finite and deterministic | Unit and production smoke | PASS |
| Gradient and frozen backbone | Finite-gradient and absence tests | PASS |
| State-dict round trip | Exact branch equality | PASS |
| CLI validation and smoke | Synthetic offline CLI plus production commands | PASS |
| Protected inputs | Exact post-run metadata inventories | PASS |

## Commands

```text
uv run --python 3.12 --extra dev pytest tests/m3_encoders
# 48 passed in 25.53 s

uv run --python 3.12 --extra dev vaaniswitch-m3 validate --config configs/models/encoders.yaml --limit 8
# PASS; 8/8 valid signed representation entries

uv run --python 3.12 --extra dev vaaniswitch-m3 smoke --config configs/models/encoders.yaml --limit 8 --batch-size 2 --timing-repetitions 3
# PASS; ImageNet-1K V1 loaded and every smoke/audit gate passed

uv run --python 3.12 --extra dev pytest
# 232 passed in 35.91 s

uv run --python 3.12 --extra dev pytest --cov=vaaniswitch.m0_data --cov=vaaniswitch.m1_vmd --cov=vaaniswitch.m2_representations --cov=vaaniswitch.m3_encoders --cov-branch --cov-report=term-missing
# 232 passed; 83.19% branch-aware coverage; 80% gate passed

uv run --python 3.12 --extra dev ruff check .
# All checks passed

uv run --python 3.12 --extra dev mypy
# Success: no issues found in 85 source files

uv run --python 3.12 python -m compileall -q src tests
# PASS
```

## Protected inputs

| Input | Files | Bytes | Path/size/mtime SHA-256 | Result |
|---|---:|---:|---|---|
| Raw TORGO | 82,772 | 15,845,129,970 | `206041a4be1fad47975d25ba38c73bb49fe68e3957ebfb046d0db15c0b14c76a` | UNCHANGED |
| Module 1 production | 26,019 | 19,981,366,416 | `b4944b720c6664d55e57180c94e1c0c6149ac080b49705035ac12ca16be54baa` | UNCHANGED |
| Module 2 production | 43,365 | 2,378,359,184 | `c951a57640c0bbada8145ef5eeeab6cc534e03adcfef0e7f00fb6307f482ab9f` | UNCHANGED |

Module 2 retains exactly 8,673 each of its five expected files and zero zero-byte files.

## Artifacts and limitations

- `artifacts/metrics/module3_encoder_summary.json`
- `artifacts/metrics/module3_parameter_audit.json`
- `artifacts/metrics/module3_full_signoff_audit.json`
- `artifacts/logs/module3_encoder_failures.csv` (header only; zero failures)
- `artifacts/logs/module3_encoder_pipeline.log`

No new implementation defect was found. The provisional-to-locked dimension change is
an architecture resolution, not a defect, so no Module 3 defect-log row was added.
Accepted upstream limitations remain unchanged. The performance audit is a short CPU
smoke measurement, not a training-throughput benchmark. No classification performance
exists because no classifier or training was implemented.

## Gate

✅ Module 3 is **COMPLETE / PASSED / SIGNED OFF**. Module 4 is **NOT STARTED** and
requires explicit authorization.
