# Module 3 — Parallel Token Encoders: Mel-CNN, VG-GCN & RP-CNN

**Status:** COMPLETE / PASSED / SIGNED OFF
**Target:** Days 6–8
**Code:** `src/vaaniswitch/m3_encoders/`
**Tests:** `tests/m3_encoders/`

## Locked architecture resolution

The provisional `D=192` values in `configs/models/encoders.yaml` and
`configs/models/fusion.yaml` were superseded before implementation. The single common
token dimension is `D=128`; the future four-head Module 4 interface will therefore use
32 dimensions per head. No active architecture configuration retains `D=192`.

The lightweight scaffold widths remain locked: graph hidden width 96 and recurrence
channels 16, 32, and 64. This decision altered no Module 0–2 data or representation.

## Signed source boundary

Module 3 reads only the production Module 2 namespace
`data/processed/representations/d8a36c6ab8c48b15e61ecd63194ffa30ede4e7e7e7ab0a30c197002652f9d65d/`.
The loader validates the Module 2 sign-off and numerical config, Module 1 and manifest
identities, exact five-file directory schema, per-artifact size/SHA-256/shape/dtype, and
Mel/RP/graph semantics. It joins condition, gender, prompt, microphone, and evaluation
fold metadata read-only from the signed manifest and speaker-fold map. It does not read
raw WAV or Module 1 arrays and does not create a token cache.

## Encoders

- Mel: fixed `[-100,+20]` dB clamp and `[0,1]` mapping, one-to-three channel
  replication, fixed ImageNet mean/std, frozen ImageNet-1K V1 ResNet18 feature trunk,
  4x4 adaptive average pool, `Linear(512,128)`, and LayerNorm.
- Visibility graph: exactly-once self-loops, `D^-1/2(A+I)D^-1/2`, GCN `2→96→128`,
  GELU/dropout, valid-node mean+max pool, `Linear(256,128)`, and LayerNorm. Padding is
  excluded from adjacency, degree, mean, and maximum operations.
- Recurrence: binary float32 conversion, three two-convolution GroupNorm/GELU blocks
  with channels `1→16→32→64`, pooling after the first two blocks, 4x4 adaptive average
  pool, `Linear(64,128)`, and LayerNorm.

## Outputs

- `[B,16,128]` Mel
- `[B,1,128]` graph
- `[B,16,128]` RP
- `[B,17,128]` nonlinear convenience concatenation

## Acceptance

- [x] Strict signed Module 2 loader and speaker-fold provenance.
- [x] All three token encoders and typed output contract.
- [x] Real ImageNet-1K V1 smoke, determinism, gradient, freeze, state, parameter, and CPU audits.
- [x] 48 Module 3 tests and all 232 project tests pass.
- [x] 83.19% branch-aware coverage; Ruff, strict mypy, and compileall pass.
- [x] Raw TORGO and signed Module 1/2 caches remain unchanged.
- [x] No attention, classifier, training, accuracy metric, ASR, or TTS was implemented.

Module 4 remains not started and requires explicit authorization.
