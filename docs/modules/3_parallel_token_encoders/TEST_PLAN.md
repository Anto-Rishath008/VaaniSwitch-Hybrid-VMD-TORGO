# Test Plan

## Required gates

- Strict production namespace, Module 2/1/manifest identities, artifact hashes, filenames,
  array dtypes/shapes/finiteness, recurrence invariants, and graph invariants.
- Signed manifest and speaker-fold provenance retained without random utterance splitting.
- Fixed label-independent Mel scaling, replication, and ImageNet normalization.
- Binary recurrence conversion without corpus statistics.
- Hand-verifiable self-loop and symmetric-degree normalization.
- Variable-node batching, prefix masks, and padding-independent GCN pooling.
- Exact output shapes: Mel `[B,16,128]`, graph `[B,1,128]`, RP `[B,16,128]`, and
  nonlinear `[B,17,128]`.
- Finite tokens and exact repeated evaluation outputs.
- Finite trainable-branch gradients with no frozen-backbone gradients and no optimizer step.
- Exact in-memory state-dict save/load round trip.
- Parameter and CPU timing audits.
- CLI validation and smoke paths; unit tests use `weights=None` and require no network.
- Full Modules 0–3 regression, branch coverage at least 80%, Ruff, strict mypy, and compileall.
- Raw TORGO and Module 1/2 production cache inventories unchanged.

Every genuine defect follows reproduce → failing regression test → targeted fix → retest →
full regression → centralized defect-log closure. The D=192 to D=128 resolution is an
architecture decision, not a software defect.
