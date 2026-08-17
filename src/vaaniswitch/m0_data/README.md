# Module 0 Code — TORGO Manifest, Quality Audit & Speaker Evaluation Protocol

This package performs prompt-driven TORGO discovery, immutable WAV auditing, canonical manifest construction, deterministic speaker-disjoint fold creation, and duplicate/leakage validation. It does not resample audio or run VMD, feature extraction, ASR, or training.

Build all Module 0 outputs:

```text
uv run --python 3.12 --extra dev vaaniswitch-m0 build --config configs/data/torgo.yaml
```

Revalidate existing outputs against each other and the current raw-tree metadata:

```text
uv run --python 3.12 --extra dev vaaniswitch-m0 validate --config configs/data/torgo.yaml
```

The authoritative completion report is `docs/modules/0_torgo_manifest_evaluation/TEST_REPORT.md`.
