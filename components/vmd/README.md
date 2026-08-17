# Module 1 — Shared VMD

This package implements dependency-light audio standardization, a project-owned NumPy VMD
solver, all-mode reconstruction, scientific metrics, atomic resumable caching, diagnostic
SVGs, and build/validation orchestration.

The default is the Module 0 CORE subset and `primary_audio_path` only. The implementation
does not rediscover microphones, modify raw TORGO, remove VMD modes, denoise audio, or run
later representation/model stages.

```text
uv run --python 3.12 --extra dev components-m1 build --config configs/vmd/default.yaml --subset core --limit 6
uv run --python 3.12 --extra dev components-m1 validate --config configs/vmd/default.yaml --subset core --limit 6
```

Omit `--limit 6` for the resumable full CORE run. The nominal float64 array cache is
18.57 GiB before metadata and filesystem overhead. See
`docs/modules/vmd/TEST_REPORT.md` for measured smoke behavior and
`docs/modules/vmd/TAU_FIDELITY_REPORT.md` for the pre-build decision retaining
tau=0 and the full-mode storage policy.
