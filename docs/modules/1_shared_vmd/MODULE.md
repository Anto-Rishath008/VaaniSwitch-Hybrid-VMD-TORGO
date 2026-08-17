# Module 1 — Shared VMD Decomposition & Safe Reconstruction

**Status:** ✅ COMPLETE / PASSED / SIGNED OFF  
**Code:** `src/vaaniswitch/m1_vmd/`  
**Tests:** `tests/m1_vmd/`

## Goal

Create the single shared VMD stage while retaining all modes by default and preserving a
safe, reproducible contract for later representation modules.

## Implemented behavior

- Reads only Module 0 `primary_audio_path` records from a configured manifest subset.
- Defaults to the 8,673-record CORE subset; EXTENDED can be selected explicitly.
- Standardizes a working copy to finite float64 mono at 16 kHz without modifying raw data.
- Applies conservative RMS normalization only; no VAD, silence deletion, noise reduction,
  pitch correction, or other aggressive processing is present.
- Applies project-owned pure-NumPy VMD using the parameters in
  `configs/vmd/default.yaml`.
- Reconstructs by summing all modes. Selected-mode reconstruction is a disabled,
  explicit future ablation and has no automatic selection rule.
- Records per-mode center frequency, energy, relative energy, RMS, and spectral centroid,
  plus reconstruction fidelity metrics.
- Uses atomic, content-addressed, semantically validated, resumable caches under
  `data/processed/vmd/<config-hash>/<speaker>/<session>/<prompt>/`.

## Commands

Smoke build and validation:

```text
uv run --python 3.12 --extra dev vaaniswitch-m1 build --config configs/vmd/default.yaml --subset core --limit 6 --force
uv run --python 3.12 --extra dev vaaniswitch-m1 validate --config configs/vmd/default.yaml --subset core --limit 6
```

Executed full CORE build and validation commands:

```text
uv run --python 3.12 --extra dev vaaniswitch-m1 build --config configs/vmd/default.yaml --subset core
uv run --python 3.12 --extra dev vaaniswitch-m1 validate --config configs/vmd/default.yaml --subset core
```

The completed production cache occupies 19,981,366,416 bytes (18.6091 GiB), including
NumPy headers and JSON metadata. The nominal array-only estimate was 18.5693 GiB.

The pre-full-build numerical check compared tau values 0, 0.1, 0.25, and 0.5 on a fixed
12-record CORE panel. Tau=0 remains the default because the nonzero candidates had severe
500-iteration convergence failure and 2.67–3.13× mean runtime despite better aggregate
fidelity. The current full float64 modes-plus-reconstruction cache is retained for strict
validation and later selected-mode ablations; see `TAU_FIDELITY_REPORT.md`.

## Acceptance

- [x] Implementation complete.
- [x] Synthetic and real TORGO smoke validation passed.
- [x] Fixed 12-record tau fidelity review completed; production `tau=0` retained.
- [x] Full 8,673-record CORE cache built.
- [x] Full CORE validation passed with 8,673/8,673 valid entries and zero failures.
- [x] Module 0+1 regression, branch coverage, Ruff, strict mypy, and compileall gates passed.
- [x] Raw immutability, Module 0 identity, primary-path binding, and cache integrity passed.
- [x] Module 1 signed off.

Forty-eight records reached the 500-iteration ceiling without satisfying the strict
update tolerance. Their finite, structurally valid, source-bound, reconstruction-valid
outputs are retained as an accepted scientific limitation. Module 2 remains not started
and requires explicit authorization.
