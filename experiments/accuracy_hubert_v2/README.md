# VaaniSwitch HuBERT accuracy study V2

Numerical-stability repair of incomplete V1. V1 is preserved; all 8,673 frozen
HuBERT entries are reused through verified read-only references. E2/E3 use CUDA
FP32 without GradScaler. The frozen HuBERT feature transform remains unchanged.

Run the complete V2 study manually after reviewing the repair evidence:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_Accuracy_Study\experiments\accuracy_hubert_v2\run_all.ps1"
```

The launcher uses the existing `.venv-accuracy`, performs CUDA checks, verifies V1
features without extraction, runs the original E1/E2/E3 candidate grids and all
three folds, freezes every final fit before outer inference, and generates the
original five reports/data files and three plots. It requires no interactive input.
All V2 checkpoints, stages, results and logs are under this directory.

Rerun the same command to resume unfinished epochs or reuse sealed stages/results.
Completed epoch boundaries include model, AdamW, scaler provenance, RNG, history,
best state and early-stopping state. OOM recovery is bounded by batch size one.
Numerical corruption fails closed; a consumed outer claim without a valid sealed
result requires review and is never automatically repeated.

See `IMPLEMENTATION_REPORT.md`, `TEST_REPORT.md`, `PRESERVATION_REPORT.md`,
`EXPERIMENT_PROTOCOL.md` and `V1_LINEAGE.json`. Validation checkpoints under
`validation/` are isolated from study `checkpoints/` and never supply study fits.

The full launcher has **not** been executed during the repair. No real outer-test
inference or accuracy result is produced by this handoff.
