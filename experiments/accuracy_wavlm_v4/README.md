# VaaniSwitch V4: WavLM speaker-disjoint research

Independent post-hoc experimental pipeline under `experiments/accuracy_wavlm_v4/` in `VaaniSwitch_Accuracy_Study`. Original Modules 0-6, raw TORGO and V1/V2/V3 are protected. V4 is not the signed shared-VMD release architecture.

Use the existing `.venv-accuracy`; no dependency installation was needed. Model: official `microsoft/wavlm-base-plus`, immutable revision `4c66d4806a428f2e922ccfa1a962776e232d487b`. See `configs/model_lock.json` for file hashes, official processor configuration and installed dependency versions. The official processor disables waveform normalization. All download/cache files are inside V4.

Read `EXPERIMENT_PROTOCOL.md`, `MODEL_SELECTION_PROTOCOL.md`, then `IMPLEMENTATION_REPORT.md`. Full development and real outer inference are different permission boundaries. The safe default launcher mode is bounded validation.

```powershell
# Reverify/download the pinned model and validate manifest/partition identities.
powershell -NoProfile -ExecutionPolicy Bypass -File ".\experiments\accuracy_wavlm_v4\run_all.ps1" -Mode Prepare
# Automated unit/regression/leakage tests only.
powershell -NoProfile -ExecutionPolicy Bypass -File ".\experiments\accuracy_wavlm_v4\run_all.ps1" -Mode Test
# Tests plus ten-record, one-epoch CUDA checks; this creates retained validation evidence.
powershell -NoProfile -ExecutionPolicy Bypass -File ".\experiments\accuracy_wavlm_v4\run_all.ps1" -Mode Validate
# Verify current protected files against the original retained session baseline.
powershell -NoProfile -ExecutionPolicy Bypass -File ".\experiments\accuracy_wavlm_v4\run_all.ps1" -Mode Preserve
```

After separate user approval, the development-only launcher is:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_Accuracy_Study\experiments\accuracy_wavlm_v4\run_all.ps1" -Mode Develop -AuthorizeDevelopment
```

It requires passing tests, matching source-bound CUDA evidence, verified model and lineage, and an 85 GiB free-space planning allowance. It reuses verified completed work, runs development selection/refits, freezes all three contexts, and stops. It never starts outer inference. All checkpoints are retained; monitor free disk space and do not delete historical evidence.

Only after another explicit authorization:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\experiments\accuracy_wavlm_v4\run_all.ps1" -Mode Outer -AuthorizeOuter
```

Do not run the last two commands during this implementation session. An unsealed outer claim stops for review. Resume a development interruption by invoking the same approved development command; settings/source/dependencies/partitions must match. Neural resume uses complete accumulation-update state; work since the last durable checkpoint is replayed. Interrupted sklearn LBFGS restarts from the beginning. Numerical/OOM failures require review; no automatic altered-configuration retry is implemented.

Production outputs are generated only from later observed results: `FINAL_EXPERIMENT_REPORT.md`, `final_results.json`, recording/fold/speaker CSVs, development ablations, selected configurations and five figures. Synthetic report tests write only under `validation/unit/` and prominently mark fixture outputs. The implementation smoke results are infrastructure evidence, not estimates of V4 classification performance.

Only `validation/`, `cache/` and `logs/` have execution evidence from this session. Full development training has NOT been executed. Real outer-test inference has NOT been executed. V4 accuracy is not yet known. Ninety percent is aspirational, with no guarantee. Reused TORGO outer results are exploratory; independent confirmation needs genuinely untouched speakers or an appropriate external corpus.
