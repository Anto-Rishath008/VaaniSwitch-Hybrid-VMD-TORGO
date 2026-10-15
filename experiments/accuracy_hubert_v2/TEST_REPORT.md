# V2 test report

2026-10-13, Asia/Calcutta. **PASS: 56 tests, 0 failures, 0 errors, 0 skips.**
Final unittest runtime: 13.755 seconds, existing .venv-accuracy, actual RTX 4070 CUDA.

Command, from the accuracy-study worktree:

```powershell
$env:ACCURACY_CUDA_FEATURE_TEST='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv-accuracy\Scripts\python.exe -B -m unittest discover -s experiments/accuracy_hubert_v2/tests -v
```

The 38 inherited tests retain dataset, speaker split, leakage, pinned model,
feature checksum/mask, training-only scaling/weighting, original architecture,
selection, checkpoint, interruption, OOM, once-only outer claim and reporting
coverage. Their outputs are under V2 only. Outer prediction/report fixtures are synthetic.

The 18 added regressions cover:

- FP32 updates even inside an ambient autocast context; no GradScaler creation.
- Non-finite input, parameter, logit, loss, individual gradient and total norm.
- Corrupt pre/post optimizer state and prevention of unsafe optimizer steps.
- Invalid scaler/RP, persistent diagnostics and no invalid checkpoint publication.
- Partial-epoch interruption with model/optimizer/RNG restoration.
- FP16 checkpoint rejection and clipping OOM exception preservation.
- Read-only cache write/missing/corruption rejection and preprocessing drift.
- Three-fold E1/E2/E3 ordering before evaluation; no extractor on the reuse path.
- Frozen V2 identity reuse and source-drift rejection.

The report-generation regression also verifies the V2 reproduction command.

Actual TORGO CUDA validation uses only Fold 1 development roles:

- E2: 4 epochs × 595 batches = 2,380 steps; repeated uninterrupted comparison
  also 2,380 steps; exact resumed model/AdamW/RNG/scaler/history/best equality.
- E3: 4 epochs × 595 batches = 2,380 steps; interrupted at epoch 2 and resumed.
- Each uses all 4,760 inner-training records from eight speakers, batch size 8.
- Each also passes two-epoch, 64-training/16-validation-record regularized panels
  at layers 9 and 12 (16 optimizer steps per panel).
- Both pass save/load, RNG roundtrip, sealed fit reuse, and repeated fixed-checkpoint
  prediction equality. No non-finite training state occurs.
- The final frozen-source E2/E3 check and real FP32 benchmarks are recorded in
  `logs/final_runtime_check.json`.

Full cache verification: **8,673/8,673 PASS**, including checksums, identities,
revision, preprocessing and all crop intervals; zero corpus re-extractions.

Python/PowerShell syntax and tracked whitespace checks pass. Full launcher
execution and real outer inference were not performed. Native E3 adaptive-pooling
backward warning remains expected; E3 bitwise resumed-training equality is not claimed.

Detailed evidence: `logs/tests_final.log`, `logs/v1_diagnosis.json`,
`logs/cuda_validation.log`, `logs/cuda_validation.json`,
`logs/feature_reuse.json`, `logs/final_runtime_check.json`.
