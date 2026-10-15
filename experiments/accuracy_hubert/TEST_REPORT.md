# Accuracy study implementation validation

Date: 2026-10-13, Asia/Calcutta. **38 tests PASS, 0 failures, 0 skipped**, 16.208 seconds.
Authoritative retained output: `logs/handoff_tests.log`. Earlier failures and
intermediate passing suites remain in their original logs.

Executed in the existing `.venv-accuracy`:

```powershell
$env:ACCURACY_CUDA_FEATURE_TEST = '1'
& .\.venv-accuracy\Scripts\python.exe -B -m unittest discover -s .\experiments\accuracy_hubert\tests -p 'test_*.py' -v
```

| Contract | Evidence |
|---|---|
| Frozen identities and CORE membership | 8,673 unique records, 235 metadata crops, 7,503 head/1,170 array primaries |
| Original outer and inner speaker roles | Original 2,819/2,873/2,981 counts; exact validation speakers; all identity intersections checked |
| Microphone selection | Usable head preference, missing-head and invalid-head valid-array fallback; alternate duplication rejected |
| Existing RP provenance | Signed nested source metadata, exact original preprocessing/code, primary-audio binding corruption rejection |
| HuBERT dimensions and indexing | Actual pinned model on CUDA; [1,3,1536] output for hidden states 6/9/12; frozen/eval backbone |
| Correct masks and pooling | Exact convolution frame lengths; padded values excluded; population std; mixed-length GroupNorm batches rejected |
| Audio window | Centered 128,000-sample crop, downmix and explicit crop interval metadata |
| Feature identity, checksum and resume | Completed feature reuse, source invalidation/corruption rejection, pinned configuration mismatch rejection |
| Train-only statistics and weighting | Scaler excludes validation/test; equal speaker loss mass; outer-role and forbidden-speaker access rejected |
| Bounded CPU fitting | Actual BLAS/OpenMP pools inside logistic fitting use at most two threads |
| No outer-label selection | Poisoned outer labels fail before access; candidate ranking depends only on inner metrics; synthetic final-fit provenance checked |
| CNN/fusion training and evaluation | E2/E3 finite forward/backward, real CUDA E3 gradients, fixed-checkpoint repeated evaluation equality |
| Checkpoint and RNG state | Model/optimizer/AMP/Python/NumPy/CPU/CUDA RNG serialization; E2 interrupted resume bit-identical to uninterrupted fit |
| OOM rollback | Injected extraction OOM halves batches; OOM after an optimizer step restores entire unfinished epoch and matches the smaller-batch reference |
| Stage recovery | Completed stages skipped, failed stages resumed, failed events retained, terminal-epoch/publication interruption recovered |
| Outer claim safety | Completed results reused without inference; interrupted/changed/corrupt claims/seals stop |
| Metrics and exports | Known confusion/AUC; arithmetic fold mean and ddof=1 SD; speaker means/errors; all five report files and three figures generated from synthetic fixtures |

Synthetic outer fixtures contain invented test identities used only for software tests;
they never load reserved TORGO signals or publish scientific results. Temporary fixture
exports stay under `smoke/` and are removed by the test harness. Real source/RP reads
in tests are restricted to Fold 1 development speakers.

The final six-record smoke is in
`smoke/cuda-1abefcda4275498b9684a48562d589c1/SMOKE_REPORT.json`.
It passed actual RTX 4070 inference, E1 CPU fitting, E2/E3 CUDA optimization, checkpoint
roundtrips, feature reuse and stage reuse. Its shape is [6,3,1536]; two recordings
were centered-cropped. Classifier parameters: E1 1,537; E2 213,633; E3 310,833.
Frozen HuBERT parameters: 94,371,712. No smoke accuracy is reported as final accuracy.

Measured PyTorch peak allocation: 943,448,064 bytes (0.879 GiB). Peak reserved:
1,270,874,112 bytes (1.184 GiB), including disposable extraction/training benchmarks.
These are PyTorch allocator measurements, not whole-device driver/context accounting.

The expected native B2 adaptive-pooling CUDA backward warning is disclosed in
ARCHITECTURE.md and EXPERIMENT_PROTOCOL.md. Seed/RNG restoration is implemented;
bitwise fresh-run or resumed E3 training equality is not claimed. Repeated evaluation
was exact in the tested cases.

Python parsing passes for all 11 source and three test files. The PowerShell launcher
parses successfully. `git diff --check` passes for existing tracked files; no tracked
file changed. `git check-ignore` confirms model/cache/smoke/log exclusions.

Defects AH-001 through AH-008 and before-fix evidence are in DEFECT_LOG.md.
Final preservation evidence is in `logs/implementation_preservation_summary.json`,
with full before/after inventories retained. The complete 8,673-record extraction,
63 candidate fits, nine final fits and real outer inference were not executed.
