# Study defect log

All defects below are confined to new study code. No original module fix or rewrite.

| ID | Defect | Resolution and regression | Status |
|---|---|---|---|
| AH-001 | Adapter initially read serialized RP source metadata at the top level. | Use representation_metadata.source. test_rp_nested_source_metadata_regression; before-fix KeyError retained in logs/AH-001_before_fix.log. | CLOSED |
| AH-002 | Copying preprocessing from the experimental checkout preserved its different line endings, failing exact original-byte provenance. | Copy original preprocessing bytes read-only; test_native_b2_code_is_exact_except_isolated_imports. Initial failure retained in logs/contract_tests.log. | CLOSED |
| AH-003 | Resuming after the terminal epoch but before complete.json tried to delete an epoch-local variable absent on zero-iteration resume. | Cleanup only persistent locals. test_terminal_epoch_checkpoint_resumes_before_summary_publication; before-fix error retained in logs/AH-003_before_fix.log. | CLOSED |
| AH-004 | Strict deterministic CUDA algorithms rejected the native B2 adaptive-pooling backward. | Preserve B2; seeded warn-only training policy, disclose nondeterminism. test_e3_cuda_backward_under_declared_seed_policy; failure retained in logs/AH-004_before_fix.log and first smoke log. | CLOSED with documented runtime limitation |
| AH-005 | Tuple-valued split identity would differ after JSON roundtrip on resumed selection. | Serialize split speaker lists explicitly. test_split_identity_is_json_roundtrip_stable and test_selection_final_fit_and_outer_seal_synthetic. | CLOSED |
| AH-006 | An extraction summary published just before stage completion could be recomputed into different timing/cache-hit metadata on resume. | Validate complete caches and reuse published summary before model construction. test_extraction_resume_after_summary_before_stage_completion. | CLOSED |
| AH-007 | The synthetic OOM extractor fixture lacked a model after backbone parameter accounting was added. | Supply a disposable model stub; test_extraction_oom_halves_batch_and_preserves_cache. Retain the initial fixture failure in logs/final_tests.log. Actual CUDA extraction already passed. | CLOSED, test fixture only |
| AH-008 | OMP/MKL limits alone do not necessarily bound a NumPy OpenBLAS build. | Set OPENBLAS_NUM_THREADS in the launcher and explicitly constrain actual LR thread pools with threadpoolctl. test_logistic_fit_bounds_actual_blas_thread_pools checks runtime pool sizes inside fitting. | CLOSED |

Additional recovery regressions cover corruption/stale cache entries, preserving failed
stage events, consumed outer claims, optimizer/RNG restoration and rollback after an
injected OOM following a successful optimizer step. No unresolved implementation defect
is known at handoff. Full-corpus execution has not been validated by running it.
