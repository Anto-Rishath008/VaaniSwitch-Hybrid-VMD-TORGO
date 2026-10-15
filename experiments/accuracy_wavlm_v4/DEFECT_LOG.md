# V4 defect log

| ID | Finding | Reproduction | Resolution / regression |
|---|---|---|---|
| V4-001 | Initial loader incorrectly required HuBERT-style waveform normalization | Official processor has do_normalize=false; validation/test_processor_reproduction.log fails load contract | Preserve official false setting and attention masks; ProcessorRegression.test_official_processor_normalization_is_preserved passes; test_processor_fixed.log retained |
| V4-002 | Scaler test used zero absolute tolerance near zero | tests_initial.log: observed -2.98e-8 vs mathematical 0; 43/44 pass | Test uses 1e-7 absolute tolerance, retaining exact weighted-train-only expectation; production implementation unchanged |
| V4-003 | Idempotent outer claim publication could accept a concurrent identical claim | test_claim_race_reproduction.log deliberately inserts competing claim before publication | Exclusive atomic write; ExecutionGates.test_outer_claim_race_cannot_forward_twice passes |
| V4-004 | Inherited warn-only determinism and memory-efficient attention caused exact interrupted/resumed parameter drift on a second actual CUDA panel | validation/validation_final.log records real resume failure; test_determinism_reproduction.log fails strict-policy contract | FIXED: strict deterministic algorithms and math SDPA only for forward/recomputation; final 53-test suite and real CUDA interrupted/resumed equality PASS in validation_final_strict.log; earlier failed panel retained |

The normal Hub acquisition took about six minutes and completed successfully with verified files. An HTTPS connectivity check was performed while waiting; no model was substituted and no acquisition failure is claimed. The downloader now also implements a bounded-timeout normal HTTPS path to the same pinned official files; the retained checkpoint was obtained by the successful Hub download.
