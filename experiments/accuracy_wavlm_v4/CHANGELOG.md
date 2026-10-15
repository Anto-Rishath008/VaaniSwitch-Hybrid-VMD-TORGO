# V4 changelog

2026-10-13: Continued the initial two-file implementation and retained its original protected-before inventory. Verified the resumed original/study state before further implementation. Added the official pinned WavLM acquisition, inherited CORE/CV contracts, V4-only frozen cache, FP32 selective fine-tuning, resumable training state, development selection, authorization/outer sealing, future reporting, launcher and automated validation. All changes are inside V4.

2026-10-13: Reproduced and fixed the processor normalization contract: official WavLM disables normalization, unlike inherited HuBERT preprocessing. Added regression coverage and retained initial failure output.

2026-10-13: Corrected the zero-valued scaler expectation tolerance in its test (2.98e-8 float32 weight-rounding difference; no production scaler change).

2026-10-13: Reproduced and fixed an outer-claim publication race by using exclusive atomic creation. Identical claims are no longer accepted idempotently. Retained failure and regression logs.

2026-10-13: Added report-only WavLM-versus-HuBERT development comparison after decision freezing, requiring identical grouped speaker partitions; synthetic negative-difference coverage added. Completed decision resume also completes report publication.

2026-10-13: Second actual CUDA panel exposed nondeterministic attention resume drift. Reproduced the inherited warn-only backend policy in a focused test and switched V4 to strict deterministic math SDPA for forward and backward recomputation, preserving both earlier panels. Full development remains unexecuted.
