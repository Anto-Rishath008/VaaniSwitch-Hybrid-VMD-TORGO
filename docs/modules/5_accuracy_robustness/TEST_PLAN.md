# Test Plan

Use synthetic fixtures for full workflows, including outer evaluation. Real smoke
is restricted to Fold 1 development speakers. Do not rewrite signed Modules0–4.
Every defect requires reproduction, a failing regression, targeted fix/retest and
an entry only in `docs/defects/DEFECT_LOG.md`.

| Requirement group | Tests |
|---|---|
| Signed identities, wrong-hash rejection, exact outer folds, deterministic 8/2/5 roles, both classes, train-only weights, source/batch guards, protocol lock, CLI dry run | `test_protocol_splits.py` |
| Exact A0 primary binding, no VMD/Module 2 source, signed numerical conditioning/Mel, separate cache, corruption/resume, branch isolation, shapes/finite outputs, pooling, Q/K/V, gradients, frozen buffers, shared initialization/head capacity | `test_models_no_vmd.py` |
| Ordered selection metrics, selected epochs, no outer selection, all candidates required, hand-computable metrics, sample std, ledger transitions/locks/IDs/failure recovery | `test_search_ledger_metrics.py` |
| Fresh final initialization, all 10 development speakers, exact epochs, development-only weight, no final validation, checkpoint provenance/corruption, bit-exact interrupted/resumed model and AdamW | `test_checkpoint_training.py` |
| Candidate selection-file/ledger/recovery consistency, all-comparison retrain gate, actual synthetic outer forwards once, immutable results/checksums, failure after claim, subgroup counts/single-class behavior/no selection mutation, predecessor sentinels | `test_workflow_outer_robustness.py` |

Follow unit tests with the instrumented bounded real smoke, module suite, full suite,
separate branch-aware coverage (configured >=80%), Ruff, strict mypy across Modules0–5
and tests, compileall, git diff --check and protected-input audit. Implementation
readiness is separate from manual scientific experiments and Module 5 sign-off.
