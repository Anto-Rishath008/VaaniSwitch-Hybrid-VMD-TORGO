# Module 0 Agent Rules

Read PROJECT_SCOPE_LOCK.md, FINAL_WORKFLOW.md, ARCHITECTURE_RATIONALE.md, ACCURACY_STRATEGY.md, TORGO_CURRENT_AUDIT.md, this MODULE.md and TEST_PLAN.md.

Work primarily in `src/vaaniswitch/m0_data/`, `tests/m0_data/`, and required configs/docs.

Hard rules: shared-VMD token architecture; all-mode default; no GMM-HMM; no ViT/large Transformer; no PyTorch-Geometric requirement; never modify raw TORGO; speaker-disjoint evaluation; regression test for every fix; stop after Module 0.
