# Test Plan

1. Accept signed Module 3 architecture and reject changed Module 3/2/1/manifest identities.
2. Enforce D=128, Q length 16, K/V length 17, direction and nonlinear sensitivity.
3. Verify Mel residual, LayerNorm, mean pools, 256-vector, single logit and optional per-head weights.
4. Verify canonical labels and direct weighted BCE logits; compute weights from training records only.
5. Verify deterministic speaker splits for each outer fold, both development classes, zero overlap,
   authoritative fold consistency and pre-forward rejection of reserved/mislabeled records.
6. Verify each metric with hand calculations, tied-rank AUC, single-class handling and threshold 0.5.
7. Verify finite nonzero aggregate gradients in every trainable component and no ResNet gradients;
   retain frozen backbone weights and BatchNorm buffers after tiny optimization.
8. Verify fixed-batch loss reduction, eval determinism and exact model/optimizer/RNG resume.
9. Verify checkpoint metadata, hashes, speaker/record selection, corruption rejection and absence of
   data arrays; keep best and last only; prevent smoke checkpoints from masquerading as baseline.
10. Exercise split audit, dry run, real/synthetic smoke, one tiny smoke epoch and checkpoint validation.
11. Test early stopping on validation macro-F1, no outer-test loads, and protected synthetic trees.
12. Compare live pre/post raw and cache metadata fingerprints and all captured signed artifact hashes.
13. Run Module 4 tests, full project tests, branch-aware coverage >=80%, Ruff, strict mypy,
    compileall and git diff --check. Keep reproduction logs for actual defects.

Use only `docs/defects/DEFECT_LOG.md`: OPEN -> reproduce -> failing regression -> targeted fix ->
targeted retest -> full regression -> CLOSED. No final outer-fold evaluation, ablation, broad search,
long baseline training, or Module 5 work is part of these gates.
