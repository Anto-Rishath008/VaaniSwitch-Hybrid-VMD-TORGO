# Module 4 — Token-Level Cross-Attention Fusion & Dysarthria Classification

**Target:** Days 9–10
**Code:** `src/vaaniswitch/m4_fusion/`
**Tests:** `tests/m4_fusion/`

## Goal
Perform token-level cross-attention and binary dysarthria classification.

## Detailed tasks
- Q=Mel 16 tokens.
- K,V=nonlinear 17 tokens.
- one MHA block.
- residual+LayerNorm.
- pool both streams and classify.
- run fusion ablations.

## Outputs
- fusion model
- checkpoint
- metrics
- ablation table

## Acceptance
- [ ] Required outputs exist.
- [ ] Tests pass.
- [ ] No scientific leakage introduced.
- [ ] Test report completed.

Do not start the next module automatically.
