# Module 3 — Parallel Token Encoders: Mel-CNN, VG-GCN & RP-CNN

**Target:** Days 6–8
**Code:** `src/components/m3_encoders/`
**Tests:** `tests/encoders/`

## Goal
Build compact token encoders.

## Detailed tasks
- Mel CNN → 4×4 → 16 tokens.
- 2-layer pure-PyTorch GCN → 1 token.
- RP CNN → 4×4 → 16 tokens.
- nonlinear sequence = 17 tokens.
- single-branch sanity baselines.

## Outputs
- [B,16,D] Mel
- [B,1,D] graph
- [B,16,D] RP
- [B,17,D] nonlinear

## Acceptance
- [ ] Required outputs exist.
- [ ] Tests pass.
- [ ] No scientific leakage introduced.
- [ ] Test report completed.

Do not start the next module automatically.
