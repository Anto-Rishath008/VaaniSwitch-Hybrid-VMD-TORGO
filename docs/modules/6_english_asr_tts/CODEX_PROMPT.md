# Codex Prompt — Module 6 Frozen English ASR, Transcript Review & Clear TTS

```text
Work ONLY on Module 6: Frozen English ASR, Transcript Review & Clear TTS.

FIRST read PROJECT_SCOPE_LOCK.md, FINAL_WORKFLOW.md, ARCHITECTURE_RATIONALE.md, ACCURACY_STRATEGY.md, TORGO_CURRENT_AUDIT.md, ACCURACY_EXPERIMENT_MATRIX.md, AGENTS.md, MODULE_TRACKER.md and this module's MODULE.md/AGENTS.md/TEST_PLAN.md.

FINAL ARCHITECTURE:
TORGO → shared VMD → all-mode reconstruction by default → Mel CNN 16 tokens + visibility graph pure-PyTorch GCN 1 token + recurrence CNN 16 tokens → nonlinear 17 tokens → token-level cross-attention (Q=Mel, K/V=nonlinear) → pooled streams → small MLP → dysarthric/control.

Separate utility: frozen English ASR → transcript → optional explicit correction → standard English TTS.

DO NOT implement GMM-HMM, EMD+VMD, default mode deletion, global-vector attention, ViT/large Transformer, PyTorch-Geometric requirement, Tamil/code-switching/RAWDysPeech/voice cloning.

Goal: Provide recognition/reconstruction without training another ASR research model.
Production: src/vaaniswitch/m6_reconstruction/
Tests: tests/m6_reconstruction/

Inspect before editing. Never modify data/raw. Use config-driven paths. Never invent labels/metrics. Use speaker-disjoint evaluation. Keep tests tiny. Do not launch full-corpus/long training unless explicitly requested. Every fix gets a regression test. Run tests, Ruff and type checks. Update TEST_REPORT.md, MODULE_TRACKER.md, CHANGELOG.md and DEFECT_LOG.md.

At end report files changed, commands, tests, outputs, blockers and exact manual next action. STOP after Module 6.
```
