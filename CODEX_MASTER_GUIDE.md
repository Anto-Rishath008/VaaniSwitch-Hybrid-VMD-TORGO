# Codex Master Guide

Final architecture phrase:
**Shared VMD all-mode reconstruction → Mel-CNN 16 tokens + visibility-GCN 1 token + recurrence-CNN 16 tokens → 17 nonlinear tokens → token-level cross-attention → dysarthric/control classification.**

Frozen English ASR + standard TTS are separate utilities.

Before every module read PROJECT_SCOPE_LOCK.md, FINAL_WORKFLOW.md, ARCHITECTURE_RATIONALE.md, ACCURACY_STRATEGY.md, TORGO_CURRENT_AUDIT.md, MODULE_TRACKER.md and active module docs.

Codex writes code/tests/smoke commands. User runs full VMD cache, representation cache, long training and final speaker-fold evaluation.

Never start the next module automatically.
