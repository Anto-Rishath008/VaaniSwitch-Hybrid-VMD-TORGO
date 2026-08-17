# Root AGENTS.md — FINAL

Read `PROJECT_SCOPE_LOCK.md` first.

Architecture: TORGO → shared VMD all-mode reconstruction → Mel CNN 16 tokens + VG-GCN 1 token + RP-CNN 16 tokens → token-level cross-attention → binary classification.

ASR/TTS are separate frozen utilities.

Never reintroduce GMM-HMM, EMD+VMD, mandatory mode removal, single-vector attention, ViT requirement, large Transformer fusion, PyTorch-Geometric requirement, Tamil/code switching, RAWDysPeech or voice cloning.

Never modify raw TORGO. Use speaker-disjoint evaluation. One logical record per speaker/session/prompt. No microphone-copy leakage.

Tune only on development speakers. Freeze settings before final folds. Train-only augmentation. Report fold mean±std. No guaranteed accuracy claims.

Every bug fix needs a regression test. Update module test report, tracker, changelog and defect log. Stop after active module.
