# Final Workflow — Start to End

## 1. TORGO logical manifest
One record = `(speaker_id, session_id, prompt_id)`.
Use head mic first, otherwise array mic. Alternate mic is metadata only. Exclude/log prompts with neither microphone.

## 2. Speaker-disjoint evaluation
Create deterministic development/evaluation speaker folds. Never split utterances randomly across speakers.

## 3. Shared VMD
Processed primary waveform → VMD K modes → **sum all modes** for the default shared reconstruction. Cache results. Selected-mode reconstruction is optional ablation only.

## 4. Representations from the same VMD reconstruction
- log-Mel → 224×224;
- recurrence plot → 224×224;
- natural visibility graph → ≤256 nodes.

## 5. Encoders
- Mel CNN → adaptive pool 4×4 → 16 tokens × D.
- visibility GCN → mean+max pool → 1 token × D.
- recurrence CNN → adaptive pool 4×4 → 16 tokens × D.
- nonlinear = 17 tokens × D.

## 6. Token cross-attention
Mel tokens query nonlinear tokens → residual + LayerNorm → mean-pool both streams → concatenate → small MLP → dysarthric/control probability.

## 7. Accuracy optimization
Tune only a small set: VMD K, D, learning rate, dropout, weight decay, graph node cap. Use train-only mild augmentation and early stopping.

Required ablations:
1. raw/no-VMD Mel CNN,
2. all-mode VMD Mel CNN,
3. nonlinear-only,
4. pooled concatenation,
5. token cross-attention,
6. optional selected-mode VMD hybrid.

Freeze settings before final grouped evaluation. Report fold-wise metrics and mean±std.

## 8. Recognition/reconstruction
Same user audio → frozen English ASR → raw transcript → explicit optional correction → standard English TTS → clear speech.

## 9. Final app
Show classification probability, VMD summary, Mel, recurrence plot, visibility graph, transcript, correction box and clear TTS audio.
