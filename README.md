# VaaniSwitch FINAL — Shared-VMD Hybrid Dysarthria Classification & Clear Speech Reconstruction

## Final research core

```text
TORGO English WAV
        ↓
Shared VMD decomposition
        ↓
ALL-MODE VMD reconstruction (default)
        ↓
        ┌───────────────────────────────┐
        │                               │
        ▼                               ▼
Log-Mel spectrogram              Nonlinear representations
        ↓                        ├─ visibility graph → small GCN → 1 token
lightweight CNN                  └─ recurrence plot → small CNN → 16 tokens
        ↓                               │
4×4 feature map                         └→ 17 nonlinear tokens total
        ↓
16 Mel tokens
        └───────────────┐
                        ▼
              token-level cross-attention
      Q=16 Mel tokens; K,V=17 nonlinear tokens
                        ↓
               residual + LayerNorm
                        ↓
         pooled Mel + pooled nonlinear
                        ↓
                     small MLP
                        ↓
               Dysarthric / Control
```

## Recognition / reconstruction utility

```text
same input speech → frozen English ASR → transcript → optional explicit correction → standard English TTS → clear speech
```

## Final design decisions

- VMD only; no EMD.
- Keep all VMD modes by default.
- Mode deletion is an optional ablation only.
- Lightweight CNN; no ViT required.
- Small pure-PyTorch GCN; no PyTorch Geometric required.
- Small recurrence 2D-CNN.
- Token-level cross-attention, not one-vector attention.
- Frozen pretrained ASR; no GMM-HMM.
- Standard English TTS; no voice cloning.
- Speaker-disjoint evaluation.
- TORGO English only.

## Module 0 dataset policy

Module 0 is complete. It preserves one logical record per speaker/session/prompt and audits both microphone recordings before selection. A structurally valid head recording is mandatory as primary when present. If the head is missing, or is structurally unusable with recorded audit evidence, a structurally valid array recording is used. If neither recording is structurally usable, the logical record is excluded. Raw TORGO files remain immutable.

Later modules must select records from the broad manifest as follows:

- CORE/CLEAN: `normal_speech_eligible == true` and `usability_status == usable`.
- EXTENDED robustness ablation: `normal_speech_eligible == true` and `usability_status in {usable, review}`.

The default core classification experiment excludes records marked `excluded` and prompt types `articulation_task`, `instruction_or_nonstandard`, or `unknown`. Modules 0 and 1 are complete and signed off. Module 2 is implementation-ready but awaits its manual full CORE build; Module 3 has not started.

Raw TORGO audio, production VMD caches, representation caches, and other large generated
artifacts are intentionally excluded from this repository. This research software is not
a medical diagnostic system, and no trained-classifier accuracy is claimed at the current
project stage.
