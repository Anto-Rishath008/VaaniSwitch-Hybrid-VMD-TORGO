# PROJECT SCOPE LOCK — FINAL

This overrides every previous VaaniSwitch plan.

## Final project
**VaaniSwitch: Shared-VMD Multirepresentation Hybrid Network for English Dysarthria Classification and Clear Speech Reconstruction**

## Dataset
TORGO English dysarthric/control speech.

## Core target
Binary classification: **dysarthric vs control**.

## Shared VMD stage
Speech → VMD → **all-mode reconstruction by default**.

Do not automatically remove irregular/high-frequency modes; they may contain dysarthria evidence. Mode-selection is a separate ablation only.

## Branch A
VMD reconstruction → log-Mel → lightweight CNN → adaptive pool 4×4 → **16 Mel tokens**.

## Branch B
VMD reconstruction →
- natural visibility graph → 2-layer pure-PyTorch GCN → mean+max pool → **1 graph token**;
- recurrence plot → small 2D-CNN → adaptive pool 4×4 → **16 recurrence tokens**.

Nonlinear sequence = **17 tokens**.

## Fusion
Q = 16 Mel tokens. K,V = 17 nonlinear tokens.
One lightweight multi-head cross-attention block → residual + LayerNorm → pool attended Mel + pool nonlinear → concatenate → small MLP → class probability.

## Evaluation
Speaker-disjoint only. No random utterance train/test split. No head/array microphone duplicate leakage. Report fold variability.

## Reconstruction
Frozen pretrained English ASR → transcript → explicit optional correction → standard English TTS.

## Never reintroduce
GMM-HMM; EMD+VMD; mandatory mode deletion; one-vector attention; ViT requirement; large Transformer fusion; PyTorch-Geometric requirement; Tamil; code switching; RAWDysPeech; voice cloning; medical diagnosis claims.
