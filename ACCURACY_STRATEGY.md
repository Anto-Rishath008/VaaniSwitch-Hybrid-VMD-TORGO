# Accuracy Strategy

## Goal
Maximize honest speaker-disjoint performance, not inflated random-utterance accuracy.

## Safeguards
- speaker-disjoint folds;
- no microphone duplicate leakage;
- train-only augmentation;
- no evaluation statistics during tuning.

## Regularization
Compact encoders, pretrained Mel CNN when available, freeze most CNN layers initially, dropout, weight decay, early stopping, one attention block.

## Tune only
VMD K, token dimension D, learning rate, dropout, weight decay, graph node cap.

## Final metrics
Accuracy, balanced accuracy, macro-F1, dysarthric sensitivity/recall, specificity, precision, ROC-AUC where valid, confusion matrix, fold mean±std.

## Baselines
Raw Mel; VMD Mel; nonlinear-only; pooled concat; token cross-attention; optional selected-mode VMD.

Do not promise a fixed 95–99% accuracy before held-out speaker evaluation.
