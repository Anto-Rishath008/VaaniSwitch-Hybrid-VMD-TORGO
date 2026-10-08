# Module 4 Code — Token-Level Cross-Attention Fusion & Dysarthria Classification

Read `docs/modules/4_token_cross_attention_fusion/MODULE.md` for the frozen development
protocol and exact manual commands. `DysarthriaClassifier` composes signed Module 3
encoders with `TokenFusionClassifier`; `FusionOutput.logits` is the BCE training input.
The CLI exposes split-audit, bounded smoke, manual train, and validation only.
