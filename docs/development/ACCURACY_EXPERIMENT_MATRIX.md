# Accuracy Experiment Matrix

| ID | VMD | Mel | VG-GCN | RP-CNN | Fusion |
|---|---|---|---|---|---|
| A0 | no | yes | no | no | none |
| A1 | all modes | yes | no | no | none |
| B1 | all modes | no | yes | no | none |
| B2 | all modes | no | no | yes | none |
| B3 | all modes | no | yes | yes | concat |
| C1 | all modes | yes | yes | yes | pooled concat |
| C2 | all modes | yes | yes | yes | token cross-attention |
| C3 | selected modes | yes | yes | yes | token cross-attention |

C2 is the intended final architecture. C3 is only an optional VMD mode-selection ablation.
