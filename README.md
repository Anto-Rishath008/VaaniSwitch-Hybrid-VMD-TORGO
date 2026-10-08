# VaaniSwitch FINAL — Shared-VMD Hybrid Dysarthria Classification & Clear Speech Reconstruction

Module 6 Phase-0 audit (2026-10-07): **PASS; implementation NOT STARTED**.
See the [audit and implementation plan](docs/modules/6_english_asr_tts/PHASE0_AUDIT_AND_PLAN.md).
The existing `small.en` / `pyttsx3` choices are retained; reference eligibility and a
separate runtime are planned. No ASR/TTS execution or WER/CER measurement has occurred.
Module 5 remains signed off and unchanged; its handoff statements below are historical.

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

The default CORE classification experiment excludes excluded records and articulation_task, instruction_or_nonstandard, or unknown prompts. Modules 0–5 are COMPLETE / PASSED / SIGNED OFF; Module5 sign-off preserves Fold1/A1 = RECOVERED_PROTOCOL_EXCEPTION and UNKNOWN prior native forwards. Module6 is NOT STARTED.

Raw TORGO audio, production VMD caches, representation caches, and other large generated
artifacts are intentionally excluded from this repository. This research software is not
a medical diagnostic system. Tiny-batch and smoke metrics are implementation checks and
do not establish classifier performance.

## Module 4 development baseline

The baseline keeps the ImageNet ResNet18 trunk frozen, trains the remaining encoders and
the one-block Mel-query fusion/classifier, and reserves outer Fold 1 without evaluating it.
See [Module 4 protocol](docs/modules/4_token_cross_attention_fusion/MODULE.md) and
[test report](docs/modules/4_token_cross_attention_fusion/TEST_REPORT.md).

Commands used for the completed manual baseline and validation (historical reference):

```powershell
uv run --python 3.12 --extra dev vaaniswitch-m4 train --config configs/models/fusion.yaml --outer-test-fold 1 --seed 42
uv run --python 3.12 --extra dev vaaniswitch-m4 validate --config configs/models/fusion.yaml --checkpoint artifacts/checkpoints/module4/fold1_seed42_6fc69cf0ad55/best.pt --outer-test-fold 1
```

The training command is bounded at 30 epochs with validation macro-F1 early stopping.
`--dry-run` audits the split without model forwards; `--smoke` runs one tiny development
epoch. The completed baseline stopped after six epochs and selected epoch 1 by validation
macro-F1 (0.8858803969843076). Independent validation on M03/MC02 reproduced the saved
metrics. These were speaker-disjoint development-validation results only; Fold 1 was
untouched during Module 4. Module 4 is signed off; its original evidence is preserved.

## Module 5 final closure

**Module 5: COMPLETE / PASSED / SIGNED OFF (exception-aware).**
**Module 6: NOT STARTED; handoff prepared only.**

All 21 frozen outer cells are independently verified: 20 NATIVE and Fold1/A1
RECOVERED_PROTOCOL_EXCEPTION. Its original prior native-forward count remains UNKNOWN;
its original consumed claim, native state and absent native result remain intact.
The eight-entry remaining sequence is COMPLETE with no next target.
Frozen foldwise mean/sample-SD aggregation and C2 descriptive robustness are complete.
No integrity blocker remains. This sign-off does not assert uninterrupted native
once-only compliance for Fold1/A1.

[Final report and Module6 handoff](docs/modules/5_accuracy_robustness/FINAL_CLOSURE_MODULE6_HANDOFF.md).
All 21 retained outer results, eight linked completions and frozen identities pass.
Primary cross-fold summary is arithmetic mean ± sample SD; C2 robustness covers
8,673 records in 70 observed subgroups. C2 does not improve macro-F1 over C1
in any fold; report simpler baselines honestly. No retuning or model reselection.

27 focused closure tests and the 830-test Module5 suite pass; Ruff passes.
Closure performed zero production model forwards/training/native-scientific writes.
The immutable package is artifacts/module5/677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9/final_closure/.
Module6 is prepared on paper only: frozen English ASR, transcript review, standard TTS.
