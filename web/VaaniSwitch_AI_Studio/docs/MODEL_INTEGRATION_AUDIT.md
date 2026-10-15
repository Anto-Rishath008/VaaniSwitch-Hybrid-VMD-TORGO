# Model integration audit

Observed on 15 October 2026. All referenced research assets were opened read-only. No training, refit, model selection, new ensemble or TORGO outer evaluation was performed.

## Runtime and model contracts

| Component | Format / input / output | Availability and limits |
| --- | --- | --- |
| Hybrid CNN–GCN Cross-Attention | Final PyTorch .pt checkpoints; shared all-mode VMD and frozen representation/normalization pipeline; sigmoid binary probability at 0.5. | Frozen corpus-bound VMD/representation normalization and model adapters have not been validated for arbitrary uploads. |
| Recurrence CNN Baseline | Final PyTorch .pt checkpoints; shared all-mode VMD and frozen representation/normalization pipeline; sigmoid binary probability at 0.5. | Frozen corpus-bound VMD/representation normalization and model adapters have not been validated for arbitrary uploads. |
| HuBERT Feature Classifier | PyTorch pickle payload containing fitted scikit-learn head/scaler; float32 mono 16 kHz, >=400 samples after resampling; centered <=8s; HuBERT FP16 features, FP32 mean + population std; 1×1536 features -> probability. | Focused real inference passed. Explicit fold contexts 1, 2, 3 required; research-only, no universal deployment. |
| Improved HuBERT Classifier | PyTorch pickle payload containing fitted scikit-learn head/scaler; float32 mono 16 kHz, >=400 samples after resampling; centered <=8s; HuBERT FP16 features, FP32 mean + population std; 1×1536 features -> probability. | Focused real inference passed. Explicit fold contexts 1, 2, 3 required; research-only, no universal deployment. |
| WavLM Adaptive Fine-Tuning | Pinned pretrained .bin backbone plus .pt fitted LR/scaler or selected upper-layer delta; float32 mono 16 kHz; exact-length [1,N] values/mask; FP32. Centered <=4s; 768 pooled values -> sigmoid. | Focused real inference passed. Explicit fold contexts 1, 2, 3 required; research-only, no universal deployment. |
| WavLM Frozen Reference | Pinned pretrained .bin backbone plus .pt fitted LR/scaler or selected upper-layer delta; float32 mono 16 kHz; exact-length [1,N] values/mask; FP32. Centered <=8s; layer12 mean/std, 1×1536 -> LR probability. | Focused real inference passed. Explicit fold contexts 1, 2, 3 required; research-only, no universal deployment. |
| Whisper small.en | OpenAI Whisper small.en .pt; mono 16 kHz float32 input, <=60s; CUDA FP16 decode -> actual raw English text; unchanged Module 6 decoding. | Focused real inference passed. Frozen utility; local processing. |
| Microsoft Zira | Windows SAPI5 Microsoft Zira installed engine; 1–2000 text characters -> actual WAV, <=180s; rate175, volume1.0; worker timeout60s. | Focused real inference passed. Frozen utility; local processing. |

All classifier labels are 0=Control, 1=Dysarthric, with fixed threshold 0.5. Probabilities are not medically calibrated confidence. Source uploads: 25MB / 60s; center crop is disclosed in live provenance. Silence is rejected for inference. Channels use arithmetic-mean downmix, resampling uses scipy resample_poly, amplitude is never peak-normalized. WavLM official processor does_normalize=false; HuBERT official processor does_normalize=true. No VMD enters pretrained-model inference.

CUDA RTX4070 Laptop GPU is validated; CPU fallback is deliberately unavailable for the frozen adapter contract. Web-only packages and local pywin32 live in the studio .venv; the existing accuracy / Module6 environments are reused read-only. Runtime versions are pinned and checked before model loading.

## Exact local checkpoint references

### Hybrid CNN–GCN Cross-Attention

- `C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_FINAL_Hybrid_VMD_TORGO_Project\artifacts\module5\677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9\experiments\runs\fold1_C2_stronger_dropout_seed42_677ad640faf5\final\d32bc51dce71ab37afd02471c5cfe874ce03f737e8191acad7f31663c6fe4e24.pt`
- `C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_FINAL_Hybrid_VMD_TORGO_Project\artifacts\module5\677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9\experiments\runs\fold2_C2_stronger_decay_seed42_677ad640faf5\final\cf54f92b2e06c6c544e4b9efb91d3e248f9207f9ce773fc510fd4f23b3649d8d.pt`
- `C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_FINAL_Hybrid_VMD_TORGO_Project\artifacts\module5\677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9\experiments\runs\fold3_C2_baseline_seed42_677ad640faf5\final\a0c0b3520ef6f04a11e29c32dd11a65fda363b3bbd8793c82f75f1e53ac37508.pt`
### Recurrence CNN Baseline

- `C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_FINAL_Hybrid_VMD_TORGO_Project\artifacts\module5\677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9\experiments\runs\fold1_B2_stronger_dropout_seed42_677ad640faf5\final\2d20c88ecc273a3df48818fb020eb49b9bd7bd6ffbfb46b7094f4303ad510973.pt`
- `C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_FINAL_Hybrid_VMD_TORGO_Project\artifacts\module5\677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9\experiments\runs\fold2_B2_stronger_decay_seed42_677ad640faf5\final\3e8a11c49869ab5ba5cc797d79c521eb5934612c38dee407e73b959e638d0910.pt`
- `C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_FINAL_Hybrid_VMD_TORGO_Project\artifacts\module5\677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9\experiments\runs\fold3_B2_baseline_seed42_677ad640faf5\final\9123550a57819f9df7237267e65da05f502fef433abaada9534ae799da9b3f89.pt`
### HuBERT Feature Classifier

- Context 1: `C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_Accuracy_Study\experiments\accuracy_hubert_v2\checkpoints\fold1_E1_final\epoch-000-4c3d387ceae84562820a38cd3c97f42a.pt`
  SHA-256 `7af26ae46b1e464cc9da123e1c2cfba1a59b9fa322367916b363a8adbf988afd`; identity `6afc5b20abb4704e464aee6d9e40537e29d07658b10d4e8a486c4635f08c643f`.
- Context 2: `C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_Accuracy_Study\experiments\accuracy_hubert_v2\checkpoints\fold2_E1_final\epoch-000-5f502af041d04acaa3c921474e9fb3a8.pt`
  SHA-256 `93bd95bf6e4ed4a2e62a1f5a6a16d90fd765c0a2d4567a342fc5d5a56cbb3db0`; identity `21e8935f8d4f9e1926b169c32cd18ef1d3807862ae1b13f4c391c4be6f6fa3f1`.
- Context 3: `C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_Accuracy_Study\experiments\accuracy_hubert_v2\checkpoints\fold3_E1_final\epoch-000-688740166e9047dab8264aa43c0e5adf.pt`
  SHA-256 `1d15715b00d9cf72ab32afce1785904f4b4023e4009660d5e345a4fdb62776ec`; identity `df3e38f4fe030ffe4a600d26af06085a7f14e83c7f11aac0215c968838fd74c0`.

Configurations:
```json
{
  "1": {
    "layer": 12
  },
  "2": {
    "layer": 6
  },
  "3": {
    "layer": 12
  }
}
```
### Improved HuBERT Classifier

- Context 1: `C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_Accuracy_Study\experiments\accuracy_hubert_v3\checkpoints\fold1\final\selected\epoch-000-77c417190bbb4745bc9665eb466c1ea5.pt`
  SHA-256 `61fbb044d3221ead1c34e2015f8952a2d2d5a3995f59e6a5af55d17aaf87606b`; identity `07cf9e1114b33392eadcfca7313dd01b3ae2531768c61ed0ebea3319be2810ce`.
- Context 2: `C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_Accuracy_Study\experiments\accuracy_hubert_v3\checkpoints\fold2\final\selected\epoch-000-c4cf9211d1f347e199584bb145ea86b4.pt`
  SHA-256 `58c4d59b93e8a60e95d771ab34550d1773ce5a3bc338f55f080c177428238823`; identity `f19a007cb71dfc21bba564987748ddec722f7fbbf3a22084d5847ac0afe9f98e`.
- Context 3: `C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_Accuracy_Study\experiments\accuracy_hubert_v3\checkpoints\fold3\final\reference\epoch-000-a4b6ad4a44c84d2eb3f3dac44f0b1945.pt`
  SHA-256 `a20b496d10e47e3b66f11625f4b6a1799df8de968ac1a4b88c55b614fdcdce80`; identity `998407601fd72d2ec2950daa69ffb98776bf6a81edc400514e51bf725b432df0`.

Configurations:
```json
{
  "1": {
    "reference": {
      "C": 0.01,
      "augmentation": false,
      "complexity": 0,
      "id": "ref_L09_C0.01_B0_N0_A0",
      "layer": 9,
      "normalization": "N0",
      "weighting": "B0"
    },
    "selected": {
      "C": 0.01,
      "augmentation": false,
      "complexity": 1,
      "id": "weight_L09_C0.01_B2_N0_A0",
      "layer": 9,
      "normalization": "N0",
      "weighting": "B2"
    }
  },
  "2": {
    "reference": {
      "C": 0.01,
      "augmentation": false,
      "complexity": 0,
      "id": "ref_L09_C0.01_B0_N0_A0",
      "layer": 9,
      "normalization": "N0",
      "weighting": "B0"
    },
    "selected": {
      "C": 0.01,
      "augmentation": false,
      "complexity": 1,
      "id": "robust_L09_C0.01_B0_N1_A0",
      "layer": 9,
      "normalization": "N1",
      "weighting": "B0"
    }
  },
  "3": {
    "reference": {
      "C": 1.0,
      "augmentation": false,
      "complexity": 0,
      "id": "ref_L09_C1_B0_N0_A0",
      "layer": 9,
      "normalization": "N0",
      "weighting": "B0"
    },
    "selected": {
      "C": 1.0,
      "augmentation": false,
      "complexity": 0,
      "id": "ref_L09_C1_B0_N0_A0",
      "layer": 9,
      "normalization": "N0",
      "weighting": "B0"
    }
  }
}
```
### WavLM Adaptive Fine-Tuning

- Context 1: `C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_Accuracy_Study\experiments\accuracy_wavlm_v4\experiments\context1\final\selected\best\state-0000732-b21f3ca25fb340eaaf58e9f8b7b393ba.pt`
  SHA-256 `b1b726d435b5f3ce66a558d28f53269b0c3725862e50b9f14e40f7cd4a232f37`; identity `04cdb28f73ff414693296096bd6e63885040c23ec2be7ac1220f77ec2a41e736`.
- Context 2: `C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_Accuracy_Study\experiments\accuracy_wavlm_v4\experiments\context2\final\selected\best\state-0000725-c3c778eaa3694a1284f2fd5ec7027b40.pt`
  SHA-256 `4dac189f3bfdbe5a4671f6370cbdeebd85f4fb77e07ebf3e7b1a2d3593ea964d`; identity `addcc9592f5d6de822846c2f46f019006ecc1503985629343ea0818891dae9a6`.
- Context 3: `C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_Accuracy_Study\experiments\accuracy_wavlm_v4\experiments\context3\final\selected\best\state-0001424-215a0349fbed4399902b5888e78fd3c2.pt`
  SHA-256 `76212226dac39657807c233a4fd434e0e6a4347fc01de6b9141e73d9697aab27`; identity `13b6ebf545d4142177c62af8124362597a9776d59df263bcb9d30555aece2568`.

The original outer result names the final fit checkpoint in its identity; original predict_neural restores final-fit best_checkpoint. This adapter restores that exact best delta, verified against complete.json. The outer-fit reference is retained separately in model_registry.json.

Configurations:
```json
{
  "1": {
    "complexity": 2,
    "depth": 2,
    "family": "B",
    "id": "B_last2_mean",
    "pooling": "mean"
  },
  "2": {
    "complexity": 3,
    "depth": 2,
    "family": "B",
    "id": "B_last2_attention",
    "pooling": "attention"
  },
  "3": {
    "complexity": 4,
    "depth": 4,
    "family": "B",
    "id": "B_last4_mean",
    "pooling": "mean"
  }
}
```
### WavLM Frozen Reference

- Context 1: `C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_Accuracy_Study\experiments\accuracy_wavlm_v4\experiments\context1\final\frozen_reference\state-0000000-f6cb2c5d01ce4659916e45294fa31f1e.pt`
  SHA-256 `6f41ca12ca6563b99d6faf3281b80896d98e78a5a002017679b317d4a4172355`; identity `37934c23fd6006468e18b20e1510fcfc80628ca6eaa2d0d4d81cefa7c985667f`.
- Context 2: `C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_Accuracy_Study\experiments\accuracy_wavlm_v4\experiments\context2\final\frozen_reference\state-0000000-940247ff07e84724a1d1618b515d252d.pt`
  SHA-256 `90174351c990da0ba3addcfa8c6b8f5e8b93af4a5dd1c91d250242f9446910ed`; identity `1cbc939905059326a6638a0c4389c73249902105503055648afe923e9c8bf0c2`.
- Context 3: `C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_Accuracy_Study\experiments\accuracy_wavlm_v4\experiments\context3\final\frozen_reference\state-0000000-81b7ac95998a464a882500139e31f9ec.pt`
  SHA-256 `582fc88b51e9b2bf32cfe78094cf4d1d3741cd72d19030ea79d4007f5b2d05c4`; identity `264fc3c93ce6c41cc3628812565c86303247c3113a94428995b5573a2c4eaafd`.

Configurations:
```json
{
  "1": {
    "C": 0.01,
    "complexity": 0,
    "family": "A",
    "id": "A_L12_mean_std_C0.01",
    "layer": 12,
    "pooling": "mean_std"
  },
  "2": {
    "C": 0.01,
    "complexity": 0,
    "family": "A",
    "id": "A_L12_mean_std_C0.01",
    "layer": 12,
    "pooling": "mean_std"
  },
  "3": {
    "C": 0.01,
    "complexity": 0,
    "family": "A",
    "id": "A_L12_mean_std_C0.01",
    "layer": 12,
    "pooling": "mean_std"
  }
}
```
### Whisper small.en

- `C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_FINAL_Hybrid_VMD_TORGO_Project\artifacts\module6\models\whisper\small.en.pt`
### Microsoft Zira

- `Windows SAPI5 installed voice`

## Pretrained provenance

- hubert: `facebook/hubert-base-ls960` revision `dba3bb02fda4248b6e082697eee756de8fe8aa8a`; local `C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_Accuracy_Study\experiments\accuracy_hubert\cache\checkpoint\dba3bb02fda4248b6e082697eee756de8fe8aa8a`.
  `config.json` SHA-256 `56be398848bbd9cbc720172b0d45b2f01cac7652f2968f0f31f5faf9f2986acc`.
  `preprocessor_config.json` SHA-256 `4a93853b74278b7c769d07f5a861e5d12ceb5db2bced5620d335f87238cb9e86`.
  `pytorch_model.bin` SHA-256 `062249fffb353eab67547a2fbc129f7c31a2f459faf641b19e8fb007cc5c48ad`.
- wavlm: `microsoft/wavlm-base-plus` revision `4c66d4806a428f2e922ccfa1a962776e232d487b`; local `C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_Accuracy_Study\experiments\accuracy_wavlm_v4\cache\model\4c66d4806a428f2e922ccfa1a962776e232d487b`.
  `README.md` SHA-256 `d417fc16b08e2f66f0af6b371e47a5f9f1f596f6c1c593f1214d360c41d7a27e`.
  `config.json` SHA-256 `fea6df1c2700a3954fc07e70588aecc9055eeb28db2ff57151a2db0d19180ed4`.
  `preprocessor_config.json` SHA-256 `99272fe8ccfab114b68b478681ea47ee3a1ce62bb788cb92dd6e4f69fb1f1da2`.
  `pytorch_model.bin` SHA-256 `3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51`.

HuBERT, WavLM and Whisper are externally pretrained. Team-fitted heads and selected WavLM upper-layer deltas are distinguished from pretrained backbone provenance. No replacement weights were downloaded.

## Read-only integration entry points

- HuBERT: `experiments.accuracy_hubert_v2.src.features.Extractor.infer`, masked pooling and train-scaler transform; V3 `training.predict` uses preserved normalizer.
- WavLM: original `model.load_backbone`, `inputs`, `pool`, `Classifier`, `training.restore_delta`, `predict_lr`; inference only, no Corpus/selection/evaluate/fit calls.
- ASR: `vaaniswitch.m6_reconstruction.asr.transcribe_audio`; verified local weights and exact Module6 decoding.
- TTS: original `select_required_voice` and VOICE_ID; app-local adaptation of public SAPI-loop workaround; all synthesis output stays under runtime.

## Focused engineering measurements

New Zira speech sample only; these are one-clip engineering observations, not standardized latency or classification-accuracy benchmarks. Model loading includes checkpoint verification and preparation. Total job time includes process/import overhead.

| Adapter / context | Inference ms | Loading seconds | Peak allocated GiB | Peak reserved GiB |
| --- | ---: | ---: | ---: | ---: |
| tts | 501.85 | Not measured | Not applicable | Not applicable |
| asr | 1352.94 | 1.85 | 1.370 | 1.533 |
| hubert-v2-context1 | 684.22 | 4.37 | 0.436 | 0.520 |
| hubert-v2-context2 | 393.46 | 4.15 | 0.436 | 0.520 |
| hubert-v2-context3 | 570.92 | 3.73 | 0.436 | 0.520 |
| hubert-v3-context1 | 765.99 | 3.72 | 0.436 | 0.520 |
| hubert-v3-context2 | 374.77 | 3.42 | 0.436 | 0.520 |
| hubert-v3-context3 | 359.39 | 3.64 | 0.436 | 0.520 |
| wavlm-reference-context1 | 411.60 | 3.63 | 0.417 | 0.477 |
| wavlm-reference-context2 | 391.96 | 3.92 | 0.417 | 0.477 |
| wavlm-reference-context3 | 288.02 | 3.43 | 0.417 | 0.477 |
| wavlm-selected-context1 | 556.73 | 3.50 | 0.403 | 0.479 |
| wavlm-selected-context2 | 463.90 | 3.88 | 0.403 | 0.479 |
| wavlm-selected-context3 | 323.48 | 4.71 | 0.403 | 0.479 |

ASR returned: Every voice deserves to be understood. This is a local speech research demonstration.

Synthesis produced non-silent finite audio and browser playback/download passed. No human listening or MOS study was conducted in this implementation session. No numerical CPU-usage benchmark was taken. Browser navigation timing observations are in accessibility_browser.json; they are local cached-load observations, not an external web-performance benchmark.
