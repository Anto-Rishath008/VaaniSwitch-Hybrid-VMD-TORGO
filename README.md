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

## Setup and Installation

This project is built using Python 3.11+ and uses `uv` for package resolution.

### Using `uv` (Recommended)
1. Install `uv` if not already present:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Sync the dependencies and setup the virtual environment:
   ```bash
   uv sync
   ```
3. Run the test suite to verify everything works:
   ```bash
   uv run pytest
   ```

### Using Standard `pip` and Virtualenv
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

---

## Execution Guide

The workflow is managed by standard CLI commands. You can run preprocessing or start the dashboard using the registered scripts:

### 1. Preprocessing Pipeline (End-to-End)
To build manifests, cache VMD, and extract representations sequentially:
```bash
# Run preprocessing end-to-end
uv run vaaniswitch-preprocess run

# Validate all generated outputs
uv run vaaniswitch-preprocess validate
```

### 2. Gradio Web Application UI
To launch the interactive dashboard for speech classification and reconstruction:
```bash
uv run vaaniswitch-app
```

### 3. Placeholders
Future modules can be run via:
```bash
# Train the models (coming soon)
uv run vaaniswitch-train

# Run evaluations (coming soon)
uv run vaaniswitch-test
```

---

## Development Documentation

The detailed research design plans and guidelines are located in the [docs/development/](docs/development/) directory:
- [PROJECT_SCOPE_LOCK.md](docs/development/PROJECT_SCOPE_LOCK.md) — Requirements and system scope
- [FINAL_WORKFLOW.md](docs/development/FINAL_WORKFLOW.md) — Complete execution workflow
- [MODULE_TRACKER.md](docs/development/MODULE_TRACKER.md) — Progress tracker for project modules
- [PROJECT_STATUS.md](docs/development/PROJECT_STATUS.md) — Active status, caching, and validation report
- [ARCHITECTURE_RATIONALE.md](docs/development/ARCHITECTURE_RATIONALE.md) — Technical choices and reasoning
- [ACCURACY_STRATEGY.md](docs/development/ACCURACY_STRATEGY.md) & [ACCURACY_EXPERIMENT_MATRIX.md](docs/development/ACCURACY_EXPERIMENT_MATRIX.md) — Evaluation rules and baselines
- [TESTING_STRATEGY.md](docs/development/TESTING_STRATEGY.md) — Validation criteria and tools
- [TORGO_CURRENT_AUDIT.md](docs/development/TORGO_CURRENT_AUDIT.md) — Input data quality metrics
- [DIRECTORY_MAP.md](docs/development/DIRECTORY_MAP.md) — Folder mapping structure

