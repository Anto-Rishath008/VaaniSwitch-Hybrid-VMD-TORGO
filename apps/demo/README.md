# VaaniSwitch Demonstration Dashboard

A streamlined, academic demonstration application built for presenting the **VaaniSwitch** shared-VMD hybrid dysarthria classification and clear speech reconstruction project to faculty and evaluators.

---

## 🚀 Quick Start (Running the App)

From the project root (`VaaniSwitch-Hybrid-VMD-TORGO`), run:

```powershell
python -m streamlit run apps/demo/app.py --server.port 8502
```

Then open your browser to: **`http://localhost:8502`** (or `http://localhost:8501`).

---

## 🧪 Running Unit Tests

Run the dedicated Module 7 test suite:

```powershell
python -m pytest tests/m7_app/test_demo_pipeline.py -v
```

All 9 tests validate:
- WAV loading & standardization (paths and byte buffers)
- Pure-NumPy K=5 VMD decomposition and reconstruction RMSE
- Multimodal feature tensors (Log-Mel, Recurrence Plot, NVG)
- Token Cross-Attention tensor invariants (softmax rows sum to 1.0)
- Interactive selected-mode filtering (Ablation C1)
- Checkpoint loading bridge & model forward pass inference
- Paper benchmark metrics data integrity

---

## 🎯 Model Integration Hook (For Your Teammate)

The UI is cleanly decoupled from PyTorch model training via `apps/demo/backend_bridge.py`.

### Option A: Via the Web UI (Zero Code Changes)
1. Open the UI in your browser.
2. In the left sidebar, expand **🔌 Model Checkpoint Hook**.
3. Type or paste your checkpoint path (e.g., `artifacts/checkpoints/module4/best.pt`).
4. The UI automatically inspects weights, parameters, and validates the model!

### Option B: In Code
1. Open `apps/demo/backend_bridge.py`.
2. Locate line 43:
   ```python
   CHECKPOINT_PATH: Path | None = None
   ```
3. Update it to point to your saved checkpoint:
   ```python
   CHECKPOINT_PATH = REPO_ROOT / "artifacts/checkpoints/module4/best.pt"
   ```
4. Save the file. Streamlit hot-reloads and immediately runs inference using the live weights!

### Option C: Environment Variable
```powershell
$env:VAANISWITCH_CHECKPOINT = "artifacts/checkpoints/module4/best.pt"
```

---

## 📂 Interactive Features Demonstrated in the UI

1. **🔬 Signal & Shared VMD**:
   - Audio waveform playback.
   - Pure-NumPy Variational Mode Decomposition ($K=5$ modes) with center frequencies.
   - **Interactive Mode Filter (Ablation C1)**: Toggle modes dynamically to observe signal reconstruction changes live.
   - Clinical acoustic guide for each mode (Pitch tremor, Formants $F_1, F_2$, frication).

2. **📊 Multimodal Representations**:
   - 224×224 Log-Mel power-dB spectrogram with colormap selector (`magma`, `viridis`, `plasma`, `inferno`).
   - 224×224 Binary Recurrence Plot ($m=3, \tau=2$) with recurrence density.
   - Natural Visibility Graph (NVG) envelope mapping and complex network density.
   - Clinical cards explaining why each representation isolates dysarthria.

3. **🧠 Attention & Classification**:
   - **Attention Head Inspector**: View individual heads (Head 1–4) or mean attention.
   - Attention allocation metrics (Graph nonlinear token vs. Recurrence patches).
   - **Interactive Clinical Decision Threshold Slider** (0.10 to 0.90, default locked at 0.50).
   - Diagnostic classification card with confidence and probability scale.

4. **🗣️ Clear Speech Reconstruction**:
   - Frozen Whisper ASR transcript display.
   - Human-in-the-Loop text verification input.
   - Clear reconstructed speech audio playback (Microsoft Zira voice).
   - Side-by-side A/B comparison player (Original vs. Reconstructed).
   - **Download button** for synthesized audio WAV.

5. **📈 Paper Benchmark Metrics**:
   - Interactive bar chart comparing all 7 Ablation configurations on Macro-F1 and ROC-AUC.
   - Signed TORGO 7-ablation comparison table.
   - Fold-by-fold confusion matrices for the proposed architecture ($N=8,673$ clean records).
   - Speech recognition error rates on the frozen 130-record panel (Control 6.71% vs. Dysarthric 44.95%).

6. **📖 Architecture & About**:
   - Academic explanation of VMD and nonlinear dynamic tokens for neuromotor speech disorders.
   - Strict speaker-disjoint partition methodology disclosure.
