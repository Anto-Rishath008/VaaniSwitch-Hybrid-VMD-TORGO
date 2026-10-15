"""VaaniSwitch Demonstration Application.

Shared-VMD Hybrid Dysarthria Classification & Clear Speech Reconstruction.
Built for academic presentation and evaluation.
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Local modules
import importlib
import backend_bridge
importlib.reload(backend_bridge)
from backend_bridge import (
    CHECKPOINT_PATH,
    PipelineResult,
    REPO_ROOT,
    inspect_and_load_checkpoint,
    run_pipeline,
)
from benchmark_data import (
    ASR_RECONSTRUCTION_BENCHMARK,
    BENCHMARK_ABLATIONS,
    DATASET_SUMMARY,
    PROPOSED_FOLD_RESULTS,
)

# -----------------------------------------------------------------------------
# Page Configuration & Academic Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VaaniSwitch | Dysarthria Classification & Speech Reconstruction",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom sleek academic styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.05rem;
        font-weight: 400;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .badge-status {
        background-color: #ECFDF5;
        border: 1px solid #10B981;
        color: #065F46;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .badge-hook {
        background-color: #EFF6FF;
        border: 1px solid #3B82F6;
        color: #1E40AF;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .badge-ckpt {
        background-color: #FEF3C7;
        border: 1px solid #F59E0B;
        color: #92400E;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
    }
    .metric-card {
        background: #F9FAFB;
        border-radius: 8px;
        padding: 14px;
        border: 1px solid #E5E7EB;
        text-align: center;
    }
    .info-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 1rem;
        font-size: 0.92rem;
        color: #334155;
    }
    .result-box-dys {
        background-color: #FEF2F2;
        border-left: 6px solid #EF4444;
        padding: 16px;
        border-radius: 6px;
        margin: 10px 0;
    }
    .result-box-ctrl {
        background-color: #F0FDF4;
        border-left: 6px solid #22C55E;
        padding: 16px;
        border-radius: 6px;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar: Audio Input & Friend Checkpoint Configuration
# -----------------------------------------------------------------------------
st.sidebar.header("🎛️ Audio Input & Controls")

sample_mode = st.sidebar.radio(
    "Choose Input Source:",
    ["Preloaded Demo Samples", "Upload Custom WAV Audio"],
    index=0,
)

sample_audio_dir = REPO_ROOT / "apps/demo/sample_audio"
selected_audio_path: Path | None = None
uploaded_audio_bytes: bytes | None = None
sample_label: str = "Input Audio"

if sample_mode == "Preloaded Demo Samples":
    sample_options = {
        "Dysarthric Speech (TORGO Speaker M03)": sample_audio_dir / "dysarthric_sample_M03.wav",
        "Control Healthy Speech (TORGO Speaker MC02)": sample_audio_dir / "control_sample_MC02.wav",
    }
    chosen_label = st.sidebar.selectbox("Select Sample Utterance:", list(sample_options.keys()))
    selected_audio_path = sample_options[chosen_label]
    sample_label = chosen_label
    
    st.sidebar.markdown("**Input Sample Playback:**")
    if selected_audio_path.exists():
        st.sidebar.audio(str(selected_audio_path), format="audio/wav")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Speech Audio (.wav):",
        type=["wav"],
        help="Upload standard 16-bit integer PCM WAV recording.",
    )
    if uploaded_file is not None:
        uploaded_audio_bytes = uploaded_file.read()
        sample_label = uploaded_file.name
        st.sidebar.markdown("**Uploaded Audio Playback:**")
        st.sidebar.audio(uploaded_audio_bytes, format="audio/wav")
    else:
        st.sidebar.info("Please upload a WAV file to process.")

st.sidebar.markdown("---")

# 🔌 Friend Model Checkpoint Panel
st.sidebar.subheader("🔌 Model Checkpoint Hook")
with st.sidebar.expander("Load Trained Checkpoint (.pt)", expanded=False):
    st.markdown(
        """
        **For Developer / Friend**:
        Provide a `.pt` checkpoint path (from `artifacts/checkpoints/...` or local disk).
        """
    )
    user_ckpt_input = st.text_input(
        "Checkpoint File Path:",
        value=str(CHECKPOINT_PATH) if CHECKPOINT_PATH else "",
        placeholder="artifacts/checkpoints/module4/best.pt",
        help="Path to trained PyTorch weights file (.pt)",
    )
    custom_ckpt_path: Path | None = None
    if user_ckpt_input.strip():
        resolved_p = Path(user_ckpt_input.strip())
        if not resolved_p.is_absolute():
            resolved_p = REPO_ROOT / resolved_p
        custom_ckpt_path = resolved_p

    ckpt_test = inspect_and_load_checkpoint(custom_ckpt_path)
    if ckpt_test["valid"]:
        st.success(f"✅ {ckpt_test['status']}")
        if "epoch" in ckpt_test.get("metadata", {}):
            st.caption(f"Epoch: {ckpt_test['metadata'].get('epoch')} | Metric: {ckpt_test['metadata'].get('best_metric')}")
    elif custom_ckpt_path:
        st.warning(f"⚠️ {ckpt_test['status']}")
    else:
        st.caption("ℹ️ No checkpoint linked. Operating in deterministic acoustic heuristic mode.")

# Visualization Settings
st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Display Options")
selected_cmap = st.sidebar.selectbox("Spectrogram Colormap:", ["magma", "viridis", "inferno", "plasma"], index=0)

# -----------------------------------------------------------------------------
# Header & Navigation Badges
# -----------------------------------------------------------------------------
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown('<div class="main-header">🎙️ VaaniSwitch (वाणीSwitch)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Shared-VMD Hybrid Dysarthria Classification & Speech Reconstruction | TORGO Corpus</div>',
        unsafe_allow_html=True,
    )

active_ckpt_status = ckpt_test["status"] if ckpt_test["valid"] else "Heuristic Fallback"
with col_status:
    st.markdown(
        f'<div class="badge-status">🟢 Signal Engine: Active</div><br>'
        f'<div class="badge-hook">🔌 Bridge: {active_ckpt_status}</div>',
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Pipeline Execution with Caching
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Running VMD Decomposition & Multimodal Feature Extraction...")
def _cached_run_pipeline(
    source: str | Path | bytes,
    name: str,
    ckpt_path_str: str | None,
) -> PipelineResult:
    p_ckpt = Path(ckpt_path_str) if ckpt_path_str else None
    return run_pipeline(source, sample_name=name, checkpoint_path=p_ckpt)


pipeline_result: PipelineResult | None = None
source_to_run = uploaded_audio_bytes if uploaded_audio_bytes is not None else selected_audio_path

if source_to_run is not None:
    try:
        pipeline_result = _cached_run_pipeline(
            source_to_run,
            name=sample_label,
            ckpt_path_str=str(custom_ckpt_path) if (custom_ckpt_path and custom_ckpt_path.exists()) else None,
        )
    except Exception as e:
        st.error(f"Error executing signal pipeline: {e}")

# -----------------------------------------------------------------------------
# Main Application Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔬 Signal & Shared VMD",
    "📊 Multimodal Representations",
    "🧠 Attention & Classification",
    "🗣️ Clear Speech Reconstruction",
    "📈 Paper Benchmark Metrics",
    "📖 Architecture & About",
])

# -----------------------------------------------------------------------------
# Tab 1: Signal & Shared VMD
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("1. Audio Standardization & Shared Variational Mode Decomposition (VMD)")
    st.markdown(
        """
        Input speech is normalized via **conservative RMS normalization** (target 0.1, max gain 10×) 
        and non-recursively decomposed into **$K=5$ Intrinsic Mode Functions (IMFs)** using a project-local pure-NumPy VMD solver 
        ($\\alpha=2000, \\tau=0$).
        """
    )

    if pipeline_result is not None:
        # Metrics row
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Sampling Rate", f"{pipeline_result.sample_rate} Hz")
        m_col2.metric("Duration", f"{pipeline_result.duration_seconds:.2f} s")
        m_col3.metric("Canonical RMSE", f"{pipeline_result.rmse:.5f}")
        m_col4.metric("Energy Conservation", f"{pipeline_result.energy_ratio:.4f}")

        st.markdown("---")

        # 🎛️ Interactive VMD Mode Selection (Demonstrating Ablation C1)
        st.markdown("#### 🎛️ Interactive Mode Filter (Ablation C1: Selective Mode Reconstruction)")
        st.caption(
            "Toggle individual VMD modes to explore how filtering specific frequency bands impacts signal fidelity. "
            "Dysarthric tremor concentrates in IMF 1, while formant articulation resides in IMFs 2–3."
        )

        selected_mode_nums = st.multiselect(
            "Active Modes for Reconstruction:",
            options=[1, 2, 3, 4, 5],
            default=[1, 2, 3, 4, 5],
            format_func=lambda m: f"IMF {m} (Center: {pipeline_result.center_frequencies_hz[m-1]:.1f} Hz)",
        )

        if not selected_mode_nums:
            st.warning("Please select at least one mode to view reconstruction.")
            active_filtered_signal = np.zeros_like(pipeline_result.raw_waveform)
            active_rmse = float(np.sqrt(np.mean(pipeline_result.raw_waveform**2)))
            active_energy = 0.0
        else:
            active_indices = np.array(selected_mode_nums) - 1
            active_filtered_signal = np.sum(pipeline_result.vmd_modes[active_indices], axis=0)
            diff_active = pipeline_result.raw_waveform - active_filtered_signal
            active_rmse = float(np.sqrt(np.mean(diff_active**2)))
            active_energy = float(np.sum(active_filtered_signal**2) / (np.sum(pipeline_result.raw_waveform**2) + 1e-12))

        f_col1, f_col2, f_col3 = st.columns(3)
        f_col1.metric("Selected Modes Count", f"{len(selected_mode_nums)} of 5")
        f_col2.metric("Active Filtered RMSE", f"{active_rmse:.5f}")
        f_col3.metric("Active Energy Ratio", f"{active_energy:.4f}")

        st.markdown("---")
        col_wave, col_modes = st.columns([1, 1])

        with col_wave:
            st.markdown("#### Waveform Comparison")
            fig, ax = plt.subplots(2, 1, figsize=(7, 4.2), sharex=True)
            time_axis = np.linspace(0, pipeline_result.duration_seconds, len(pipeline_result.raw_waveform))
            
            ax[0].plot(time_axis, pipeline_result.raw_waveform, color="#2563EB", lw=0.9)
            ax[0].set_title("Input Speech Waveform (Normalized)", fontsize=10, fontweight="bold")
            ax[0].grid(True, alpha=0.3)
            ax[0].set_ylabel("Amplitude")

            ax[1].plot(time_axis, active_filtered_signal, color="#059669", lw=0.9)
            mode_str = ", ".join(f"IMF {m}" for m in selected_mode_nums) if selected_mode_nums else "None"
            ax[1].set_title(f"Reconstructed Signal [{mode_str}]", fontsize=10, fontweight="bold")
            ax[1].grid(True, alpha=0.3)
            ax[1].set_xlabel("Time (seconds)")
            ax[1].set_ylabel("Amplitude")
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_modes:
            st.markdown("#### $K=5$ Decomposed VMD Modes")
            fig_modes, axes = plt.subplots(5, 1, figsize=(7, 5.0), sharex=True)
            colors = ["#4F46E5", "#0891B2", "#0D9488", "#CA8A04", "#DC2626"]
            mode_descriptions = [
                "Fundamental Pitch & Vocal Tremor",
                "Formant F1 (Vocal Tract Opening)",
                "Formant F2 (Tongue Body Articulation)",
                "Formant F3 & Spectral Transients",
                "High-Frequency Fricatives & Turbulence",
            ]

            for i in range(5):
                cf = pipeline_result.center_frequencies_hz[i]
                axes[i].plot(time_axis, pipeline_result.vmd_modes[i], color=colors[i], lw=0.8)
                axes[i].set_ylabel(f"IMF {i+1}", fontsize=8)
                axes[i].set_title(f"Mode {i+1} ({cf:.1f} Hz) — {mode_descriptions[i]}", fontsize=8, loc="right", pad=2)
                axes[i].grid(True, alpha=0.2)
            axes[-1].set_xlabel("Time (seconds)")
            
            plt.tight_layout()
            st.pyplot(fig_modes)
            plt.close(fig_modes)

        # Mode Clinical Guide
        with st.expander("ℹ️ Clinical & Acoustic Significance of Each Mode", expanded=False):
            st.markdown(
                """
                - **IMF 1 (0–500 Hz)**: Isolates glottal vibration and pitch tremor. In dysarthric speech, irregular vocal fold vibration manifests as erratic amplitude modulation in this mode.
                - **IMFs 2 & 3 (500–2500 Hz)**: Captures first and second vowel formants ($F_1, F_2$). In dysarthria, restricted articulatory tongue movement centralizes these formants, reducing vowel distinctiveness.
                - **IMFs 4 & 5 (>2500 Hz)**: Captures consonant bursts, fricative turbulence, and aspiration noise, preserving speech onset markers.
                """
            )
    else:
        st.warning("Please select or upload an audio file from the sidebar.")

# -----------------------------------------------------------------------------
# Tab 2: Multimodal Representations
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("2. Deterministic Multimodal Feature Representations")
    st.markdown(
        """
        From the all-mode VMD reconstruction, the framework computes **three complementary representations**:
        1. **Log-Mel Spectrogram** (Acoustic spectral energy across 80 Mel filters, fixed 8s duration).
        2. **Binary Recurrence Plot** (Phase-space trajectory recurrence, embedding dimension $m=3$, delay $\\tau=2$).
        3. **Natural Visibility Graph (NVG)** (Time-series envelope mapped to complex network graph preserving nonlinear dynamics).
        """
    )

    if pipeline_result is not None:
        r_col1, r_col2, r_col3 = st.columns(3)

        with r_col1:
            st.markdown("##### 1. Log-Mel Spectrogram")
            fig_mel, ax_mel = plt.subplots(figsize=(4.2, 4.2))
            im_mel = ax_mel.imshow(pipeline_result.log_mel, origin="lower", cmap=selected_cmap, aspect="auto")
            ax_mel.set_title("224×224 Log-Mel (Power dB)", fontsize=10, fontweight="bold")
            ax_mel.set_xlabel("Time Frames")
            ax_mel.set_ylabel("Mel Frequency Bins")
            fig_mel.colorbar(im_mel, ax=ax_mel, fraction=0.046, pad=0.04)
            st.pyplot(fig_mel)
            plt.close(fig_mel)
            st.markdown(
                '<div class="info-card"><strong>Acoustic Role:</strong> Captures formant trajectories and spectral envelope energy across 80 Mel filterbanks.</div>',
                unsafe_allow_html=True,
            )

        with r_col2:
            st.markdown("##### 2. Binary Recurrence Plot")
            fig_rec, ax_rec = plt.subplots(figsize=(4.2, 4.2))
            ax_rec.imshow(pipeline_result.recurrence_plot, origin="lower", cmap="binary", aspect="equal")
            ax_rec.set_title(f"224×224 RP (Density: {pipeline_result.recurrence_density:.1%})", fontsize=10, fontweight="bold")
            ax_rec.set_xlabel("State Index $i$")
            ax_rec.set_ylabel("State Index $j$")
            st.pyplot(fig_rec)
            plt.close(fig_rec)
            st.markdown(
                f'<div class="info-card"><strong>Phase Space Role:</strong> Identifies non-periodicities and motor speech timing irregularities. Recurrence Density: <code>{pipeline_result.recurrence_density:.2%}</code>.</div>',
                unsafe_allow_html=True,
            )

        with r_col3:
            st.markdown("##### 3. Natural Visibility Graph (NVG)")
            fig_graph, ax_graph = plt.subplots(figsize=(4.2, 4.2))
            nodes = pipeline_result.graph_nodes
            ax_graph.plot(nodes[:, 0], nodes[:, 1], color="#7C3AED", lw=1.2, marker=".", markersize=3)
            ax_graph.set_title(f"NVG Envelope ({len(nodes)} Nodes)", fontsize=10, fontweight="bold")
            ax_graph.set_xlabel("Normalized Time")
            ax_graph.set_ylabel("Standardized Amplitude")
            ax_graph.grid(True, alpha=0.3)
            st.pyplot(fig_graph)
            plt.close(fig_graph)
            st.markdown(
                f'<div class="info-card"><strong>Dynamical Role:</strong> Quantifies neuromotor speech instability through graph topology. Graph Edge Density: <code>{pipeline_result.graph_density:.2%}</code>.</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("Awaiting audio input.")

# -----------------------------------------------------------------------------
# Tab 3: Token Attention & Classification
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("3. Token Cross-Attention Fusion & Hybrid Classification")
    st.markdown(
        """
        The **Parallel Encoders** yield $16$ Mel tokens (via frozen ResNet-18), $1$ Graph token (via pure-PyTorch GCN), 
        and $16$ Recurrence tokens (via 2D-CNN).
        The $16$ Mel spectral queries dynamically attend to the $17$ nonlinear tokens via **4-head cross-attention**.
        """
    )

    if pipeline_result is not None:
        c_left, c_right = st.columns([1.2, 1])

        with c_left:
            st.markdown("#### Token Cross-Attention Heatmaps")
            head_view = st.selectbox(
                "Select Attention Head to Inspect:",
                ["All 4 Heads Average", "Head 1", "Head 2", "Head 3", "Head 4"],
                index=0,
            )

            fig_attn, ax_a = plt.subplots(figsize=(6.5, 4.2))
            if head_view == "All 4 Heads Average":
                display_attn = pipeline_result.attention_weights.mean(axis=0)  # shape: (16, 17)
                head_title = "4-Head Mean Cross-Attention Matrix"
            else:
                head_idx = int(head_view.split()[-1]) - 1
                display_attn = pipeline_result.attention_weights[head_idx]
                head_title = f"Cross-Attention Matrix — {head_view}"

            im_a = ax_a.imshow(display_attn, cmap="viridis", aspect="auto")
            ax_a.set_title(head_title, fontsize=10, fontweight="bold")
            ax_a.set_xlabel("Nonlinear Keys [0: NVG Graph | 1-16: Recurrence Patches]", fontsize=8)
            ax_a.set_ylabel("Mel Query Tokens [0-15]", fontsize=8)
            fig_attn.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
            plt.tight_layout()
            st.pyplot(fig_attn)
            plt.close(fig_attn)

            # Token Importance Distribution
            graph_attn_share = float(display_attn[:, 0].mean())
            rec_attn_share = float(display_attn[:, 1:].mean())
            st.caption(
                f"📊 **Attention Allocation**: Graph Nonlinear Token: `{graph_attn_share:.2%}` | "
                f"Recurrence Temporal Patches: `{rec_attn_share:.2%}` (Avg per token)"
            )

        with c_right:
            st.markdown("#### Diagnostic Classification Result")

            # Interactive Decision Threshold
            threshold = st.slider(
                "Clinical Decision Threshold:",
                min_value=0.10,
                max_value=0.90,
                value=0.50,
                step=0.05,
                help="Default locked protocol threshold is 0.50.",
            )

            is_dys = pipeline_result.probability_dysarthric >= threshold
            predicted_label = "DYSARTHRIC" if is_dys else "CONTROL (Healthy)"

            if is_dys:
                st.markdown(
                    f"""
                    <div class="result-box-dys">
                        <h3 style="color:#B91C1C; margin:0;">🚨 Classification: {predicted_label}</h3>
                        <p style="margin-top:6px; color:#4B5563;">
                            <strong>Estimated Probability:</strong> {pipeline_result.probability_dysarthric:.1%}<br>
                            <strong>Operating Threshold:</strong> {threshold:.2f}<br>
                            <strong>Confidence:</strong> {pipeline_result.confidence:.1%}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-box-ctrl">
                        <h3 style="color:#15803D; margin:0;">✅ Classification: {predicted_label}</h3>
                        <p style="margin-top:6px; color:#4B5563;">
                            <strong>Estimated Probability:</strong> {pipeline_result.probability_dysarthric:.1%}<br>
                            <strong>Operating Threshold:</strong> {threshold:.2f}<br>
                            <strong>Confidence:</strong> {pipeline_result.confidence:.1%}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.progress(pipeline_result.probability_dysarthric)
            st.caption(f"Scale: 0.0 (Control) ───[ {threshold:.2f} Threshold ]─── 1.0 (Dysarthric)")

            st.markdown("---")
            st.markdown("##### 🔌 Backend Model Status")
            if pipeline_result.model_loaded:
                st.success(f"🟢 **Trained Model Active**: {pipeline_result.model_hook_info}")
                if pipeline_result.model_metadata:
                    m_data = pipeline_result.model_metadata
                    st.json({k: v for k, v in m_data.items() if k != "path"})
            else:
                st.markdown(
                    f"""
                    <div style="background-color:#EFF6FF; border-left:4px solid #3B82F6;
                                padding:12px 16px; border-radius:6px; color:#1E40AF; font-size:0.92rem;">
                        <strong>Signal &amp; Attention Mode</strong>: Pure-NumPy VMD &amp; Cross-Attention are active.<br>
                        <em>{pipeline_result.model_hook_info}</em>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("Awaiting audio input.")

# -----------------------------------------------------------------------------
# Tab 4: Clear Speech Reconstruction
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("4. Speech Recognition & Clear Speech Reconstruction (Module 6)")
    st.markdown(
        """
        Designed for assistive speech rehabilitation: Distorted speech is transcribed by **frozen Whisper (`small.en`)**, 
        optionally reviewed by the speaker, and re-synthesized into intelligible speech via **standard English TTS** (Microsoft Zira).
        """
    )

    if pipeline_result is not None:
        rec_col1, rec_col2 = st.columns(2)

        with rec_col1:
            st.markdown("#### Stage 1: Automatic Speech Recognition (ASR)")
            st.text_area(
                "Whisper Raw Transcript (Decoded without Lexical Repair):",
                value=pipeline_result.asr_transcript,
                height=100,
                key="raw_transcript_box",
            )
            st.caption("Reference WER: 44.95% on dysarthric speech vs. 6.71% on control speech.")

            st.markdown("#### Stage 2: Human-in-the-Loop Review (Optional)")
            reviewed_text = st.text_input(
                "Verify or edit transcript before synthesis:",
                value=pipeline_result.asr_transcript,
            )

        with rec_col2:
            st.markdown("#### Stage 3: Clear Synthesized Utterance")
            if pipeline_result.reconstructed_audio_path and pipeline_result.reconstructed_audio_path.exists():
                st.audio(str(pipeline_result.reconstructed_audio_path), format="audio/wav")
                st.success("Clear speech output synthesized with Microsoft Zira voice (rate=175 wpm).")

                # Audio download button
                with open(pipeline_result.reconstructed_audio_path, "rb") as f_wav:
                    st.download_button(
                        label="⬇️ Download Synthesized Clear Audio (.wav)",
                        data=f_wav.read(),
                        file_name="vaaniswitch_reconstructed_speech.wav",
                        mime="audio/wav",
                    )
            else:
                st.info("Synthesized audio available for demo utterances.")

            st.markdown("#### 🎧 A/B Demonstration Comparison")
            st.write("**Original Input Utterance:**")
            if selected_audio_path and selected_audio_path.exists():
                st.audio(str(selected_audio_path), format="audio/wav")
            elif uploaded_audio_bytes:
                st.audio(uploaded_audio_bytes, format="audio/wav")

            st.write("**Reconstructed Clear Utterance:**")
            if pipeline_result.reconstructed_audio_path and pipeline_result.reconstructed_audio_path.exists():
                st.audio(str(pipeline_result.reconstructed_audio_path), format="audio/wav")
    else:
        st.info("Awaiting audio input.")

# -----------------------------------------------------------------------------
# Tab 5: Benchmark Metrics (Paper Results)
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("5. Audited TORGO Cross-Validation & Ablation Results")
    st.markdown(
        """
        These results represent **strict, signed research evidence** evaluated over 
        **8,673 CORE clean records** across **3 speaker-disjoint folds** (Module 5 Final Sign-Off).
        """
    )

    # Ablations Table
    st.markdown("#### 7-Ablation Benchmark Comparison (Arithmetic Mean ± Sample SD)")
    st.dataframe(
        BENCHMARK_ABLATIONS,
        column_config={
            "ablation_id": "Ablation",
            "description": "Configuration Architecture",
            "accuracy": "Accuracy",
            "balanced_acc": "Balanced Acc",
            "macro_f1": "Macro-F1",
            "precision": "Precision",
            "sensitivity": "Sensitivity",
            "specificity": "Specificity",
            "roc_auc": "ROC-AUC",
        },
        use_container_width=True,
        hide_index=True,
    )

    # Visual Ablation Comparison Chart
    st.markdown("#### Architecture Comparison: Macro-F1 & ROC-AUC Across 7 Ablations")
    fig_bench, ax_b = plt.subplots(figsize=(10, 3.8))
    ablations = [row["ablation_id"] for row in BENCHMARK_ABLATIONS]
    f1_vals = [float(row["macro_f1"].split("%")[0]) for row in BENCHMARK_ABLATIONS]
    auc_vals = [float(row["roc_auc"].split("±")[0].strip()) * 100 for row in BENCHMARK_ABLATIONS]

    x_idx = np.arange(len(ablations))
    width = 0.35

    ax_b.bar(x_idx - width/2, f1_vals, width, label="Macro-F1 (%)", color="#3B82F6")
    ax_b.bar(x_idx + width/2, auc_vals, width, label="ROC-AUC (×100)", color="#10B981")
    ax_b.set_ylabel("Score (%)")
    ax_b.set_xticks(x_idx)
    ax_b.set_xticklabels(ablations)
    ax_b.grid(True, alpha=0.3, axis="y")
    ax_b.legend(loc="lower right")
    ax_b.set_ylim(20, 100)
    plt.tight_layout()
    st.pyplot(fig_bench)
    plt.close(fig_bench)

    st.markdown("#### Per-Fold Confusion Matrices for Proposed A1 Architecture")
    f_cols = st.columns(3)
    for idx, fold_res in enumerate(PROPOSED_FOLD_RESULTS):
        with f_cols[idx]:
            st.markdown(f"**Fold {fold_res['fold']}** (Macro-F1: `{fold_res['macro_f1']:.4f}`, AUC: `{fold_res['roc_auc']:.4f}`)")
            st.caption(f"Test Speakers: {fold_res['test_speakers']}")
            matrix_data = [
                {"Label": "Pred Control", "Actual Control (TN)": fold_res["tn"], "Actual Dysarthric (FN)": fold_res["fn"]},
                {"Label": "Pred Dysarthric", "Actual Control (FP)": fold_res["fp"], "Actual Dysarthric (TP)": fold_res["tp"]},
            ]
            st.table(matrix_data)

    st.markdown("#### Module 6 Speech Recognition Benchmark (Frozen 130-Record Panel)")
    asr_subgroups = ASR_RECONSTRUCTION_BENCHMARK["subgroups"]
    st.dataframe(asr_subgroups, use_container_width=True, hide_index=True)
    st.info(f"**Listening Sign-Off**: {ASR_RECONSTRUCTION_BENCHMARK['human_listening_confirmation']}")

# -----------------------------------------------------------------------------
# Tab 6: Architecture & About
# -----------------------------------------------------------------------------
with tab6:
    st.subheader("6. System Architecture & Scientific Methodology")
    
    st.markdown(
        """
        ### Why Shared VMD for Dysarthric Speech?
        Traditional speech processing relies solely on Fourier STFT Log-Mel representations. However, 
        **dysarthria causes neuromotor speech instability**, articulatory vowel centralization, and irregular acoustic tremor.
        
        1. **Variational Mode Decomposition (VMD)**:
           Non-recursively decomposes speech into amplitude- and frequency-modulated sub-signals (modes), isolating resonant vocal tract formants from tremor frequencies without empirical mode mixing.
        2. **Multi-representation Complementarity**:
           - **Log-Mel**: Encodes spectral envelope and acoustic energy across formants.
           - **Binary Recurrence Plot**: Captures determinism and non-linear trajectory recurrence in reconstructed phase space ($m=3, \\tau=2$).
           - **Natural Visibility Graph (NVG)**: Preserves nonlinear dynamical structural chaos across signal envelopes.
        3. **Token Cross-Attention**:
           Rather than naive vector concatenation, 16 Mel queries dynamically attend to 17 nonlinear tokens via multi-head cross-attention with residual LayerNorm.
        
        ### Dataset Policy & Scientific Disclosures
        - **Dataset**: TORGO English database (F01–F04, FC01–FC03, M01–M05, MC01–MC04).
        - **Speaker Disjoint**: Zero speaker overlap across folds; strictly audited for leakage.
        - **Disclaimer**: This research prototype is intended for academic evaluation and clinical speech technology research, not as an automated medical diagnostic tool.
        """
    )
    
    st.markdown("---")
    st.markdown(
        """
        **VaaniSwitch Engineering Team** | *Speech Processing Final Project*  
        All modules verified against git branch `release/vaaniswitch-final-m0-m6`.
        """
    )
