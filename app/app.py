"""
Gradio web app for the Deepfake Audio Detection model.
Multi-tab structure: Welcome / Detector / Performance / Technical.

Deployed on Hugging Face Spaces.
"""

import os
import json
import time
from pathlib import Path

import gradio as gr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from huggingface_hub import hf_hub_download

# Add repo root to path
import sys
APP_DIR = Path(__file__).parent
sys.path.insert(0, str(APP_DIR))

from src.inference.predict import DeepfakeDetector


# ============================================================
# Configuration
# ============================================================

EXAMPLES_DIR = APP_DIR / "examples"
MODEL_REPO = "Sara1708/deepfake-audio-wav2vec2"
MODEL_FILENAME = "stage2_best.pt"

# Color palette (consistent across all charts)
COLOR_BONAFIDE = "#16a34a"    # green
COLOR_SPOOF = "#dc2626"       # red
COLOR_NEUTRAL = "#6b7280"     # gray
COLOR_PRIMARY = "#7c3aed"     # purple (matches gradio theme)
COLOR_BG_LIGHT = "#f3f4f6"


# ============================================================
# Download and load model once at startup
# ============================================================

print(f"Downloading checkpoint from HF Hub: {MODEL_REPO}")
checkpoint_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
print(f"Checkpoint at: {checkpoint_path}")

print("Loading detector...")
detector = DeepfakeDetector(checkpoint_path=checkpoint_path, device="cpu")
print("Model loaded.")


# ============================================================
# Load example metadata
# ============================================================

with open(EXAMPLES_DIR / "metadata.json") as f:
    METADATA = json.load(f)

EXAMPLE_FILES = [
    [str(EXAMPLES_DIR / ex["filename"]), ex["display_name"]]
    for ex in METADATA["examples"]
]


# ============================================================
# Plotting utilities
# ============================================================

def style_axis(ax):
    """Apply consistent styling to a matplotlib axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle="-", linewidth=0.5)
    ax.tick_params(axis="both", which="major", labelsize=9)


def make_per_window_plot(window_scores, threshold=0.5):
    """Per-window spoof probability bar chart."""
    fig, ax = plt.subplots(figsize=(8, 3.2))
    n = len(window_scores)
    indices = list(range(1, n + 1))
    colors = [COLOR_SPOOF if s > threshold else COLOR_BONAFIDE for s in window_scores]
    
    bars = ax.bar(indices, window_scores, color=colors, edgecolor="white", linewidth=1.2)
    ax.axhline(y=threshold, color=COLOR_NEUTRAL, linestyle="--", linewidth=1, 
               label=f"decision threshold ({threshold})")
    
    for bar, score in zip(bars, window_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
                f"{score:.2f}", ha="center", va="bottom", fontsize=9, color="#374151", weight="bold")
    
    ax.set_xlabel("Window (4-second segment)", fontsize=10)
    ax.set_ylabel("P(spoof)", fontsize=10)
    ax.set_title("Per-window spoof probability", fontsize=11, weight="bold", pad=10)
    ax.set_ylim(0, 1.15)
    ax.set_xticks(indices)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95, edgecolor="none")
    style_axis(ax)
    plt.tight_layout()
    return fig


def make_per_codec_plot():
    """Bar chart of per-codec EER from 2021 LA results."""
    codecs = ["none", "opus", "g722", "ulaw", "alaw", "pstn", "gsm"]
    eers = [5.24, 5.30, 5.42, 7.81, 8.37, 11.14, 11.53]
    
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [COLOR_BONAFIDE if e < 7 else (COLOR_NEUTRAL if e < 10 else COLOR_SPOOF) for e in eers]
    bars = ax.bar(codecs, eers, color=colors, edgecolor="white", linewidth=1.2)
    
    for bar, eer in zip(bars, eers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{eer:.2f}%", ha="center", va="bottom", fontsize=9, weight="bold", color="#374151")
    
    ax.set_xlabel("Audio codec", fontsize=10)
    ax.set_ylabel("Equal Error Rate (%)", fontsize=10)
    ax.set_title("EER by codec on ASVspoof 2021 LA eval (148K utterances)", 
                 fontsize=11, weight="bold", pad=10)
    ax.set_ylim(0, max(eers) * 1.2)
    style_axis(ax)
    plt.tight_layout()
    return fig


def make_per_attack_plot():
    """Bar chart of per-attack EER from 2019 LA eval."""
    attacks = ["A13", "A09", "A12", "A11", "A16", "A18", "A08", "A17", "A19", "A07", "A14", "A15", "A10"]
    eers = [0.24, 0.60, 0.99, 1.05, 2.31, 2.72, 0.63, 3.82, 3.79, 5.81, 6.05, 7.53, 15.54]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = []
    for e in eers:
        if e < 2:
            colors.append(COLOR_BONAFIDE)
        elif e < 7:
            colors.append(COLOR_NEUTRAL)
        else:
            colors.append(COLOR_SPOOF)
    
    bars = ax.bar(attacks, eers, color=colors, edgecolor="white", linewidth=1.2)
    
    for bar, eer in zip(bars, eers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{eer:.1f}%", ha="center", va="bottom", fontsize=8, weight="bold", color="#374151")
    
    ax.set_xlabel("Attack ID (synthesis method)", fontsize=10)
    ax.set_ylabel("Equal Error Rate (%)", fontsize=10)
    ax.set_title("EER by attack on ASVspoof 2019 LA eval (71K utterances)", 
                 fontsize=11, weight="bold", pad=10)
    ax.set_ylim(0, max(eers) * 1.15)
    style_axis(ax)
    plt.tight_layout()
    return fig


def make_wavefake_plot():
    """Bar chart of per-vocoder EER from WaveFake."""
    vocoders = ["jsut_pwg*", "jsut_mb*", "ljspeech_mb_melgan", "ljspeech_pwg",
                "ljspeech_waveglow", "ljspeech_full_band", "ljspeech_melgan",
                "ljspeech_hifiGAN", "ljspeech_melgan_lg"]
    eers = [0.83, 1.13, 21.92, 26.12, 29.60, 30.60, 31.12, 33.23, 33.85]
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = []
    for v, e in zip(vocoders, eers):
        if "jsut" in v:
            colors.append(COLOR_NEUTRAL)
        elif e < 25:
            colors.append("#fbbf24")
        else:
            colors.append(COLOR_SPOOF)
    
    bars = ax.bar(vocoders, eers, color=colors, edgecolor="white", linewidth=1.2)
    
    for bar, eer in zip(bars, eers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{eer:.1f}%", ha="center", va="bottom", fontsize=8, weight="bold", color="#374151")
    
    ax.set_xlabel("Vocoder pipeline", fontsize=10)
    ax.set_ylabel("Equal Error Rate (%)", fontsize=10)
    ax.set_title("EER by vocoder on WaveFake (model trained ONLY on ASVspoof attacks)", 
                 fontsize=11, weight="bold", pad=10)
    ax.set_ylim(0, max(eers) * 1.15)
    plt.xticks(rotation=30, ha="right")
    style_axis(ax)
    
    fig.text(0.02, 0.02, "* JSUT (Japanese) numbers reflect domain shortcut, not real spoofing detection", 
             fontsize=8, color=COLOR_NEUTRAL, style="italic")
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


# ============================================================
# Prediction handler
# ============================================================

def predict_audio(audio_path):
    if audio_path is None:
        return ("Please upload an audio file or select an example.", None, None, None)
    
    start = time.time()
    try:
        result = detector.predict(audio_path, return_per_window=True)
    except Exception as e:
        return (f"Error: {type(e).__name__}: {e}", None, None, None)
    elapsed_ms = (time.time() - start) * 1000
    
    pred = result["prediction"]
    confidence = result["confidence"] * 100
    
    if pred == "spoof":
        badge = (f"<div style='padding:1rem;border-radius:0.5rem;"
                 f"background:#fee2e2;border-left:4px solid {COLOR_SPOOF};'>"
                 f"<h3 style='margin:0;color:{COLOR_SPOOF};'>SPOOF detected</h3>"
                 f"<p style='margin:0.5rem 0 0 0;font-size:1.1rem;'><b>Confidence: {confidence:.1f}%</b></p>"
                 f"</div>")
    else:
        badge = (f"<div style='padding:1rem;border-radius:0.5rem;"
                 f"background:#dcfce7;border-left:4px solid {COLOR_BONAFIDE};'>"
                 f"<h3 style='margin:0;color:{COLOR_BONAFIDE};'>BONAFIDE (likely real)</h3>"
                 f"<p style='margin:0.5rem 0 0 0;font-size:1.1rem;'><b>Confidence: {confidence:.1f}%</b></p>"
                 f"</div>")
    
    details = (f"**Spoof probability:** {result['spoof_probability']:.4f}\n\n"
               f"**Bonafide probability:** {result['bonafide_probability']:.4f}\n\n"
               f"**Audio duration:** {result['utterance_duration_sec']:.2f} seconds\n\n"
               f"**Windows analyzed:** {result['n_windows']}\n\n"
               f"**Inference time:** {elapsed_ms:.0f} ms (CPU)")
    
    fig = make_per_window_plot(result["window_scores"], threshold=result["threshold_used"])
    
    raw_json = {
        "spoof_probability": result["spoof_probability"],
        "bonafide_probability": result["bonafide_probability"],
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "duration_sec": result["utterance_duration_sec"],
        "n_windows": result["n_windows"],
        "window_scores": result["window_scores"],
        "inference_ms": round(elapsed_ms, 1),
    }
    
    return badge, details, fig, raw_json


# ============================================================
# Custom CSS for visual polish
# ============================================================

CUSTOM_CSS = """
.gradio-container {
    font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
    max-width: 1200px !important;
}
.tab-nav button {
    font-size: 1rem !important;
    font-weight: 600 !important;
}
.metric-card {
    background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
    padding: 1.5rem;
    border-radius: 0.75rem;
    text-align: center;
    border: 1px solid #d1d5db;
}
.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: #111827;
    line-height: 1.2;
}
.metric-label {
    font-size: 0.875rem;
    color: #6b7280;
    margin-top: 0.5rem;
}
.context-card {
    background: white;
    padding: 1.25rem;
    border-radius: 0.5rem;
    border: 1px solid #e5e7eb;
    margin-bottom: 1rem;
}
.context-card h4 {
    color: #7c3aed;
    margin: 0 0 0.5rem 0;
    font-size: 1.05rem;
}
.context-card p {
    margin: 0;
    color: #4b5563;
    line-height: 1.6;
}
.cta-section {
    text-align: center;
    padding: 2rem 1rem;
    background: linear-gradient(135deg, #ede9fe 0%, #ddd6fe 100%);
    border-radius: 1rem;
    margin: 2rem 0;
}
"""


# ============================================================
# Build the multi-tab Gradio interface
# ============================================================

with gr.Blocks(
    title="Deepfake Audio Detection",
    theme=gr.themes.Soft(primary_hue="purple"),
    css=CUSTOM_CSS,
) as demo:
    
    gr.Markdown("""
    # Deepfake Audio Detection
    *Wav2Vec 2.0 fine-tuned on ASVspoof 2019 LA  •  Cross-dataset evaluated on ASVspoof 2021 LA & WaveFake*
    """)
    
    with gr.Tabs() as tabs:
        
        # ============================================================
        # TAB 1: WELCOME
        # ============================================================
        with gr.Tab("Welcome", id=0):
            gr.Markdown("""
            ## Is this voice real?
            ### Modern AI can clone any voice from just a few seconds of audio.
            
            Voice deepfakes have become a serious concern. AI systems can now generate speech that sounds almost
            indistinguishable from a real person — and they can do it from very short samples. This creates real
            problems for security, journalism, and trust in digital media. Detecting AI-generated speech 
            reliably is an active research area, and this demo shows one approach.
            """)
            
            gr.Markdown("### Why this matters")
            
            with gr.Row():
                with gr.Column():
                    gr.HTML("""
                    <div class='context-card'>
                    <h4>Phone scams</h4>
                    <p>Voice clones are increasingly used to impersonate family members in
                    "emergency call" scams, asking for money or sensitive information. Reported cases
                    have surged since 2022.</p>
                    </div>
                    """)
                with gr.Column():
                    gr.HTML("""
                    <div class='context-card'>
                    <h4>Misinformation</h4>
                    <p>Fabricated political speeches, fake celebrity endorsements, and false
                    statements attributed to public figures have circulated widely on social media.</p>
                    </div>
                    """)
                with gr.Column():
                    gr.HTML("""
                    <div class='context-card'>
                    <h4>Trust in evidence</h4>
                    <p>Courts now have to grapple with whether audio recordings are authentic.
                    The same is true for journalism and historical archives.</p>
                    </div>
                    """)
            
            gr.Markdown("## Try the detector")
            gr.Markdown("Upload your own audio, record from your microphone, or click an example.")
            cta_btn = gr.Button("Open the detector", variant="primary", size="lg")
            
            gr.Markdown("""
            ---
            **Built by:** Sara Iqbal & Areeba Arif  •  FAST-NUCES Spring 2026 Deep Learning Project
            
            **Source code:** [github.com/Saracasm/deepfake-audio-detection](https://github.com/Saracasm/deepfake-audio-detection)  
            **Model weights:** [Sara1708/deepfake-audio-wav2vec2](https://huggingface.co/Sara1708/deepfake-audio-wav2vec2)
            """)
        
        
        # ============================================================
        # TAB 2: DETECTOR
        # ============================================================
        with gr.Tab("Detector", id=1):
            gr.Markdown("""
            ### Audio analysis
            Upload audio, record yourself, or click an example below. The detector returns a prediction with confidence,
            plus per-window analysis showing how the model integrates evidence over time.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="Audio input",
                    )
                    analyze_btn = gr.Button("Analyze", variant="primary", size="lg")
                    
                    gr.Examples(
                        examples=EXAMPLE_FILES,
                        inputs=audio_input,
                        label="Example clips (click to load)",
                    )
                
                with gr.Column(scale=1):
                    badge_output = gr.HTML(label=None)
                    details_output = gr.Markdown(label="Details")
            
            plot_output = gr.Plot(label="Per-window analysis")
            
            with gr.Accordion("Raw output (JSON)", open=False):
                raw_output = gr.JSON(label=None)
            
            analyze_btn.click(
                fn=predict_audio,
                inputs=audio_input,
                outputs=[badge_output, details_output, plot_output, raw_output],
            )
        
        
        # ============================================================
        # TAB 3: PERFORMANCE
        # ============================================================
        with gr.Tab("Performance", id=2):
            gr.Markdown("### Headline results")
            
            with gr.Row():
                gr.HTML("""
                <div class='metric-card'>
                <div class='metric-value' style='color:#16a34a;'>5.55%</div>
                <div class='metric-label'><b>ASVspoof 2019 LA</b><br/>(unseen attacks A07-A19)</div>
                </div>
                """)
                gr.HTML("""
                <div class='metric-card'>
                <div class='metric-value' style='color:#7c3aed;'>9.09%</div>
                <div class='metric-label'><b>ASVspoof 2021 LA</b><br/>(codec-degraded audio)</div>
                </div>
                """)
                gr.HTML("""
                <div class='metric-card'>
                <div class='metric-value' style='color:#dc2626;'>26.33%</div>
                <div class='metric-label'><b>WaveFake</b><br/>(novel vocoder pipelines)</div>
                </div>
                """)
            
            gr.Markdown("""
            #### Comparison to published baselines
            
            | System | 2019 LA EER | 2021 LA EER |
            |---|---|---|
            | Official LFCC-GMM baseline | 8.09% | 25.56% |
            | Official CQCC-GMM baseline | 9.57% | 19.30% |
            | Official LFCC-LCNN baseline | – | 9.26% |
            | Official RawNet2 baseline | – | 9.50% |
            | **This work (Wav2Vec 2.0)** | **5.55%** | **9.09%** |
            
            Our model outperforms LFCC-GMM on 2019 LA by 2.54 pp and matches the strongest neural
            baselines (LFCC-LCNN, RawNet2) on 2021 LA — without any codec-specific training augmentation.
            """)
            
            gr.Markdown("---")
            gr.Markdown("### Performance by audio codec (ASVspoof 2021 LA)")
            gr.Markdown("Real-world speech goes through codecs (compression for transmission). The model handles modern codecs well but struggles with aggressive cellular compression.")
            gr.Plot(value=make_per_codec_plot(), label=None)
            
            gr.Markdown("---")
            gr.Markdown("### Performance by attack type (ASVspoof 2019 LA eval)")
            gr.Markdown("13 different synthesis methods (A07-A19), all unseen during training. A10 is the model's persistent weakness across both datasets.")
            gr.Plot(value=make_per_attack_plot(), label=None)
            
            gr.Markdown("---")
            gr.Markdown("### The WaveFake story (honest negative result)")
            gr.Markdown("""
            On WaveFake the model performs significantly worse — particularly on LJSpeech-based vocoders
            (22-34% EER). This is because WaveFake tests pure neural vocoder synthesis, while the model
            was trained on ASVspoof's mix of TTS + voice conversion attacks. **The model has learned
            ASVspoof-specific synthesis artifacts but not universal vocoder detection.**
            
            JSUT (Japanese) numbers look artificially good because the bonafide examples are English LJSpeech —
            the model is detecting language/domain, not actual spoofing artifacts. The LJSpeech-based numbers
            are the methodologically meaningful results.
            """)
            gr.Plot(value=make_wavefake_plot(), label=None)
        
        
        # ============================================================
        # ============================================================
        # TAB 4: TECHNICAL
        # ============================================================
        with gr.Tab("Under the hood", id=3):
            gr.Markdown("## Architecture")
            
            gr.HTML("""
            <div style="background:#1f2937;color:#e5e7eb;padding:1.5rem;border-radius:0.5rem;font-family:monospace;font-size:0.95rem;line-height:1.7;">
            <div style="text-align:center;color:#a78bfa;font-weight:600;margin-bottom:0.5rem;">Pipeline</div>
            raw waveform (16 kHz, 4 sec, 64,000 samples)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;|<br>
            &nbsp;&nbsp;&nbsp;&nbsp;v<br>
            <span style="color:#fbbf24;">Wav2Vec 2.0 Base backbone (95M params, 12 transformer layers)</span><br>
            &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;Stage 1: fully frozen<br>
            &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;Stage 2: top 2 layers + final LayerNorm unfrozen (~14M trainable)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;v<br>
            mean pooling over time<br>
            &nbsp;&nbsp;&nbsp;&nbsp;|<br>
            &nbsp;&nbsp;&nbsp;&nbsp;v<br>
            <span style="color:#34d399;">linear classification head (768 -> 2)</span><br>
            &nbsp;&nbsp;&nbsp;&nbsp;|<br>
            &nbsp;&nbsp;&nbsp;&nbsp;v<br>
            softmax -> P(spoof), P(bonafide)
            </div>
            """)
            
            gr.Markdown("## Two-stage training rationale")
            
            with gr.Row():
                gr.HTML("""
                <div class='context-card'>
                <h4>Stage 1: frozen backbone, head only</h4>
                <p>Train only the linear classification head, keeping all 95M Wav2Vec parameters frozen.
                This proves that pretrained Wav2Vec representations already carry strong anti-spoofing signal.</p>
                <p style='margin-top:1rem;'><b>Result:</b> <span style='color:#7c3aed;font-size:1.2rem;font-weight:700;'>10.09% dev EER</span><br>
                with just <b>1,538</b> trainable parameters.</p>
                </div>
                """)
                gr.HTML("""
                <div class='context-card'>
                <h4>Stage 2: top 2 layers unfrozen</h4>
                <p>Unfreeze top 2 transformer layers + final LayerNorm. Lower LR from 1e-3 to 1e-5
                with 10% warmup + linear decay. Enable mixed precision (fp16) for speed.</p>
                <p style='margin-top:1rem;'><b>Result:</b> <span style='color:#16a34a;font-size:1.2rem;font-weight:700;'>0.69% dev EER</span><br>
                a <b>93% relative error reduction</b> with 14.18M trainable params (15% of model).</p>
                </div>
                """)
            
            gr.Markdown("## Key design decisions")
            
            gr.Markdown("""
- **Class-weighted cross-entropy** to handle 9:1 spoof:bonafide imbalance (bonafide=4.92, spoof=0.56)
- **4-second windowing with 50% overlap** to handle clips of varying length
- **Mean aggregation** over per-window scores produces final utterance prediction
- **Mixed precision training** reduced wall-clock time from ~6h to 2h 56m on T4
            """)
            
            gr.Markdown("## Limitations (honest disclosure)")
            
            gr.HTML("""
            <div style='background:#fef3c7;border-left:4px solid #f59e0b;padding:1rem 1.5rem;border-radius:0.5rem;margin:1rem 0;'>
            <p><b>WaveFake out-of-domain generalization is poor</b> (~29% EER on LJSpeech vocoders).
            The model learned ASVspoof-specific synthesis artifacts, not universal vocoder detection.
            Future work: train on a mixed corpus including pure vocoder samples.</p>
            </div>
            <div style='background:#fef3c7;border-left:4px solid #f59e0b;padding:1rem 1.5rem;border-radius:0.5rem;margin:1rem 0;'>
            <p><b>Codec sensitivity:</b> GSM and PSTN telephone codecs degrade EER by ~6 percentage points.
            Codec augmentation during training would likely close this gap.</p>
            </div>
            <div style='background:#fef3c7;border-left:4px solid #f59e0b;padding:1rem 1.5rem;border-radius:0.5rem;margin:1rem 0;'>
            <p><b>A10 attack family is consistently challenging</b> (15.54% EER on this attack alone).
            This is a stable model weakness across both 2019 and 2021 evaluations.</p>
            </div>
            <div style='background:#fee2e2;border-left:4px solid #dc2626;padding:1rem 1.5rem;border-radius:0.5rem;margin:1rem 0;'>
            <p><b>Not a production deepfake detector.</b> Real-world deepfakes use synthesis methods this
            model has never seen. Use this as a research demonstration, not for security-critical decisions.</p>
            </div>
            """)
            
            gr.Markdown("## Source and citations")
            
            gr.Markdown("""
**Source code, training notebooks, full evaluation results:**  
[github.com/Saracasm/deepfake-audio-detection](https://github.com/Saracasm/deepfake-audio-detection)

**Model weights and card:**  
[huggingface.co/Sara1708/deepfake-audio-wav2vec2](https://huggingface.co/Sara1708/deepfake-audio-wav2vec2)

### Datasets used
- ASVspoof 2019 LA — Wang et al., 2020
- ASVspoof 2021 LA — Yamagishi et al., 2021
- WaveFake — Frank & Schonherr, 2021

### Backbone model
- Wav2Vec 2.0 Base — Baevski et al., 2020 (Facebook AI Research)
            """)


    # Wire up the CTA button to switch to the Detector tab
    cta_btn.click(fn=lambda: gr.Tabs(selected=1), outputs=tabs)


if __name__ == "__main__":
    demo.launch()
