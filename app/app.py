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
        empty_badge = """
        <div class='result-placeholder'>
            <div class='result-placeholder-icon'>⚠️</div>
            <div class='result-placeholder-text'>Please upload an audio file or select an example first.</div>
        </div>
        """
        return (empty_badge, None, None, None)
    
    start = time.time()
    try:
        result = detector.predict(audio_path, return_per_window=True)
    except Exception as e:
        error_badge = f"""
        <div class='result-error'>
            <div class='result-placeholder-icon'>❌</div>
            <div class='result-placeholder-text'><b>Error:</b> {type(e).__name__}: {e}</div>
        </div>
        """
        return (error_badge, None, None, None)
    elapsed_ms = (time.time() - start) * 1000
    
    pred = result["prediction"]
    confidence = result["confidence"] * 100
    spoof_pct = result["spoof_probability"] * 100
    bona_pct = result["bonafide_probability"] * 100
    
    if pred == "spoof":
        badge_class = "result-card-spoof"
        icon = "⚠"
        verdict = "Likely synthetic"
        verdict_sub = "This audio shows characteristics of AI-generated speech."
    else:
        badge_class = "result-card-bonafide"
        icon = "✓"
        verdict = "Likely authentic"
        verdict_sub = "This audio shows characteristics of natural human speech."
    
    badge = f"""
    <div class='{badge_class}'>
        <div class='result-card-header'>
            <div class='result-card-icon'>{icon}</div>
            <div class='result-card-text'>
                <div class='result-card-verdict'>{verdict}</div>
                <div class='result-card-verdict-sub'>{verdict_sub}</div>
            </div>
        </div>
        <div class='result-card-confidence'>
            <div class='confidence-label'>
                <span>Confidence</span>
                <span class='confidence-value'>{confidence:.1f}%</span>
            </div>
            <div class='confidence-bar-track'>
                <div class='confidence-bar-fill' style='width: {confidence:.1f}%;'></div>
            </div>
        </div>
        <div class='result-card-probs'>
            <div class='prob-row'>
                <span class='prob-label'>Synthetic</span>
                <div class='prob-bar-track'>
                    <div class='prob-bar-fill prob-bar-spoof' style='width: {spoof_pct:.1f}%;'></div>
                </div>
                <span class='prob-pct'>{spoof_pct:.1f}%</span>
            </div>
            <div class='prob-row'>
                <span class='prob-label'>Authentic</span>
                <div class='prob-bar-track'>
                    <div class='prob-bar-fill prob-bar-bonafide' style='width: {bona_pct:.1f}%;'></div>
                </div>
                <span class='prob-pct'>{bona_pct:.1f}%</span>
            </div>
        </div>
        <div class='result-card-meta'>
            <span>{result['utterance_duration_sec']:.2f}s audio</span>
            <span class='meta-dot'>·</span>
            <span>{result['n_windows']} windows</span>
            <span class='meta-dot'>·</span>
            <span>{elapsed_ms:.0f}ms on CPU</span>
        </div>
    </div>
    """
    
    details = (f"**Spoof probability:** {result['spoof_probability']:.4f}\n\n"
               f"**Bonafide probability:** {result['bonafide_probability']:.4f}\n\n"
               f"**Audio duration:** {result['utterance_duration_sec']:.2f} seconds\n\n"
               f"**Windows analyzed:** {result['n_windows']}\n\n"
               f"**Inference time:** {elapsed_ms:.0f} ms (CPU)\n\n"
               f"**Threshold used:** {result['threshold_used']:.4f}")
    
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
/* ============================================================
   STAGE 1: FOUNDATION — Modern AI aesthetic
   Color system, typography, spacing, transitions
   ============================================================ */

/* Import Inter for clean modern look */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ---------- Color tokens ---------- */
:root {
    --brand-purple-50:  #f5f3ff;
    --brand-purple-100: #ede9fe;
    --brand-purple-300: #c4b5fd;
    --brand-purple-400: #a78bfa;
    --brand-purple-500: #8b5cf6;
    --brand-purple-600: #7c3aed;
    --brand-purple-700: #6d28d9;
    --brand-pink-400:   #f472b6;
    --brand-pink-500:   #ec4899;
    --accent-green:     #10b981;
    --accent-amber:     #f59e0b;
    --accent-red:       #ef4444;
    --gradient-brand:   linear-gradient(135deg, #7c3aed 0%, #ec4899 100%);
    --gradient-soft:    linear-gradient(135deg, rgba(124, 58, 237, 0.08) 0%, rgba(236, 72, 153, 0.08) 100%);
    --gradient-hero:    radial-gradient(ellipse at top, rgba(124, 58, 237, 0.15) 0%, transparent 50%),
                        radial-gradient(ellipse at bottom, rgba(236, 72, 153, 0.10) 0%, transparent 50%);
}

/* ---------- Container & typography ---------- */
.gradio-container {
    font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
    font-feature-settings: 'cv11', 'ss01';
}

/* Tighter headings */
.gradio-container h1 {
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    line-height: 1.1 !important;
}
.gradio-container h2 {
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.2 !important;
}
.gradio-container h3 {
    font-weight: 600 !important;
    letter-spacing: -0.015em !important;
}
.gradio-container h4 {
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
}

/* Body text breathing room */
.gradio-container p {
    line-height: 1.65 !important;
}

/* Monospace for code/pipeline blocks */
.gradio-container code, .gradio-container pre {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
}

/* ---------- Tab navigation polish ---------- */
.tab-nav {
    border-bottom: 1px solid var(--border-color-primary, rgba(255,255,255,0.1)) !important;
    margin-bottom: 1.5rem !important;
}
.tab-nav button {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    transition: all 0.2s ease !important;
    border-radius: 0.5rem 0.5rem 0 0 !important;
}
.tab-nav button:hover {
    background: var(--gradient-soft) !important;
}
.tab-nav button.selected {
    border-bottom: 2px solid var(--brand-purple-500) !important;
    color: var(--brand-purple-400) !important;
}

/* ---------- Metric cards (Performance tab) ---------- */
.metric-card {
    background: var(--background-fill-secondary, rgba(124, 58, 237, 0.04));
    color: var(--body-text-color, #111827);
    padding: 1.75rem 1.5rem;
    border-radius: 0.875rem;
    text-align: center;
    border: 1px solid var(--border-color-primary, rgba(124, 58, 237, 0.15));
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 24px -8px rgba(124, 58, 237, 0.25);
}
.metric-value {
    font-size: 2.75rem;
    font-weight: 800;
    background: var(--gradient-brand);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.metric-label {
    font-size: 0.8125rem;
    color: var(--body-text-color-subdued, #6b7280);
    margin-top: 0.5rem;
    opacity: 0.8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 500;
}

/* ---------- Context cards (Welcome tab) ---------- */
.context-card {
    background: var(--background-fill-secondary, #ffffff);
    color: var(--body-text-color, #111827);
    padding: 1.5rem;
    border-radius: 0.875rem;
    border: 1px solid var(--border-color-primary, rgba(124, 58, 237, 0.15));
    margin-bottom: 1rem;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.context-card:hover {
    transform: translateY(-2px);
    border-color: var(--brand-purple-400) !important;
}
.context-card h4 {
    color: var(--brand-purple-400);
    margin: 0 0 0.5rem 0;
    font-size: 1.05rem;
}
.context-card p {
    margin: 0;
    color: var(--body-text-color, #4b5563);
    line-height: 1.65;
    opacity: 0.9;
}

/* ---------- Stage cards (Under the hood tab) ---------- */
.stage-card {
    background: var(--background-fill-secondary, #f9fafb);
    color: var(--body-text-color, #111827);
    border: 1px solid var(--border-color-primary, rgba(124, 58, 237, 0.15));
    padding: 1.5rem;
    border-radius: 0.875rem;
    margin: 0.5rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stage-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 24px -8px rgba(124, 58, 237, 0.2);
}
.stage-card p, .stage-card b {
    color: var(--body-text-color, #111827);
}

/* ---------- Limitation alerts ---------- */
.limitation-warn {
    background: rgba(251, 191, 36, 0.08);
    border-left: 3px solid var(--accent-amber);
    padding: 1rem 1.25rem;
    border-radius: 0.5rem;
    margin: 0.75rem 0;
    color: var(--body-text-color, #111827);
}
.limitation-warn p, .limitation-warn b {
    color: var(--body-text-color, #111827);
    margin: 0;
}
.limitation-danger {
    background: rgba(239, 68, 68, 0.08);
    border-left: 3px solid var(--accent-red);
    padding: 1rem 1.25rem;
    border-radius: 0.5rem;
    margin: 0.75rem 0;
    color: var(--body-text-color, #111827);
}
.limitation-danger p, .limitation-danger b {
    color: var(--body-text-color, #111827);
    margin: 0;
}

/* ---------- CTA section ---------- */
.cta-section {
    text-align: center;
    padding: 2.5rem 1.5rem;
    background: var(--gradient-soft);
    border-radius: 1.25rem;
    margin: 2.5rem 0;
    color: var(--body-text-color, #111827);
    border: 1px solid var(--border-color-primary, rgba(124, 58, 237, 0.15));
}

/* ---------- Buttons polish ---------- */
.gradio-container button.lg {
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.gradio-container button.lg:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 16px -4px rgba(124, 58, 237, 0.3) !important;
}
.gradio-container button.primary {
    background: var(--gradient-brand) !important;
    border: none !important;
}

/* ---------- Theme toggle ---------- */
#theme-toggle-row {
    justify-content: flex-end;
    margin-bottom: 0.5rem;
}
#theme-toggle-btn {
    max-width: 140px !important;
    min-width: 140px !important;
    font-size: 0.85rem !important;
}

/* ---------- Subtle animated gradient background (very low opacity) ---------- */
html, body {
    overflow-x: hidden !important;
    max-width: 100vw;
}
body::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    background: var(--gradient-hero);
    pointer-events: none;
    z-index: -1;
    opacity: 0.6;
    animation: gradientShift 20s ease-in-out infinite;
}
@keyframes gradientShift {
    0%, 100% { opacity: 0.5; }
    50% { opacity: 0.8; }
}

/* ---------- Reduce motion for accessibility ---------- */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}

/* ============================================================
   STAGE 2: WELCOME HERO
   ============================================================ */

/* Hero container with animated glow */
.hero-section {
    position: relative;
    text-align: center;
    padding: 3rem 1.5rem 1.5rem 1.5rem;
    margin: 0 0 0.5rem 0;
    overflow: hidden;
    border-radius: 1.5rem;
}
.hero-section::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 600px;
    height: 600px;
    transform: translate(-50%, -50%);
    background: radial-gradient(circle,
        rgba(124, 58, 237, 0.25) 0%,
        rgba(236, 72, 153, 0.15) 40%,
        transparent 70%);
    z-index: -1;
    animation: heroGlow 8s ease-in-out infinite;
    filter: blur(40px);
}
@keyframes heroGlow {
    0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 0.7; }
    50% { transform: translate(-50%, -50%) scale(1.15); opacity: 1; }
}

/* Hero eyebrow tag */
.hero-eyebrow {
    display: inline-block;
    padding: 0.4rem 1rem;
    background: rgba(124, 58, 237, 0.12);
    border: 1px solid rgba(124, 58, 237, 0.25);
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--brand-purple-400);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

/* Massive gradient hero headline */
.hero-title {
    font-size: clamp(2.5rem, 6vw, 4.5rem) !important;
    font-weight: 800 !important;
    line-height: 1.05 !important;
    letter-spacing: -0.04em !important;
    margin: 0 0 1rem 0 !important;
    background: linear-gradient(90deg, #7c3aed 0%, #a78bfa 30%, #ec4899 70%, #fb7185 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Hero subtitle */
.hero-subtitle {
    font-size: clamp(1.1rem, 2.2vw, 1.4rem) !important;
    font-weight: 500 !important;
    color: var(--body-text-color, #4b5563);
    opacity: 0.85;
    max-width: 720px;
    margin: 0 auto 0 auto !important;
    line-height: 1.5 !important;
    letter-spacing: -0.01em !important;
}

/* Section eyebrow + heading combo */
.section-header {
    text-align: center;
    margin: 1.5rem 0 1.5rem 0;
}
.section-eyebrow {
    display: block;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--brand-purple-400);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
.section-title {
    font-size: 2rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    margin: 0 !important;
}

/* Redesigned context cards with icon, bigger, animated */
.context-card-v2 {
    background: var(--background-fill-secondary, #ffffff);
    border: 1px solid var(--border-color-primary, rgba(124, 58, 237, 0.15));
    padding: 2rem 1.75rem;
    border-radius: 1rem;
    height: 100%;
    transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    position: relative;
    overflow: hidden;
}
.context-card-v2::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--brand-purple-500), transparent);
    opacity: 0;
    transition: opacity 0.25s ease;
}
.context-card-v2:hover {
    transform: translateY(-4px);
    border-color: rgba(124, 58, 237, 0.4) !important;
    box-shadow: 0 20px 40px -12px rgba(124, 58, 237, 0.2);
}
.context-card-v2:hover::before {
    opacity: 1;
}
.context-card-icon {
    width: 56px;
    height: 56px;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.15) 0%, rgba(236, 72, 153, 0.15) 100%);
    display: flex !important;
    align-items: center;
    justify-content: center;
    font-size: 1.75rem !important;
    line-height: 1 !important;
    margin-bottom: 1.25rem;
    border: 1px solid rgba(124, 58, 237, 0.25);
}
.context-card-icon span {
    font-size: 1.75rem !important;
    line-height: 1 !important;
    display: inline-block;
}
.context-card-v2 h4 {
    color: var(--body-text-color, #111827) !important;
    margin: 0 0 0.6rem 0 !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
}
.context-card-v2 p {
    margin: 0;
    color: var(--body-text-color, #4b5563) !important;
    line-height: 1.6 !important;
    opacity: 0.85;
    font-size: 0.95rem;
}

/* CTA section v2 — gradient bg with stronger presence */
.cta-section-v2 {
    text-align: center;
    padding: 2.5rem 2rem;
    background: linear-gradient(135deg,
        rgba(124, 58, 237, 0.12) 0%,
        rgba(236, 72, 153, 0.12) 100%);
    border-radius: 1.5rem;
    margin: 2rem 0 1.5rem 0;
    border: 1px solid rgba(124, 58, 237, 0.2);
    position: relative;
    overflow: hidden;
}
.cta-section-v2::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(167, 139, 250, 0.1) 0%, transparent 50%);
    animation: ctaGlow 12s ease-in-out infinite;
    pointer-events: none;
}
@keyframes ctaGlow {
    0%, 100% { transform: translate(0, 0); }
    50% { transform: translate(20px, -20px); }
}
.cta-title {
    font-size: 2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
    margin: 0 0 0.75rem 0 !important;
    color: #a78bfa;
    background: linear-gradient(135deg, #a78bfa 0%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: block;
}
.cta-subtitle {
    font-size: 1.05rem;
    color: var(--body-text-color, #4b5563);
    opacity: 0.85;
    margin: 0 0 1.75rem 0;
}

/* Footer credits */
.welcome-footer {
    margin-top: 4rem;
    padding-top: 2rem;
    border-top: 1px solid var(--border-color-primary, rgba(124, 58, 237, 0.15));
    text-align: center;
    font-size: 0.9rem;
    color: var(--body-text-color, #6b7280);
    opacity: 0.75;
    line-height: 1.8;
}
.welcome-footer a {
    color: var(--brand-purple-400) !important;
    text-decoration: none;
    font-weight: 500;
}
.welcome-footer a:hover {
    text-decoration: underline;
}


/* New card title (replaces h4 which gets stripped by Gradio sanitizer) */
.card-title {
    color: var(--body-text-color, #111827) !important;
    margin: 0 0 0.6rem 0 !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
    line-height: 1.3 !important;
}


/* ============================================================
   STAGE 3: DETECTOR POLISH
   ============================================================ */

/* Detector intro paragraph */
.detector-intro {
    max-width: 720px;
    margin: 0.75rem auto 0 auto !important;
    font-size: 1.02rem !important;
    color: var(--body-text-color, #4b5563);
    opacity: 0.85;
    line-height: 1.6 !important;
}

/* Step labels (numbered guidance) */
.step-label {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--body-text-color-subdued, #6b7280);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 0.75rem 0 0.6rem 0;
    opacity: 0.85;
}
.step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: var(--gradient-brand);
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: none;
    letter-spacing: 0;
}

/* Result placeholder (shown before any analysis) */
.result-placeholder {
    background: var(--background-fill-secondary, rgba(124, 58, 237, 0.04));
    border: 2px dashed var(--border-color-primary, rgba(124, 58, 237, 0.2));
    border-radius: 1rem;
    padding: 3rem 2rem;
    text-align: center;
    color: var(--body-text-color-subdued, #6b7280);
    min-height: 200px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
}
.result-placeholder-icon {
    font-size: 2.5rem;
    opacity: 0.6;
}
.result-placeholder-text {
    font-size: 0.95rem;
    opacity: 0.85;
    line-height: 1.5;
}
.result-error {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 1rem;
    padding: 1.5rem;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
}

/* Result cards — bonafide (green) and spoof (red) variants */
.result-card-bonafide, .result-card-spoof {
    border-radius: 1rem;
    padding: 1.75rem 1.5rem;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}
.result-card-bonafide {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(16, 185, 129, 0.03) 100%);
    border-color: rgba(16, 185, 129, 0.3);
}
.result-card-spoof {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(239, 68, 68, 0.03) 100%);
    border-color: rgba(239, 68, 68, 0.3);
}
.result-card-bonafide::before, .result-card-spoof::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.result-card-bonafide::before { background: linear-gradient(90deg, transparent, #10b981, transparent); }
.result-card-spoof::before    { background: linear-gradient(90deg, transparent, #ef4444, transparent); }

.result-card-header {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 1.25rem;
}
.result-card-icon {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    font-weight: 700;
    flex-shrink: 0;
}
.result-card-bonafide .result-card-icon {
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.result-card-spoof .result-card-icon {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
}
.result-card-text { flex: 1; }
.result-card-verdict {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--body-text-color, #111827);
    letter-spacing: -0.01em;
    line-height: 1.2;
    margin-bottom: 0.25rem;
}
.result-card-verdict-sub {
    font-size: 0.9rem;
    color: var(--body-text-color, #4b5563);
    opacity: 0.8;
    line-height: 1.5;
}

/* Confidence section */
.result-card-confidence {
    margin: 1rem 0;
}
.confidence-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--body-text-color-subdued, #6b7280);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}
.confidence-value {
    color: var(--body-text-color, #111827);
    font-size: 1rem;
    text-transform: none;
    letter-spacing: 0;
}
.confidence-bar-track {
    width: 100%;
    height: 8px;
    background: var(--border-color-primary, rgba(0,0,0,0.1));
    border-radius: 999px;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    background: var(--gradient-brand);
    border-radius: 999px;
    transition: width 0.5s ease-out;
}

/* Probability rows (synthetic vs authentic) */
.result-card-probs {
    margin: 1.25rem 0;
    padding: 1rem;
    background: var(--background-fill-secondary, rgba(0,0,0,0.02));
    border-radius: 0.75rem;
}
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0.5rem 0;
}
.prob-label {
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--body-text-color-subdued, #6b7280);
    width: 80px;
    flex-shrink: 0;
}
.prob-bar-track {
    flex: 1;
    height: 6px;
    background: var(--border-color-primary, rgba(0,0,0,0.08));
    border-radius: 999px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.5s ease-out;
}
.prob-bar-spoof    { background: #ef4444; }
.prob-bar-bonafide { background: #10b981; }
.prob-pct {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--body-text-color, #111827);
    width: 50px;
    text-align: right;
    font-variant-numeric: tabular-nums;
}

/* Result card meta footer */
.result-card-meta {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    font-size: 0.8rem;
    color: var(--body-text-color-subdued, #9ca3af);
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border-color-primary, rgba(0,0,0,0.06));
}
.meta-dot {
    opacity: 0.5;
}

/* Analyze button override */
.analyze-button {
    width: 100% !important;
    margin-top: 0.25rem !important;
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
    
    # Theme toggle button at top-right
    with gr.Row(elem_id="theme-toggle-row"):
        theme_btn = gr.Button("☾ Dark mode", elem_id="theme-toggle-btn", size="sm")
    theme_btn.click(
        fn=None,
        inputs=None,
        outputs=theme_btn,
        js="""() => {
            document.body.classList.toggle('dark');
            const isDark = document.body.classList.contains('dark');
            return isDark ? '☀️ Light mode' : '☾ Dark mode';
        }"""
    )
    
    gr.Markdown("""
    # Deepfake Audio Detection
    *Wav2Vec 2.0 fine-tuned on ASVspoof 2019 LA  •  Cross-dataset evaluated on ASVspoof 2021 LA & WaveFake*
    """)
    
    with gr.Tabs() as tabs:
        
        # ============================================================
        # TAB 1: WELCOME
        # ============================================================
        with gr.Tab("Welcome", id=0):
            # Hero section
            gr.HTML("""
            <div class='hero-section'>
                <div class='hero-eyebrow'>Deep Learning Audio Forensics</div>
                <h1 class='hero-title'>Is this voice real?</h1>
                <p class='hero-subtitle'>
                    Modern AI can clone any voice from just a few seconds of audio.
                    This detector uses Wav2Vec 2.0 to tell synthetic speech apart from authentic recordings —
                    with 0.69% Equal Error Rate on the ASVspoof 2019 LA benchmark.
                </p>
            </div>
            """)
            
            # Why this matters section
            gr.HTML("""
            <div class='section-header'>
                <div class='section-eyebrow'>Why this matters</div>
                <div class='section-title'>Voice deepfakes are already in the wild</div>
            </div>
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.HTML("""
                    <div class='context-card-v2'>
                        <div class='context-card-icon'><span style='font-size:1.6rem;line-height:1;'>📞</span></div>
                        <div class='card-title'>Phone scams</div>
                        <p>Voice clones are increasingly used to impersonate family members in
                        "emergency call" scams. Reported cases have surged since 2022, with losses
                        running into millions of dollars annually.</p>
                    </div>
                    """)
                with gr.Column():
                    gr.HTML("""
                    <div class='context-card-v2'>
                        <div class='context-card-icon'><span style='font-size:1.6rem;line-height:1;'>📰</span></div>
                        <div class='card-title'>Misinformation</div>
                        <p>Fabricated political speeches, fake celebrity endorsements, and false
                        statements attributed to public figures have circulated widely on social
                        media platforms in recent election cycles.</p>
                    </div>
                    """)
                with gr.Column():
                    gr.HTML("""
                    <div class='context-card-v2'>
                        <div class='context-card-icon'><span style='font-size:1.6rem;line-height:1;'>⚖️</span></div>
                        <div class='card-title'>Trust in evidence</div>
                        <p>Courts now have to grapple with whether audio recordings are authentic.
                        The same challenge applies to investigative journalism and historical
                        archive verification.</p>
                    </div>
                    """)
            
            # CTA section
            gr.HTML("""
            <div class='cta-section-v2'>
                <div class='cta-title'>Try the detector</div>
                <div class='cta-subtitle'>
                    Upload your own audio, record from your microphone, or pick an example to start.
                </div>
            </div>
            """)
            cta_btn = gr.Button("Open the detector  →", variant="primary", size="lg")
            
            gr.HTML("""
            <div class='welcome-footer'>
                <strong>Built by</strong> Sara Iqbal & Areeba Arif &nbsp;·&nbsp; FAST-NUCES Spring 2026 Deep Learning Project<br>
                <a href='https://github.com/Saracasm/deepfake-audio-detection' target='_blank'>Source code on GitHub</a>
                &nbsp;·&nbsp;
                <a href='https://huggingface.co/Sara1708/deepfake-audio-wav2vec2' target='_blank'>Model weights on Hugging Face</a>
            </div>
            """)
        
        
        # ============================================================
        # TAB 2: DETECTOR
        # ============================================================
        with gr.Tab("Detector", id=1):
            gr.HTML("""
            <div class='section-header' style='margin-top: 1rem;'>
                <div class='section-eyebrow'>The detector</div>
                <div class='section-title'>Test the model on any audio</div>
                <p class='detector-intro'>
                    Upload audio, record yourself, or pick an example. The detector returns a calibrated
                    prediction with confidence, plus per-window analysis showing how evidence accumulates over time.
                </p>
            </div>
            """)
            
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.HTML("<div class='step-label'><span class='step-number'>1</span> Provide audio</div>")
                    audio_input = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="",
                        elem_classes=["audio-input-styled"],
                    )
                    
                    gr.HTML("<div class='step-label' style='margin-top: 1.25rem;'><span class='step-number'>2</span> Run the detector</div>")
                    analyze_btn = gr.Button("Analyze audio  →", variant="primary", size="lg", elem_classes=["analyze-button"])
                    
                    gr.HTML("<div class='step-label' style='margin-top: 1.5rem;'>Or try an example</div>")
                    gr.Examples(
                        examples=EXAMPLE_FILES,
                        inputs=audio_input,
                        label="",
                    )
                
                with gr.Column(scale=1):
                    gr.HTML("<div class='step-label'><span class='step-number'>3</span> Result</div>")
                    badge_output = gr.HTML(value="""
                    <div class='result-placeholder'>
                        <div class='result-placeholder-icon'>🎤</div>
                        <div class='result-placeholder-text'>Run the detector to see prediction</div>
                    </div>
                    """, label=None)
                    
                    with gr.Accordion("Detailed scores", open=False, elem_classes=["details-accordion"]):
                        details_output = gr.Markdown(label="")
            
            gr.HTML("<div class='step-label' style='margin-top: 2rem;'>Per-window analysis</div>")
            plot_output = gr.Plot(label="")
            
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
                <div class='stage-card'>
                <h4 style='color:#7c3aed;margin-top:0;'>Stage 1: frozen backbone, head only</h4>
                <p>Train only the linear classification head, keeping all 95M Wav2Vec parameters frozen.
                This proves that pretrained Wav2Vec representations already carry strong anti-spoofing signal.</p>
                <p style='margin-top:1rem;'><b>Result:</b> <span style='color:#a78bfa;font-size:1.2rem;font-weight:700;'>10.09% dev EER</span><br>
                with just <b>1,538</b> trainable parameters.</p>
                </div>
                """)
                gr.HTML("""
                <div class='stage-card'>
                <h4 style='color:#7c3aed;margin-top:0;'>Stage 2: top 2 layers unfrozen</h4>
                <p>Unfreeze top 2 transformer layers + final LayerNorm. Lower LR from 1e-3 to 1e-5
                with 10% warmup + linear decay. Enable mixed precision (fp16) for speed.</p>
                <p style='margin-top:1rem;'><b>Result:</b> <span style='color:#34d399;font-size:1.2rem;font-weight:700;'>0.69% dev EER</span><br>
                a <b style='color:#34d399;'>93% relative error reduction</b> with 14.18M trainable params (15% of model).</p>
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
            <div class='limitation-warn'>
            <p><b>WaveFake out-of-domain generalization is poor</b> (~29% EER on LJSpeech vocoders).
            The model learned ASVspoof-specific synthesis artifacts, not universal vocoder detection.
            Future work: train on a mixed corpus including pure vocoder samples.</p>
            </div>
            <div class='limitation-warn'>
            <p><b>Codec sensitivity:</b> GSM and PSTN telephone codecs degrade EER by ~6 percentage points.
            Codec augmentation during training would likely close this gap.</p>
            </div>
            <div class='limitation-warn'>
            <p><b>A10 attack family is consistently challenging</b> (15.54% EER on this attack alone).
            This is a stable model weakness across both 2019 and 2021 evaluations.</p>
            </div>
            <div class='limitation-danger'>
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
