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

def predict_audio_router(upload_path, record_path):
    """
    Routes between the two audio inputs (upload tab vs record tab).
    Whichever one has a value gets used. Upload takes precedence if both somehow set.
    """
    audio_path = upload_path if upload_path is not None else record_path
    return predict_audio(audio_path)


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
    
    # Plain-language hint about difficulty based on confidence
    if confidence >= 97:
        difficulty_hint = "clear case"
    elif confidence >= 80:
        difficulty_hint = "moderately confident"
    elif confidence >= 65:
        difficulty_hint = "borderline"
    else:
        difficulty_hint = "uncertain — interpret with caution"
    
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
            <details class='confidence-explainer'>
                <summary>What does this number mean?</summary>
                <div class='confidence-explainer-body'>
                    <p>
                        <strong>Confidence is how much probability the model puts behind its
                        prediction.</strong> If it says "Likely synthetic" at 66%, it means the model
                        sees a 66% chance this audio is synthetic and a 34% chance it's authentic.
                        That IS the answer — the prediction label is just the side with more probability.
                    </p>
                    <p>
                        <strong>High confidence does not always mean the model is right.</strong>
                        On the example clips below, the model is 100% confident on the easy ones and
                        less confident on the harder ones — that's expected. But it can also be 100%
                        confident and <em>wrong</em>, especially on attack types it struggles with
                        (like A10, the hardest example). When a deepfake is made by a method the
                        model hasn't learned to detect, it may see no spoofing signal at all and
                        confidently call it authentic.
                    </p>
                    <p>
                        <strong>Bottom line:</strong> treat any single prediction as one piece of
                        evidence, not a definitive answer. High confidence means the model sees
                        strong signal — but it can't detect what it hasn't been trained to detect.
                        Try the examples in order (easy → hardest) to see how confidence varies.
                    </p>
                </div>
            </details>
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
            <span class='meta-dot'>·</span>
            <span class='meta-difficulty'>{difficulty_hint}</span>
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


/* Input tabs (Upload file / Record mic) — smaller, segmented control feel */
.input-tabs > .tab-nav {
    border-bottom: none !important;
    margin-bottom: 0.5rem !important;
    background: var(--background-fill-secondary, rgba(124, 58, 237, 0.04));
    border-radius: 0.5rem;
    padding: 0.25rem;
    width: fit-content;
}
.input-tabs > .tab-nav button {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.45rem 0.9rem !important;
    border-radius: 0.4rem !important;
    border: none !important;
    background: transparent !important;
    color: var(--body-text-color-subdued, #6b7280) !important;
    transition: all 0.15s ease !important;
}
.input-tabs > .tab-nav button:hover {
    background: var(--background-fill-primary, rgba(124, 58, 237, 0.08)) !important;
    color: var(--body-text-color, #111827) !important;
}
.input-tabs > .tab-nav button.selected {
    background: var(--background-fill-primary, white) !important;
    color: var(--brand-purple-400) !important;
    border-bottom: none !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}


/* Recording instructions banner */
.record-instructions {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.08) 0%, rgba(236, 72, 153, 0.08) 100%);
    border: 1px solid rgba(124, 58, 237, 0.2);
    border-radius: 0.625rem;
    padding: 0.75rem 1rem;
    margin: 0.25rem 0 0.75rem 0;
    font-size: 0.85rem;
    line-height: 1.5;
    color: var(--body-text-color, #4b5563);
}
.record-instructions-icon {
    font-size: 1.25rem;
    flex-shrink: 0;
    line-height: 1.4;
}
.record-instructions-text {
    flex: 1;
    opacity: 0.9;
}
.record-instructions-text strong {
    color: var(--body-text-color, #111827);
    font-weight: 600;
}

/* Force record waveform area to have visible height */
.audio-record-styled .waveform-container,
.audio-record-styled audio {
    min-height: 80px !important;
}


/* ============================================================
   STAGE 4: PERFORMANCE TAB POLISH
   ============================================================ */

/* Subsection header (smaller than section header) */
.subsection-header {
    text-align: left;
    margin: 2rem 0 1rem 0;
}
.subsection-eyebrow {
    display: block;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--brand-purple-400);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}
.subsection-title {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.015em !important;
    color: var(--body-text-color, #111827);
    margin: 0 0 0.5rem 0 !important;
}
.subsection-caption {
    font-size: 0.95rem !important;
    color: var(--body-text-color, #4b5563) !important;
    opacity: 0.85;
    line-height: 1.6 !important;
    margin: 0 !important;
    max-width: 780px;
}

/* Performance metric cards (replaces older .metric-card on this tab) */
.perf-metric-card {
    background: var(--background-fill-secondary, rgba(124, 58, 237, 0.04));
    border: 1px solid var(--border-color-primary, rgba(124, 58, 237, 0.15));
    border-radius: 1rem;
    padding: 1.5rem 1.25rem;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    position: relative;
    overflow: hidden;
}
.perf-metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.perf-metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 16px 32px -12px rgba(124, 58, 237, 0.2);
}
.perf-metric-good::before { background: linear-gradient(90deg, transparent, #10b981, transparent); }
.perf-metric-mid::before  { background: linear-gradient(90deg, transparent, #a78bfa, transparent); }
.perf-metric-warn::before { background: linear-gradient(90deg, transparent, #f59e0b, transparent); }

.perf-metric-good:hover { border-color: rgba(16, 185, 129, 0.4) !important; }
.perf-metric-mid:hover  { border-color: rgba(167, 139, 250, 0.5) !important; }
.perf-metric-warn:hover { border-color: rgba(245, 158, 11, 0.4) !important; }

.perf-metric-tag {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    margin-bottom: 0.85rem;
}
.perf-metric-good .perf-metric-tag {
    background: rgba(16, 185, 129, 0.12);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.25);
}
.perf-metric-mid .perf-metric-tag {
    background: rgba(167, 139, 250, 0.12);
    color: var(--brand-purple-400);
    border: 1px solid rgba(167, 139, 250, 0.3);
}
.perf-metric-warn .perf-metric-tag {
    background: rgba(245, 158, 11, 0.12);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.perf-metric-value {
    font-size: 2.75rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.05;
    margin-bottom: 0.5rem;
    font-variant-numeric: tabular-nums;
}
.perf-metric-good .perf-metric-value { color: #10b981; }
.perf-metric-mid  .perf-metric-value { color: var(--brand-purple-400); }
.perf-metric-warn .perf-metric-value { color: #f59e0b; }

.perf-metric-name {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--body-text-color, #111827);
    letter-spacing: -0.01em;
    margin-bottom: 0.25rem;
}
.perf-metric-detail {
    font-size: 0.8rem;
    color: var(--body-text-color-subdued, #6b7280);
    opacity: 0.8;
}

/* Comparison table */
.comparison-table-wrap {
    background: var(--background-fill-secondary, rgba(124, 58, 237, 0.04));
    border: 1px solid var(--border-color-primary, rgba(124, 58, 237, 0.15));
    border-radius: 1rem;
    padding: 1.5rem;
    margin: 1rem 0;
    overflow: hidden;
}
.comparison-table {
    width: 100%;
    border-collapse: collapse;
    font-variant-numeric: tabular-nums;
}
.comparison-table thead {
    border-bottom: 2px solid var(--border-color-primary, rgba(124, 58, 237, 0.2));
}
.comparison-table th {
    padding: 0.75rem 0.5rem;
    text-align: left;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--body-text-color-subdued, #6b7280);
}
.comparison-table th:nth-child(2),
.comparison-table th:nth-child(3) {
    text-align: right;
}
.comparison-table td {
    padding: 0.75rem 0.5rem;
    font-size: 0.95rem;
    color: var(--body-text-color, #111827);
    border-bottom: 1px solid var(--border-color-primary, rgba(0,0,0,0.05));
}
.comparison-table td:nth-child(2),
.comparison-table td:nth-child(3) {
    text-align: right;
    font-weight: 500;
}
.comparison-table tbody tr:last-child td {
    border-bottom: none;
}
.comparison-row-highlight td {
    background: linear-gradient(90deg, rgba(124, 58, 237, 0.08) 0%, rgba(236, 72, 153, 0.08) 100%);
    border-bottom: none !important;
    color: var(--body-text-color, #111827) !important;
}
.comparison-row-highlight td:first-child {
    border-radius: 0.5rem 0 0 0.5rem;
}
.comparison-row-highlight td:last-child {
    border-radius: 0 0.5rem 0.5rem 0;
}
.comparison-caption {
    margin: 1.25rem 0 0 0 !important;
    font-size: 0.9rem !important;
    color: var(--body-text-color, #4b5563) !important;
    opacity: 0.85;
    line-height: 1.6 !important;
}

/* Chart wrapper — subtle frame around each plot */
.chart-wrap {
    background: var(--background-fill-secondary, rgba(124, 58, 237, 0.03));
    border: 1px solid var(--border-color-primary, rgba(124, 58, 237, 0.12));
    border-radius: 1rem !important;
    padding: 1rem !important;
    margin-top: 0.5rem;
}


/* ============================================================
   LIGHT MODE OVERRIDES
   Card backgrounds and borders are tuned for dark mode (low alpha
   over dark bg). On light backgrounds, those tints become invisible.
   This block bumps alpha + uses solid neutrals only when NOT in dark mode.
   ============================================================ */
body:not(.dark) .perf-metric-card {
    background: #ffffff !important;
    border: 1px solid rgba(124, 58, 237, 0.18) !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}
body:not(.dark) .perf-metric-card:hover {
    box-shadow: 0 16px 32px -12px rgba(124, 58, 237, 0.18);
}

body:not(.dark) .comparison-table-wrap {
    background: #ffffff !important;
    border: 1px solid rgba(124, 58, 237, 0.18) !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}
body:not(.dark) .comparison-table td {
    border-bottom: 1px solid rgba(0, 0, 0, 0.06) !important;
}

body:not(.dark) .chart-wrap {
    background: #ffffff !important;
    border: 1px solid rgba(124, 58, 237, 0.18) !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

body:not(.dark) .context-card-v2 {
    background: #ffffff !important;
    border: 1px solid rgba(124, 58, 237, 0.18) !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

body:not(.dark) .stage-card {
    background: #ffffff !important;
    border: 1px solid rgba(124, 58, 237, 0.18) !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

body:not(.dark) .result-placeholder {
    background: rgba(124, 58, 237, 0.025) !important;
    border-color: rgba(124, 58, 237, 0.25) !important;
}

body:not(.dark) .result-card-bonafide {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.06) 0%, #ffffff 100%) !important;
    border-color: rgba(16, 185, 129, 0.35) !important;
}
body:not(.dark) .result-card-spoof {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.06) 0%, #ffffff 100%) !important;
    border-color: rgba(239, 68, 68, 0.35) !important;
}

body:not(.dark) .result-card-probs {
    background: rgba(124, 58, 237, 0.03) !important;
}

body:not(.dark) .input-tabs > .tab-nav {
    background: rgba(124, 58, 237, 0.05) !important;
}
body:not(.dark) .input-tabs > .tab-nav button.selected {
    background: #ffffff !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}

body:not(.dark) .record-instructions {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.05) 0%, rgba(236, 72, 153, 0.05) 100%) !important;
    border-color: rgba(124, 58, 237, 0.25) !important;
}

body:not(.dark) .cta-section-v2 {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.06) 0%, rgba(236, 72, 153, 0.06) 100%) !important;
    border-color: rgba(124, 58, 237, 0.25) !important;
}

body:not(.dark) .hero-eyebrow {
    background: rgba(124, 58, 237, 0.08) !important;
    border-color: rgba(124, 58, 237, 0.3) !important;
}


/* ============================================================
   ARCHITECTURE DIAGRAMS — entrance animation, no pulsing
   ============================================================ */
.arch-diagram-wrap {
    margin: 1.5rem 0;
    padding: 1.5rem;
    background: var(--background-fill-secondary, rgba(124, 58, 237, 0.03));
    border: 1px solid var(--border-color-primary, rgba(124, 58, 237, 0.15));
    border-radius: 1rem;
    color: var(--body-text-color, #111827);
    overflow: hidden;
}
body:not(.dark) .arch-diagram-wrap {
    background: #ffffff !important;
    border: 1px solid rgba(124, 58, 237, 0.18) !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}
.arch-svg {
    display: block;
    max-width: 100%;
    height: auto;
}

/* Entrance animation — each element fades in + slides down slightly */
@keyframes archDrawIn {
    from {
        opacity: 0;
        transform: translateY(8px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
.arch-anim {
    animation: archDrawIn 0.5s ease-out both;
    transform-origin: center;
}

/* Hover effect — boxes brighten slightly */
.arch-svg rect {
    transition: opacity 0.2s ease, fill-opacity 0.2s ease;
}
.arch-svg g:hover rect {
    fill-opacity: 1;
}

/* Reduce motion — respect user preference */
@media (prefers-reduced-motion: reduce) {
    .arch-anim {
        animation: none;
        opacity: 1;
        transform: none;
    }
}


/* ============================================================
   PLAIN-LANGUAGE OVERFITTING SECTION
   ============================================================ */
.plain-card {
    background: var(--background-fill-secondary, rgba(124, 58, 237, 0.03));
    border: 1px solid var(--border-color-primary, rgba(124, 58, 237, 0.15));
    border-radius: 1rem;
    padding: 2rem 1.75rem;
    margin: 1rem 0;
}
body:not(.dark) .plain-card {
    background: #ffffff !important;
    border: 1px solid rgba(124, 58, 237, 0.18) !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.plain-card-eyebrow {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--brand-purple-400);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    background: rgba(124, 58, 237, 0.1);
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    margin-bottom: 0.75rem;
}
.plain-card-title {
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.015em !important;
    color: var(--body-text-color, #111827);
    margin-bottom: 1.25rem !important;
    line-height: 1.3;
}
.plain-card-body {
    font-size: 1rem !important;
    line-height: 1.7 !important;
    color: var(--body-text-color, #374151) !important;
    margin: 0 0 1rem 0 !important;
}
.plain-card-body:last-child {
    margin-bottom: 0 !important;
}
.plain-card-body strong {
    color: var(--body-text-color, #111827);
    font-weight: 600;
}

.analogy-diagram-wrap {
    margin: 1.5rem 0;
    padding: 1rem;
    background: rgba(124, 58, 237, 0.025);
    border-radius: 0.75rem;
    border: 1px solid rgba(124, 58, 237, 0.08);
}
body:not(.dark) .analogy-diagram-wrap {
    background: rgba(124, 58, 237, 0.02) !important;
    border-color: rgba(124, 58, 237, 0.1) !important;
}

/* Takeaway grid in Part 3 */
.takeaway-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin: 1rem 0 1.25rem 0;
}
@media (max-width: 768px) {
    .takeaway-grid { grid-template-columns: 1fr; }
}
.takeaway-item {
    display: flex;
    align-items: flex-start;
    gap: 0.85rem;
    padding: 1rem 1.1rem;
    border-radius: 0.75rem;
    border: 1px solid;
}
.takeaway-good {
    background: rgba(16, 185, 129, 0.06);
    border-color: rgba(16, 185, 129, 0.25);
}
.takeaway-warn {
    background: rgba(245, 158, 11, 0.06);
    border-color: rgba(245, 158, 11, 0.25);
}
body:not(.dark) .takeaway-good {
    background: rgba(16, 185, 129, 0.04) !important;
    border-color: rgba(16, 185, 129, 0.3) !important;
}
body:not(.dark) .takeaway-warn {
    background: rgba(245, 158, 11, 0.04) !important;
    border-color: rgba(245, 158, 11, 0.3) !important;
}
.takeaway-icon {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.95rem;
    font-weight: 700;
    flex-shrink: 0;
}
.takeaway-good .takeaway-icon {
    background: rgba(16, 185, 129, 0.18);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.35);
}
.takeaway-warn .takeaway-icon {
    background: rgba(245, 158, 11, 0.18);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.35);
}
.takeaway-body {
    font-size: 0.94rem;
    line-height: 1.55;
    color: var(--body-text-color, #374151);
}
.takeaway-body strong {
    color: var(--body-text-color, #111827);
    font-weight: 600;
}

/* Bottom quote */
.plain-card-bottom-quote {
    margin-top: 1.5rem;
    padding: 1.25rem 1.5rem;
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.08) 0%, rgba(236, 72, 153, 0.08) 100%);
    border-left: 3px solid var(--brand-purple-400);
    border-radius: 0 0.75rem 0.75rem 0;
    font-style: italic;
    font-size: 0.98rem;
    line-height: 1.6;
    color: var(--body-text-color, #374151);
}
body:not(.dark) .plain-card-bottom-quote {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.06) 0%, rgba(236, 72, 153, 0.06) 100%) !important;
}


/* ============================================================
   CONFIDENCE EXPLAINER (inside result card)
   ============================================================ */
.confidence-explainer {
    margin-top: 0.75rem;
    border-radius: 0.5rem;
    background: rgba(124, 58, 237, 0.04);
    border: 1px solid rgba(124, 58, 237, 0.12);
    overflow: hidden;
}
body:not(.dark) .confidence-explainer {
    background: rgba(124, 58, 237, 0.025) !important;
    border-color: rgba(124, 58, 237, 0.18) !important;
}

.confidence-explainer summary {
    cursor: pointer;
    padding: 0.65rem 0.9rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--brand-purple-400);
    list-style: none;
    user-select: none;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    transition: background 0.15s ease;
}
.confidence-explainer summary::-webkit-details-marker {
    display: none;
}
.confidence-explainer summary::before {
    content: '▸';
    font-size: 0.7rem;
    transition: transform 0.2s ease;
    opacity: 0.7;
}
.confidence-explainer[open] summary::before {
    transform: rotate(90deg);
}
.confidence-explainer summary:hover {
    background: rgba(124, 58, 237, 0.06);
}

.confidence-explainer-body {
    padding: 0 1rem 0.9rem 1rem;
    border-top: 1px solid rgba(124, 58, 237, 0.1);
    margin-top: 0.1rem;
}
.confidence-explainer-body p {
    margin: 0.85rem 0 0 0;
    font-size: 0.84rem;
    line-height: 1.6;
    color: var(--body-text-color, #374151);
    opacity: 0.92;
}
.confidence-explainer-body p:first-child {
    margin-top: 0.85rem;
}
.confidence-explainer-body strong {
    color: var(--body-text-color, #111827);
    font-weight: 600;
    opacity: 1;
}


/* Difficulty hint in result card meta line */
.meta-difficulty {
    font-style: italic;
    opacity: 0.85;
}


/* Verdict callout — direct answer */
.verdict-callout {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(245, 158, 11, 0.05) 100%);
    border-left: 4px solid #f59e0b;
    border-radius: 0 0.75rem 0.75rem 0;
    padding: 1.25rem 1.5rem;
    margin: 1rem 0 1.25rem 0;
}
body:not(.dark) .verdict-callout {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, rgba(245, 158, 11, 0.03) 100%) !important;
}
.verdict-line {
    margin: 0 !important;
    font-size: 1.05rem;
    line-height: 1.55;
    color: var(--body-text-color, #111827);
}
.verdict-line strong {
    color: var(--body-text-color, #111827);
    font-weight: 600;
}

/* Aim callout — what the project is for */
.aim-callout {
    margin-top: 1.5rem;
    padding: 1.5rem 1.6rem;
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.08) 0%, rgba(236, 72, 153, 0.08) 100%);
    border: 1px solid rgba(124, 58, 237, 0.25);
    border-radius: 0.875rem;
}
body:not(.dark) .aim-callout {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.05) 0%, rgba(236, 72, 153, 0.05) 100%) !important;
    border-color: rgba(124, 58, 237, 0.3) !important;
}
.aim-eyebrow {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--brand-purple-400);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.85rem;
}
.aim-body {
    margin: 0 0 0.85rem 0 !important;
    font-size: 0.96rem;
    line-height: 1.65;
    color: var(--body-text-color, #374151);
}
.aim-body:last-child {
    margin-bottom: 0 !important;
}
.aim-body strong {
    color: var(--body-text-color, #111827);
    font-weight: 600;
}

/* Note above example clips in Detector tab */
.examples-note {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.05) 0%, rgba(236, 72, 153, 0.05) 100%);
    border: 1px solid rgba(124, 58, 237, 0.18);
    border-radius: 0.625rem;
    padding: 0.85rem 1rem;
    font-size: 0.83rem;
    line-height: 1.55;
    color: var(--body-text-color, #374151);
    margin: 0.25rem 0 0.85rem 0;
}
body:not(.dark) .examples-note {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.04) 0%, rgba(236, 72, 153, 0.04) 100%) !important;
    border-color: rgba(124, 58, 237, 0.22) !important;
}
.examples-note strong {
    color: var(--body-text-color, #111827);
    font-weight: 600;
}


body:not(.dark) .verdict-callout {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(16, 185, 129, 0.03) 100%) !important;
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
                    
                    with gr.Tabs(elem_classes=["input-tabs"]):
                        with gr.Tab("Upload file"):
                            audio_upload = gr.Audio(
                                sources=["upload"],
                                type="filepath",
                                label="",
                                elem_classes=["audio-input-styled"],
                            )
                        with gr.Tab("Record mic"):
                            gr.HTML("""
                            <div class='record-instructions'>
                                <div class='record-instructions-icon'>🎤</div>
                                <div class='record-instructions-text'>
                                    <strong>Click the record button below</strong>, speak for 3 to 10 seconds, then click stop.
                                    A live waveform will show your audio being captured.
                                </div>
                            </div>
                            """)
                            audio_record = gr.Audio(
                                sources=["microphone"],
                                type="filepath",
                                label="",
                                format="wav",
                                show_download_button=True,
                                waveform_options=gr.WaveformOptions(
                                    waveform_color="#a78bfa",
                                    waveform_progress_color="#ec4899",
                                    show_recording_waveform=True,
                                    show_controls=True,
                                    skip_length=2,
                                    sample_rate=16000,
                                ),
                                elem_classes=["audio-input-styled", "audio-record-styled"],
                            )
                    
                    gr.HTML("<div class='step-label' style='margin-top: 1.25rem;'><span class='step-number'>2</span> Run the detector</div>")
                    analyze_btn = gr.Button("Analyze audio  →", variant="primary", size="lg", elem_classes=["analyze-button"])
                    
                    gr.HTML("<div class='step-label' style='margin-top: 1.5rem;'>Or try an example</div>")
                    gr.HTML("""
                    <div class='examples-note'>
                        <strong>Try all 5 examples in order</strong> — they go from easy to hardest.
                        You'll see the model handle easy cases confidently, become uncertain on medium
                        ones, and <strong>get the hardest one (A10) completely wrong</strong>. Why?
                        A10 uses Tacotron 2 + WaveRNN — a system so advanced that even human listeners
                        can't tell its output from real speech. The acoustic features literally overlap
                        with authentic speech, leaving our model (and any acoustic-feature-based
                        detector) with no signal to detect. We included this example so you can see
                        where the limits are, not just where it succeeds.
                    </div>
                    """)
                    gr.Examples(
                        examples=EXAMPLE_FILES,
                        inputs=audio_upload,
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
                fn=predict_audio_router,
                inputs=[audio_upload, audio_record],
                outputs=[badge_output, details_output, plot_output, raw_output],
            )
        
        
        # ============================================================
        # TAB 3: PERFORMANCE
        # ============================================================
        with gr.Tab("Performance", id=2):
            # Section header
            gr.HTML("""
            <div class='section-header' style='margin-top: 1rem;'>
                <div class='section-eyebrow'>Evaluation</div>
                <div class='section-title'>How well does the model actually perform?</div>
                <p class='detector-intro'>
                    Three datasets, two regimes (in-domain and out-of-domain), and full transparency about
                    where the model wins and where it struggles. Results are reported as Equal Error Rate (EER) —
                    lower is better.
                </p>
            </div>
            """)
            
            # Headline metric cards
            gr.HTML("""
            <div class='subsection-header'>
                <span class='subsection-eyebrow'>Headline results</span>
                <div class='subsection-title'>Three benchmarks at a glance</div>
            </div>
            """)
            
            with gr.Row():
                gr.HTML("""
                <div class='perf-metric-card perf-metric-good'>
                    <div class='perf-metric-tag'>In-domain</div>
                    <div class='perf-metric-value'>5.55%</div>
                    <div class='perf-metric-name'>ASVspoof 2019 LA</div>
                    <div class='perf-metric-detail'>Unseen attacks A07–A19</div>
                </div>
                """)
                gr.HTML("""
                <div class='perf-metric-card perf-metric-mid'>
                    <div class='perf-metric-tag'>Cross-dataset</div>
                    <div class='perf-metric-value'>9.09%</div>
                    <div class='perf-metric-name'>ASVspoof 2021 LA</div>
                    <div class='perf-metric-detail'>Codec-degraded audio</div>
                </div>
                """)
                gr.HTML("""
                <div class='perf-metric-card perf-metric-warn'>
                    <div class='perf-metric-tag'>Out-of-domain</div>
                    <div class='perf-metric-value'>26.33%</div>
                    <div class='perf-metric-name'>WaveFake</div>
                    <div class='perf-metric-detail'>Novel vocoder pipelines</div>
                </div>
                """)
            
            # Baseline comparison
            gr.HTML("""
            <div class='subsection-header' style='margin-top: 2.5rem;'>
                <span class='subsection-eyebrow'>Benchmark comparison</span>
                <div class='subsection-title'>How we compare to published baselines</div>
            </div>
            """)
            
            gr.HTML("""
            <div class='comparison-table-wrap'>
                <table class='comparison-table'>
                    <thead>
                        <tr>
                            <th>System</th>
                            <th>2019 LA EER</th>
                            <th>2021 LA EER</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td>Official LFCC-GMM baseline</td><td>8.09%</td><td>25.56%</td></tr>
                        <tr><td>Official CQCC-GMM baseline</td><td>9.57%</td><td>19.30%</td></tr>
                        <tr><td>Official LFCC-LCNN baseline</td><td>—</td><td>9.26%</td></tr>
                        <tr><td>Official RawNet2 baseline</td><td>—</td><td>9.50%</td></tr>
                        <tr class='comparison-row-highlight'>
                            <td><strong>This work (Wav2Vec 2.0)</strong></td>
                            <td><strong>5.55%</strong></td>
                            <td><strong>9.09%</strong></td>
                        </tr>
                    </tbody>
                </table>
                <p class='comparison-caption'>
                    Outperforms LFCC-GMM on 2019 LA by 2.54 pp and matches the strongest neural baselines
                    (LFCC-LCNN, RawNet2) on 2021 LA — without any codec-specific training augmentation.
                </p>
            </div>
            """)
            
            # Per-codec analysis
            gr.HTML("""
            <div class='subsection-header' style='margin-top: 3rem;'>
                <span class='subsection-eyebrow'>Codec robustness</span>
                <div class='subsection-title'>Performance by audio codec (ASVspoof 2021 LA)</div>
                <p class='subsection-caption'>
                    Real-world speech goes through codecs for transmission. The model handles modern codecs
                    well but struggles with aggressive cellular compression.
                </p>
            </div>
            """)
            with gr.Row():
                with gr.Column(elem_classes=["chart-wrap"]):
                    gr.Plot(value=make_per_codec_plot(), label=None)
            
            # Per-attack analysis
            gr.HTML("""
            <div class='subsection-header' style='margin-top: 3rem;'>
                <span class='subsection-eyebrow'>Attack-family robustness</span>
                <div class='subsection-title'>Performance by attack type (ASVspoof 2019 LA eval)</div>
                <p class='subsection-caption'>
                    13 different synthesis methods (A07–A19), all unseen during training. A10 is the
                    model's persistent weakness across both 2019 and 2021 evaluations.
                </p>
            </div>
            """)
            with gr.Row():
                with gr.Column(elem_classes=["chart-wrap"]):
                    gr.Plot(value=make_per_attack_plot(), label=None)
            
            # WaveFake story
            gr.HTML("""
            <div class='subsection-header' style='margin-top: 3rem;'>
                <span class='subsection-eyebrow'>Out-of-domain limits</span>
                <div class='subsection-title'>The WaveFake story — an honest negative result</div>
                <p class='subsection-caption'>
                    On WaveFake the model performs significantly worse, particularly on LJSpeech-based
                    vocoders (22–34% EER). WaveFake tests pure neural vocoder synthesis, while the model
                    was trained on ASVspoof's mix of TTS and voice-conversion attacks.
                    <br><br>
                    <strong>Interpretation:</strong> the model has learned ASVspoof-specific synthesis
                    artifacts, not universal vocoder detection. JSUT (Japanese) numbers look artificially
                    good because the bonafide examples are English LJSpeech — the model is partly detecting
                    language and domain, not spoofing artifacts. The LJSpeech-based numbers are the
                    methodologically meaningful results.
                </p>
            </div>
            """)
            with gr.Row():
                with gr.Column(elem_classes=["chart-wrap"]):
                    gr.Plot(value=make_wavefake_plot(), label=None)
            
            # ============================================================
            # Is the model overfit? (honest analysis)
            gr.HTML("""
            <div class='subsection-header' style='margin-top: 4rem;'>
                <span class='subsection-eyebrow'>Plain-language analysis</span>
                <div class='subsection-title'>So — is our model overfit?</div>
                <p class='subsection-caption'>
                    A fair question to ask of any deep learning model. We'll explain what overfitting is,
                    walk through what our numbers show, and give you a straight answer.
                </p>
            </div>
            
            <!-- PART 1: What is overfitting? -->
            <div class='plain-card'>
                <div class='plain-card-eyebrow'>Part 1</div>
                <div class='plain-card-title'>What is overfitting?</div>
                <p class='plain-card-body'>
                    Overfitting is when a model <strong>memorises specific examples</strong> instead of
                    <strong>learning general patterns</strong>. Sometimes called "rote learning" — the model gets very good at
                    recognising things that look like its training data, but anything that looks even slightly
                    different feels wrong to it and it gets confused.
                </p>
                <p class='plain-card-body'>
                    A good model learns the underlying signal. A deepfake detector should learn what makes a synthetic
                    voice sound synthetic — patterns that show up across many different fake-voice methods, not
                    just the specific ones it studied. If it only recognises fake voices that look exactly like the
                    ones it trained on, it has overfit.
                </p>
                <p class='plain-card-body'>
                    The way you spot overfitting is to test the model on examples it has never seen — and ideally on
                    examples that are <em>quite different</em> from what it trained on. If performance drops gracefully,
                    the model is generalising. If it falls off a cliff, the model has overfit.
                </p>
            </div>
            
            <!-- PART 2: Where does our model land? -->
            <div class='plain-card' style='margin-top: 1.5rem;'>
                <div class='plain-card-eyebrow'>Part 2</div>
                <div class='plain-card-title'>Where does our model actually land?</div>
                <p class='plain-card-body'>
                    We tested the detector on four progressively harder challenges. Each step further from what it
                    trained on tells us how well it generalises.
                </p>
                
                <div class='analogy-diagram-wrap'>
                    <svg width="100%" viewBox="0 0 680 360" role="img" xmlns="http://www.w3.org/2000/svg" class='arch-svg'>
                        <title>Model performance across four difficulty levels</title>
                        <desc>The detector's error rate increases as the test data moves further from what the model trained on.</desc>
                        <text x="40" y="32" font-size="12" fill="currentColor" opacity="0.65" font-weight="500">EASIER →→→→→→→→→→→→→→→→→→→→→→→→→→→ HARDER</text>
                        <line x1="40" y1="200" x2="640" y2="200" stroke="currentColor" stroke-width="0.8" opacity="0.4"/>
                        
                        <g class='arch-anim' style='animation-delay: 0.15s;'>
                            <circle cx="115" cy="200" r="9" fill="#10b981" stroke="#10b981" stroke-width="0.6"/>
                            <text x="115" y="186" text-anchor="middle" font-size="11" fill="#10b981" font-weight="700">0.69%</text>
                            <text x="115" y="172" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.6">error rate</text>
                            <text x="115" y="225" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">Familiar voices</text>
                            <text x="115" y="244" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7">Same examples it</text>
                            <text x="115" y="258" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7">trained on</text>
                            <text x="115" y="285" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.55">~1 wrong in 145</text>
                        </g>
                        <g class='arch-anim' style='animation-delay: 0.35s;'>
                            <circle cx="290" cy="200" r="11" fill="#10b981" stroke="#10b981" stroke-width="0.6" opacity="0.85"/>
                            <text x="290" y="186" text-anchor="middle" font-size="11" fill="#10b981" font-weight="700">5.55%</text>
                            <text x="290" y="172" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.6">error rate</text>
                            <text x="290" y="225" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">New fakes,</text>
                            <text x="290" y="240" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">same style</text>
                            <text x="290" y="258" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7">13 fake-voice methods</text>
                            <text x="290" y="272" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7">it had never heard</text>
                            <text x="290" y="299" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.55">~1 wrong in 18</text>
                        </g>
                        <g class='arch-anim' style='animation-delay: 0.55s;'>
                            <circle cx="465" cy="200" r="13" fill="#f59e0b" stroke="#f59e0b" stroke-width="0.6" opacity="0.9"/>
                            <text x="465" y="186" text-anchor="middle" font-size="11" fill="#f59e0b" font-weight="700">9.09%</text>
                            <text x="465" y="172" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.6">error rate</text>
                            <text x="465" y="225" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">Phone-quality</text>
                            <text x="465" y="240" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">audio</text>
                            <text x="465" y="258" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7">Compressed audio</text>
                            <text x="465" y="272" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7">like real phone calls</text>
                            <text x="465" y="299" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.55">~1 wrong in 11</text>
                        </g>
                        <g class='arch-anim' style='animation-delay: 0.75s;'>
                            <circle cx="615" cy="200" r="17" fill="#ef4444" stroke="#ef4444" stroke-width="0.6" opacity="0.9"/>
                            <text x="615" y="186" text-anchor="middle" font-size="11" fill="#ef4444" font-weight="700">26.33%</text>
                            <text x="615" y="172" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.6">error rate</text>
                            <text x="615" y="232" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">Brand new</text>
                            <text x="615" y="247" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">fake-voice tech</text>
                            <text x="615" y="265" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7">Made by a method</text>
                            <text x="615" y="279" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7">it never studied</text>
                            <text x="615" y="306" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.55">~1 wrong in 4</text>
                        </g>
                        <line x1="40" y1="335" x2="640" y2="335" stroke="currentColor" stroke-width="0.4" opacity="0.2"/>
                    </svg>
                </div>
                
                <p class='plain-card-body'>
                    Two things to notice. First — the model degrades gradually, not catastrophically. It doesn't go
                    from 0.69% to 50% (which would mean random guessing on anything new). That tells us it
                    <strong>did learn real patterns</strong>, not just memorise specific clips.
                </p>
                <p class='plain-card-body'>
                    Second — there's still a <strong>big gap</strong>. Going from 0.69% on familiar territory to 26.33%
                    on brand new fake-voice technology is a 38× jump. That's not catastrophic, but it's also not
                    great. The model clearly learned features that matter for the kinds of fake voices it studied —
                    and those features don't fully transfer to fake voices made by methods it has never seen.
                </p>
            </div>
            
            <!-- PART 3: The honest verdict -->
            <div class='plain-card' style='margin-top: 1.5rem;'>
                <div class='plain-card-eyebrow'>Part 3</div>
                <div class='plain-card-title'>The honest verdict</div>
                
                <div class='verdict-callout'>
                    <p class='verdict-line'>
                        <strong>The honest answer: it's a mix.</strong> The model learned real patterns
                        and generalises to most unseen attacks — but it has a genuine blind spot, and
                        its confidence can be dangerously high even when it's wrong.
                    </p>
                </div>
                
                <p class='plain-card-body'>
                    <strong>What works well:</strong> When tested on 13 fake-voice methods it had never
                    seen during training, it achieved a 5.55% error rate — roughly 94 out of 100 predictions
                    correct on completely new fakes. It becomes appropriately uncertain on medium-difficulty
                    attacks (66% confidence on A07). And it handles noisy, real-world audio without
                    false-alarming (93.7% confidence on a noisy real voice). These are signs of a model
                    that learned real anti-spoofing patterns, not just memorised its training data.
                </p>
                <p class='plain-card-body'>
                    <strong>What doesn't work:</strong> Two real problems. First, the model has a
                    <strong>complete blind spot for A10 attacks</strong> — it classifies the hardest
                    spoof example as "100% authentic," completely wrong. But there's a specific reason:
                    A10 is a Tacotron 2 + WaveRNN system whose output is so natural that even <strong>human
                    listeners cannot distinguish it from real speech</strong>. The ASVspoof 2019 paper
                    itself confirms that A10's acoustic features literally overlap with authentic speech
                    in feature space. Since our model relies on acoustic representations (Wav2Vec 2.0
                    features), it faces the same fundamental limit human ears do — there's no acoustic
                    signal to detect.
                </p>
                <p class='plain-card-body'>
                    Second, on the WaveFake dataset (modern neural vocoders like MelGAN
                    and HiFi-GAN — the same technology used in real-world voice cloning today), the error
                    rate jumps to 26.33%. These vocoders produce different artifacts from what the model
                    trained on. Since our project's goal is detecting AI voice cloning broadly, this is
                    a real coverage gap.
                </p>
                <p class='plain-card-body'>
                    <strong>What this means:</strong> The model is not classically "overfit" in the sense of
                    having memorised its training data — the 5.55% result on unseen attacks proves that. But
                    it does have <strong>limited coverage</strong>: it learned to detect certain types of
                    synthesis artifacts (the ones present in ASVspoof) and is blind to others (A10, neural
                    vocoders). For the project's stated goal of detecting AI voice cloning broadly, this is
                    a meaningful gap.
                </p>
                
                <div class='aim-callout'>
                    <div class='aim-eyebrow'>What our project actually demonstrates</div>
                    <p class='aim-body'>
                        <strong>1. Wav2Vec 2.0 features work for deepfake detection.</strong> Pretrained speech
                        representations carry strong anti-spoofing signal. With minimal fine-tuning (15% of the
                        model), we match or beat published neural baselines on the standard ASVspoof benchmarks.
                        This validates the transfer-learning approach.
                    </p>
                    <p class='aim-body'>
                        <strong>2. Single-corpus training has real limits — and we measured exactly where.</strong>
                        The A10 blind spot reveals a fundamental challenge: when a synthesis system produces
                        speech that is acoustically indistinguishable from real speech (even to humans),
                        acoustic-feature-based detection reaches its theoretical limit. The WaveFake results
                        show that cross-family generalization requires cross-family training data. Both findings
                        are concrete, measured, and reproducible.
                    </p>
                    <p class='aim-body'>
                        <strong>3. The path forward is clear.</strong> Universal AI voice cloning detection
                        requires multi-corpus, multi-family training — combining ASVspoof, WaveFake, and newer
                        datasets covering the latest synthesis methods. This project establishes the baseline
                        that such future work would build on, with measured evidence showing exactly where the
                        current approach succeeds and where it falls short.
                    </p>
                    <p class='aim-body'>
                        We chose to include the failures (A10, WaveFake) rather than hide them because honest
                        evaluation is more valuable than inflated numbers. A detector that reports 5.55% EER
                        with known blind spots is more useful than one that reports 5.55% EER and pretends it
                        works on everything.
                    </p>
                </div>
                
                <div class='plain-card-bottom-quote'>
                    "Treat this as a research demonstration of how Wav2Vec features behave for deepfake detection,
                    not a security tool. If you need to verify whether a real-world recording is a deepfake, no
                    single model — including this one — should be trusted as the final answer."
                </div>
            </div>
            """)
        
        
        # ============================================================
        # ============================================================
        # TAB 4: TECHNICAL
        # ============================================================
        with gr.Tab("Under the hood", id=3):
            gr.Markdown("## Architecture")
            
            gr.HTML("""
            <div class='arch-diagram-wrap'>
                <svg width="100%" viewBox="0 0 680 380" role="img" xmlns="http://www.w3.org/2000/svg" class='arch-svg'>
                    <title>Wav2Vec 2.0 architecture for deepfake detection</title>
                    <desc>Raw waveform feeds into a CNN feature encoder, then a 12-layer transformer stack, mean-pooled into a 768-dim embedding, then a linear classifier produces spoof and bonafide probabilities.</desc>
                    <defs>
                        <marker id="arch-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                            <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                        </marker>
                        <linearGradient id="arch-brand-grad" x1="0" y1="0" x2="1" y2="1">
                            <stop offset="0%" stop-color="#a78bfa" stop-opacity="0.9"/>
                            <stop offset="100%" stop-color="#ec4899" stop-opacity="0.9"/>
                        </linearGradient>
                    </defs>

                    <g class='arch-anim' style='animation-delay: 0.05s;'>
                        <path d="M30 165 L30 195 L34 188 L38 192 L42 175 L46 200 L50 168 L54 195 L58 178 L62 192 L66 170 L70 198 L74 165 L78 200 L82 172 L86 195 L86 165 Z" fill="#a78bfa" fill-opacity="0.4" stroke="#a78bfa" stroke-width="0.6"/>
                        <text x="58" y="225" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.85">Waveform</text>
                        <text x="58" y="240" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.6">16 kHz · 4 s</text>
                    </g>

                    <line x1="92" y1="180" x2="118" y2="180" stroke="currentColor" stroke-width="1.5" opacity="0.45" marker-end="url(#arch-arrow)" class='arch-anim' style='animation-delay: 0.2s;'/>

                    <g class='arch-anim' style='animation-delay: 0.3s;'>
                        <rect x="118" y="142" width="22" height="76" rx="3" fill="#a78bfa" fill-opacity="0.18" stroke="#a78bfa" stroke-width="0.6"/>
                        <rect x="142" y="148" width="22" height="64" rx="3" fill="#a78bfa" fill-opacity="0.28" stroke="#a78bfa" stroke-width="0.6"/>
                        <rect x="166" y="154" width="22" height="52" rx="3" fill="#a78bfa" fill-opacity="0.4" stroke="#a78bfa" stroke-width="0.6"/>
                        <rect x="190" y="160" width="22" height="40" rx="3" fill="#a78bfa" fill-opacity="0.55" stroke="#a78bfa" stroke-width="0.6"/>
                        <text x="165" y="232" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.85">CNN encoder</text>
                        <text x="165" y="247" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.6">7 conv layers</text>
                    </g>

                    <line x1="218" y1="180" x2="244" y2="180" stroke="currentColor" stroke-width="1.5" opacity="0.45" marker-end="url(#arch-arrow)" class='arch-anim' style='animation-delay: 0.45s;'/>

                    <g class='arch-anim' style='animation-delay: 0.55s;'>
                        <rect x="248" y="62" width="180" height="20" rx="3" fill="#7c3aed" fill-opacity="0.85" stroke="#7c3aed" stroke-width="0.5"/>
                        <text x="338" y="76" text-anchor="middle" font-size="12" fill="#ffffff" font-weight="500">LayerNorm</text>

                        <rect x="248" y="86" width="180" height="22" rx="3" fill="#a78bfa" fill-opacity="0.95" stroke="#a78bfa" stroke-width="0.5"/>
                        <text x="338" y="101" text-anchor="middle" font-size="12" fill="#ffffff" font-weight="500">Layer 12</text>

                        <rect x="248" y="112" width="180" height="22" rx="3" fill="#a78bfa" fill-opacity="0.9" stroke="#a78bfa" stroke-width="0.5"/>
                        <text x="338" y="127" text-anchor="middle" font-size="12" fill="#ffffff" font-weight="500">Layer 11</text>

                        <rect x="248" y="138" width="180" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <rect x="248" y="156" width="180" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <rect x="248" y="174" width="180" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <rect x="248" y="192" width="180" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <rect x="248" y="210" width="180" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <text x="338" y="220" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.6">Layers 10 — 6</text>

                        <rect x="248" y="228" width="180" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <rect x="248" y="246" width="180" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <rect x="248" y="264" width="180" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <rect x="248" y="282" width="180" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <rect x="248" y="300" width="180" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <text x="338" y="310" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.6">Layers 5 — 1</text>

                        <line x1="248" y1="62" x2="248" y2="314" stroke="#7c3aed" stroke-width="0.5" opacity="0.4"/>
                        <line x1="428" y1="62" x2="428" y2="314" stroke="#7c3aed" stroke-width="0.5" opacity="0.4"/>

                        <text x="338" y="335" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.75" font-weight="500">Transformer stack · 12 layers · 95M params</text>
                    </g>

                    <line x1="436" y1="180" x2="462" y2="180" stroke="currentColor" stroke-width="1.5" opacity="0.45" marker-end="url(#arch-arrow)" class='arch-anim' style='animation-delay: 0.75s;'/>

                    <g class='arch-anim' style='animation-delay: 0.85s;'>
                        <ellipse cx="486" cy="180" rx="22" ry="36" fill="#10b981" fill-opacity="0.22" stroke="#10b981" stroke-width="0.6"/>
                        <text x="486" y="178" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.85">Mean</text>
                        <text x="486" y="192" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.85">pool</text>
                        <text x="486" y="232" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.6">768-dim</text>
                    </g>

                    <line x1="510" y1="180" x2="536" y2="180" stroke="currentColor" stroke-width="1.5" opacity="0.45" marker-end="url(#arch-arrow)" class='arch-anim' style='animation-delay: 1.0s;'/>

                    <g class='arch-anim' style='animation-delay: 1.1s;'>
                        <rect x="540" y="156" width="100" height="48" rx="6" fill="url(#arch-brand-grad)" stroke="#7c3aed" stroke-width="0.6"/>
                        <text x="590" y="174" text-anchor="middle" font-size="12" fill="#ffffff" font-weight="500">Linear</text>
                        <text x="590" y="190" text-anchor="middle" font-size="12" fill="#ffffff">768 → 2</text>
                        <text x="590" y="225" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.6">P(spoof)</text>
                        <text x="590" y="240" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.6">P(bonafide)</text>
                    </g>

                    <g transform="translate(40, 290)" class='arch-anim' style='animation-delay: 1.3s;'>
                        <rect x="0" y="0" width="14" height="14" rx="3" fill="#a78bfa" fill-opacity="0.9"/>
                        <text x="20" y="11" font-size="12" fill="currentColor" opacity="0.7">Trainable in Stage 2 (14M params)</text>
                        <rect x="0" y="22" width="14" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32"/>
                        <text x="20" y="33" font-size="12" fill="currentColor" opacity="0.7">Frozen (81M params)</text>
                    </g>
                </svg>
            </div>
            """)
            
            gr.Markdown("## Two-stage training rationale")
            
            gr.HTML("""
            <div class='arch-diagram-wrap'>
                <svg width="100%" viewBox="0 0 680 460" role="img" xmlns="http://www.w3.org/2000/svg" class='arch-svg'>
                    <title>Two-stage fine-tuning strategy</title>
                    <desc>Stage 1 trains only the linear classification head with all transformer layers frozen, achieving 10.09% EER. Stage 2 unfreezes the top 2 transformer layers plus the final LayerNorm, achieving 0.69% EER.</desc>
                    <defs>
                        <linearGradient id="ft-head-grad" x1="0" y1="0" x2="1" y2="1">
                            <stop offset="0%" stop-color="#a78bfa" stop-opacity="0.95"/>
                            <stop offset="100%" stop-color="#ec4899" stop-opacity="0.95"/>
                        </linearGradient>
                    </defs>

                    <text x="170" y="32" text-anchor="middle" font-size="14" fill="currentColor" font-weight="500" class='arch-anim' style='animation-delay: 0.05s;'>Stage 1: head only</text>
                    <text x="170" y="50" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.65" class='arch-anim' style='animation-delay: 0.1s;'>1,538 trainable params</text>

                    <g class='arch-anim' style='animation-delay: 0.2s;'>
                        <rect x="100" y="70" width="140" height="20" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <text x="170" y="84" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.6">LayerNorm</text>
                        <rect x="100" y="94" width="140" height="20" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <text x="170" y="108" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.6">Layer 12</text>
                        <rect x="100" y="118" width="140" height="20" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <text x="170" y="132" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.6">Layer 11</text>
                        <rect x="100" y="142" width="140" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <rect x="100" y="160" width="140" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <rect x="100" y="178" width="140" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <rect x="100" y="196" width="140" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <rect x="100" y="214" width="140" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <text x="170" y="224" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.55">Layers 10 — 6</text>
                        <rect x="100" y="232" width="140" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <rect x="100" y="250" width="140" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <rect x="100" y="268" width="140" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <rect x="100" y="286" width="140" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <rect x="100" y="304" width="140" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <text x="170" y="314" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.55">Layers 5 — 1</text>
                        <rect x="100" y="322" width="140" height="20" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <text x="170" y="336" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.6">CNN encoder</text>
                    </g>

                    <g class='arch-anim' style='animation-delay: 0.4s;'>
                        <rect x="100" y="354" width="140" height="28" rx="4" fill="url(#ft-head-grad)" stroke="#7c3aed" stroke-width="0.6"/>
                        <text x="170" y="372" text-anchor="middle" font-size="12" fill="#ffffff" font-weight="500">Linear head</text>
                    </g>

                    <text x="170" y="408" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.7" class='arch-anim' style='animation-delay: 0.5s;'>Dev EER</text>
                    <text x="170" y="434" text-anchor="middle" font-size="22" fill="#a78bfa" font-weight="700" class='arch-anim' style='animation-delay: 0.55s;'>10.09%</text>

                    <line x1="320" y1="60" x2="320" y2="430" stroke="#9ca3af" stroke-width="0.4" stroke-dasharray="4 4" opacity="0.5"/>

                    <text x="510" y="32" text-anchor="middle" font-size="14" fill="currentColor" font-weight="500" class='arch-anim' style='animation-delay: 0.6s;'>Stage 2: top 2 layers + head</text>
                    <text x="510" y="50" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.65" class='arch-anim' style='animation-delay: 0.65s;'>14.18M trainable params</text>

                    <g class='arch-anim' style='animation-delay: 0.75s;'>
                        <rect x="440" y="70" width="140" height="20" rx="3" fill="#a78bfa" fill-opacity="0.95" stroke="#a78bfa" stroke-width="0.6"/>
                        <text x="510" y="84" text-anchor="middle" font-size="12" fill="#ffffff" font-weight="500">LayerNorm</text>
                        <rect x="440" y="94" width="140" height="20" rx="3" fill="#a78bfa" fill-opacity="0.95" stroke="#a78bfa" stroke-width="0.6"/>
                        <text x="510" y="108" text-anchor="middle" font-size="12" fill="#ffffff" font-weight="500">Layer 12</text>
                        <rect x="440" y="118" width="140" height="20" rx="3" fill="#a78bfa" fill-opacity="0.95" stroke="#a78bfa" stroke-width="0.6"/>
                        <text x="510" y="132" text-anchor="middle" font-size="12" fill="#ffffff" font-weight="500">Layer 11</text>
                        <rect x="440" y="142" width="140" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <rect x="440" y="160" width="140" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <rect x="440" y="178" width="140" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <rect x="440" y="196" width="140" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <rect x="440" y="214" width="140" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <text x="510" y="224" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.55">Layers 10 — 6</text>
                        <rect x="440" y="232" width="140" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <rect x="440" y="250" width="140" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <rect x="440" y="268" width="140" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <rect x="440" y="286" width="140" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <rect x="440" y="304" width="140" height="14" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <text x="510" y="314" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.55">Layers 5 — 1</text>
                        <rect x="440" y="322" width="140" height="20" rx="3" fill="#9ca3af" fill-opacity="0.32" stroke="#9ca3af" stroke-width="0.4"/>
                        <text x="510" y="336" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.6">CNN encoder</text>
                    </g>

                    <g class='arch-anim' style='animation-delay: 0.95s;'>
                        <rect x="440" y="354" width="140" height="28" rx="4" fill="url(#ft-head-grad)" stroke="#7c3aed" stroke-width="0.6"/>
                        <text x="510" y="372" text-anchor="middle" font-size="12" fill="#ffffff" font-weight="500">Linear head</text>
                    </g>

                    <text x="510" y="408" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.7" class='arch-anim' style='animation-delay: 1.05s;'>Dev EER</text>
                    <text x="510" y="434" text-anchor="middle" font-size="22" fill="#10b981" font-weight="700" class='arch-anim' style='animation-delay: 1.1s;'>0.69%</text>
                </svg>
            </div>
            """)
            
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
