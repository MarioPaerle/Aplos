"""
Vathos ModdedFormer Inspector — ModelLens
==========================================
Run with:
    streamlit run ModelLens.py

Usage in your script:
    from ModelLens import run_inspector
    run_inspector(model)
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, math, sys
from typing import Optional, Dict, List, Tuple

try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ModelLens · Vathos Inspector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
#  THEME DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────

THEMES = {
    "🟣 Claude Minimal": {
        "css": """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #0A0A12 !important; color: #E2E2F0;
    font-family: 'Inter', system-ui, sans-serif !important;
}
[data-testid="stSidebar"] {
    background-color: #0D0D18 !important;
    border-right: 1px solid rgba(167,139,250,0.15) !important;
}
[data-testid="stSidebar"] * { color: #C0B8F0 !important; }
h1 { color: #A78BFA !important; letter-spacing: -0.5px; font-weight: 600 !important; }
h2, h3 { color: #7C6FCD !important; font-weight: 500 !important; }
h4 { color: #9CA3AF !important; }
.stMetric label { color: #6B7280 !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
.stMetric [data-testid="stMetricValue"] { color: #A78BFA !important; font-size: 1.05rem !important; font-weight: 600; }
div[data-testid="stTabs"] button { color: #6B7280 !important; font-weight: 400; }
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #A78BFA !important; border-bottom: 2px solid #A78BFA !important; font-weight: 600;
}
.stSelectbox label, .stMultiSelect label, .stSlider label { color: #6B7280 !important; font-size: 0.78rem !important; }
.stExpander { border: 1px solid rgba(167,139,250,0.12) !important; border-radius: 8px !important; }
.stExpander summary { color: #7C6FCD !important; }
div[data-testid="stButton"] > button {
    background: transparent !important; border: 1px solid rgba(167,139,250,0.3) !important;
    color: #A78BFA !important; border-radius: 6px !important;
    transition: all 0.2s ease;
}
div[data-testid="stButton"] > button:hover {
    background: rgba(167,139,250,0.08) !important; border-color: #A78BFA !important;
}
div[data-testid="stButton"] > button[kind="primary"] {
    background: #A78BFA !important; color: #0A0A12 !important; border: none !important; font-weight: 600 !important;
}
.stAlert { border-radius: 6px !important; border-left-width: 3px !important; }
[data-testid="stDataFrame"] { border-radius: 8px !important; }
.badge { display:inline-block; padding:2px 9px; border-radius:999px; font-size:0.7rem; font-weight:600; margin-right:4px; letter-spacing:0.02em; }
.badge-attn  { background:rgba(124,111,205,0.15); color:#A78BFA; border:1px solid rgba(167,139,250,0.3); }
.badge-ffn   { background:rgba(52,211,153,0.1); color:#6EE7B7; border:1px solid rgba(52,211,153,0.25); }
.badge-skip  { background:rgba(251,113,133,0.1); color:#FCA5A5; border:1px solid rgba(251,113,133,0.25); }
.badge-zero  { background:rgba(251,191,36,0.1); color:#FCD34D; border:1px solid rgba(251,191,36,0.25); }
.badge-param { background:rgba(56,189,248,0.1); color:#7DD3FC; border:1px solid rgba(56,189,248,0.25); }
.badge-pca   { background:rgba(52,211,153,0.1); color:#6EE7B7; border:1px solid rgba(52,211,153,0.25); }
.tree-node   { background:rgba(167,139,250,0.04); border:1px solid rgba(167,139,250,0.12); border-radius:8px; padding:8px 14px; margin:3px 0; font-size:0.8rem; }
""",
        "rc": {
            "figure.facecolor": "#0A0A12", "axes.facecolor": "#0D0D18",
            "axes.edgecolor": "#1E1A30", "axes.labelcolor": "#9CA3AF",
            "xtick.color": "#6B7280", "ytick.color": "#6B7280",
            "text.color": "#E2E2F0", "grid.color": "#151520", "grid.alpha": 0.7,
            "legend.facecolor": "#0D0D18", "legend.edgecolor": "#1E1A30",
            "font.family": "sans-serif",
        },
        "colors": ["#A78BFA", "#60A5FA", "#34D399", "#F472B6", "#FB923C",
                   "#FBBF24", "#38BDF8", "#A3E635", "#E879F9", "#F87171"],
        "title": "🔬 ModelLens · Vathos Inspector",
        "accent": "#A78BFA",
    },

    "🟠 EVA Magi System": {
        "css": """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #020A04 !important; color: #39FF6A;
    font-family: 'Share Tech Mono', monospace !important;
    background-image:
        repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(57,255,106,0.03) 40px),
        repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(57,255,106,0.03) 40px) !important;
}
[data-testid="stSidebar"] {
    background-color: #010602 !important;
    border-right: 2px solid #1AFF5C !important;
    box-shadow: inset -8px 0 20px rgba(26,255,92,0.05) !important;
}
[data-testid="stSidebar"] * { color: #1AFF5C !important; font-family: 'Share Tech Mono', monospace !important; }
h1 {
    color: #FF6600 !important; font-family: 'Orbitron', monospace !important;
    font-weight: 900 !important; text-transform: uppercase !important;
    letter-spacing: 3px !important;
    text-shadow: 0 0 20px rgba(255,102,0,0.8), 0 0 40px rgba(255,102,0,0.4) !important;
}
h2, h3 {
    color: #FF8C00 !important; font-family: 'Share Tech Mono', monospace !important;
    text-transform: uppercase !important; letter-spacing: 2px !important;
    text-shadow: 0 0 10px rgba(255,140,0,0.5) !important;
    border-bottom: 1px solid rgba(255,102,0,0.3) !important; padding-bottom: 4px !important;
}
h4 { color: #39FF6A !important; letter-spacing: 1px !important; }
.stMetric label { color: #1AFF5C !important; font-size: 0.68rem !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; }
.stMetric [data-testid="stMetricValue"] {
    color: #FF6600 !important; font-size: 1.1rem !important; font-weight: 700 !important;
    font-family: 'Share Tech Mono', monospace !important;
    text-shadow: 0 0 12px rgba(255,102,0,0.6) !important;
}
div[data-testid="stTabs"] button {
    color: #39FF6A !important; font-family: 'Share Tech Mono', monospace !important;
    text-transform: uppercase !important; font-size: 0.72rem !important; letter-spacing: 0.08em !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #FF6600 !important; border-bottom: 2px solid #FF6600 !important;
    text-shadow: 0 0 8px rgba(255,102,0,0.6) !important;
}
.stSelectbox label, .stMultiSelect label, .stSlider label {
    color: #1AFF5C !important; text-transform: uppercase !important;
    font-size: 0.7rem !important; letter-spacing: 0.08em !important;
}
.stExpander {
    border: 1px solid rgba(26,255,92,0.3) !important;
    background: rgba(0,30,10,0.4) !important;
}
.stExpander summary { color: #FF8C00 !important; text-transform: uppercase !important; }
div[data-testid="stButton"] > button {
    background: transparent !important; border: 1px solid #1AFF5C !important;
    color: #1AFF5C !important; font-family: 'Share Tech Mono', monospace !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important;
    box-shadow: 0 0 8px rgba(26,255,92,0.2) !important;
    transition: all 0.15s !important;
}
div[data-testid="stButton"] > button:hover {
    background: rgba(26,255,92,0.1) !important;
    box-shadow: 0 0 16px rgba(26,255,92,0.4) !important;
}
div[data-testid="stButton"] > button[kind="primary"] {
    background: #FF6600 !important; color: #000 !important; border-color: #FF6600 !important;
    font-weight: 700 !important; box-shadow: 0 0 20px rgba(255,102,0,0.5) !important;
}
.stAlert { border-left: 3px solid #FF6600 !important; background: rgba(0,20,8,0.8) !important; }
.badge { display:inline-block; padding:2px 8px; border-radius:2px; font-size:0.68rem; font-weight:700; margin-right:4px;
         font-family:'Share Tech Mono',monospace; text-transform:uppercase; letter-spacing:0.06em; }
.badge-attn  { background:rgba(255,102,0,0.15); color:#FF8C00; border:1px solid #FF6600; }
.badge-ffn   { background:rgba(26,255,92,0.1); color:#1AFF5C; border:1px solid #1AFF5C; }
.badge-skip  { background:rgba(255,50,50,0.12); color:#FF6060; border:1px solid #FF4040; }
.badge-zero  { background:rgba(255,255,0,0.1); color:#FFFF00; border:1px solid #DDDD00; }
.badge-param { background:rgba(26,255,92,0.08); color:#39FF6A; border:1px solid rgba(26,255,92,0.5); }
.badge-pca   { background:rgba(26,255,92,0.08); color:#39FF6A; border:1px solid rgba(26,255,92,0.5); }
.tree-node   { background:rgba(0,30,10,0.6); border:1px solid rgba(26,255,92,0.25); border-left:3px solid #FF6600;
               border-radius:0 4px 4px 0; padding:8px 14px; margin:3px 0; font-size:0.78rem;
               font-family:'Share Tech Mono',monospace; }
""",
        "rc": {
            "figure.facecolor": "#020A04", "axes.facecolor": "#020D05",
            "axes.edgecolor": "#0A2A10", "axes.labelcolor": "#1AFF5C",
            "xtick.color": "#39FF6A", "ytick.color": "#39FF6A",
            "text.color": "#39FF6A", "grid.color": "#071A09", "grid.alpha": 0.8,
            "legend.facecolor": "#010602", "legend.edgecolor": "#1AFF5C",
            "font.family": "monospace",
        },
        "colors": ["#FF6600", "#1AFF5C", "#FF8C00", "#39FF6A", "#FF4040",
                   "#FFFF00", "#00FFCC", "#FF69B4", "#FF3300", "#CCFF00"],
        "title": "⬡ MAGI SYSTEM · MODEL ANALYSIS TERMINAL",
        "accent": "#FF6600",
    },

    "🔵 Jarvis HUD": {
        "css": """
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #020814 !important; color: #8FCFFF;
    font-family: 'Rajdhani', sans-serif !important;
    background-image:
        radial-gradient(ellipse at 20% 50%, rgba(0,140,255,0.04) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(0,200,255,0.03) 0%, transparent 50%) !important;
}
[data-testid="stSidebar"] {
    background-color: #010610 !important;
    border-right: 1px solid rgba(0,180,255,0.25) !important;
    background-image: linear-gradient(180deg, rgba(0,140,255,0.04) 0%, transparent 100%) !important;
    box-shadow: 4px 0 30px rgba(0,140,255,0.06) !important;
}
[data-testid="stSidebar"] * { color: #5BB8FF !important; font-family: 'Rajdhani', sans-serif !important; }
h1 {
    color: #00C8FF !important; font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 4px !important;
    text-shadow: 0 0 30px rgba(0,200,255,0.7), 0 0 60px rgba(0,140,255,0.3) !important;
    border-bottom: 1px solid rgba(0,200,255,0.2) !important; padding-bottom: 8px !important;
}
h2, h3 {
    color: #00A8E8 !important; font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 2px !important;
    text-shadow: 0 0 15px rgba(0,168,232,0.4) !important;
}
h4 { color: #5BB8FF !important; letter-spacing: 1px !important; font-weight: 500 !important; }
.stMetric label {
    color: #3A7CA5 !important; font-size: 0.7rem !important;
    text-transform: uppercase !important; letter-spacing: 0.12em !important; font-weight: 500 !important;
}
.stMetric [data-testid="stMetricValue"] {
    color: #00C8FF !important; font-size: 1.1rem !important; font-weight: 700 !important;
    font-family: 'Share Tech Mono', monospace !important;
    text-shadow: 0 0 15px rgba(0,200,255,0.5) !important;
}
div[data-testid="stTabs"] button {
    color: #3A7CA5 !important; text-transform: uppercase !important;
    font-size: 0.74rem !important; letter-spacing: 0.08em !important; font-weight: 500 !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00C8FF !important; border-bottom: 2px solid #00C8FF !important;
    text-shadow: 0 0 10px rgba(0,200,255,0.5) !important;
}
.stSelectbox label, .stMultiSelect label, .stSlider label {
    color: #3A7CA5 !important; text-transform: uppercase !important;
    font-size: 0.7rem !important; letter-spacing: 0.08em !important;
}
.stExpander {
    border: 1px solid rgba(0,168,232,0.2) !important;
    background: rgba(0,20,50,0.3) !important; border-radius: 4px !important;
}
.stExpander summary { color: #00A8E8 !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; }
div[data-testid="stButton"] > button {
    background: rgba(0,140,255,0.06) !important; border: 1px solid rgba(0,200,255,0.35) !important;
    color: #00C8FF !important; font-family: 'Rajdhani', sans-serif !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important; font-weight: 600 !important;
    border-radius: 2px !important; transition: all 0.2s !important;
    box-shadow: 0 0 8px rgba(0,140,255,0.15), inset 0 0 8px rgba(0,140,255,0.04) !important;
}
div[data-testid="stButton"] > button:hover {
    background: rgba(0,200,255,0.12) !important;
    box-shadow: 0 0 20px rgba(0,200,255,0.3), inset 0 0 12px rgba(0,200,255,0.06) !important;
}
div[data-testid="stButton"] > button[kind="primary"] {
    background: rgba(0,200,255,0.15) !important; border-color: #00C8FF !important;
    color: #00E8FF !important; font-weight: 700 !important;
    box-shadow: 0 0 25px rgba(0,200,255,0.4), inset 0 0 15px rgba(0,200,255,0.08) !important;
}
.stAlert {
    border-left: 2px solid #00A8E8 !important; background: rgba(0,20,50,0.6) !important;
    border-radius: 2px !important;
}
[data-testid="stDataFrame"] { border: 1px solid rgba(0,140,255,0.15) !important; border-radius: 4px !important; }
.badge { display:inline-block; padding:2px 10px; border-radius:2px; font-size:0.7rem; font-weight:600; margin-right:4px;
         font-family:'Rajdhani',sans-serif; text-transform:uppercase; letter-spacing:0.06em; }
.badge-attn  { background:rgba(0,200,255,0.12); color:#00C8FF; border:1px solid rgba(0,200,255,0.4); }
.badge-ffn   { background:rgba(0,255,180,0.1); color:#00FFCC; border:1px solid rgba(0,255,180,0.3); }
.badge-skip  { background:rgba(255,140,0,0.1); color:#FFA040; border:1px solid rgba(255,140,0,0.35); }
.badge-zero  { background:rgba(255,220,0,0.08); color:#FFE040; border:1px solid rgba(255,220,0,0.3); }
.badge-param { background:rgba(91,184,255,0.1); color:#5BB8FF; border:1px solid rgba(91,184,255,0.3); }
.badge-pca   { background:rgba(0,255,180,0.1); color:#00FFCC; border:1px solid rgba(0,255,180,0.3); }
.tree-node   { background:rgba(0,20,50,0.5); border:1px solid rgba(0,168,232,0.2);
               border-left:2px solid #00A8E8; border-radius:0 4px 4px 0;
               padding:8px 14px; margin:3px 0; font-size:0.8rem; }
""",
        "rc": {
            "figure.facecolor": "#020814", "axes.facecolor": "#030B18",
            "axes.edgecolor": "#0A2040", "axes.labelcolor": "#5BB8FF",
            "xtick.color": "#3A7CA5", "ytick.color": "#3A7CA5",
            "text.color": "#8FCFFF", "grid.color": "#071428", "grid.alpha": 0.7,
            "legend.facecolor": "#010610", "legend.edgecolor": "#0A3060",
            "font.family": "sans-serif",
        },
        "colors": ["#00C8FF", "#00FFCC", "#5BB8FF", "#FF8C00", "#FFE040",
                   "#FF6080", "#80FFCC", "#A0C8FF", "#FF4080", "#40E0FF"],
        "title": "◈ JARVIS · NEURAL ARCHITECTURE ANALYSIS",
        "accent": "#00C8FF",
    },
}


def _get_theme() -> str:
    return st.session_state.get("_theme", "🟣 Claude Minimal")


def _apply_theme(theme_name: str):
    """Inject CSS + update matplotlib RC + COLORS for the selected theme."""
    t = THEMES[theme_name]
    st.markdown(f"<style>{t['css']}</style>", unsafe_allow_html=True)
    plt.rcParams.update(t["rc"])
    return t["colors"]


# ──────────────────────────────────────────────────────────────────────────────
#  Apply selected theme immediately (called at module level so CSS is first)
# ──────────────────────────────────────────────────────────────────────────────
_ACTIVE_THEME = THEMES[st.session_state.get("_theme", "🟣 Claude Minimal")]
st.markdown(f"<style>{_ACTIVE_THEME['css']}</style>", unsafe_allow_html=True)
plt.rcParams.update(_ACTIVE_THEME["rc"])
COLORS: List[str] = _ACTIVE_THEME["colors"]



# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    return buf.getvalue()


def _tensor_stats(t: torch.Tensor) -> dict:
    f = t.detach().float().cpu()
    v = f.numpy().flatten()
    return {
        "shape": tuple(t.shape), "numel": t.numel(),
        "mean": float(v.mean()), "std": float(v.std()),
        "min": float(v.min()), "max": float(v.max()),
        "l2": float(np.linalg.norm(v)),
        "sparsity": float((np.abs(v) < 1e-6).mean()),
    }


def _svd_values(t: torch.Tensor) -> Optional[np.ndarray]:
    f = t.detach().float().cpu()
    if f.ndim == 2:
        try: return torch.linalg.svdvals(f).numpy()
        except Exception: return None
    if f.ndim == 3:
        results = []
        for i in range(f.shape[0]):
            try: results.append(torch.linalg.svdvals(f[i]).numpy())
            except Exception: pass
        return np.stack(results) if results else None
    return None


# ──────────────────────────────────────────────────────────────────────────────
#  PCA HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _pca_core(mat_np: np.ndarray, threshold: float = 0.99):
    """Shared SVD kernel. mat_np must be (n_observations, n_features)."""
    if mat_np.shape[0] < 2 or mat_np.shape[1] < 1:
        return 1, None
    mat_np = mat_np - mat_np.mean(axis=0, keepdims=True)
    try:
        _, s, _ = np.linalg.svd(mat_np, full_matrices=False)
        var = s ** 2; total = var.sum()
        if total < 1e-12: return 1, np.ones(len(s))
        cum = np.cumsum(var) / total
        n = int(np.searchsorted(cum, threshold)) + 1
        return min(n, len(s)), cum
    except Exception:
        return -1, None


def _activation_pca_99(act: torch.Tensor, threshold: float = 0.99) -> int:
    """
    Activation PCA — 'core features' as in the Value Residual Learning paper.
    act shape: (B, L, D) or (L, D) → reshaped to (B*L, D).
    Each *token position* is one observation; counts how many D-dimensional directions
    explain `threshold` of variance across the token population.
    Small value = representations are low-dimensional (good compression).
    """
    f = act.detach().float().cpu()
    if f.ndim == 3:
        B, L, D = f.shape; mat = f.reshape(B * L, D).numpy()
    elif f.ndim == 2:
        mat = f.numpy()
    else:
        return -1
    n, _ = _pca_core(mat, threshold)
    return n


def _activation_pca_curve(act: torch.Tensor) -> Optional[np.ndarray]:
    """Cumulative explained-variance curve for an activation tensor (token-population view)."""
    f = act.detach().float().cpu()
    if f.ndim == 3:
        B, L, D = f.shape; mat = f.reshape(B * L, D).numpy()
    elif f.ndim == 2:
        mat = f.numpy()
    else:
        return None
    _, cum = _pca_core(mat)
    return cum


def _pca_components_99(t: torch.Tensor, threshold: float = 0.99) -> int:
    """Weight-rank PCA (rows = output neurons). Measures weight rank utilisation.
    NOTE: for representation/activation dimensionality use _activation_pca_99()."""
    f = t.detach().float().cpu()
    if f.numel() < 2 or f.ndim < 2: return 1
    n, _ = _pca_core(f.reshape(f.shape[0], -1).numpy(), threshold)
    return n


def _pca_explained_curve(t: torch.Tensor) -> Optional[np.ndarray]:
    """Cumulative explained-variance curve for a weight tensor (weight-rank view)."""
    f = t.detach().float().cpu()
    if f.ndim < 2 or f.numel() < 2: return None
    _, cum = _pca_core(f.reshape(f.shape[0], -1).numpy())
    return cum


# ──────────────────────────────────────────────────────────────────────────────
#  SIMILARITY HELPERS  (new)
# ──────────────────────────────────────────────────────────────────────────────

def _cosine_sim_matrix(vectors: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    """Pairwise cosine similarity matrix from a dict of flat vectors."""
    labels = list(vectors.keys())
    n = len(labels)
    norm_vecs = []
    for k in labels:
        v = vectors[k].flatten().astype(np.float32)
        norm = np.linalg.norm(v)
        norm_vecs.append(v / (norm + 1e-12))
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            mat[i, j] = float(np.dot(norm_vecs[i], norm_vecs[j]))
    return mat, labels


def plot_similarity_heatmap(sim: np.ndarray, labels: List[str], title: str,
                             vmin: float = -1.0, vmax: float = 1.0) -> bytes:
    n = len(labels)
    cell = max(0.45, min(1.2, 14.0 / n))
    fig_w = max(6, n * cell + 1.5)
    fig_h = max(5, n * cell * 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(sim, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    fs = max(5, min(10, int(180 / n)))
    ax.set_xticklabels(labels, rotation=90, fontsize=fs)
    ax.set_yticklabels(labels, fontsize=fs)
    ax.set_title(title, color="#A78BFA", fontsize=10, pad=10)

    if n <= 20:
        for i in range(n):
            for j in range(n):
                val = sim[i, j]
                txt_color = "white" if abs(val) > 0.55 else "#C0C0C0"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=max(4, 7 - n // 5), color=txt_color)

    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    fig.tight_layout()
    return _fig_to_bytes(fig)


def _is_attention(mixer) -> bool:
    cls = type(mixer).__name__.lower()
    return "attention" in cls or "attn" in cls


def _collect_named_params(block) -> Dict[str, torch.Tensor]:
    return {name: p for name, p in block.named_parameters()}


def _get_attention_weights(mixer, x: torch.Tensor) -> Optional[np.ndarray]:
    if hasattr(mixer, "get_attention_weights"):
        try:
            with torch.no_grad():
                return mixer.get_attention_weights(x).detach().float().cpu().numpy()
        except Exception:
            pass
    try:
        with torch.no_grad():
            B, L, D = x.shape
            if hasattr(mixer, "qkv"):
                n_heads, head_dim = mixer.n_heads, mixer.head_dim
                qkv = mixer.qkv(x).view(B, L, 3, n_heads, head_dim)
                q, k, _ = qkv.unbind(dim=2)
                q, k = q.transpose(1, 2).float(), k.transpose(1, 2).float()
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
                if getattr(mixer, "causal", True):
                    mask = torch.tril(torch.ones(L, L, device=x.device)).bool()
                    scores = scores.masked_fill(~mask, float("-inf"))
                return torch.softmax(scores, dim=-1)[0].cpu().numpy()
            elif hasattr(mixer, "qk"):
                n_heads, head_dim = mixer.n_heads, mixer.head_dim
                qk = mixer.qk(x).view(B, L, 2, n_heads, head_dim)
                q, k = qk.unbind(dim=2)
                q, k = q.transpose(1, 2).float(), k.transpose(1, 2).float()
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
                if getattr(mixer, "causal", True):
                    mask = torch.tril(torch.ones(L, L, device=x.device)).bool()
                    scores = scores.masked_fill(~mask, float("-inf"))
                return torch.softmax(scores, dim=-1)[0].cpu().numpy()
    except Exception:
        pass
    return None


def _get_head_outputs(mixer, x: torch.Tensor) -> Optional[np.ndarray]:
    """
    Extracts the per-head output vectors BEFORE the out-projection:
        o_h = softmax(Q_h K_h^T / sqrt(d)) @ V_h      shape: (H, L, head_dim)

    This is the actual *information* each head produces — as opposed to the
    attention pattern (H, L, L) which only captures *where* heads look.
    Returned as float32 numpy array (H, L, head_dim) for batch item 0.
    """
    try:
        with torch.no_grad():
            B, L, D = x.shape
            x_f = x.float()

            if hasattr(mixer, "qkv"):
                n_heads, head_dim = mixer.n_heads, mixer.head_dim
                qkv = mixer.qkv(x_f).view(B, L, 3, n_heads, head_dim)
                q, k, v = qkv.unbind(dim=2)
                q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                # pos-emb if any
                if getattr(mixer, "pos_emb", None) is not None:
                    try: q, k = mixer.pos_emb(q, k, start_pos=0)
                    except Exception: pass
                scale = math.sqrt(head_dim)
                scores = torch.matmul(q, k.transpose(-2, -1)) / scale
                if getattr(mixer, "causal", True):
                    mask = torch.tril(torch.ones(L, L, device=x.device)).bool()
                    scores = scores.masked_fill(~mask, float("-inf"))
                attn_w = torch.softmax(scores, dim=-1)         # (B, H, L, L)
                head_out = torch.matmul(attn_w, v)             # (B, H, L, head_dim)
                return head_out[0].cpu().numpy()               # (H, L, head_dim)

            elif hasattr(mixer, "qk"):
                n_heads, head_dim = mixer.n_heads, mixer.head_dim
                qk = mixer.qk(x_f).view(B, L, 2, n_heads, head_dim)
                q, k = qk.unbind(dim=2)
                q, k = q.transpose(1, 2), k.transpose(1, 2)
                # V = x itself (NOV mixer: No-Output-Projection-Value)
                v = x_f.view(B, L, n_heads, head_dim).transpose(1, 2)
                if getattr(mixer, "pos_emb", None) is not None:
                    try: q, k = mixer.pos_emb(q, k, start_pos=0)
                    except Exception: pass
                scale = math.sqrt(head_dim)
                scores = torch.matmul(q, k.transpose(-2, -1)) / scale
                if getattr(mixer, "causal", True):
                    mask = torch.tril(torch.ones(L, L, device=x.device)).bool()
                    scores = scores.masked_fill(~mask, float("-inf"))
                attn_w = torch.softmax(scores, dim=-1)
                head_out = torch.matmul(attn_w, v)
                return head_out[0].cpu().numpy()

            elif hasattr(mixer, "q_proj") and hasattr(mixer, "kv_proj"):
                # GQA style (GroupedQueryAttention)
                n_heads = mixer.n_heads; head_dim = mixer.head_dim
                n_kv = mixer.n_kv_heads; n_rep = n_heads // n_kv
                q = mixer.q_proj(x_f).view(B, L, n_heads, head_dim).transpose(1, 2)
                kv = mixer.kv_proj(x_f).view(B, L, n_kv, 2, head_dim)
                k, v = kv.unbind(dim=3)
                k, v = k.transpose(1, 2), v.transpose(1, 2)
                if n_rep > 1:
                    k = k.unsqueeze(2).expand(B, n_kv, n_rep, L, head_dim).reshape(B, n_heads, L, head_dim)
                    v = v.unsqueeze(2).expand(B, n_kv, n_rep, L, head_dim).reshape(B, n_heads, L, head_dim)
                scale = math.sqrt(head_dim)
                scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / scale
                if getattr(mixer, "causal", True):
                    mask = torch.tril(torch.ones(L, L, device=x.device)).bool()
                    scores = scores.masked_fill(~mask, float("-inf"))
                attn_w = torch.softmax(scores, dim=-1)
                head_out = torch.matmul(attn_w, v.float())
                return head_out[0].cpu().numpy()

    except Exception:
        pass
    return None


def _get_ffn_activations(cm, x: torch.Tensor, temp: float) -> Optional[Dict[str, np.ndarray]]:
    try:
        with torch.no_grad():
            pre_act = cm.expand(x)
            post_act = cm.activation(pre_act) if hasattr(cm, "activation") else pre_act
            concept_attn = torch.softmax(pre_act / temp, dim=-1)
            return {
                "pre_act": pre_act[0].cpu().numpy(),
                "post_act": post_act[0].cpu().numpy(),
                "concept_attn": concept_attn[0].cpu().numpy(),
            }
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_weight_distribution(tensors: Dict[str, torch.Tensor], bins=80) -> bytes:
    n = len(tensors)
    cols = min(n, 3); rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    if n == 1: axes = np.array([[axes]])
    elif rows == 1: axes = axes.reshape(1, -1)
    for idx, (name, t) in enumerate(tensors.items()):
        ax = axes[idx // cols][idx % cols]
        v = t.detach().float().cpu().numpy().flatten()
        s = _tensor_stats(t)
        ax.hist(v, bins=bins, color=COLORS[idx % len(COLORS)], alpha=0.8, linewidth=0)
        ax.set_title(f"{name}\nμ={s['mean']:.3e}  σ={s['std']:.3e}", fontsize=8)
        ax.set_xlabel("value", fontsize=7); ax.set_ylabel("count", fontsize=7)
        ax.grid(True, ls="--", alpha=0.4); ax.tick_params(labelsize=7)
    for idx in range(n, rows * cols): axes[idx // cols][idx % cols].set_visible(False)
    fig.suptitle("Weight Distributions", fontsize=10, color="#A78BFA")
    fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_svd_spectrum(tensors: Dict[str, torch.Tensor], log_scale=True) -> Optional[bytes]:
    eligible = {k: v for k, v in tensors.items() if v.ndim >= 2}
    if not eligible: return None
    n = len(eligible); cols = min(n, 3); rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    if n == 1: axes = np.array([[axes]])
    elif rows == 1: axes = axes.reshape(1, -1)
    for idx, (name, t) in enumerate(eligible.items()):
        ax = axes[idx // cols][idx % cols]
        sv = _svd_values(t)
        if sv is None:
            ax.text(0.5, 0.5, "SVD failed", ha="center", va="center", transform=ax.transAxes); continue
        if sv.ndim == 2:
            for h in range(sv.shape[0]): ax.plot(sv[h], color=COLORS[idx % len(COLORS)], alpha=0.3, lw=0.8)
            ax.plot(sv.mean(0), color=COLORS[idx % len(COLORS)], lw=2, label="mean"); ax.legend(fontsize=7)
        else:
            ax.plot(sv, color=COLORS[idx % len(COLORS)], lw=1.5, marker=".", markersize=3)
        cond = sv.flatten()[0] / (sv.flatten()[-1] + 1e-12)
        ax.set_title(f"{name}\ncond={cond:.1f}  rank≈{int((sv.flatten()>sv.flatten()[0]*0.01).sum())}", fontsize=8)
        ax.set_xlabel("index", fontsize=7); ax.set_ylabel("σ", fontsize=7)
        if log_scale: ax.set_yscale("log")
        ax.grid(True, ls="--", which="both", alpha=0.4); ax.tick_params(labelsize=7)
    for idx in range(n, rows * cols): axes[idx // cols][idx % cols].set_visible(False)
    fig.suptitle("Singular Value Spectra", fontsize=10, color="#A78BFA")
    fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_gradient_distribution(tensors: Dict[str, torch.Tensor], bins=80) -> Optional[bytes]:
    g = {k: v.grad for k, v in tensors.items() if v.grad is not None}
    return plot_weight_distribution(g, bins=bins) if g else None


def plot_gradient_svd(tensors: Dict[str, torch.Tensor], log_scale=True) -> Optional[bytes]:
    g = {k: v.grad for k, v in tensors.items() if v.grad is not None and v.grad.ndim >= 2}
    return plot_svd_spectrum(g, log_scale=log_scale) if g else None


def plot_heatmap(data: np.ndarray, title: str, xlabel: str, ylabel: str,
                 tokens: Optional[List[str]] = None, cmap: str = "magma") -> bytes:
    rows, cols = data.shape
    fig_w = max(8, cols * 0.15) if tokens else 10
    fig_h = max(6, rows * 0.3) if tokens else 6
    fig, ax = plt.subplots(figsize=(min(fig_w, 20), min(fig_h, 15)))
    im = ax.imshow(data, aspect="auto", cmap=cmap)
    ax.set_title(title, color="#A78BFA", pad=20); ax.set_ylabel(ylabel); ax.set_xlabel(xlabel)
    if tokens and len(tokens) == rows:
        clean = [t.replace('Ġ', ' ') for t in tokens]
        fs = max(4, min(10, int(400 / rows)))
        ax.set_yticks(range(rows)); ax.set_yticklabels(clean, fontsize=fs)
        if rows == cols:
            ax.set_xticks(range(cols)); ax.set_xticklabels(clean, rotation=90, fontsize=fs)
    fig.colorbar(im, ax=ax); fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_attention_weights(attn_weights: np.ndarray, tokens: Optional[List[str]] = None) -> bytes:
    if attn_weights.ndim == 2: attn_weights = attn_weights[np.newaxis]
    H, L_q, L_k = attn_weights.shape
    cols = min(H, 4); rows = math.ceil(H / cols)
    if tokens is not None:
        cell = max(0.2, min(0.6, 12.0 / L_q))
        fig_w = max(4 * cols, cell * L_q * cols); fig_h = max(3.5 * rows, cell * L_q * rows + 1)
    else:
        fig_w = 4 * cols; fig_h = 3.5 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.array(axes).flatten() if H > 1 else [axes]
    clean_tokens = [t.replace('Ġ', ' ') for t in tokens] if tokens else None
    for h in range(H):
        ax = axes[h]; w = attn_weights[h]
        im = ax.imshow(w, aspect="auto", cmap="magma", vmin=0, vmax=w.max()+1e-9)
        ax.set_title(f"Head {h}", fontsize=9, color="#C4B5FD")
        if clean_tokens and len(clean_tokens) == L_q:
            fs = max(4, min(10, int(300 / L_q)))
            ax.set_xticks(range(L_k)); ax.set_yticks(range(L_q))
            ax.set_xticklabels(clean_tokens, rotation=90, fontsize=fs, color="#9CA3AF")
            ax.set_yticklabels(clean_tokens, fontsize=fs, color="#9CA3AF")
        else:
            ax.set_xlabel("Key (Source)", fontsize=7); ax.set_ylabel("Query (Target)", fontsize=7)
            ax.tick_params(labelsize=6)
        fig.colorbar(im, ax=ax, shrink=0.8)
    for h in range(H, len(axes)): axes[h].set_visible(False)
    fig.suptitle("Attention Topology Analysis", fontsize=11, color="#A78BFA", y=1.02)
    fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_activation_distribution(pre_act: np.ndarray, post_act: np.ndarray,
                                  act_fn: Optional[nn.Module]) -> bytes:
    """
    Three-panel activation analysis:
      Panel 1 — f(x) curve: the activation function drawn over its mathematical domain
      Panel 2 — Input distribution overlaid on f(x): p(x) histogram coloured by f(x) sign/magnitude
      Panel 3 — Output distribution p(f(x)): sparsity check with dead-neuron annotation

    All three share the same x-domain so the mapping is visually obvious.
    """
    C = COLORS  # pick up current theme palette
    pre_flat  = pre_act.flatten()
    post_flat = post_act.flatten()

    # Domain: cover actual range plus a comfortable margin, at least [-6, 6]
    data_min, data_max = float(pre_flat.min()), float(pre_flat.max())
    x_min = min(-6.0, data_min - abs(data_min) * 0.1 - 0.5)
    x_max = max( 6.0, data_max + abs(data_max) * 0.1 + 0.5)
    x_dense = np.linspace(x_min, x_max, 2000)

    # ── Evaluate f(x) analytically, fall back to empirical scatter ────────────
    y_dense = x_dense.copy()          # identity fallback
    fn_label = "identity (fallback)"
    fn_success = False
    if act_fn is not None:
        try:
            dev = "cpu"
            if hasattr(act_fn, "parameters") and list(act_fn.parameters()):
                dev = next(act_fn.parameters()).device
            with torch.no_grad():
                xt = torch.tensor(x_dense, dtype=torch.float32, device=dev)
                yt = act_fn(xt)
            y_dense = yt.cpu().numpy()
            fn_label = type(act_fn).__name__
            fn_success = True
        except Exception:
            pass

    if not fn_success:
        # Empirical: sort pre by value, map to post
        sort_idx = np.argsort(pre_flat)
        # subsample to ≤ 5000 pts
        step = max(1, len(sort_idx) // 5000)
        x_emp = pre_flat[sort_idx[::step]]
        y_emp = post_flat[sort_idx[::step]]
        fn_label = "empirical (sorted scatter)"

    # ── Figure: 1 row × 3 panels ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    accent = C[0] if C else "#A78BFA"
    inp_col = C[3] if len(C) > 3 else "#F472B6"
    out_col = C[2] if len(C) > 2 else "#34D399"
    dead_col = "#EF4444"

    # ── Panel 1: f(x) curve only ──────────────────────────────────────────────
    ax1 = axes[0]
    if fn_success:
        ax1.plot(x_dense, y_dense, color=accent, lw=2.5, label=f"$f(x)$ — {fn_label}", zorder=4)
    else:
        ax1.scatter(x_emp, y_emp, color=accent, s=2, alpha=0.3, label="empirical mapping")
        ax1.plot(x_emp, y_emp, color=accent, lw=0.5, alpha=0.2)

    ax1.axvline(0, color="#4B5563", lw=1.2, ls="--", alpha=0.6)
    ax1.axhline(0, color="#4B5563", lw=1.2, ls="--", alpha=0.6)
    ax1.set_xlim(x_min, x_max)
    ax1.set_xlabel("$x$ (pre-activation)", fontsize=9)
    ax1.set_ylabel("$f(x)$", fontsize=9)
    ax1.set_title(f"Activation Function  $f(x)$\n{fn_label}", color=accent, fontsize=10)
    ax1.legend(fontsize=8); ax1.grid(True, ls="--", alpha=0.3)

    # Annotate derivative sign regions (negative slope = gradient kill zone for ReLU)
    if fn_success:
        neg_mask = y_dense < 0
        if neg_mask.any():
            ax1.fill_between(x_dense, y_dense, 0,
                             where=neg_mask, alpha=0.08, color=dead_col, label="f(x)<0")

    # ── Panel 2: f(x) + input distribution overlaid ───────────────────────────
    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    # f(x) curve (same as panel 1 but lighter)
    if fn_success:
        ax2.plot(x_dense, y_dense, color=accent, lw=2.0, alpha=0.9, zorder=4)
    ax2.axvline(0, color="#4B5563", lw=1, ls="--", alpha=0.5)
    ax2.axhline(0, color="#4B5563", lw=1, ls="--", alpha=0.5)
    ax2.set_xlim(x_min, x_max)
    ax2.set_xlabel("$x$ (pre-activation)", fontsize=9)
    ax2.set_ylabel("$f(x)$", color=accent, fontsize=9)
    ax2.tick_params(axis='y', labelcolor=accent)

    # Input distribution histogram on twin axis
    ax2_twin.hist(pre_flat, bins=200, range=(x_min, x_max),
                  color=inp_col, alpha=0.45, density=True, zorder=2)
    ax2_twin.set_ylabel("Density  $p(x)$", color=inp_col, fontsize=9)
    ax2_twin.tick_params(axis='y', labelcolor=inp_col)

    # Mark actual data min/max
    ax2.axvline(data_min, color=dead_col, lw=1, ls=":", alpha=0.7)
    ax2.axvline(data_max, color=dead_col, lw=1, ls=":", alpha=0.7)
    ax2.text(data_min, ax2.get_ylim()[1] * 0.95 if ax2.get_ylim()[1] != 0 else 1,
             f"{data_min:.2f}", color=dead_col, fontsize=7, ha='center', va='top')
    ax2.text(data_max, ax2.get_ylim()[1] * 0.95 if ax2.get_ylim()[1] != 0 else 1,
             f"{data_max:.2f}", color=dead_col, fontsize=7, ha='center', va='top')

    ax2.set_title("$f(x)$ + Input Distribution  $p(x)$\n"
                  f"range [{data_min:.2f}, {data_max:.2f}]", color=accent, fontsize=10)

    # ── Panel 3: output distribution ──────────────────────────────────────────
    ax3 = axes[2]
    dead_frac = float((np.abs(post_flat) < 1e-6).mean())
    out_mean  = float(post_flat.mean())
    out_std   = float(post_flat.std())

    # Use percentile clip to avoid extreme outlier bins swamping the plot
    p01, p999 = np.percentile(post_flat, [0.1, 99.9])
    ax3.hist(post_flat, bins=150, range=(p01, max(p999, 1e-6)),
             color=out_col, alpha=0.75, density=True)
    ax3.axvline(0,        color="#9CA3AF", lw=1.2, ls="--", alpha=0.7, label="0")
    ax3.axvline(out_mean, color=accent,   lw=1.5, ls="--", alpha=0.9, label=f"μ={out_mean:.3f}")

    ax3.set_xlabel("$f(x)$ (post-activation)", fontsize=9)
    ax3.set_ylabel("Density", fontsize=9)
    ax3.set_title(f"Output Distribution  $p(f(x))$\n"
                  f"dead={dead_frac:.1%}  μ={out_mean:.3f}  σ={out_std:.3f}",
                  color=out_col, fontsize=10)
    ax3.legend(fontsize=8); ax3.grid(True, ls="--", alpha=0.3)

    # Dead-neuron bar annotation
    if dead_frac > 0.01:
        ax3.text(0.97, 0.96, f"💀 {dead_frac:.1%} dead",
                 transform=ax3.transAxes, ha='right', va='top',
                 fontsize=9, color=dead_col,
                 bbox=dict(fc='none', ec=dead_col, lw=0.8, pad=3))

    fig.tight_layout(pad=2.0)
    return _fig_to_bytes(fig)


def plot_comparison_svd(layer_data, param_key, log_scale=True) -> bytes:
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, (lname, params) in enumerate(layer_data.items()):
        if param_key not in params: continue
        sv = _svd_values(params[param_key])
        if sv is None: continue
        if sv.ndim == 2: sv = sv.mean(0)
        ax.plot(sv, label=lname, color=COLORS[idx % len(COLORS)], lw=1.5, marker=".", markersize=3, alpha=0.85)
    ax.set_xlabel("singular value index"); ax.set_ylabel("σ")
    ax.set_title(f"SVD comparison: {param_key}", color="#A78BFA")
    if log_scale: ax.set_yscale("log")
    ax.legend(fontsize=8); ax.grid(True, ls="--", which="both", alpha=0.4)
    fig.tight_layout(); return _fig_to_bytes(fig)


def plot_comparison_hist(layer_data, param_key, bins=60) -> bytes:
    fig, ax = plt.subplots(figsize=(9, 4))
    for idx, (lname, params) in enumerate(layer_data.items()):
        if param_key not in params: continue
        v = params[param_key].detach().float().cpu().numpy().flatten()
        ax.hist(v, bins=bins, alpha=0.45, label=lname, color=COLORS[idx % len(COLORS)], linewidth=0)
    ax.set_xlabel("value"); ax.set_ylabel("count")
    ax.set_title(f"Distribution comparison: {param_key}", color="#A78BFA")
    ax.legend(fontsize=8); ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout(); return _fig_to_bytes(fig)


def plot_skip_lambdas(model) -> Optional[bytes]:
    lambdas = {k: float(v.detach()) for k, v in model.skip_lambdas.items()}
    if not lambdas: return None
    fig, ax = plt.subplots(figsize=(max(6, len(lambdas) * 1.4), 4))
    keys = list(lambdas.keys()); vals = [lambdas[k] for k in keys]
    bars = ax.bar(range(len(keys)), vals, color=COLORS[:len(keys)], alpha=0.85)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace("route_","").replace("_to_","→") for k in keys], rotation=30, ha="right", fontsize=8)
    ax.axhline(0, color="#9CA3AF", lw=0.8, ls="--")
    ax.set_title("Skip-connection Gate Values (λ)", color="#A78BFA"); ax.set_ylabel("λ (gate)")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.003, f"{v:.4f}", ha="center", va="bottom", fontsize=7, color="#D8D8F0")
    fig.tight_layout(); return _fig_to_bytes(fig)


def plot_zeroskip_params(model) -> Optional[bytes]:
    if not model.zeroskip: return None
    vals = [float(p.detach()) for p in model.zeroskip_params]
    fig, ax = plt.subplots(figsize=(max(6, len(vals)*0.7), 4))
    ax.plot(vals, marker="o", color="#FBBF24", lw=1.5, markersize=6)
    ax.axhline(0, color="#9CA3AF", lw=0.8, ls="--"); ax.set_xlabel("layer index"); ax.set_ylabel("zeroskip α")
    ax.set_title("ZeroSkip Parameters", color="#A78BFA"); ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout(); return _fig_to_bytes(fig)


def plot_param_norms(model) -> bytes:
    layer_names, l2_norms = [], []
    for i, block in enumerate(model.blocks):
        for name, p in block.named_parameters():
            layer_names.append(f"L{i}.{name}"); l2_norms.append(float(p.detach().norm(2)))
    fig, ax = plt.subplots(figsize=(max(8, len(layer_names)*0.35), 5))
    ax.bar(range(len(layer_names)), l2_norms, color="#60A5FA", alpha=0.8)
    ax.set_xticks(range(len(layer_names))); ax.set_xticklabels(layer_names, rotation=90, fontsize=5.5)
    ax.set_ylabel("L2 norm"); ax.set_title("Per-parameter L2 Norms (all blocks)", color="#A78BFA")
    ax.grid(True, ls="--", alpha=0.4, axis="y"); fig.tight_layout(); return _fig_to_bytes(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def render_model_overview(model):
    st.markdown("## 🏗️ Model Overview")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    has_grad = any(p.grad is not None for p in model.parameters())

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Layers", model.n_layer); c2.metric("Embed dim", model.embed_dim)
    c3.metric("Total params", f"{total:,}"); c4.metric("Trainable", f"{trainable:,}")
    c5.metric("Has gradients", "✅" if has_grad else "❌")
    c6,c7,c8,c9,c10 = st.columns(5)
    c6.metric("Vocab size", model.vocab_size); c7.metric("Max len", model.max_len)
    c8.metric("Attn layers", sum(1 for b in model.blocks if _is_attention(b.spatial_mixer)))
    c9.metric("Skip conns", sum(s is not None for s in model.skips))
    c10.metric("ZeroSkip", "✅" if model.zeroskip else "❌")

    st.markdown("### Layer Graph")
    st.markdown(_build_layer_graph_html(model), unsafe_allow_html=True)

    if any(s is not None for s in model.skips) and model.skip_lambdas:
        st.markdown("### Skip-Connection Gate Values")
        img = plot_skip_lambdas(model)
        if img: st.image(img)
    if model.zeroskip:
        st.markdown("### ZeroSkip Coefficients")
        img = plot_zeroskip_params(model)
        if img: st.image(img)
    st.markdown("### Parameter L2 Norms")
    st.image(plot_param_norms(model))


def _build_layer_graph_html(model) -> str:
    rows = ['<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%;font-size:0.78rem">']
    rows.append("<tr>"
        "<th style='padding:6px 10px;text-align:left;color:#818CF8'>Layer</th>"
        "<th style='padding:6px 10px;color:#818CF8'>d_model</th>"
        "<th style='padding:6px 10px;color:#818CF8'>FFN dim</th>"
        "<th style='padding:6px 10px;color:#818CF8'>Spatial mixer</th>"
        "<th style='padding:6px 10px;color:#818CF8'>Skip from</th>"
        "<th style='padding:6px 10px;color:#818CF8'>Skip to</th>"
        "<th style='padding:6px 10px;color:#818CF8'>Params</th>"
        "<th style='padding:6px 10px;color:#818CF8'>PCA 99%</th></tr>")
    for i, block in enumerate(model.blocks):
        sm_name = type(block.spatial_mixer).__name__
        is_attn = _is_attention(block.spatial_mixer)
        badge = f'<span class="badge badge-{"attn" if is_attn else "ffn"}">{sm_name}</span>'
        skip_from = skip_to = ""
        if model.skips[i] is not None:
            skip_to = f'<span class="badge badge-skip">→ L{model.skips[i]}</span>'
        for src, tgt in enumerate(model.skips):
            if tgt == i: skip_from = f'<span class="badge badge-skip">L{src} →</span>'
        n_params = sum(p.numel() for p in block.parameters())
        # Quick PCA for channel mixer expand weight
        pca_str = "—"
        cm = block.channel_mixer
        if hasattr(cm, "expand"):
            pca_str = str(_pca_components_99(cm.expand.weight))
        bg = "#13132A" if i % 2 == 0 else "#0D0D1A"
        rows.append(
            f"<tr style='background:{bg}'>"
            f"<td style='padding:5px 10px;color:#A78BFA;font-weight:700'>Layer {i}</td>"
            f"<td style='padding:5px 10px;text-align:center'>{model.d_models[i]}</td>"
            f"<td style='padding:5px 10px;text-align:center'>{model.M_dims[i]}</td>"
            f"<td style='padding:5px 10px'>{badge}</td>"
            f"<td style='padding:5px 10px'>{skip_from}</td>"
            f"<td style='padding:5px 10px'>{skip_to}</td>"
            f"<td style='padding:5px 10px'><span class='badge badge-param'>{n_params:,}</span></td>"
            f"<td style='padding:5px 10px'><span class='badge badge-pca'>{pca_str}</span></td>"
            "</tr>")
    rows.append("</table></div>")
    return "".join(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER INSPECTOR  (modified to include PCA)
# ══════════════════════════════════════════════════════════════════════════════

def render_layer_inspector(model):
    st.markdown("## 🔍 Layer Inspector")
    layer_idx = st.selectbox("Select layer", list(range(model.n_layer)),
                             format_func=lambda i: f"Layer {i}  —  {type(model.blocks[i].spatial_mixer).__name__}")
    block = model.blocks[layer_idx]
    sm, cm = block.spatial_mixer, block.channel_mixer

    tags = []
    if _is_attention(sm): tags.append('<span class="badge badge-attn">Attention</span>')
    if model.skips[layer_idx] is not None:
        tags.append(f'<span class="badge badge-skip">→ skip L{model.skips[layer_idx]}</span>')
    for src, tgt in enumerate(model.skips):
        if tgt == layer_idx: tags.append(f'<span class="badge badge-skip">skip from L{src}</span>')
    if model.zeroskip:
        zval = float(model.zeroskip_params[layer_idx].detach())
        tags.append(f'<span class="badge badge-zero">zeroskip α={zval:.4f}</span>')
    st.markdown(" ".join(tags), unsafe_allow_html=True)

    comp_choice = st.radio("Component", ["Spatial Mixer (SM)", "Channel Mixer (FFN)", "Both"],
                           horizontal=True, key="layer_comp_choice")
    selected_modules = ({"spatial": sm} if comp_choice == "Spatial Mixer (SM)"
                        else {"channel": cm} if comp_choice == "Channel Mixer (FFN)"
                        else {"spatial": sm, "channel": cm})

    all_params: Dict[str, torch.Tensor] = {}
    for prefix, module in selected_modules.items():
        for name, p in module.named_parameters():
            all_params[f"{prefix}.{name}"] = p

    if not all_params:
        st.info("No parameters found in this component."); return

    tabs = st.tabs(["📊 Distributions", "📉 SVD Spectrum", "🌊 Gradients",
                    "🌊 Gradient SVD", "👁️ Attention Weights", "📋 Stats Table"])

    with tabs[0]:
        st.image(plot_weight_distribution(all_params))
    with tabs[1]:
        log_s = st.toggle("Log scale", value=True, key=f"svd_log_{layer_idx}")
        img = plot_svd_spectrum(all_params, log_scale=log_s)
        st.image(img) if img else st.info("No 2-D parameters to plot SVD for.")
    with tabs[2]:
        if not any(p.grad is not None for p in all_params.values()):
            st.warning("No gradients available.")
        else:
            img = plot_gradient_distribution(all_params)
            if img: st.image(img)
    with tabs[3]:
        if not any(p.grad is not None for p in all_params.values()):
            st.warning("No gradients available.")
        else:
            log_gs = st.toggle("Log scale", value=True, key=f"gsvd_log_{layer_idx}")
            img = plot_gradient_svd(all_params, log_scale=log_gs)
            st.image(img) if img else st.info("Gradients are 1-D or unavailable.")
    with tabs[4]:
        if not _is_attention(sm):
            st.info("Spatial mixer is not an attention layer.")
        else:
            seq_len = st.slider("Sequence length (dummy)", 8, 256, 32, key=f"attn_seq_{layer_idx}")
            if st.button("Compute attention weights", key=f"attn_btn_{layer_idx}"):
                dummy_x = torch.randn(1, seq_len, model.d_models[layer_idx])
                sm.eval()
                w = _get_attention_weights(sm, dummy_x)
                if w is None: st.error("Could not extract attention weights.")
                else:
                    st.image(plot_attention_weights(w))
                    st.caption(f"Shape: {w.shape}  |  max={w.max():.4f}  min={w.min():.4f}")
    with tabs[5]:
        import pandas as pd
        rows = []
        for pname, p in all_params.items():
            s = _tensor_stats(p)
            sv = _svd_values(p)
            eff_rank = int((sv.flatten() > sv.flatten()[0]*0.01).sum()) if sv is not None else "—"
            pca = _pca_components_99(p)
            rows.append({
                "Parameter": pname, "Shape": str(s["shape"]), "Numel": f"{s['numel']:,}",
                "Mean": f"{s['mean']:.4e}", "Std": f"{s['std']:.4e}",
                "Min": f"{s['min']:.4e}", "Max": f"{s['max']:.4e}",
                "L2": f"{s['l2']:.4e}", "Sparsity": f"{s['sparsity']:.2%}",
                "PCA 99%": pca if pca > 0 else "N/A",
                "Eff. Rank": eff_rank,
                "Has grad": "✅" if p.grad is not None else "❌",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-LAYER COMPARISON  (modified to include PCA)
# ══════════════════════════════════════════════════════════════════════════════

def render_comparison(model):
    st.markdown("## ⚖️ Multi-Layer Comparison")
    layers_sel = st.multiselect("Select layers to compare", list(range(model.n_layer)),
                                default=list(range(min(model.n_layer, 4))),
                                format_func=lambda i: f"Layer {i}")
    if len(layers_sel) < 2: st.info("Select at least 2 layers."); return

    layer_data: Dict[str, Dict[str, torch.Tensor]] = {}
    all_param_keys: set = set()
    for i in layers_sel:
        block = model.blocks[i]; params = {}
        for prefix, mod in [("sm", block.spatial_mixer), ("ffn", block.channel_mixer)]:
            for name, p in mod.named_parameters(): params[f"{prefix}.{name}"] = p
        layer_data[f"L{i}"] = params; all_param_keys.update(params.keys())

    common_keys = [k for k in sorted(all_param_keys) if all(k in layer_data[f"L{i}"] for i in layers_sel)]
    if not common_keys: st.warning("No common parameter names across selected layers."); return

    param_key = st.selectbox("Parameter to compare", common_keys)
    mode = st.radio("Compare by", ["SVD Spectrum", "Distribution"], horizontal=True, key="compare_mode")
    log_s = st.toggle("Log scale", value=True, key="cmp_log")

    img = (plot_comparison_svd(layer_data, param_key, log_scale=log_s)
           if mode == "SVD Spectrum" else plot_comparison_hist(layer_data, param_key))
    st.image(img)

    st.markdown("#### Per-layer stats for selected parameter")
    import pandas as pd
    rows = []
    for lname, params in layer_data.items():
        if param_key not in params: continue
        s = _tensor_stats(params[param_key])
        pca = _pca_components_99(params[param_key])
        sv = _svd_values(params[param_key])
        eff_rank = int((sv.flatten() > sv.flatten()[0]*0.01).sum()) if sv is not None else None
        rows.append({"Layer": lname, "numel": s["numel"], "mean": round(s["mean"],6),
                     "std": round(s["std"],6), "l2": round(s["l2"],4),
                     "sparsity": round(s["sparsity"],4), "PCA 99%": pca, "Eff. Rank": eff_rank})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ATTENTION EXPLORER  (unchanged except PCA metric added)
# ══════════════════════════════════════════════════════════════════════════════

def render_attention_explorer(model):
    st.markdown("## 🧠 Manifold Explorer")
    attn_layers = [i for i, b in enumerate(model.blocks) if _is_attention(b.spatial_mixer)]
    if not attn_layers: st.info("No attention layers found."); return

    layer_idx = st.selectbox("Seleziona Layer", attn_layers,
                             format_func=lambda i: f"Layer {i} ({type(model.blocks[i].spatial_mixer).__name__})")
    block = model.blocks[layer_idx]
    sm, cm = block.spatial_mixer, block.channel_mixer
    d = model.d_models[layer_idx]

    st.markdown(f"**Mixer:** `{type(sm).__name__}`  |  **n_heads:** `{getattr(sm,'n_heads','?')}`  |  "
                f"**head_dim:** `{getattr(sm,'head_dim','?')}`  |  **causal:** `{getattr(sm,'causal','?')}`")

    input_mode = st.radio("Test Strategy", ["Synthetic Pattern (Dummy)", "Real Text (HuggingFace Tokenizer)"],
                          horizontal=True)

    if input_mode == "Synthetic Pattern (Dummy)":
        seq_len = st.slider("Sequence length", 8, 512, 64)
        pattern = st.selectbox("Input pattern", ["random", "ones", "zeros", "eye"])
        if st.button("Analyse"):
            x_map = {"random": torch.randn, "ones": torch.ones, "zeros": torch.zeros}
            if pattern in x_map: x = x_map[pattern](1, seq_len, d)
            else:
                x = torch.zeros(1, seq_len, d)
                for j in range(min(seq_len, d)): x[0, j, j] = 1.0
            sm.eval(); cm.eval()
            st.session_state["ffn_explorer_cache"] = {
                "layer_idx": layer_idx, "spatial_x": x, "channel_x": x,
                "tokens": None, "mode": input_mode, "d_model": d}

    elif input_mode == "Real Text (HuggingFace Tokenizer)":
        if not HAS_TRANSFORMERS: st.error("`transformers` not found."); return
        c1, c2 = st.columns([1, 2])
        tok_id = c1.text_input("Tokenizer HF Repo", "gpt2")
        use_bos = c1.checkbox("Add BOS token", value=True)
        prompt = c2.text_area("Prompt", "The mathematical structure of a transformer is an associative memory.")
        if st.button("Analyse with Hook"):
            with st.spinner("Running forward hooks..."):
                try:
                    tokenizer = transformers.AutoTokenizer.from_pretrained(tok_id)
                    full = (tokenizer.bos_token if use_bos and tokenizer.bos_token else "") + prompt
                    inputs = tokenizer(full, return_tensors="pt")
                    dev = next(model.parameters()).device
                    input_ids = inputs["input_ids"].to(dev)
                    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                except Exception as e:
                    st.error(f"Tokenizer error: `{e}`"); return
                captured = {"spatial_x": None, "channel_x": None}
                h1 = sm.register_forward_pre_hook(lambda m,i: captured.__setitem__("spatial_x", i[0].detach()))
                h2 = cm.register_forward_pre_hook(lambda m,i: captured.__setitem__("channel_x", i[0].detach()))
                try:
                    model.eval()
                    with torch.no_grad(): model(input_ids)
                except Exception as e:
                    st.error(f"Forward pass error: `{e}`")
                finally:
                    h1.remove(); h2.remove()
                if captured["spatial_x"] is None: st.error("Hook failed."); return
                st.session_state["ffn_explorer_cache"] = {
                    "layer_idx": layer_idx, "spatial_x": captured["spatial_x"],
                    "channel_x": captured["channel_x"], "tokens": tokens,
                    "mode": input_mode, "d_model": d}

    cache = st.session_state.get("ffn_explorer_cache")
    if cache and cache["layer_idx"] == layer_idx and cache["mode"] == input_mode:
        tab_attn, tab_head_sim, tab_core, tab_ffn = st.tabs([
            "👁️ Spatial Attention (QK)",
            "🔗 Head Similarity (on input)",
            "🔵 Core Features per Layer",
            "💎 Channel Expansion (XW)"])

        with tab_attn:
            w_attn = _get_attention_weights(sm, cache["spatial_x"])
            head_outputs = _get_head_outputs(sm, cache["spatial_x"])
            cache["w_attn"] = w_attn
            cache["head_outputs"] = head_outputs          # (H, L, head_dim) numpy or None
            _render_attention_output(w_attn, tokens=cache["tokens"])

        with tab_head_sim:
            w_attn_cached = cache.get("w_attn")
            w_attn_hs = w_attn_cached if w_attn_cached is not None else _get_attention_weights(sm, cache["spatial_x"])
            head_outputs = cache.get("head_outputs")
            _render_head_similarity(w_attn_hs, head_outputs, layer_idx)

        with tab_core:
            _render_core_features_per_layer(model, cache["spatial_x"])

        with tab_ffn:
            default_temp = float(math.sqrt(cache["d_model"]))
            temp_scale = st.slider("Temperature τ", 0.1, max(100.0, default_temp*2), default_temp, 0.1)
            ffn_data = _get_ffn_activations(cm, cache["channel_x"], temp_scale)
            if ffn_data is not None:
                view_mode = st.radio("Matrix Projection",
                    ["FFN Activations $Act(X W_{expand})$","Raw Logits $X W_{expand}$",
                     "Concept Attention $Softmax(X W_{expand}/τ)$"], horizontal=True, key="view_mode_ffn")
                mat = (ffn_data["post_act"] if "Act" in view_mode
                       else ffn_data["pre_act"] if "Raw" in view_mode else ffn_data["concept_attn"])
                title = ("FFN Activations" if "Act" in view_mode
                         else "Raw Concept Logits" if "Raw" in view_mode else "Concept Attention")
                sparsity = float((np.abs(mat) < 1e-6).mean())
                l2_norm = float(np.linalg.norm(mat))
                try:
                    svd_vals = np.linalg.svd(mat, compute_uv=False)
                    eff_rank = int(np.sum(svd_vals > svd_vals[0] * 1e-3))
                except np.linalg.LinAlgError:
                    svd_vals = None; eff_rank = "N/A"
                pca_act = _activation_pca_99(torch.tensor(mat))  # token-population PCA
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Sparsity", f"{sparsity:.2%}"); c2.metric("L2 Norm", f"{l2_norm:.2f}")
                c3.metric("Eff. Rank (SVD>0.1%)", eff_rank)
                c4.metric("🔵 Core features (PCA 99%)", pca_act)
                display_n = min(128, mat.shape[1])
                st.image(plot_heatmap(mat[:, :display_n], f"{title} (first {display_n} of {mat.shape[1]})",
                                      "Neurons", "Tokens", cache["tokens"]))
                act_fn = getattr(cm, "activation", None)
                img_act = plot_activation_distribution(ffn_data["pre_act"], ffn_data["post_act"], act_fn)
                if img_act: st.image(img_act)
                if svd_vals is not None:
                    fig_svd, ax_svd = plt.subplots(figsize=(6, 2.5))
                    ax_svd.plot(svd_vals, color=COLORS[1], lw=1.5, marker=".", markersize=3)
                    ax_svd.set_yscale("log"); ax_svd.set_title("SVD Spectrum", color="#A78BFA", fontsize=10)
                    ax_svd.set_xlabel("σ index", fontsize=8); ax_svd.grid(True, ls="--", alpha=0.4)
                    fig_svd.tight_layout(); st.image(_fig_to_bytes(fig_svd))
            else:
                st.warning("Cannot extract FFN activations.")


def _render_attention_output(w, tokens=None):
    if w is None: st.error("Cannot compute weights for this mixer type."); return
    st.image(plot_attention_weights(w, tokens=tokens))
    w_calc = w if w.ndim == 3 else w[np.newaxis]
    entropies = []
    for h in range(w_calc.shape[0]):
        wh = w_calc[h] + 1e-12
        entropies.append(float(-(wh * np.log(wh)).sum(axis=-1).mean()))
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(len(entropies)), entropies, color=COLORS)
    ax.set_xlabel("Head index"); ax.set_ylabel("Avg Entropy (nats)")
    ax.set_title("Per-head Shannon Entropy $\\mathbb{E}[\\mathcal{H}]$", color="#A78BFA")
    ax.grid(True, ls="--", alpha=0.4, axis="y"); fig.tight_layout()
    st.image(_fig_to_bytes(fig))


def _render_head_similarity(w_attn, head_outputs, layer_idx: int):
    """
    Two complementary views of inter-head similarity:

    1. Attention-Pattern Similarity
       Compare the (L×L) attention distribution of each head.
       Answers: "do heads look at the same positions?"

    2. Head-Output Similarity (new — on actual forward-pass values)
       Compare the (L × head_dim) output tensor of each head BEFORE the
       out-projection:  o_h = softmax(QK^T/√d) @ V_h
       Answers: "do heads produce similar *representations*?"
       This is the meaningful redundancy check — two heads that attend
       differently but write the same subspace to the residual stream are
       still redundant.
    """
    has_patterns = w_attn is not None
    has_outputs  = head_outputs is not None  # (H, L, head_dim)

    if not has_patterns and not has_outputs:
        st.error("No attention data available — run an analysis first.")
        return

    sub_tabs = []
    if has_patterns: sub_tabs.append("🗺️ Attention Pattern Similarity")
    if has_outputs:  sub_tabs.append("💡 Head Output Similarity")
    tabs = st.tabs(sub_tabs)
    tab_ptr = 0

    # ── 1. Attention-Pattern Similarity ───────────────────────────────────────
    if has_patterns:
        with tabs[tab_ptr]:
            tab_ptr += 1
            w_calc = w_attn if w_attn.ndim == 3 else w_attn[np.newaxis]   # (H, L, L)
            H, L_q, L_k = w_calc.shape

            st.caption(
                f"Layer {layer_idx} · {H} heads · attention pattern ({L_q}×{L_k}) "
                f"each flattened to {L_q*L_k}-d vector before cosine sim.")

            if H < 2:
                st.info("Only 1 head — no pairwise similarity.");
            else:
                head_vecs = {f"H{h}": w_calc[h].flatten() for h in range(H)}
                sim_pat, labels = _cosine_sim_matrix(head_vecs)
                off_pat = sim_pat[~np.eye(H, dtype=bool)].flatten()

                vmin = max(-1.0, float(off_pat.min()) - 0.02)
                st.image(plot_similarity_heatmap(
                    sim_pat, labels,
                    f"L{layer_idx} — Attention-Pattern Similarity",
                    vmin=vmin, vmax=1.0))

                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Mean", f"{off_pat.mean():.4f}")
                c2.metric("Max",  f"{off_pat.max():.4f}")
                c3.metric("Min",  f"{off_pat.min():.4f}")
                c4.metric("Std",  f"{off_pat.std():.4f}")

                fig, ax = plt.subplots(figsize=(7, 2.8))
                ax.hist(off_pat, bins=max(10, H*3), color=COLORS[0], alpha=0.8, density=True)
                ax.axvline(off_pat.mean(), color=COLORS[3], lw=1.5, ls="--",
                           label=f"μ = {off_pat.mean():.3f}")
                ax.set_title("Off-diagonal distribution (attention patterns)", color="#A78BFA")
                ax.set_xlabel("Cosine similarity"); ax.set_ylabel("Density")
                ax.legend(fontsize=8); ax.grid(True, ls="--", alpha=0.4)
                fig.tight_layout(); st.image(_fig_to_bytes(fig))

                if off_pat.mean() > 0.80:
                    st.warning(f"⚠️  High pattern similarity ({off_pat.mean():.3f}) — "
                               f"heads may be **looking at similar positions**.")
                elif off_pat.mean() < 0.30:
                    st.success(f"✅  Low pattern similarity ({off_pat.mean():.3f}) — "
                               f"heads attend to **diverse** positions.")

    # ── 2. Head-Output Similarity ─────────────────────────────────────────────
    if has_outputs:
        with tabs[tab_ptr]:
            H, L, head_dim = head_outputs.shape  # (H, L, head_dim)

            st.caption(
                f"Layer {layer_idx} · {H} heads · head output shape "
                f"({L}×{head_dim}) — `o_h = softmax(QKᵀ/√d) @ V_h` before out-proj.")

            sim_mode = st.radio(
                "Similarity space",
                ["Full output sequence (L × head_dim flattened)",
                 "Mean-pooled over tokens (head_dim only)",
                 "Per-token then averaged"],
                horizontal=False, key=f"hout_sim_mode_{layer_idx}")

            if sim_mode.startswith("Full"):
                head_vecs = {f"H{h}": head_outputs[h].flatten() for h in range(H)}
                sim_title = "Head Output Similarity (full sequence flattened)"
            elif sim_mode.startswith("Mean"):
                head_vecs = {f"H{h}": head_outputs[h].mean(0) for h in range(H)}
                sim_title = "Head Output Similarity (mean-pooled over tokens)"
            else:
                # average cosine sim computed token-by-token
                per_tok_sims = np.zeros((H, H), dtype=np.float32)
                for t in range(L):
                    vecs = {f"H{h}": head_outputs[h, t] for h in range(H)}
                    s, _ = _cosine_sim_matrix(vecs)
                    per_tok_sims += s
                per_tok_sims /= L
                labels_out = [f"H{h}" for h in range(H)]
                off_out = per_tok_sims[~np.eye(H, dtype=bool)].flatten()
                vmin = max(-1.0, float(off_out.min()) - 0.02)
                st.image(plot_similarity_heatmap(
                    per_tok_sims, labels_out,
                    f"L{layer_idx} — Head Output Similarity (avg over {L} tokens)",
                    vmin=vmin, vmax=1.0))
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Mean", f"{off_out.mean():.4f}")
                c2.metric("Max",  f"{off_out.max():.4f}")
                c3.metric("Min",  f"{off_out.min():.4f}")
                c4.metric("Std",  f"{off_out.std():.4f}")
                if off_out.mean() > 0.80:
                    st.warning(f"⚠️  High output similarity ({off_out.mean():.3f}) — "
                               f"heads write **redundant representations** to the residual stream.")
                elif off_out.mean() < 0.30:
                    st.success(f"✅  Low output similarity ({off_out.mean():.3f}) — "
                               f"heads write **diverse** representations.")
                # Extra: per-head L2 norm bar (output magnitude)
                norms = np.linalg.norm(head_outputs, axis=-1).mean(1)   # (H,)
                fig, ax = plt.subplots(figsize=(max(5, H*0.7), 2.8))
                ax.bar(range(H), norms, color=COLORS[:H] if H <= len(COLORS) else COLORS*10)
                ax.set_xlabel("Head index"); ax.set_ylabel("Mean L2 norm")
                ax.set_title("Per-head mean output magnitude", color="#A78BFA")
                ax.grid(True, ls="--", alpha=0.4, axis="y")
                fig.tight_layout(); st.image(_fig_to_bytes(fig))
                return  # per-token path fully rendered above

            # Shared path for Full / Mean-pooled
            if H < 2:
                st.info("Only 1 head — no pairwise similarity."); return

            sim_out, labels_out = _cosine_sim_matrix(head_vecs)
            off_out = sim_out[~np.eye(H, dtype=bool)].flatten()
            vmin = max(-1.0, float(off_out.min()) - 0.02)

            st.image(plot_similarity_heatmap(
                sim_out, labels_out, f"L{layer_idx} — {sim_title}",
                vmin=vmin, vmax=1.0))

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Mean", f"{off_out.mean():.4f}")
            c2.metric("Max",  f"{off_out.max():.4f}")
            c3.metric("Min",  f"{off_out.min():.4f}")
            c4.metric("Std",  f"{off_out.std():.4f}")

            fig, ax = plt.subplots(figsize=(7, 2.8))
            ax.hist(off_out, bins=max(10, H*3), color=COLORS[1], alpha=0.8, density=True)
            ax.axvline(off_out.mean(), color=COLORS[3], lw=1.5, ls="--",
                       label=f"μ = {off_out.mean():.3f}")
            ax.set_title("Off-diagonal distribution (head outputs)", color="#A78BFA")
            ax.set_xlabel("Cosine similarity"); ax.set_ylabel("Density")
            ax.legend(fontsize=8); ax.grid(True, ls="--", alpha=0.4)
            fig.tight_layout(); st.image(_fig_to_bytes(fig))

            if off_out.mean() > 0.80:
                st.warning(f"⚠️  High output similarity ({off_out.mean():.3f}) — "
                           f"heads write **redundant representations** to the residual stream.")
            elif off_out.mean() < 0.30:
                st.success(f"✅  Low output similarity ({off_out.mean():.3f}) — "
                           f"heads write **diverse** representations.")

            # ── Bonus: per-position cross-head similarity heatmap ──────────────
            st.markdown("#### Per-position output similarity")
            st.caption(
                "For each token position t, compute cos_sim(o_H_i(t), o_H_j(t)), "
                "then show the mean similarity *across heads* vs token position. "
                "Dips indicate positions where heads diverge.")
            mean_sim_per_pos = []
            for t in range(L):
                vecs_t = {f"H{h}": head_outputs[h, t] for h in range(H)}
                s_t, _ = _cosine_sim_matrix(vecs_t)
                off_t = s_t[~np.eye(H, dtype=bool)].mean()
                mean_sim_per_pos.append(float(off_t))

            fig2, ax2 = plt.subplots(figsize=(max(7, L*0.15), 3))
            xs = range(L)
            ax2.plot(xs, mean_sim_per_pos, color=COLORS[2], lw=1.5, alpha=0.8)
            ax2.fill_between(xs, mean_sim_per_pos, alpha=0.15, color=COLORS[2])
            ax2.axhline(np.mean(mean_sim_per_pos), color=COLORS[3], lw=1, ls="--",
                        label=f"global mean = {np.mean(mean_sim_per_pos):.3f}")
            ax2.set_xlabel("Token position"); ax2.set_ylabel("Mean cross-head cos sim")
            ax2.set_title("Head-output agreement per token position", color="#A78BFA")
            ax2.set_ylim(-0.05, 1.05); ax2.legend(fontsize=8)
            ax2.grid(True, ls="--", alpha=0.4); fig2.tight_layout()
            st.image(_fig_to_bytes(fig2))

            # ── Per-head L2 norms ──────────────────────────────────────────────
            norms = np.linalg.norm(head_outputs, axis=-1).mean(1)  # (H,)
            fig3, ax3 = plt.subplots(figsize=(max(5, H*0.7), 2.6))
            ax3.bar(range(H), norms, color=COLORS[:H] if H <= len(COLORS) else COLORS*10)
            ax3.set_xlabel("Head index"); ax3.set_ylabel("Mean L2 norm of output")
            ax3.set_title("Per-head output magnitude", color="#A78BFA")
            ax3.grid(True, ls="--", alpha=0.4, axis="y"); fig3.tight_layout()
            st.image(_fig_to_bytes(fig3))
            st.caption("💡 Large norm differences between heads often indicate "
                       "**unequal contribution** to the residual stream — some heads dominate.")


def _render_core_features_per_layer(model, ref_input: torch.Tensor):
    """
    Replicates Figure 9 (Right) of the Value Residual Learning paper.
    Hooks every Block output, captures hidden states (B, L, D), reshapes to
    (B*L, D) and counts how many PCA components explain 99% of the variance.
    """
    st.caption(
        "Counts **# PCA components** explaining 99% of variance in the residual-stream "
        "hidden states at each layer — each *token position* is one observation. "
        "Mirrors Figure 9 (Right) of *Value Residual Learning for Transformer Language Models*.")

    seq_len = ref_input.shape[1] if ref_input.ndim == 3 else ref_input.shape[0]
    n_seqs = st.slider("Sequences for estimation", 1, 8, 2, key="core_n_seqs")

    if st.button("▶ Compute Core Features per Layer", key="core_run"):
        with st.spinner("Running forward hooks on all layers..."):
            layer_acts: Dict[int, torch.Tensor] = {}
            embed_act = [None]
            handles = []

            def _make_block_hook(i):
                def hook(m, inp, out):
                    t = out[0] if isinstance(out, (tuple, list)) else out
                    if isinstance(t, torch.Tensor):
                        layer_acts[i] = t.detach().float().cpu()
                return hook

            def _embed_hook(m, inp, out):
                t = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(t, torch.Tensor):
                    embed_act[0] = t.detach().float().cpu()

            handles.append(model.embedder.register_forward_hook(_embed_hook))
            for i, block in enumerate(model.blocks):
                handles.append(block.register_forward_hook(_make_block_hook(i)))

            try:
                model.eval()
                dummy = torch.randint(0, model.vocab_size, (n_seqs, seq_len))
                with torch.no_grad():
                    model(dummy)
            except Exception as e:
                st.error(f"Forward pass failed: `{e}`")
                for h in handles: h.remove()
                return
            finally:
                for h in handles: h.remove()

            layer_names, core_vals = [], []
            if embed_act[0] is not None:
                layer_names.append("Embed")
                core_vals.append(_activation_pca_99(embed_act[0]))
            for i in range(model.n_layer):
                if i in layer_acts:
                    layer_names.append(f"L{i}")
                    core_vals.append(_activation_pca_99(layer_acts[i]))

            st.session_state["_core_features"] = (layer_names, core_vals, model.embed_dim)

    if "_core_features" in st.session_state:
        layer_names, core_vals, embed_dim = st.session_state["_core_features"]
        fig, ax = plt.subplots(figsize=(max(8, len(layer_names)*0.75), 4.2))
        xs = range(len(layer_names))
        ax.plot(xs, core_vals, marker="o", color=COLORS[0], lw=2.2, markersize=7, label="This model")
        ax.axhline(embed_dim, color="#9CA3AF", lw=1, ls="--", alpha=0.6, label=f"d_model = {embed_dim}")
        ax.fill_between(xs, core_vals, alpha=0.12, color=COLORS[0])
        ax.set_xlabel("Layer index"); ax.set_ylabel("# Core Features (PCA 99%)")
        ax.set_title("Core Features per Layer — Hidden-State Intrinsic Dimensionality",
                     color="#A78BFA", fontsize=11)
        ax.set_xticks(list(xs)); ax.set_xticklabels(layer_names, rotation=45 if len(layer_names)>12 else 0)
        ax.set_ylim(0, embed_dim + 10)
        ax.legend(fontsize=9); ax.grid(True, ls="--", alpha=0.4)
        fig.tight_layout()
        st.image(_fig_to_bytes(fig))

        valid = [v for v in core_vals if v > 0]
        if valid:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Min core features", min(valid)); c2.metric("Max core features", max(valid))
            c3.metric("Mean", f"{sum(valid)/len(valid):.1f}"); c4.metric("d_model", embed_dim)
            ratio = sum(valid)/len(valid)/embed_dim
            st.caption(
                f"Representations use on average **{ratio:.1%}** of available dimensions. "
                f"A model with well-structured representations should show values well below d_model "
                f"(compare to ~{int(embed_dim*0.22)} for a 384-dim model from the paper).")


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL ANATOMY PAGE  (new)
# ══════════════════════════════════════════════════════════════════════════════

def render_model_anatomy(model):
    st.markdown("## 🌳 Model Anatomy")
    st.caption("Hierarchical model tree — navigate every parameter, inspect SVD / PCA / distribution / mock activation.")

    # ── Component selector ────────────────────────────────────────────────────
    component_options = ["📦 Embedder", "📤 Unembedder", "🔠 Final Norm"]
    for i in range(model.n_layer):
        sm_name = type(model.blocks[i].spatial_mixer).__name__
        component_options.insert(i + 1, f"🧱 Block {i}  [{sm_name}]")

    col1, col2, col3 = st.columns([1.4, 1.2, 2.4])

    with col1:
        sel_comp = st.selectbox("Component", component_options, key="anat_comp")

    # Resolve modules based on selection
    if sel_comp == "📦 Embedder":
        modules = {"embedder": model.embedder}
    elif sel_comp == "📤 Unembedder":
        modules = {"unembedder": model.unembedder}
    elif sel_comp == "🔠 Final Norm":
        modules = {"final_norm": model.final_norm}
    else:
        layer_idx = int(sel_comp.split("Block ")[1].split(" ")[0])
        block = model.blocks[layer_idx]
        with col2:
            sub = st.selectbox("Sub-module",
                               ["Spatial Mixer", "Channel Mixer", "Both"],
                               key="anat_sub")
        modules = ({"spatial_mixer": block.spatial_mixer} if sub == "Spatial Mixer"
                   else {"channel_mixer": block.channel_mixer} if sub == "Channel Mixer"
                   else {"spatial_mixer": block.spatial_mixer, "channel_mixer": block.channel_mixer})

    # Collect parameters
    all_params: Dict[str, torch.Tensor] = {}
    for prefix, module in modules.items():
        for name, p in module.named_parameters():
            all_params[f"{prefix}.{name}"] = p

    if not all_params:
        st.info("No parameters in selected component."); return

    with col3:
        sel_param = st.selectbox("Parameter", list(all_params.keys()), key="anat_param")

    t = all_params[sel_param]
    sv = _svd_values(t)
    stats = _tensor_stats(t)
    pca_99 = _pca_components_99(t)
    cond = float(sv.flatten()[0] / (sv.flatten()[-1] + 1e-12)) if sv is not None else float("nan")
    eff_rank = int((sv.flatten() > sv.flatten()[0] * 0.01).sum()) if sv is not None else 0

    # ── Header card ───────────────────────────────────────────────────────────
    st.markdown(f"### 📐 `{sel_param}`")
    st.markdown(
        f'<div class="tree-node">shape: <b>{tuple(t.shape)}</b> &nbsp;|&nbsp; '
        f'dtype: <b>{t.dtype}</b> &nbsp;|&nbsp; '
        f'numel: <b>{stats["numel"]:,}</b> &nbsp;|&nbsp; '
        f'device: <b>{t.device}</b></div>',
        unsafe_allow_html=True)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Mean", f"{stats['mean']:.3e}")
    c2.metric("Std", f"{stats['std']:.3e}")
    c3.metric("Sparsity", f"{stats['sparsity']:.2%}")
    c4.metric("L2 Norm", f"{stats['l2']:.3e}")
    c5.metric("🔵 PCA 99% var", pca_99 if pca_99 > 0 else "N/A")
    c6.metric("Eff. Rank (SVD)", eff_rank)

    # ── Inspection tabs ───────────────────────────────────────────────────────
    tab_dist, tab_svd, tab_pca, tab_mock, tab_grad = st.tabs([
        "📊 Distribution", "📉 SVD Spectrum", "🔵 PCA Curve",
        "⚡ Mock Activation", "🌊 Gradients"])

    with tab_dist:
        st.image(plot_weight_distribution({sel_param: t}))

    with tab_svd:
        log_s = st.toggle("Log scale", value=True, key="anat_svd_log")
        if sv is not None:
            _render_svd_with_pca(t, sel_param, log_s)
        else:
            st.info("1-D parameter — no SVD available.")

    with tab_pca:
        _render_pca_curve_tab(t, sel_param)

    with tab_mock:
        _render_mock_activation_tab(t, sel_param)

    with tab_grad:
        if t.grad is None:
            st.warning("No gradient available. Run a forward+backward pass first.")
        else:
            st.image(plot_weight_distribution({"gradient of " + sel_param: t.grad}))
            g_pca = _pca_components_99(t.grad)
            st.metric("Gradient PCA 99%", g_pca if g_pca > 0 else "N/A")

    # ── Full component parameter table ────────────────────────────────────────
    with st.expander("📋 All parameters in this component", expanded=False):
        import pandas as pd
        rows = []
        for pname, p in all_params.items():
            s = _tensor_stats(p)
            pca = _pca_components_99(p)
            sv2 = _svd_values(p)
            er = int((sv2.flatten() > sv2.flatten()[0]*0.01).sum()) if sv2 is not None else "—"
            rows.append({
                "Parameter": pname, "Shape": str(s["shape"]), "Numel": f"{s['numel']:,}",
                "Mean": f"{s['mean']:.3e}", "Std": f"{s['std']:.3e}",
                "Sparsity": f"{s['sparsity']:.2%}", "L2": f"{s['l2']:.3e}",
                "PCA 99%": pca if pca > 0 else "N/A", "Eff. Rank": er,
                "Grad": "✅" if p.grad is not None else "❌",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ── Global model PCA summary ──────────────────────────────────────────────
    with st.expander("🌐 Full model PCA-99 overview (all blocks)", expanded=False):
        _render_global_pca_overview(model)


def _render_svd_with_pca(t: torch.Tensor, name: str, log_scale: bool):
    """SVD spectrum plot annotated with 99% PCA line."""
    sv = _svd_values(t)
    if sv is None: st.info("No SVD for 1-D tensors."); return
    sv_flat = sv.flatten() if sv.ndim == 1 else sv.mean(0)
    pca_99 = _pca_components_99(t)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    if sv.ndim == 2:
        for h in range(sv.shape[0]):
            ax.plot(sv[h], color=COLORS[0], alpha=0.25, lw=0.8)
        ax.plot(sv.mean(0), color=COLORS[0], lw=2, label="mean")
        ax.legend(fontsize=8)
    else:
        ax.plot(sv_flat, color=COLORS[0], lw=1.5, marker=".", markersize=3)

    if pca_99 > 0 and pca_99 <= len(sv_flat):
        ax.axvline(pca_99 - 1, color=COLORS[3], lw=1.5, ls="--", alpha=0.85,
                   label=f"PCA 99% ({pca_99} components)")
        ax.legend(fontsize=8)

    cond = sv_flat[0] / (sv_flat[-1] + 1e-12)
    ax.set_title(f"`{name}`  cond={cond:.1f}  rank≈{int((sv_flat > sv_flat[0]*0.01).sum())}",
                 color="#A78BFA", fontsize=10)
    ax.set_xlabel("index"); ax.set_ylabel("σ")
    if log_scale: ax.set_yscale("log")
    ax.grid(True, ls="--", which="both", alpha=0.4)
    fig.tight_layout()
    st.image(_fig_to_bytes(fig))


def _render_pca_curve_tab(t: torch.Tensor, name: str):
    """PCA explained-variance curve with multi-threshold annotations."""
    curve = _pca_explained_curve(t)
    if curve is None:
        st.info("PCA curve not available for 1-D tensors."); return

    thresholds = [0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
    n_comps = {}
    for thr in thresholds:
        n = int(np.searchsorted(curve, thr)) + 1
        n_comps[thr] = min(n, len(curve))

    cols = st.columns(len(thresholds))
    for col, thr in zip(cols, thresholds):
        col.metric(f"PCA @ {int(thr*100)}%", n_comps[thr])

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(range(1, len(curve)+1), curve * 100, color=COLORS[0], lw=2, label="Cumul. var %")

    # Annotate key thresholds
    colors_thr = [COLORS[4], COLORS[2], COLORS[1], COLORS[3], COLORS[5], COLORS[8]]
    for thr, c in zip([0.90, 0.95, 0.99, 0.999], colors_thr[2:]):
        n = n_comps[thr]
        ax.axhline(thr*100, color=c, lw=1, ls="--", alpha=0.7)
        ax.axvline(n, color=c, lw=1, ls="--", alpha=0.6)
        ax.text(n + 0.3, thr*100 - 4, f"n={n}", color=c, fontsize=7)

    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title(f"PCA Explained Variance — `{name}`", color="#A78BFA")
    ax.legend(fontsize=8); ax.grid(True, ls="--", alpha=0.4)
    ax.set_xlim(1, len(curve)); ax.set_ylim(0, 102)
    fig.tight_layout()
    st.image(_fig_to_bytes(fig))

    # Intrinsic dimensionality
    st.caption(
        f"**Intrinsic dimensionality:** the weight matrix of shape `{tuple(t.shape)}` has "
        f"**{n_comps[0.99]} principal components** (out of {t.shape[0]} rows) needed for 99% reconstruction fidelity. "
        f"This measures actual rank utilization vs. theoretical maximum.")


def _render_mock_activation_tab(t: torch.Tensor, name: str):
    """Output distribution when a random input is projected through this weight."""
    st.caption("Computes **W @ rand_input** to show the output activation landscape of this weight matrix.")

    if t.ndim < 2:
        st.info("Mock activation requires 2-D+ tensors (weight matrices)."); return

    c1, c2 = st.columns(2)
    n_samples = c1.slider("Random input vectors", 64, 2048, 512, key=f"mock_n_{hash(name)%9999}")
    dist = c2.selectbox("Input distribution",
                        ["Normal N(0,1)", "Uniform U(-1,1)", "Sparse 10%", "Unit sphere"],
                        key=f"mock_dist_{hash(name)%9999}")

    if st.button("▶ Run Mock Activation", key=f"mock_run_{hash(name)%9999}"):
        with torch.no_grad():
            w = t.detach().float()
            in_feat = w.shape[-1]

            if dist == "Normal N(0,1)":
                x = torch.randn(n_samples, in_feat)
            elif dist == "Uniform U(-1,1)":
                x = torch.rand(n_samples, in_feat) * 2 - 1
            elif dist == "Sparse 10%":
                x = torch.zeros(n_samples, in_feat)
                mask = torch.rand(n_samples, in_feat) < 0.1
                x[mask] = torch.randn(mask.sum())
            else:  # Unit sphere
                x = torch.randn(n_samples, in_feat)
                x = x / (x.norm(dim=1, keepdim=True) + 1e-12)

            # Compute W @ x^T  →  (out, N)
            if w.ndim == 2:
                out = (w @ x.T).T  # (N, out)
            elif w.ndim == 3:
                out = torch.stack([w[h] @ x.T for h in range(w.shape[0])], dim=0)
                out = out.reshape(-1, out.shape[-1]).T  # (N*H, out) → messy but informative
            else:
                st.warning(f"Unsupported ndim={w.ndim}"); return

            out_np = out.numpy()
            out_flat = out_np.flatten()

            pca_out = _pca_components_99(out)

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Output Mean", f"{out_flat.mean():.3e}")
            c2.metric("Output Std", f"{out_flat.std():.3e}")
            c3.metric("Output Sparsity", f"{(np.abs(out_flat) < 1e-6).mean():.2%}")
            c4.metric("🔵 Output PCA 99%", pca_out if pca_out > 0 else "N/A")

            fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
            axes[0].hist(out_flat, bins=80, color=COLORS[1], alpha=0.8, density=True)
            axes[0].set_title("Output Activation Distribution", color="#A78BFA")
            axes[0].set_xlabel("Activation value"); axes[0].set_ylabel("Density")
            axes[0].grid(True, ls="--", alpha=0.4)

            neuron_means = out_np.mean(0); neuron_stds = out_np.std(0)
            xs = np.arange(len(neuron_means))
            axes[1].bar(xs, neuron_stds, color=COLORS[0], alpha=0.65, label="std per neuron")
            axes[1].plot(xs, np.abs(neuron_means), color=COLORS[3], lw=1, label="|mean| per neuron")
            axes[1].set_title("Per-neuron Mean & Std", color="#A78BFA")
            axes[1].set_xlabel("Output neuron index"); axes[1].legend(fontsize=8)
            axes[1].grid(True, ls="--", alpha=0.4)

            fig.tight_layout()
            st.image(_fig_to_bytes(fig))


def _render_global_pca_overview(model):
    """Table of PCA-99 for every parameter in the model."""
    import pandas as pd
    rows = []
    for i, block in enumerate(model.blocks):
        for comp, module in [("SM", block.spatial_mixer), ("FFN", block.channel_mixer)]:
            for pname, p in module.named_parameters():
                pca = _pca_components_99(p)
                sv = _svd_values(p)
                er = int((sv.flatten() > sv.flatten()[0]*0.01).sum()) if sv is not None else "—"
                rows.append({
                    "Layer": i, "Component": comp, "Parameter": pname,
                    "Shape": str(tuple(p.shape)), "Numel": p.numel(),
                    "PCA 99%": pca if pca > 0 else "N/A", "Eff. Rank": er,
                    "Sparsity": f"{_tensor_stats(p)['sparsity']:.2%}",
                })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # PCA heatmap across layers
    ffn_expand_pca = []
    for i, block in enumerate(model.blocks):
        cm = block.channel_mixer
        if hasattr(cm, "expand"):
            ffn_expand_pca.append(_pca_components_99(cm.expand.weight))
    if ffn_expand_pca:
        fig, ax = plt.subplots(figsize=(max(6, len(ffn_expand_pca)*0.8), 3))
        ax.bar(range(len(ffn_expand_pca)), ffn_expand_pca, color=COLORS[2], alpha=0.85)
        ax.set_xlabel("Layer"); ax.set_ylabel("PCA 99% components")
        ax.set_title("FFN Expand Weight — PCA 99% per Layer", color="#A78BFA")
        ax.grid(True, ls="--", alpha=0.4, axis="y"); fig.tight_layout()
        st.image(_fig_to_bytes(fig))


# ══════════════════════════════════════════════════════════════════════════════
#  SIMILARITY ANALYSIS PAGE  (new)
# ══════════════════════════════════════════════════════════════════════════════

def render_similarity_matrix(model):
    st.markdown("## 🔄 Similarity Analysis")
    st.caption("Pairwise cosine similarity across MLP weights, residual stream activations, and token embeddings.")

    mode = st.radio(
        "Analysis Mode",
        ["📐 MLP Weight Similarity (across layers)",
         "🌊 Residual Stream Similarity (activations)",
         "🔗 Cross-Layer Weight Similarity",
         "💬 Embedding Token Similarity"],
        key="sim_mode")

    st.markdown("---")
    if "MLP Weight" in mode:
        _render_mlp_weight_similarity(model)
    elif "Residual Stream" in mode:
        _render_residual_stream_similarity(model)
    elif "Cross-Layer" in mode:
        _render_crosslayer_weight_similarity(model)
    elif "Embedding Token" in mode:
        _render_embedding_similarity(model)


def _render_mlp_weight_similarity(model):
    """Cosine similarity of MLP weights across all layers."""
    sample_cm = model.blocks[0].channel_mixer
    param_names = [name for name, _ in sample_cm.named_parameters()]
    if not param_names: st.info("No MLP parameters found."); return

    param_name = st.selectbox("Parameter", param_names, key="sim_mlp_p")

    layer_vecs: Dict[str, np.ndarray] = {}
    for i, block in enumerate(model.blocks):
        try:
            p = dict(block.channel_mixer.named_parameters())[param_name]
            layer_vecs[f"L{i}"] = p.detach().float().cpu().flatten().numpy()
        except KeyError:
            pass

    if len(layer_vecs) < 2:
        st.warning("Not enough layers with this parameter."); return

    sim, labels = _cosine_sim_matrix(layer_vecs)
    vmin = max(-1.0, float(sim[~np.eye(len(labels), dtype=bool)].min()) - 0.05)
    st.image(plot_similarity_heatmap(sim, labels,
                                      f"MLP `{param_name}` — Cross-layer Cosine Similarity",
                                      vmin=vmin, vmax=1.0))

    off = sim[~np.eye(len(labels), dtype=bool)]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Mean", f"{off.mean():.4f}"); c2.metric("Max", f"{off.max():.4f}")
    c3.metric("Min", f"{off.min():.4f}"); c4.metric("Std", f"{off.std():.4f}")

    # Trend: similarity of each layer to L0
    if len(labels) > 2:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(range(1, len(labels)), sim[0, 1:], marker="o", color=COLORS[2], lw=2)
        ax.axhline(0, color="#9CA3AF", lw=0.8, ls="--")
        ax.set_title(f"Similarity to Layer 0 — `{param_name}`", color="#A78BFA")
        ax.set_xlabel("Layer"); ax.set_ylabel("Cosine sim to L0")
        ax.set_xticks(range(1, len(labels))); ax.set_xticklabels(labels[1:])
        ax.grid(True, ls="--", alpha=0.4); fig.tight_layout()
        st.image(_fig_to_bytes(fig))

    # PCA of each layer's weight
    with st.expander("PCA 99% for this parameter across layers"):
        import pandas as pd
        pca_rows = []
        for lname, vec in layer_vecs.items():
            i = int(lname[1:])
            p = dict(model.blocks[i].channel_mixer.named_parameters())[param_name]
            pca_rows.append({"Layer": lname, "PCA 99%": _pca_components_99(p),
                             "Shape": str(tuple(p.shape))})
        st.dataframe(pd.DataFrame(pca_rows), use_container_width=True, hide_index=True)


def _render_residual_stream_similarity(model):
    """Similarity of residual stream activations at each layer boundary."""
    st.info("Captures the mean-pooled residual stream activation at each Block output via forward hooks.", icon="ℹ️")

    col1, col2 = st.columns(2)
    seq_len = col1.slider("Dummy sequence length", 8, 256, 48, key="sim_res_seq")
    n_seqs = col2.slider("Number of sequences", 1, 8, 2, key="sim_res_n")
    pool_mode = col1.radio("Pooling", ["Mean over L and B", "Last token", "First token"],
                           horizontal=False, key="sim_res_pool")

    if st.button("▶ Compute Residual Similarity", key="sim_res_run"):
        with st.spinner("Running forward hooks..."):
            activations: Dict[str, torch.Tensor] = {}
            handles = []

            def make_block_hook(name):
                def hook(module, inp, out):
                    activations[name] = out.detach().float().cpu()
                return hook

            def make_emb_hook(name):
                def hook(module, inp, out):
                    activations[name] = out.detach().float().cpu()
                return hook

            handles.append(model.embedder.register_forward_hook(make_emb_hook("Embed")))
            for i, block in enumerate(model.blocks):
                handles.append(block.register_forward_hook(make_block_hook(f"L{i}")))

            try:
                model.eval()
                x = torch.randint(0, model.vocab_size, (n_seqs, seq_len))
                with torch.no_grad(): model(x)
            finally:
                for h in handles: h.remove()

            def pool(act):  # (B, L, D) → (D,)
                if pool_mode == "Mean over L and B": return act.mean(dim=(0,1)).numpy()
                elif pool_mode == "Last token": return act[:, -1, :].mean(0).numpy()
                else: return act[:, 0, :].mean(0).numpy()

            layer_reps = {k: pool(v) for k, v in activations.items() if v is not None}
            sim, labels = _cosine_sim_matrix(layer_reps)
            st.session_state["_sim_residual"] = (sim, labels)

    if "_sim_residual" in st.session_state:
        sim, labels = st.session_state["_sim_residual"]
        n = len(labels)
        vmin = float(sim[~np.eye(n, dtype=bool)].min()) - 0.02
        st.image(plot_similarity_heatmap(sim, labels,
                                          "Residual Stream Cosine Similarity (mean-pooled per layer)",
                                          vmin=vmin, vmax=1.0))

        # Trend plot: similarity between consecutive layers
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
        consec = [sim[i, i+1] for i in range(len(labels)-1)]
        axes[0].plot(range(len(consec)), consec, marker="o", color=COLORS[1], lw=2)
        axes[0].set_title("Consecutive Layer Similarity", color="#A78BFA")
        axes[0].set_xlabel("Layer pair (i → i+1)"); axes[0].set_ylabel("Cosine sim")
        axes[0].set_xticks(range(len(consec)))
        axes[0].set_xticklabels([f"{labels[i]}→{labels[i+1]}" for i in range(len(consec))],
                                 rotation=45, fontsize=7)
        axes[0].grid(True, ls="--", alpha=0.4)

        sim_to_first = sim[0, :]
        axes[1].plot(range(n), sim_to_first, marker="o", color=COLORS[2], lw=2)
        axes[1].set_title("Similarity to Input Embedding", color="#A78BFA")
        axes[1].set_xlabel("Layer"); axes[1].set_ylabel("Cosine sim to Embed")
        axes[1].set_xticks(range(n)); axes[1].set_xticklabels(labels, rotation=45, fontsize=7)
        axes[1].grid(True, ls="--", alpha=0.4)
        fig.tight_layout(); st.image(_fig_to_bytes(fig))

        st.caption("💡 If consecutive layers have very high similarity (> 0.99), those layers may contribute "
                   "little to the representation. Rapid drop in similarity to the embedding layer signals "
                   "effective information processing.")


def _render_crosslayer_weight_similarity(model):
    """Cross-layer similarity for any selected weight (spatial or channel mixer)."""
    # Collect all unique param keys across spatial and channel mixers
    all_keys = []
    seen = set()
    for block in model.blocks:
        for comp, module in [("spatial", block.spatial_mixer), ("channel", block.channel_mixer)]:
            for name, _ in module.named_parameters():
                key = f"{comp}.{name}"
                if key not in seen: all_keys.append(key); seen.add(key)

    if not all_keys: st.info("No parameters found."); return
    selected_key = st.selectbox("Parameter", all_keys, key="sim_cross_k")
    comp, param_name = selected_key.split(".", 1)

    layer_vecs: Dict[str, np.ndarray] = {}
    for i, block in enumerate(model.blocks):
        module = block.spatial_mixer if comp == "spatial" else block.channel_mixer
        try:
            p = dict(module.named_parameters())[param_name]
            layer_vecs[f"L{i}"] = p.detach().float().cpu().flatten().numpy()
        except KeyError:
            pass

    if len(layer_vecs) < 2:
        st.warning("Parameter not found in enough layers."); return

    sim, labels = _cosine_sim_matrix(layer_vecs)
    vmin = max(-1.0, float(sim[~np.eye(len(labels), dtype=bool)].min()) - 0.05)
    st.image(plot_similarity_heatmap(sim, labels, f"Cross-layer similarity: `{selected_key}`",
                                      vmin=vmin, vmax=1.0))

    if len(labels) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))

        # Sim to L0
        axes[0].plot(range(1, len(labels)), sim[0, 1:], marker="o", color=COLORS[0], lw=2)
        axes[0].axhline(0, color="#9CA3AF", lw=0.8, ls="--")
        axes[0].set_title("Similarity to L0", color="#A78BFA"); axes[0].set_xlabel("Layer")
        axes[0].set_ylabel("Cosine sim to L0")
        axes[0].set_xticks(range(1, len(labels))); axes[0].set_xticklabels(labels[1:])
        axes[0].grid(True, ls="--", alpha=0.4)

        # PCA 99% per layer
        pca_vals = []
        for i, block in enumerate(model.blocks):
            module = block.spatial_mixer if comp == "spatial" else block.channel_mixer
            try:
                p = dict(module.named_parameters())[param_name]
                pca_vals.append(_pca_components_99(p))
            except KeyError:
                pca_vals.append(0)

        axes[1].bar(range(len(pca_vals)), pca_vals, color=COLORS[2], alpha=0.85)
        axes[1].set_title("PCA 99% per Layer", color="#A78BFA"); axes[1].set_xlabel("Layer")
        axes[1].set_ylabel("# PCA components"); axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels(labels); axes[1].grid(True, ls="--", alpha=0.4, axis="y")

        fig.tight_layout(); st.image(_fig_to_bytes(fig))


def _render_embedding_similarity(model):
    """Token embedding pairwise similarities and PCA analysis."""
    W = model.embedder.embedding.weight.detach().float().cpu()
    V, D = W.shape

    st.markdown(f"Embedding matrix: **{V} tokens × {D} dimensions**")
    pca_emb = _pca_components_99(W)
    c1,c2,c3 = st.columns(3)
    c1.metric("Vocab size", V); c2.metric("Embed dim", D)
    c3.metric("🔵 PCA 99% (full vocab)", pca_emb)

    sample_size = st.slider("Token sample for similarity heatmap", 50, min(1000, V), 200, key="sim_emb_n")
    norm_mode = st.radio("Normalization", ["Cosine (unit norm)", "Raw dot product"],
                         horizontal=True, key="sim_emb_norm")

    if st.button("▶ Compute Embedding Similarity", key="sim_emb_run"):
        with st.spinner("Computing..."):
            idx = np.sort(np.random.choice(V, min(sample_size, V), replace=False))
            W_sub = W[idx].numpy()

            if norm_mode == "Cosine (unit norm)":
                norms = np.linalg.norm(W_sub, axis=1, keepdims=True) + 1e-12
                W_norm = W_sub / norms
            else:
                W_norm = W_sub / (np.sqrt(D) + 1e-12)

            sim_full = W_norm @ W_norm.T  # (sample, sample)
            off_sim = sim_full[~np.eye(len(idx), dtype=bool)].flatten()
            st.session_state["_sim_embed"] = (sim_full, off_sim, idx, W_sub)

    if "_sim_embed" in st.session_state:
        sim_full, off_sim, idx, W_sub = st.session_state["_sim_embed"]
        n_sample = len(idx)

        # Distribution + small heatmap
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].hist(off_sim, bins=80, color=COLORS[0], alpha=0.8, density=True)
        axes[0].axvline(off_sim.mean(), color=COLORS[3], lw=1.5, ls="--", label=f"μ={off_sim.mean():.4f}")
        axes[0].axvline(0, color="#9CA3AF", lw=1, ls="--")
        axes[0].set_title("Pairwise Token Similarity Distribution", color="#A78BFA")
        axes[0].set_xlabel(norm_mode); axes[0].set_ylabel("Density")
        axes[0].legend(fontsize=8); axes[0].grid(True, ls="--", alpha=0.4)

        show_n = min(60, n_sample)
        im = axes[1].imshow(sim_full[:show_n, :show_n], cmap="RdBu_r",
                             vmin=float(sim_full.min()), vmax=1.0, aspect="auto")
        axes[1].set_title(f"Embedding similarity heatmap (first {show_n} sampled tokens)", color="#A78BFA")
        fig.colorbar(im, ax=axes[1])
        fig.tight_layout(); st.image(_fig_to_bytes(fig))

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Mean sim", f"{off_sim.mean():.4f}"); c2.metric("Std sim", f"{off_sim.std():.4f}")
        c3.metric("Max sim", f"{off_sim.max():.4f}"); c4.metric("Min sim", f"{off_sim.min():.4f}")
        c5.metric("% > 0.5", f"{(off_sim > 0.5).mean():.2%}")

        # PCA explained variance for the sampled embeddings
        curve = _pca_explained_curve(torch.tensor(W_sub))
        if curve is not None:
            pca_90 = int(np.searchsorted(curve, 0.90)) + 1
            pca_99 = int(np.searchsorted(curve, 0.99)) + 1
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            ax2.plot(range(1, len(curve)+1), curve*100, color=COLORS[2], lw=2)
            ax2.axhline(99, color=COLORS[3], lw=1, ls="--")
            ax2.axvline(pca_99, color=COLORS[3], lw=1, ls="--",
                        label=f"99% @ {pca_99} components")
            ax2.axvline(pca_90, color=COLORS[1], lw=1, ls="--",
                        label=f"90% @ {pca_90} components")
            ax2.set_title("Embedding PCA Explained Variance (sampled tokens)", color="#A78BFA")
            ax2.set_xlabel("# components"); ax2.set_ylabel("Cumul. var %")
            ax2.set_xlim(1, len(curve)); ax2.set_ylim(0, 102)
            ax2.legend(fontsize=8); ax2.grid(True, ls="--", alpha=0.4)
            fig2.tight_layout(); st.image(_fig_to_bytes(fig2))


# ══════════════════════════════════════════════════════════════════════════════
#  LOADER
# ══════════════════════════════════════════════════════════════════════════════

def _try_load_pt(raw: bytes):
    buf = io.BytesIO(raw)
    try:
        obj = torch.load(buf, map_location="cpu", weights_only=False)
    except Exception as e:
        return None, None, str(e)
    if isinstance(obj, nn.Module): return obj, "full_model", None
    if not isinstance(obj, dict): return None, None, f"Unexpected type: {type(obj).__name__}"
    if "model_state_dict" in obj: return obj["model_state_dict"], "vathos_checkpoint", None
    for key in ("model", "module", "net", "network"):
        if key in obj and isinstance(obj[key], nn.Module): return obj[key], f"dict['{key}']", None
    if obj and all(isinstance(v, torch.Tensor) for v in obj.values()): return obj, "state_dict", None
    return None, None, "Unrecognised checkpoint format."


def _infer_config_from_state_dict(sd: dict) -> dict:
    cfg = {}
    block_indices = {int(k.split(".")[1]) for k in sd if k.startswith("blocks.") and k.split(".")[1].isdigit()}
    cfg["n_layers"] = max(block_indices) + 1 if block_indices else 0
    for k, v in sd.items():
        if "embedder.embedding.weight" in k:
            cfg["vocab_size"] = v.shape[0]; cfg["embed_dim"] = v.shape[1]; break
    d_models, M_dims = [], []
    for i in range(cfg.get("n_layers", 0)):
        exp_key = f"blocks.{i}.channel_mixer.expand.weight"
        con_key = f"blocks.{i}.channel_mixer.contract.weight"
        if exp_key in sd and con_key in sd:
            M_dims.append(sd[exp_key].shape[0]); d_models.append(sd[con_key].shape[0])
        else:
            d_models.append(cfg.get("embed_dim","?")); M_dims.append("?")
    cfg["d_models"] = d_models; cfg["M_dims"] = M_dims
    cfg["has_zeroskip"] = any("zeroskip_params" in k for k in sd)
    cfg["skip_keys"] = [k for k in sd if "skip_lambdas" in k]
    for i in range(cfg.get("n_layers", 0)):
        if f"blocks.{i}.spatial_mixer.qkv.weight" in sd:
            cfg["spatial_type"] = "MultiheadAttentionMixer"; break
        elif f"blocks.{i}.spatial_mixer.qk.weight" in sd:
            cfg["spatial_type"] = "MultiheadAttentionMixerNOV"; break
    else:
        cfg["spatial_type"] = "unknown"
    return cfg


def _render_state_dict_panel(sd: dict, fmt: str, source_name: str):
    is_vathos = (fmt == "vathos_checkpoint")
    (st.info if is_vathos else st.warning)(
        f"**{'Vathos checkpoint' if is_vathos else 'Bare state dict'}** (`{source_name}`).", icon="ℹ️")
    cfg = _infer_config_from_state_dict(sd)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Layers", cfg["n_layers"]); c2.metric("Embed dim", cfg.get("embed_dim","?"))
    c3.metric("Vocab size", cfg.get("vocab_size","?")); c4.metric("Spatial type", cfg.get("spatial_type","?"))
    import pandas as pd
    rows = [{"layer": i, "d_model": d, "M_dim (FFN)": m, "Spatial Mixer": cfg.get("spatial_type","?")}
            for i,(d,m) in enumerate(zip(cfg["d_models"], cfg["M_dims"]))]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    default_code = _build_default_snippet(cfg)
    code = st.text_area("Constructor code", value=default_code, height=280, key="reconstruct_code")
    label = "🔧 Reconstruct & load_checkpoint()" if is_vathos else "🔧 Reconstruct & load_state_dict()"
    if st.button(label, use_container_width=True, type="primary"):
        _reconstruct_from_code(code, sd, source_name, fmt)


def _build_default_snippet(cfg: dict) -> str:
    n = cfg["n_layers"]; d = cfg.get("embed_dim",128); v = cfg.get("vocab_size",1000)
    ds = cfg["d_models"]; ms = cfg["M_dims"]
    stype = cfg.get("spatial_type","MultiheadAttentionMixer")
    spatial_class = stype if stype not in ("unknown", "") else "GroupedQueryAttention"
    # Build integer list literals (NOT string lists — str([…]) would give ['384', …])
    d_str = "[" + ", ".join(str(x) for x in ds) + "]"
    m_str = "[" + ", ".join(str(x) for x in ms) + "]"
    return (f"import sys\nfrom Vathos.blocks import *\n"
            f"spatials = [Builder({spatial_class}, n_heads=8)] * {n}\n"
            f"skips    = [None] * {n}\n\n"
            f"model = ModdedFormer(\n    vocab_size={v},\n    embed_dim={d},\n"
            f"    d_models={d_str},\n    spatials=spatials,\n"
            f"    M_dims={m_str},\n    norm=RMSNorm,\n    weights_tying=True,\n    skips=skips,\n)")


def _reconstruct_from_code(code: str, sd: dict, source_name: str, fmt: str):
    ns = {}
    try: exec(compile(code, "<inspector>", "exec"), ns)
    except Exception as e: st.error(f"❌ Constructor error:\n```\n{e}\n```"); return
    model = ns.get("model")
    if not isinstance(model, nn.Module):
        st.error("❌ Code must assign an `nn.Module` to `model`."); return
    if fmt == "vathos_checkpoint":
        import os
        path = st.session_state.get("_pending_source_name","")
        if os.path.isfile(path):
            try: model.load_checkpoint(path)
            except Exception as e: st.error(f"❌ load_checkpoint failed:\n```\n{e}\n```"); return
            st.success("✅ `model.load_checkpoint()` succeeded.")
        else:
            st.warning("Bytes-only upload — falling back to load_state_dict().")
            _apply_state_dict(model, sd)
    else:
        _apply_state_dict(model, sd)
    st.session_state["model"] = model
    st.session_state["_load_fmt"] = fmt; st.session_state["_load_name"] = source_name
    st.session_state.pop("_pending_state_dict", None); st.session_state.pop("_pending_source_name", None)
    st.rerun()


def _apply_state_dict(model: nn.Module, sd: dict):
    try: missing, unexpected = model.load_state_dict(sd, strict=False)
    except Exception as e: st.error(f"❌ load_state_dict failed:\n```\n{e}\n```"); return
    if missing: st.warning(f"Missing keys ({len(missing)})")
    if unexpected: st.warning(f"Unexpected keys ({len(unexpected)})")
    st.success("✅ Weights loaded.")


def _load_raw(raw: bytes, name: str):
    import os
    obj, fmt, err = _try_load_pt(raw)
    if err: st.sidebar.error(f"❌ {err}"); return
    if isinstance(obj, nn.Module):
        st.session_state["model"] = obj; st.session_state["_load_fmt"] = fmt
        st.session_state["_load_name"] = name
        st.sidebar.success(f"✅ Loaded `{os.path.basename(name)}` ({fmt})")
    elif isinstance(obj, dict):
        st.session_state["_pending_state_dict"] = obj
        st.session_state["_pending_source_name"] = name
        st.session_state["_load_fmt"] = fmt
        st.session_state.pop("model", None)
        st.sidebar.info(f"Checkpoint loaded (`{fmt}`).")
    else:
        st.sidebar.error("❌ Could not interpret checkpoint.")


def _render_loader_sidebar() -> Optional[nn.Module]:
    st.sidebar.markdown("## 📂 Load Model")
    source = st.sidebar.radio("Source", ["Upload .pt / .pth file", "Local file path"],
                               label_visibility="collapsed")
    if source == "Upload .pt / .pth file":
        uploaded = st.sidebar.file_uploader("Drop checkpoint here", type=["pt","pth","bin"])
        if uploaded is not None and st.session_state.get("_last_uploaded") != uploaded.name:
            st.session_state["_last_uploaded"] = uploaded.name
            _load_raw(uploaded.read(), uploaded.name); st.rerun()
    else:
        path_input = st.sidebar.text_input("Absolute path to .pt / .pth", key="local_path_input")
        if st.sidebar.button("Load from path", use_container_width=True):
            import os
            if not path_input: st.sidebar.warning("Enter a path first.")
            elif not os.path.isfile(path_input): st.sidebar.error(f"File not found: `{path_input}`")
            else:
                with open(path_input, "rb") as fh: _load_raw(fh.read(), path_input)
                st.rerun()

    model = st.session_state.get("model", None)
    if model is not None:
        fmt = st.session_state.get("_load_fmt","unknown")
        name = st.session_state.get("_load_name", type(model).__name__)
        total = sum(p.numel() for p in model.parameters())
        st.sidebar.markdown("---"); st.sidebar.markdown("**Loaded model**")
        st.sidebar.markdown(f"- `{name}`\n- format: `{fmt}`\n- type: `{type(model).__name__}`\n- params: `{total:,}`")
        if st.sidebar.button("🗑️ Unload", use_container_width=True):
            for k in ("model","_last_uploaded","_load_fmt","_load_name","_pending_state_dict","_pending_source_name"):
                st.session_state.pop(k, None)
            st.rerun()
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING HISTORY
# ══════════════════════════════════════════════════════════════════════════════

def render_training_history(model):
    st.markdown("## 📈 Training History")
    has_losses = hasattr(model, "_losses_dict") and len(getattr(model,"_losses_dict",{})) > 0
    has_metrics = hasattr(model, "_metrics_per_epoch") and len(getattr(model,"_metrics_per_epoch",{})) > 0
    if not has_losses and not has_metrics:
        st.info("No training history. Load a **Vathos checkpoint** that has `register_loss()` data.", icon="ℹ️")
        return

    steps = getattr(model,"steps","–"); epochs = getattr(model,"epochs","–")
    best_loss = getattr(model,"best_loss",float("inf"))
    checkpoints = getattr(model,"checkpoints","–"); spe = getattr(model,"steps_per_epoch","–")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Steps",steps); c2.metric("Epochs",epochs)
    c3.metric("Best loss", f"{best_loss:.6g}" if isinstance(best_loss,float) else best_loss)
    c4.metric("Checkpoints",checkpoints); c5.metric("Steps/epoch",spe)
    st.markdown("---")

    DARK_BG="#0e1117"; GRID="#2a2a2a"; FG="#e0e0e0"
    C1="#4c9be8"; C2="#f5a623"
    CMAP=["#4c9be8","#f5a623","#50c87a","#e85d7a","#a78bfa","#fb923c"]

    def _base_fig(title, xlabel, ylabel):
        fig, ax = plt.subplots(figsize=(10,3.8), facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG); ax.set_title(title,color=FG,fontsize=13)
        ax.set_xlabel(xlabel,color=FG,fontsize=10); ax.set_ylabel(ylabel,color=FG,fontsize=10)
        ax.tick_params(colors=FG)
        for spine in ax.spines.values(): spine.set_edgecolor(GRID)
        ax.grid(True,color=GRID,linewidth=0.7,alpha=0.6)
        return fig, ax

    tab_names = ["Loss Curves"]
    if has_metrics:
        tab_names += [f"Metric: {m}" for m in getattr(model,"_metrics_per_epoch",{})]
    tab_names.append("Raw Tables")
    tabs = st.tabs(tab_names)

    with tabs[0]:
        losses_dict = getattr(model,"_losses_dict",{})
        losses_ep = getattr(model,"_losses_per_epoch_dict",{})
        win = 1
        if len(losses_dict) > 20:
            win = st.slider("Smoothing window",1,max(2,len(losses_dict)//20),1,key="loss_smooth")
        fig, ax = _base_fig("Training Loss","Steps","Loss")
        if losses_dict:
            xs = list(losses_dict.keys()); ys = list(losses_dict.values())
            if win > 1:
                kernel = np.ones(win)/win; ys_sm = np.convolve(ys, kernel, mode="valid")
                ax.plot(xs, ys, color=C1, lw=0.6, alpha=0.25, label="Loss (raw)")
                ax.plot(xs[win-1:], ys_sm, color=C1, lw=1.4, label=f"Loss (smooth w={win})")
            else:
                ax.plot(xs, ys, color=C1, lw=0.8, alpha=0.55, label="Loss (step)")
        if losses_ep:
            ex=list(losses_ep.keys()); ey=list(losses_ep.values())
            ax.plot(ex,ey,color=C2,lw=2.2,marker="o",markersize=4,label="Loss per epoch")
            best = getattr(model,"best_loss",None)
            if best and isinstance(best,float) and best != float("inf"):
                ax.axhline(best,color="#e85d7a",lw=1,ls="--",alpha=0.7,label=f"Best {best:.5g}")
        ax.legend(facecolor=DARK_BG,edgecolor=GRID,labelcolor=FG,fontsize=9)
        st.image(_fig_to_bytes(fig),use_container_width=True); plt.close(fig)
        if losses_ep:
            fig2,ax2=_base_fig("Loss per Epoch","Epoch","Mean Loss")
            ey_list=list(losses_ep.values())
            cols=[("#e85d7a" if v==min(ey_list) else C2) for v in ey_list]
            ax2.bar(range(len(ey_list)),ey_list,color=cols,edgecolor=GRID,lw=0.5)
            ax2.set_xticks(range(len(ey_list)))
            st.image(_fig_to_bytes(fig2),use_container_width=True); plt.close(fig2)
        losses_this=getattr(model,"_losses_this_epoch",[])
        if losses_this:
            fig3,ax3=_base_fig("Current Epoch Loss","Step (this epoch)","Loss")
            ax3.plot(losses_this,color=C1,lw=0.9)
            st.image(_fig_to_bytes(fig3),use_container_width=True); plt.close(fig3)

    metrics_ep = getattr(model,"_metrics_per_epoch",{}); metrics_all = getattr(model,"_metrics",{})
    for t_idx, mname in enumerate(metrics_ep, start=1):
        with tabs[t_idx]:
            ep_vals = metrics_ep.get(mname,[]); step_vals = metrics_all.get(mname,[])
            col = CMAP[t_idx % len(CMAP)]
            fig, ax = _base_fig(f"Metric: {mname}","Steps / Epochs",mname)
            if step_vals: ax.plot(range(len(step_vals)),step_vals,color=col,lw=0.8,alpha=0.45,label=f"{mname} (step)")
            if ep_vals:
                scale = len(step_vals)/len(ep_vals) if step_vals and ep_vals else 1
                ax.plot([i*scale for i in range(len(ep_vals))],ep_vals,color=col,lw=2.2,marker="o",markersize=5,label=f"{mname}/epoch")
            ax.legend(facecolor=DARK_BG,edgecolor=GRID,labelcolor=FG,fontsize=9)
            st.image(_fig_to_bytes(fig),use_container_width=True); plt.close(fig)
            if ep_vals:
                fig2,ax2=_base_fig(f"{mname} per Epoch","Epoch",mname)
                best_idx=ep_vals.index(max(ep_vals))
                bar_colors=[(col if i!=best_idx else "#50c87a") for i in range(len(ep_vals))]
                ax2.bar(range(len(ep_vals)),ep_vals,color=bar_colors,edgecolor=GRID,lw=0.5)
                ax2.set_xticks(range(len(ep_vals)))
                st.image(_fig_to_bytes(fig2),use_container_width=True); plt.close(fig2)

    with tabs[-1]:
        import pandas as pd
        losses_dict = getattr(model,"_losses_dict",{})
        losses_ep = getattr(model,"_losses_per_epoch_dict",{})
        st.markdown("#### Step-level losses")
        if losses_dict:
            st.dataframe(pd.DataFrame({"step":list(losses_dict.keys()),"loss":list(losses_dict.values())}),
                         use_container_width=True, hide_index=True)
        else: st.caption("No step-level data.")
        st.markdown("#### Epoch-level losses")
        if losses_ep:
            df=pd.DataFrame({"step":list(losses_ep.keys()),"mean_loss":list(losses_ep.values())})
            df.insert(0,"epoch",range(len(df)))
            st.dataframe(df, use_container_width=True, hide_index=True)
        else: st.caption("No epoch-level data.")
        for mname, vals in metrics_ep.items():
            st.markdown(f"#### Metric: `{mname}` (per epoch)")
            st.dataframe(pd.DataFrame({"epoch":range(len(vals)), mname:vals}),
                         use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  RESEARCH TOOLS PAGE
# ══════════════════════════════════════════════════════════════════════════════

def render_research_tools(model):
    st.markdown("## 🔬 Research Tools")
    st.caption("Diagnostics and analyses designed for studying transformer internals.")

    tool = st.radio("Tool", [
        "💀 Dead Neuron Audit",
        "📊 Gradient Flow Health",
        "🌊 Isotropy / Anisotropy",
        "📈 Effective Rank Profile",
        "⚖️ Weight Norm Imbalance",
        "🎯 Activation Statistics Deep Dive",
    ], horizontal=False, key="research_tool")

    st.markdown("---")

    if "Dead Neuron" in tool:
        _research_dead_neurons(model)
    elif "Gradient Flow" in tool:
        _research_gradient_flow(model)
    elif "Isotropy" in tool:
        _research_isotropy(model)
    elif "Effective Rank" in tool:
        _research_effective_rank_profile(model)
    elif "Weight Norm Imbalance" in tool:
        _research_norm_imbalance(model)
    elif "Activation Statistics" in tool:
        _research_activation_stats(model)


# ──────────────────────────────────────────────────────────────────────────────

def _research_dead_neurons(model):
    """
    For each layer's channel mixer, run N random batches and track which
    neurons *never* produce a non-zero output.  Dead neurons waste capacity.
    For ReLU/ReLU² this is permanent; for SiLU/GeLU it's a sign of saturation.
    """
    st.markdown("### 💀 Dead Neuron Audit")
    st.caption(
        "A neuron is **dead** if its post-activation output is < 1e-6 for "
        "every token in every batch tested.  For ReLU/ReLU² this is permanent "
        "(gradient = 0 forever).  For SiLU/GELU it signals chronic saturation.")

    col1, col2 = st.columns(2)
    n_batches = col1.slider("Random batches", 1, 16, 4, key="dead_batches")
    seq_len   = col2.slider("Sequence length", 8, 128, 32, key="dead_seq")

    if st.button("▶ Run Dead Neuron Audit", key="dead_run"):
        with st.spinner("Running forward passes..."):
            results = []
            for i, block in enumerate(model.blocks):
                cm = block.channel_mixer
                if not hasattr(cm, "expand"): continue

                all_acts = []
                try:
                    with torch.no_grad():
                        for _ in range(n_batches):
                            x = torch.randn(2, seq_len, model.d_models[i])
                            pre = cm.expand(x)
                            post = cm.activation(pre) if hasattr(cm, "activation") else pre
                            all_acts.append(post.abs().detach().cpu())
                    stacked = torch.cat(all_acts, dim=0)   # (n_batches*2, L, M)
                    max_per_neuron = stacked.reshape(-1, stacked.shape[-1]).max(0).values.numpy()
                    dead_mask = max_per_neuron < 1e-6
                    dead_frac = float(dead_mask.mean())
                    results.append({"layer": i, "M_dim": stacked.shape[-1],
                                    "dead_count": int(dead_mask.sum()),
                                    "dead_frac": dead_frac,
                                    "max_act_mean": float(max_per_neuron.mean()),
                                    "max_act_min":  float(max_per_neuron.min()),
                                    "max_act_p1":   float(np.percentile(max_per_neuron, 1)),})
                except Exception as e:
                    results.append({"layer": i, "error": str(e)})
            st.session_state["_dead_results"] = results

    if "_dead_results" in st.session_state:
        results = st.session_state["_dead_results"]
        good = [r for r in results if "dead_frac" in r]
        if not good: st.warning("No channel mixers with `expand` found."); return

        import pandas as pd
        df = pd.DataFrame(good)
        st.dataframe(df[["layer","M_dim","dead_count","dead_frac","max_act_p1","max_act_mean"]],
                     use_container_width=True, hide_index=True)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        xs = [r["layer"] for r in good]
        dead_fracs = [r["dead_frac"] for r in good]
        colors_bar = [COLORS[3] if d > 0.05 else COLORS[2] for d in dead_fracs]
        axes[0].bar(xs, [d*100 for d in dead_fracs], color=colors_bar, alpha=0.85)
        axes[0].axhline(5, color=COLORS[3], lw=1, ls="--", alpha=0.7, label="5% threshold")
        axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Dead neurons (%)")
        axes[0].set_title("Dead Neuron Fraction per Layer", color=COLORS[0])
        axes[0].legend(fontsize=8); axes[0].grid(True, ls="--", alpha=0.4, axis="y")

        for r in good:
            try:
                cm = model.blocks[r["layer"]].channel_mixer
                with torch.no_grad():
                    x = torch.randn(4, seq_len, model.d_models[r["layer"]])
                    pre = cm.expand(x)
                    post = cm.activation(pre) if hasattr(cm, "activation") else pre
                    max_per = post.abs().reshape(-1, post.shape[-1]).max(0).values.numpy()
                axes[1].plot(sorted(max_per), color=COLORS[r["layer"] % len(COLORS)],
                             lw=1.0, alpha=0.7, label=f"L{r['layer']}")
            except Exception: pass
        axes[1].set_xlabel("Neuron rank (sorted by max activation)")
        axes[1].set_ylabel("Max activation seen")
        axes[1].set_title("Neuron Activity Spectrum (sorted)", color=COLORS[0])
        axes[1].legend(fontsize=7, ncol=2); axes[1].grid(True, ls="--", alpha=0.4)
        axes[1].set_yscale("log")
        fig.tight_layout(); st.image(_fig_to_bytes(fig))

        total_dead = sum(r["dead_count"] for r in good)
        total_neurons = sum(r["M_dim"] for r in good)
        if total_dead / total_neurons > 0.10:
            st.warning(f"⚠️  {total_dead:,} / {total_neurons:,} neurons ({total_dead/total_neurons:.1%}) "
                       f"are dead across all layers.  Consider a lower learning rate, "
                       f"leaky activations, or better weight initialisation.")
        else:
            st.success(f"✅  Dead neuron count looks healthy: "
                       f"{total_dead:,} / {total_neurons:,} ({total_dead/total_neurons:.1%})")


def _research_gradient_flow(model):
    """
    Compute a synthetic loss backward pass and visualise:
    - Gradient L2 norm per parameter
    - Gradient/weight norm ratio (relative gradient signal)
    - Vanishing gradient detection across layers
    """
    st.markdown("### 📊 Gradient Flow Health")
    st.caption(
        "Runs one synthetic forward+backward pass and analyses gradient norms. "
        "Vanishing gradients (ratio < 1e-4) or exploding gradients (ratio > 10) "
        "both indicate training instability.  The **gradient/weight ratio** is the "
        "most informative signal — it should be roughly uniform across layers.")

    seq_len = st.slider("Sequence length", 8, 128, 32, key="grad_seq")

    if st.button("▶ Compute Gradient Flow", key="grad_run"):
        with st.spinner("Forward + backward pass..."):
            try:
                model.train()
                x = torch.randint(0, model.vocab_size, (2, seq_len))
                logits = model(x)
                # cross-entropy with random targets as synthetic loss
                targets = torch.randint(0, model.vocab_size, (2, seq_len))
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, model.vocab_size), targets.reshape(-1))
                loss.backward()
                model.eval()

                grad_data = []
                for i, block in enumerate(model.blocks):
                    for comp, module in [("SM", block.spatial_mixer), ("FFN", block.channel_mixer)]:
                        for pname, p in module.named_parameters():
                            if p.grad is None: continue
                            g_norm = float(p.grad.norm(2))
                            w_norm = float(p.norm(2))
                            ratio  = g_norm / (w_norm + 1e-12)
                            grad_data.append({
                                "layer": i, "comp": comp, "param": pname,
                                "grad_norm": g_norm, "weight_norm": w_norm,
                                "ratio": ratio,
                                "label": f"L{i}.{comp}.{pname.split('.')[0]}",
                            })

                # zero grads after capture
                for p in model.parameters():
                    if p.grad is not None: p.grad.zero_()

                st.session_state["_grad_data"] = grad_data
                st.session_state["_grad_loss"] = float(loss.item())
            except Exception as e:
                st.error(f"Backward pass failed: `{e}`")
                model.eval()

    if "_grad_data" in st.session_state:
        grad_data = st.session_state["_grad_data"]
        if not grad_data: st.warning("No gradient data captured."); return

        import pandas as pd
        df = pd.DataFrame(grad_data)
        st.caption(f"Synthetic loss: **{st.session_state.get('_grad_loss', 0):.4f}**")

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        labels = df["label"].tolist()
        xs = range(len(labels))
        fs = max(5, min(8, int(220 / len(labels))))

        # gradient norms
        bar_c = [COLORS[3] if g < 1e-5 else (COLORS[0] if g < 10 else COLORS[1])
                 for g in df["grad_norm"]]
        axes[0].bar(xs, df["grad_norm"], color=bar_c, alpha=0.85)
        axes[0].set_yscale("log"); axes[0].set_ylabel("Gradient L2 Norm")
        axes[0].set_title("Gradient L2 Norms per Parameter", color=COLORS[0])
        axes[0].set_xticks(list(xs)); axes[0].set_xticklabels(labels, rotation=90, fontsize=fs)
        axes[0].grid(True, ls="--", alpha=0.4, axis="y")

        # ratio
        ratio_c = [COLORS[3] if r < 1e-4 else (COLORS[0] if r < 10 else COLORS[1])
                   for r in df["ratio"]]
        axes[1].bar(xs, df["ratio"], color=ratio_c, alpha=0.85)
        axes[1].axhline(1e-4, color=COLORS[3], lw=1, ls="--", label="vanish threshold 1e-4")
        axes[1].axhline(10,   color=COLORS[1], lw=1, ls="--", label="explode threshold 10")
        axes[1].set_yscale("log"); axes[1].set_ylabel("‖grad‖ / ‖weight‖")
        axes[1].set_title("Gradient / Weight Ratio  (ideal: uniform, ~1e-3 to 0.1)", color=COLORS[0])
        axes[1].set_xticks(list(xs)); axes[1].set_xticklabels(labels, rotation=90, fontsize=fs)
        axes[1].legend(fontsize=8); axes[1].grid(True, ls="--", alpha=0.4, axis="y")

        fig.tight_layout(); st.image(_fig_to_bytes(fig))

        vanish = (df["ratio"] < 1e-4).sum()
        explode = (df["ratio"] > 10).sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Vanishing (ratio < 1e-4)", vanish)
        c2.metric("Exploding (ratio > 10)", explode)
        c3.metric("Healthy range", len(df) - vanish - explode)

        if vanish: st.warning(f"⚠️  {vanish} parameters have near-zero gradients — possible vanishing gradient.")
        if explode: st.warning(f"⚠️  {explode} parameters have very large grad/weight ratios — check LR / clipping.")
        if not vanish and not explode: st.success("✅  Gradient flow looks healthy across all parameters.")


def _research_isotropy(model):
    """
    Measures representation isotropy: how uniformly weight matrices use all
    directions in space.  High anisotropy → rank collapse / dominated by few dims.
    Uses the partition function average cosine similarity: I = 1 - (1/n²) Σ cos²(θ_ij)
    Benevolently: I close to 1 = isotropic; I close to 0 = collapsed.
    """
    st.markdown("### 🌊 Isotropy / Anisotropy")
    st.caption(
        "**Isotropy** measures how uniformly a weight matrix fills its embedding space. "
        "An isotropic matrix uses all directions equally (score → 1). "
        "An anisotropic / rank-collapsed matrix concentrates on a few axes (score → 0). "
        "Formula: $I = 1 - \\frac{1}{n}\\sum_i \\cos^2\\theta_i$ based on the singular vector distribution.")

    if st.button("▶ Compute Isotropy", key="iso_run"):
        with st.spinner("Running SVD on all weight matrices..."):
            iso_data = []
            for i, block in enumerate(model.blocks):
                for comp, module in [("SM", block.spatial_mixer), ("FFN", block.channel_mixer)]:
                    for pname, p in module.named_parameters():
                        if p.ndim < 2: continue
                        try:
                            W = p.detach().float().cpu().reshape(p.shape[0], -1).numpy()
                            sv = np.linalg.svd(W, compute_uv=False)
                            # Isotropy = 1 - variance of normalised singular values
                            sv_norm = sv / (sv.sum() + 1e-12)
                            isotropy = 1.0 - float(np.sum(sv_norm ** 2)) * len(sv)
                            isotropy = max(0.0, min(1.0, (isotropy / (1 - 1/len(sv))) if len(sv) > 1 else 0.0))
                            iso_data.append({
                                "layer": i, "comp": comp, "param": pname,
                                "isotropy": isotropy,
                                "n_sv": len(sv),
                                "sv_max": float(sv[0]),
                                "sv_min": float(sv[-1]),
                                "cond": float(sv[0] / (sv[-1] + 1e-12)),
                            })
                        except Exception:
                            pass
            st.session_state["_iso_data"] = iso_data

    if "_iso_data" in st.session_state:
        data = st.session_state["_iso_data"]
        if not data: st.info("No 2D+ parameters found."); return

        import pandas as pd
        df = pd.DataFrame(data)

        # Per-layer mean isotropy
        layer_iso = df.groupby("layer")["isotropy"].mean().reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        xs = layer_iso["layer"].values
        bar_c = [COLORS[2] if v > 0.6 else (COLORS[0] if v > 0.3 else COLORS[3])
                 for v in layer_iso["isotropy"]]
        axes[0].bar(xs, layer_iso["isotropy"], color=bar_c, alpha=0.85)
        axes[0].axhline(0.5, color="#9CA3AF", lw=1, ls="--", alpha=0.7, label="0.5 boundary")
        axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Mean Isotropy Score")
        axes[0].set_title("Mean Weight Isotropy per Layer\n(1=isotropic, 0=collapsed)", color=COLORS[0])
        axes[0].set_ylim(0, 1.05); axes[0].legend(fontsize=8)
        axes[0].grid(True, ls="--", alpha=0.4, axis="y")

        # Scatter: isotropy vs condition number
        axes[1].scatter(df["cond"], df["isotropy"],
                        c=[COLORS[i % len(COLORS)] for i in df["layer"]],
                        s=40, alpha=0.75)
        axes[1].set_xscale("log")
        axes[1].set_xlabel("Condition number κ (log scale)")
        axes[1].set_ylabel("Isotropy")
        axes[1].set_title("Isotropy vs Condition Number\n(high κ = anisotropic expected)", color=COLORS[0])
        axes[1].grid(True, ls="--", alpha=0.4)
        fig.tight_layout(); st.image(_fig_to_bytes(fig))

        st.dataframe(df[["layer","comp","param","isotropy","cond","sv_max","sv_min"]].round(4),
                     use_container_width=True, hide_index=True)


def _research_effective_rank_profile(model):
    """
    Plots effective rank (SVD-based, 99% energy) across every layer for
    both spatial mixer and channel mixer, to reveal rank collapse patterns.
    """
    st.markdown("### 📈 Effective Rank Profile")
    st.caption(
        "**Effective rank** = number of singular values that capture 99% of the matrix energy. "
        "Uniform high rank = full utilisation of parameter budget. "
        "Sudden drops often correspond to layers that collapsed during training.")

    params_to_track = st.multiselect(
        "Parameters to track", ["expand", "contract", "qkv", "qk", "out", "q_proj", "kv_proj"],
        default=["expand", "qkv", "qk"], key="er_params")
    if not params_to_track: params_to_track = ["expand"]

    data = {}
    for i, block in enumerate(model.blocks):
        for comp, module in [("SM", block.spatial_mixer), ("FFN", block.channel_mixer)]:
            for pname, p in module.named_parameters():
                key = pname.split(".")[0]
                if key not in params_to_track: continue
                full_key = f"{comp}.{key}"
                if full_key not in data: data[full_key] = {}
                if p.ndim >= 2:
                    sv = _svd_values(p)
                    if sv is not None:
                        sv_f = sv.flatten() if sv.ndim == 1 else sv.mean(0)
                        energy = np.cumsum(sv_f ** 2) / (np.sum(sv_f ** 2) + 1e-12)
                        er = int(np.searchsorted(energy, 0.99)) + 1
                        data[full_key][i] = er

    if not data: st.info("No matching parameters found."); return

    fig, ax = plt.subplots(figsize=(max(8, model.n_layer * 0.8), 4.5))
    for idx, (param_key, layer_vals) in enumerate(data.items()):
        if not layer_vals: continue
        xs = sorted(layer_vals.keys())
        ys = [layer_vals[x] for x in xs]
        ax.plot(xs, ys, marker="o", color=COLORS[idx % len(COLORS)],
                lw=2, markersize=6, label=param_key, alpha=0.9)

    ax.set_xlabel("Layer index"); ax.set_ylabel("Effective rank (99% energy)")
    ax.set_title("Effective Rank Profile across Layers", color=COLORS[0])
    ax.legend(fontsize=9); ax.grid(True, ls="--", alpha=0.4)
    ax.set_xticks(range(model.n_layer))
    fig.tight_layout(); st.image(_fig_to_bytes(fig))

    st.caption(
        "💡 Ideally curves are **flat and high** (full rank utilisation). "
        "A downward slope toward later layers may indicate the model is "
        "increasingly redundant or that training didn't differentiate later layers.")


def _research_norm_imbalance(model):
    """
    Checks whether spatial mixer and channel mixer weights are balanced,
    and whether the out-projection (which writes to residual stream) is healthy.
    Large norm imbalance between layers can indicate rank collapse or
    learning rate issues.
    """
    st.markdown("### ⚖️ Weight Norm Imbalance")
    st.caption(
        "Compares L2 norms between spatial mixers, channel mixers, and within "
        "each block.  High variance across layers → uneven training. "
        "The **norm ratio SM/FFN** tells you which pathway dominates residual updates.")

    sm_norms, ffn_norms, sm_ffn_ratios = [], [], []
    out_norms = []

    for i, block in enumerate(model.blocks):
        sm_n = sum(p.norm(2).item()**2 for p in block.spatial_mixer.parameters()) ** 0.5
        ffn_n = sum(p.norm(2).item()**2 for p in block.channel_mixer.parameters()) ** 0.5
        sm_norms.append(sm_n); ffn_norms.append(ffn_n)
        sm_ffn_ratios.append(sm_n / (ffn_n + 1e-12))

        # out-projection specifically
        for module in [block.spatial_mixer, block.channel_mixer]:
            for pname, p in module.named_parameters():
                if "out" in pname or "contract" in pname:
                    out_norms.append((i, f"{type(module).__name__}.{pname}", float(p.norm(2))))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    xs = range(model.n_layer)

    axes[0].plot(xs, sm_norms,  marker="o", color=COLORS[0], lw=2, label="Spatial Mixer")
    axes[0].plot(xs, ffn_norms, marker="s", color=COLORS[2], lw=2, label="Channel Mixer")
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Total L2 Norm")
    axes[0].set_title("SM vs FFN Total Norms", color=COLORS[0])
    axes[0].legend(fontsize=9); axes[0].grid(True, ls="--", alpha=0.4)

    ratio_colors = [COLORS[3] if r > 3 or r < 0.33 else COLORS[2] for r in sm_ffn_ratios]
    axes[1].bar(xs, sm_ffn_ratios, color=ratio_colors, alpha=0.85)
    axes[1].axhline(1.0, color="#9CA3AF", lw=1, ls="--", label="balance (ratio=1)")
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("‖SM‖ / ‖FFN‖")
    axes[1].set_title("SM / FFN Norm Ratio per Layer\n(>3 or <0.33 flagged)", color=COLORS[0])
    axes[1].legend(fontsize=8); axes[1].grid(True, ls="--", alpha=0.4, axis="y")

    if out_norms:
        out_layers = [o[0] for o in out_norms]
        out_vals   = [o[2] for o in out_norms]
        axes[2].scatter(out_layers, out_vals,
                        c=[COLORS[i % len(COLORS)] for i in out_layers], s=60, alpha=0.8)
        axes[2].set_xlabel("Layer"); axes[2].set_ylabel("L2 Norm")
        axes[2].set_title("Out-projection Norms\n(residual-stream writers)", color=COLORS[0])
        axes[2].grid(True, ls="--", alpha=0.4)
    else:
        axes[2].text(0.5, 0.5, "No out/contract projections found",
                     ha="center", va="center", transform=axes[2].transAxes)

    fig.tight_layout(); st.image(_fig_to_bytes(fig))

    max_ratio = max(sm_ffn_ratios); min_ratio = min(sm_ffn_ratios)
    c1, c2, c3 = st.columns(3)
    c1.metric("Max SM/FFN ratio", f"{max_ratio:.3f}")
    c2.metric("Min SM/FFN ratio", f"{min_ratio:.3f}")
    c3.metric("Ratio std", f"{np.std(sm_ffn_ratios):.3f}")
    if max_ratio > 3:
        st.warning(f"⚠️  Layer {sm_ffn_ratios.index(max_ratio)} has SM much larger than FFN "
                   f"(ratio {max_ratio:.2f}) — spatial pathway may dominate residual updates.")


def _research_activation_stats(model):
    """
    Comprehensive per-layer activation statistics: mean, std, kurtosis, skewness,
    and a grid of histograms.  Helps detect mode collapse, exploding activations,
    or distribution shift across layers.
    """
    st.markdown("### 🎯 Activation Statistics Deep Dive")
    st.caption(
        "Captures residual-stream activations at each layer and computes "
        "mean, std, kurtosis (tail weight), and skewness. "
        "High kurtosis → heavy tails / spiky activations.  "
        "Increasing std across layers → potential activation explosion.")

    col1, col2 = st.columns(2)
    n_seqs  = col1.slider("Batch size", 1, 8, 2, key="actstat_n")
    seq_len = col2.slider("Sequence length", 8, 128, 48, key="actstat_seq")

    if st.button("▶ Compute Activation Statistics", key="actstat_run"):
        with st.spinner("Hooking all layers..."):
            layer_acts: Dict[int, torch.Tensor] = {}
            handles = []

            def _make_hook(i):
                def hook(m, inp, out):
                    t = out[0] if isinstance(out, (tuple, list)) else out
                    if isinstance(t, torch.Tensor):
                        layer_acts[i] = t.detach().float().cpu()
                return hook

            for i, block in enumerate(model.blocks):
                handles.append(block.register_forward_hook(_make_hook(i)))
            try:
                model.eval()
                x = torch.randint(0, model.vocab_size, (n_seqs, seq_len))
                with torch.no_grad(): model(x)
            except Exception as e:
                st.error(f"Forward pass failed: {e}")
                for h in handles: h.remove(); return
            finally:
                for h in handles: h.remove()

            stats = []
            for i, act in sorted(layer_acts.items()):
                flat = act.reshape(-1).numpy()
                from scipy import stats as scipy_stats
                kurt = float(scipy_stats.kurtosis(flat, fisher=True))
                skew = float(scipy_stats.skew(flat))
                stats.append({"layer": i,
                               "mean": float(flat.mean()), "std": float(flat.std()),
                               "kurtosis": kurt, "skewness": skew,
                               "p1": float(np.percentile(flat, 1)),
                               "p99": float(np.percentile(flat, 99)),
                               "_flat": flat})
            st.session_state["_actstat"] = stats

    if "_actstat" in st.session_state:
        stats = st.session_state["_actstat"]
        if not stats: return

        import pandas as pd
        df_display = pd.DataFrame([{k: v for k, v in s.items() if k != "_flat"} for s in stats])
        df_display = df_display.round(4)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        xs = [s["layer"] for s in stats]

        axes[0,0].plot(xs, [s["mean"] for s in stats], marker="o", color=COLORS[0], lw=2)
        axes[0,0].fill_between(xs,
            [s["mean"]-s["std"] for s in stats],
            [s["mean"]+s["std"] for s in stats], alpha=0.15, color=COLORS[0])
        axes[0,0].set_title("Mean ± Std per Layer", color=COLORS[0])
        axes[0,0].set_xlabel("Layer"); axes[0,0].axhline(0, color="#9CA3AF", lw=0.8, ls="--")
        axes[0,0].grid(True, ls="--", alpha=0.4)

        axes[0,1].plot(xs, [s["std"] for s in stats], marker="o", color=COLORS[1], lw=2)
        axes[0,1].set_title("Std per Layer  (growing = potential explosion)", color=COLORS[1])
        axes[0,1].set_xlabel("Layer"); axes[0,1].grid(True, ls="--", alpha=0.4)

        axes[1,0].plot(xs, [s["kurtosis"] for s in stats], marker="o", color=COLORS[3], lw=2)
        axes[1,0].axhline(0, color="#9CA3AF", lw=0.8, ls="--", label="Gaussian (kurt=0)")
        axes[1,0].set_title("Excess Kurtosis  (>3 = heavy tails)", color=COLORS[3])
        axes[1,0].set_xlabel("Layer"); axes[1,0].legend(fontsize=8); axes[1,0].grid(True, ls="--", alpha=0.4)

        axes[1,1].plot(xs, [s["skewness"] for s in stats], marker="o", color=COLORS[2], lw=2)
        axes[1,1].axhline(0, color="#9CA3AF", lw=0.8, ls="--", label="symmetric")
        axes[1,1].set_title("Skewness  (0 = symmetric)", color=COLORS[2])
        axes[1,1].set_xlabel("Layer"); axes[1,1].legend(fontsize=8); axes[1,1].grid(True, ls="--", alpha=0.4)

        fig.suptitle("Activation Statistics Across Layers", color=COLORS[0], fontsize=12)
        fig.tight_layout(); st.image(_fig_to_bytes(fig))

        # Mini histogram grid
        n_layers = len(stats)
        cols_g = min(8, n_layers)
        rows_g = math.ceil(n_layers / cols_g)
        fig2, axes2 = plt.subplots(rows_g, cols_g, figsize=(cols_g * 2, rows_g * 2))
        axes2_flat = np.array(axes2).flatten() if n_layers > 1 else [axes2]
        for idx, s in enumerate(stats):
            ax = axes2_flat[idx]
            flat = s["_flat"]
            p01, p99 = s["p1"], s["p99"]
            ax.hist(flat, bins=50, range=(p01, p99),
                    color=COLORS[idx % len(COLORS)], alpha=0.8, density=True, linewidth=0)
            ax.set_title(f"L{s['layer']}", fontsize=7); ax.tick_params(labelsize=5)
            ax.set_xticks([]); ax.set_yticks([])
        for idx in range(n_layers, len(axes2_flat)):
            axes2_flat[idx].set_visible(False)
        fig2.suptitle("Per-layer activation histograms (p1–p99)", color=COLORS[0], fontsize=9)
        fig2.tight_layout(); st.image(_fig_to_bytes(fig2))


def _render_welcome():
    st.markdown("## Welcome to ModelLens 🔬")
    st.info("No model loaded yet. Use the **sidebar** to load one.", icon="👈")
    st.markdown("""
### How to load a model

#### Option 1 — Save from your script, load via sidebar
```python
torch.save(model, "my_model.pt")
# or native Vathos checkpoint (preserves training history):
model.save_checkpoint("my_model-checkpoint.pt")
```
Then in the sidebar choose **Local file path** or **Upload**.

#### Option 2 — Run from script
```python
from ModelLens import run_inspector
run_inspector(model)
```

### Page guide
| Page | What it shows |
|---|---|
| **Model Overview** | Architecture table, skip-λ gates, parameter L2 norms |
| **Training History** | Loss curves, metric curves, raw tables |
| **Layer Inspector** | Weights/gradients/SVD/attention for any layer |
| **🌳 Model Anatomy** | Hierarchical tree, PCA curve, mock activation per tensor |
| **🔄 Similarity Analysis** | Head similarity, MLP cross-layer similarity, residual stream |
| **Multi-Layer Comparison** | Side-by-side SVD / distribution across layers |
| **Attention Explorer** | Attention maps + FFN concept manifold |
""")


def _render_app():
    # ── Re-apply theme CSS & update COLORS for this run ──────────────────────
    theme_names = list(THEMES.keys())
    active = THEMES[st.session_state.get("_theme", theme_names[0])]
    st.markdown(f"<style>{active['css']}</style>", unsafe_allow_html=True)
    plt.rcParams.update(active["rc"])
    global COLORS
    COLORS = active["colors"]

    st.title(active["title"])

    # ── Single sidebar block ──────────────────────────────────────────────────
    with st.sidebar:
        # Theme picker
        st.markdown("### 🎨 UI Theme")
        current = st.session_state.get("_theme", theme_names[0])
        chosen = st.radio("", theme_names,
                          index=theme_names.index(current),
                          label_visibility="collapsed",
                          key="_theme_radio")
        if chosen != st.session_state.get("_theme"):
            st.session_state["_theme"] = chosen
            st.rerun()
        st.markdown("---")

        # Model loader + nav
        model = _render_loader_sidebar()
        if model is not None:
            st.markdown("---")
            st.markdown("### Navigate")
            page = st.radio("", [
                "Model Overview", "Training History", "Layer Inspector",
                "🌳 Model Anatomy", "🔄 Similarity Analysis",
                "Multi-Layer Comparison", "Attention Explorer",
                "🔬 Research Tools",
            ], label_visibility="collapsed", key="nav_page")

    pending_sd = st.session_state.get("_pending_state_dict", None)
    if pending_sd is not None and model is None:
        _render_state_dict_panel(sd=pending_sd, fmt=st.session_state.get("_load_fmt","state_dict"),
                                 source_name=st.session_state.get("_pending_source_name","checkpoint"))
        return

    if model is None:
        _render_welcome(); return

    if page == "Model Overview":
        render_model_overview(model)
    elif page == "Training History":
        render_training_history(model)
    elif page == "Layer Inspector":
        render_layer_inspector(model)
    elif page == "🌳 Model Anatomy":
        render_model_anatomy(model)
    elif page == "🔄 Similarity Analysis":
        render_similarity_matrix(model)
    elif page == "Multi-Layer Comparison":
        render_comparison(model)
    elif page == "Attention Explorer":
        render_attention_explorer(model)
    elif page == "🔬 Research Tools":
        render_research_tools(model)


# ──────────────────────────────────────────────────────────────────────────────
#  PRELOAD & ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

_PRELOADED_MODEL = None
if _PRELOADED_MODEL is not None and "model" not in st.session_state:
    st.session_state["model"] = _PRELOADED_MODEL
    st.session_state["_load_fmt"] = "preloaded"
    st.session_state["_load_name"] = type(_PRELOADED_MODEL).__name__


def run_inspector(model_to_inspect=None):
    """Utility to launch inspector from another script: run_inspector(model)"""
    from streamlit.web import cli as stcli
    if model_to_inspect is not None:
        global _PRELOADED_MODEL
        _PRELOADED_MODEL = model_to_inspect
    sys.argv = ["streamlit", "run", __file__]
    sys.exit(stcli.main())


if __name__ == '__main__':
    _render_app()