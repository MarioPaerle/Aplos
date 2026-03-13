"""
Vathos ModdedFormer Inspector
==============================
Run with:
    streamlit run vathos_inspector.py

Usage in your script:
    from vathos_inspector import run_inspector
    run_inspector(model)          # blocking, launches Streamlit
    # or just run the file directly and set MODEL at the bottom
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import io, math, sys
from typing import Optional, Dict, List, Tuple

# Importiamo transformers per la tokenizzazione
try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vathos Inspector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
#  DARK THEME CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #0D0D1A !important;
    color: #D8D8F0;
}
[data-testid="stSidebar"] {
    background-color: #11111F !important;
    border-right: 1px solid #252540;
}
[data-testid="stSidebar"] * { color: #C0C0E0 !important; }
h1 { color: #A78BFA !important; letter-spacing: 1px; }
h2, h3 { color: #818CF8 !important; }
h4 { color: #93C5FD !important; }
.stMetric label { color: #9CA3AF !important; font-size: 0.78rem !important; }
.stMetric [data-testid="stMetricValue"] { color: #C4B5FD !important; font-size: 1.1rem !important; }
div[data-testid="stTabs"] button { color: #9CA3AF; }
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #A78BFA !important;
    border-bottom: 2px solid #A78BFA;
}
.stSelectbox label, .stMultiSelect label, .stSlider label { color: #9CA3AF !important; }
.stExpander { border: 1px solid #252540 !important; }
.stExpander summary { color: #818CF8 !important; }
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 700;
    margin-right: 4px;
}
.badge-attn  { background: #312E81; color: #A5B4FC; }
.badge-ffn   { background: #1F2937; color: #86EFAC; }
.badge-skip  { background: #7F1D1D; color: #FCA5A5; }
.badge-zero  { background: #713F12; color: #FCD34D; }
.badge-param { background: #164E63; color: #67E8F9; }
.layer-card {
    background: #13132A;
    border: 1px solid #252550;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  MATPLOTLIB DARK THEME
# ──────────────────────────────────────────────────────────────────────────────
DARK = {
    "figure.facecolor": "#0D0D1A",
    "axes.facecolor": "#0D0D1A",
    "axes.edgecolor": "#252550",
    "axes.labelcolor": "#C0C0E0",
    "xtick.color": "#9CA3AF",
    "ytick.color": "#9CA3AF",
    "text.color": "#D8D8F0",
    "grid.color": "#1E1E3A",
    "grid.alpha": 0.6,
    "legend.facecolor": "#11111F",
    "legend.edgecolor": "#252550",
}
plt.rcParams.update(DARK)

COLORS = ["#A78BFA", "#60A5FA", "#34D399", "#F472B6", "#FB923C",
          "#FBBF24", "#38BDF8", "#A3E635", "#E879F9", "#F87171"]


# ──────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    return buf.getvalue()


def _tensor_stats(t: torch.Tensor) -> dict:
    f = t.detach().float().cpu()
    v = f.numpy().flatten()
    return {
        "shape": tuple(t.shape),
        "numel": t.numel(),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "min": float(v.min()),
        "max": float(v.max()),
        "l2": float(np.linalg.norm(v)),
        "sparsity": float((np.abs(v) < 1e-6).mean()),
    }


def _svd_values(t: torch.Tensor) -> Optional[np.ndarray]:
    f = t.detach().float().cpu()
    if f.ndim == 2:
        try:
            return torch.linalg.svdvals(f).numpy()
        except Exception:
            return None
    if f.ndim == 3:
        results = []
        for i in range(f.shape[0]):
            try:
                results.append(torch.linalg.svdvals(f[i]).numpy())
            except Exception:
                pass
        return np.stack(results) if results else None
    return None


def _collect_named_params(block) -> Dict[str, torch.Tensor]:
    params = {}
    for name, p in block.named_parameters():
        params[name] = p
    return params


def _is_attention(mixer) -> bool:
    cls_name = type(mixer).__name__.lower()
    return "attention" in cls_name or "attn" in cls_name


def _get_attention_weights(mixer, x: torch.Tensor) -> Optional[np.ndarray]:
    if hasattr(mixer, "get_attention_weights"):
        try:
            with torch.no_grad():
                w = mixer.get_attention_weights(x)
            return w.detach().float().cpu().numpy()
        except Exception:
            pass
    try:
        with torch.no_grad():
            B, L, D = x.shape
            if hasattr(mixer, "qkv"):
                proj = mixer.qkv
                n_heads = mixer.n_heads
                head_dim = mixer.head_dim
                qkv = proj(x).view(B, L, 3, n_heads, head_dim)
                q, k, v = qkv.unbind(dim=2)
                q, k = q.transpose(1, 2).float(), k.transpose(1, 2).float()
                scale = math.sqrt(head_dim)
                scores = torch.matmul(q, k.transpose(-2, -1)) / scale
                if getattr(mixer, "causal", True):
                    mask = torch.tril(torch.ones(L, L, device=x.device)).bool()
                    scores = scores.masked_fill(~mask, float("-inf"))
                weights = torch.softmax(scores, dim=-1)
                return weights[0].cpu().numpy()
            elif hasattr(mixer, "qk"):
                proj = mixer.qk
                n_heads = mixer.n_heads
                head_dim = mixer.head_dim
                qk = proj(x).view(B, L, 2, n_heads, head_dim)
                q, k = qk.unbind(dim=2)
                q, k = q.transpose(1, 2).float(), k.transpose(1, 2).float()
                scale = math.sqrt(head_dim)
                scores = torch.matmul(q, k.transpose(-2, -1)) / scale
                if getattr(mixer, "causal", True):
                    mask = torch.tril(torch.ones(L, L, device=x.device)).bool()
                    scores = scores.masked_fill(~mask, float("-inf"))
                weights = torch.softmax(scores, dim=-1)
                return weights[0].cpu().numpy()
    except Exception:
        pass
    return None


def _get_ffn_activations(cm, x: torch.Tensor, temp: float) -> Optional[Dict[str, np.ndarray]]:
    """
    Estrae le proiezioni nel Channel Mixer analizzate come un'operazione di
    attenzione sui concetti.
    Ritorna:
    - pre_act: X @ W_expand (Logits non normalizzati)
    - post_act: Act(X @ W_expand) (Attivazione sparsa)
    - concept_attn: Softmax(X @ W_expand / temp) (Distribuzione sui concetti)
    """
    try:
        with torch.no_grad():
            pre_act = cm.expand(x)  # [B, L, M]
            post_act = cm.activation(pre_act) if hasattr(cm, "activation") else pre_act

            # Calcolo della "Concept Attention" con temperatura dinamica
            concept_attn = torch.softmax(pre_act / temp, dim=-1)

            return {
                "pre_act": pre_act[0].cpu().numpy(),
                "post_act": post_act[0].cpu().numpy(),
                "concept_attn": concept_attn[0].cpu().numpy()
            }
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────────────────────
#  PLOT FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def plot_weight_distribution(tensors: Dict[str, torch.Tensor], bins=80) -> bytes:
    n = len(tensors)
    cols = min(n, 3)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (name, t) in enumerate(tensors.items()):
        ax = axes[idx // cols][idx % cols]
        v = t.detach().float().cpu().numpy().flatten()
        st_val = _tensor_stats(t)
        ax.hist(v, bins=bins, color=COLORS[idx % len(COLORS)], alpha=0.8, linewidth=0)
        ax.set_title(f"{name}\nμ={st_val['mean']:.3e}  σ={st_val['std']:.3e}", fontsize=8)
        ax.set_xlabel("value", fontsize=7)
        ax.set_ylabel("count", fontsize=7)
        ax.grid(True, ls="--", alpha=0.4)
        ax.tick_params(labelsize=7)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Weight Distributions", fontsize=10, color="#A78BFA")
    fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_svd_spectrum(tensors: Dict[str, torch.Tensor], log_scale=True) -> bytes:
    eligible = {k: v for k, v in tensors.items() if v.ndim >= 2}
    if not eligible:
        return None
    n = len(eligible)
    cols = min(n, 3)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (name, t) in enumerate(eligible.items()):
        ax = axes[idx // cols][idx % cols]
        sv = _svd_values(t)
        if sv is None:
            ax.text(0.5, 0.5, "SVD failed", ha="center", va="center", transform=ax.transAxes)
            continue
        if sv.ndim == 2:
            for h in range(sv.shape[0]):
                ax.plot(sv[h], color=COLORS[idx % len(COLORS)], alpha=0.3, linewidth=0.8)
            ax.plot(sv.mean(0), color=COLORS[idx % len(COLORS)], linewidth=2, label="mean")
            ax.legend(fontsize=7)
        else:
            ax.plot(sv, color=COLORS[idx % len(COLORS)], linewidth=1.5, marker=".", markersize=3)

        cond = sv.flatten()[0] / (sv.flatten()[-1] + 1e-12)
        ax.set_title(f"{name}\ncond={cond:.1f}  rank≈{int((sv.flatten() > sv.flatten()[0] * 0.01).sum())}",
                     fontsize=8)
        ax.set_xlabel("index", fontsize=7)
        ax.set_ylabel("σ", fontsize=7)
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, ls="--", which="both", alpha=0.4)
        ax.tick_params(labelsize=7)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Singular Value Spectra", fontsize=10, color="#A78BFA")
    fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_gradient_distribution(tensors: Dict[str, torch.Tensor], bins=80) -> Optional[bytes]:
    grad_tensors = {k: v.grad for k, v in tensors.items() if v.grad is not None}
    if not grad_tensors:
        return None
    return plot_weight_distribution(grad_tensors, bins=bins)


def plot_gradient_svd(tensors: Dict[str, torch.Tensor], log_scale=True) -> Optional[bytes]:
    grad_tensors = {k: v.grad for k, v in tensors.items() if v.grad is not None and v.grad.ndim >= 2}
    if not grad_tensors:
        return None
    return plot_svd_spectrum(grad_tensors, log_scale=log_scale)


def plot_heatmap(data: np.ndarray, title: str, xlabel: str, ylabel: str,
                 tokens: Optional[List[str]] = None, cmap: str = "magma") -> bytes:
    """Plot generico per matrici L x L o L x M."""
    rows, cols = data.shape

    # Scaling euristico della figura
    fig_w = max(8, cols * 0.15) if tokens else 10
    fig_h = max(6, rows * 0.3) if tokens else 6

    fig, ax = plt.subplots(figsize=(min(fig_w, 20), min(fig_h, 15)))
    im = ax.imshow(data, aspect="auto", cmap=cmap)

    ax.set_title(title, color="#A78BFA", pad=20)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if tokens and len(tokens) == rows:
        clean_tokens = [t.replace('Ġ', ' ').replace(' ', ' ') for t in tokens]
        fs = max(4, min(10, int(400 / rows)))
        ax.set_yticks(range(rows))
        ax.set_yticklabels(clean_tokens, fontsize=fs)
        if rows == cols:  # Caso Attenzione
            ax.set_xticks(range(cols))
            ax.set_xticklabels(clean_tokens, rotation=90, fontsize=fs)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_attention_weights(attn_weights: np.ndarray, tokens: Optional[List[str]] = None) -> bytes:
    if attn_weights.ndim == 2:
        attn_weights = attn_weights[np.newaxis]

    H, L_q, L_k = attn_weights.shape
    cols = min(H, 4)
    rows = math.ceil(H / cols)

    if tokens is not None:
        cell_size = max(0.2, min(0.6, 12.0 / L_q))
        fig_w = max(4 * cols, cell_size * L_q * cols)
        fig_h = max(3.5 * rows, cell_size * L_q * rows + 1)
    else:
        fig_w = 4 * cols
        fig_h = 3.5 * rows

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.array(axes).flatten() if H > 1 else [axes]

    if tokens is not None:
        clean_tokens = [t.replace('Ġ', ' ').replace(' ', ' ') for t in tokens]
    else:
        clean_tokens = None

    for h in range(H):
        ax = axes[h]
        w = attn_weights[h]
        im = ax.imshow(w, aspect="auto", cmap="magma", vmin=0, vmax=w.max() + 1e-9)
        ax.set_title(f"Head {h}", fontsize=9, color="#C4B5FD")

        if clean_tokens is not None and len(clean_tokens) == L_q:
            ax.set_xticks(range(L_k))
            ax.set_yticks(range(L_q))
            fs = max(4, min(10, int(300 / L_q)))
            ax.set_xticklabels(clean_tokens, rotation=90, fontsize=fs, color="#9CA3AF")
            ax.set_yticklabels(clean_tokens, fontsize=fs, color="#9CA3AF")
        else:
            ax.set_xlabel("Key Position (Source)", fontsize=7)
            ax.set_ylabel("Query Position (Target)", fontsize=7)
            ax.tick_params(labelsize=6)

        fig.colorbar(im, ax=ax, shrink=0.8)

    for h in range(H, len(axes)):
        axes[h].set_visible(False)

    fig.suptitle("Attention Topology Analysis", fontsize=11, color="#A78BFA", y=1.02)
    fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_activation_distribution(pre_act: np.ndarray, post_act: np.ndarray, act_fn: Optional[nn.Module]) -> bytes:
    """
    Traccia la densità (PDF empirica) delle attivazioni prima della non linearità,
    sovrapposta alla funzione matematica f(x), per diagnosticare le code di distribuzione,
    i regimi di gating (per SiLU/GeLU) e l'eventuale incidenza di gradienti identici a zero
    (dead neurons per ReLU o saturazioni). Traccia anche il risultato marginale post-attivazione.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    pre_flat = pre_act.flatten()
    post_flat = post_act.flatten()

    # ── Subplot 1: Pre-act vs Activation Function ──
    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    # Niente clipping: calcoliamo il vero min e max per esporre esplicitamente le code e gli outlier
    true_min = float(pre_flat.min())
    true_max = float(pre_flat.max())

    # Assicuriamo un dominio visivo che includa almeno [-5, 5] per contestualizzare la non-linearità,
    # ma che si estenda fino agli estremi reali per non nascondere anomalie.
    x_min = min(-5.0, true_min - 1.0)
    x_max = max(5.0, true_max + 1.0)
    x_vals = np.linspace(x_min, x_max, 1000)

    # Calcolo della f(x)
    y_vals = x_vals  # Fallback: funzione lineare/identità
    if act_fn is not None:
        try:
            device = "cpu"
            # Se la funzione ha parametri apprendibili (es. PReLU), deduciamo il device
            if hasattr(act_fn, "parameters") and list(act_fn.parameters()):
                device = next(act_fn.parameters()).device
            with torch.no_grad():
                xt = torch.tensor(x_vals, dtype=torch.float32, device=device)
                yt = act_fn(xt)
                y_vals = yt.cpu().numpy()
        except Exception:
            pass # Se la valutazione fallisce o la topologia non è standard, lasciamo l'identità

    # Plot F(x) map (Matematica dell'attivazione)
    ax1.plot(x_vals, y_vals, color="#38BDF8", lw=3.0, label="$f(x)$ map", zorder=4)
    ax1.axvline(0, color="#4B5563", lw=1.5, ls="--", zorder=1)
    ax1.axhline(0, color="#4B5563", lw=1.5, ls="--", zorder=1)

    # Evidenziamo esplicitamente min e max sull'asse x per identificare istantaneamente leak di range
    ax1.plot(true_min, 0, marker='v', color='#EF4444', markersize=7, zorder=5)
    ax1.plot(true_max, 0, marker='v', color='#EF4444', markersize=7, zorder=5)
    ax1.text(true_min, 0.05, f"Min:\n{true_min:.1f}", color='#EF4444', ha='center', va='bottom', fontsize=8, transform=ax1.get_xaxis_transform())
    ax1.text(true_max, 0.05, f"Max:\n{true_max:.1f}", color='#EF4444', ha='center', va='bottom', fontsize=8, transform=ax1.get_xaxis_transform())

    ax1.set_xlabel("Pre-activation $x = (X W_{exp})$", fontsize=10)
    ax1.set_ylabel("Activation Output $f(x)$", color="#38BDF8", fontsize=10)
    ax1.tick_params(axis='y', labelcolor="#38BDF8")

    # Plot PDF della Pre-attivazione su asse gemello per non schiacciare f(x)
    ax1_twin.hist(pre_flat, bins=150, range=(x_min, x_max), color="#F472B6", alpha=0.5, density=True, zorder=2)
    ax1_twin.set_ylabel("Densità Empirica Input $p(x)$", color="#F472B6", fontsize=10)
    ax1_twin.tick_params(axis='y', labelcolor="#F472B6")

    ax1.set_title("Overlay: Activation $f(x)$ & Input Distribution", color="#A78BFA", fontsize=11)
    ax1.legend(loc="upper left", fontsize=8)

    # ── Subplot 2: Post-act distribution ──
    ax2 = axes[1]

    # Per limitare l'effetto di sparse activation estrema (molti zeri) sul binning
    post_min, post_max = np.percentile(post_flat, [0.0, 99.9])

    ax2.hist(post_flat, bins=120, range=(post_min, max(post_max, 1e-3)), color="#F472B6", alpha=0.6, density=True)
    ax2.axvline(0, color="#9CA3AF", lw=1, ls="--", alpha=0.5)

    ax2.set_xlabel("Post-activation $Act(X W_{exp})$", fontsize=9)
    ax2.set_ylabel("Density", fontsize=9)
    ax2.set_title("Post-Activation Distribution (Sparsity check)", color="#A78BFA", fontsize=10)
    ax2.grid(True, ls="--", alpha=0.3)

    fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_comparison_svd(layer_data: Dict[str, Dict[str, torch.Tensor]], param_key: str, log_scale=True) -> bytes:
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, (layer_name, params) in enumerate(layer_data.items()):
        if param_key not in params:
            continue
        sv = _svd_values(params[param_key])
        if sv is None:
            continue
        if sv.ndim == 2:
            sv = sv.mean(0)
        ax.plot(sv, label=layer_name, color=COLORS[idx % len(COLORS)], linewidth=1.5, marker=".", markersize=3,
                alpha=0.85)

    ax.set_xlabel("singular value index")
    ax.set_ylabel("σ")
    ax.set_title(f"SVD comparison: {param_key}", color="#A78BFA")
    if log_scale:
        ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", which="both", alpha=0.4)
    fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_comparison_hist(layer_data: Dict[str, Dict[str, torch.Tensor]], param_key: str, bins=60) -> bytes:
    fig, ax = plt.subplots(figsize=(9, 4))
    for idx, (layer_name, params) in enumerate(layer_data.items()):
        if param_key not in params:
            continue
        v = params[param_key].detach().float().cpu().numpy().flatten()
        ax.hist(v, bins=bins, alpha=0.45, label=layer_name, color=COLORS[idx % len(COLORS)], linewidth=0)

    ax.set_xlabel("value")
    ax.set_ylabel("count")
    ax.set_title(f"Distribution comparison: {param_key}", color="#A78BFA")
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_skip_lambdas(model) -> bytes:
    lambdas = {k: float(v.detach()) for k, v in model.skip_lambdas.items()}
    if not lambdas:
        return None
    fig, ax = plt.subplots(figsize=(max(6, len(lambdas) * 1.4), 4))
    keys = list(lambdas.keys())
    vals = [lambdas[k] for k in keys]
    bars = ax.bar(range(len(keys)), vals, color=COLORS[:len(keys)], alpha=0.85)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace("route_", "").replace("_to_", "→") for k in keys], rotation=30, ha="right",
                       fontsize=8)
    ax.axhline(0, color="#9CA3AF", linewidth=0.8, ls="--")
    ax.set_title("Skip-connection Gate Values (λ)", color="#A78BFA")
    ax.set_ylabel("λ (gate)")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003, f"{v:.4f}", ha="center", va="bottom", fontsize=7,
                color="#D8D8F0")
    fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_zeroskip_params(model) -> bytes:
    if not model.zeroskip:
        return None
    vals = [float(p.detach()) for p in model.zeroskip_params]
    fig, ax = plt.subplots(figsize=(max(6, len(vals) * 0.7), 4))
    ax.plot(vals, marker="o", color="#FBBF24", linewidth=1.5, markersize=6)
    ax.axhline(0, color="#9CA3AF", linewidth=0.8, ls="--")
    ax.set_xlabel("layer index")
    ax.set_ylabel("zeroskip α")
    ax.set_title("ZeroSkip Parameters (x₀ coefficient per layer)", color="#A78BFA")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_param_norms(model) -> bytes:
    layer_names, l2_norms = [], []
    for i, block in enumerate(model.blocks):
        for name, p in block.named_parameters():
            layer_names.append(f"L{i}.{name}")
            l2_norms.append(float(p.detach().norm(2)))

    fig, ax = plt.subplots(figsize=(max(8, len(layer_names) * 0.35), 5))
    xs = range(len(layer_names))
    ax.bar(xs, l2_norms, color="#60A5FA", alpha=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(layer_names, rotation=90, fontsize=5.5)
    ax.set_ylabel("L2 norm")
    ax.set_title("Per-parameter L2 Norms (all blocks)", color="#A78BFA")
    ax.grid(True, ls="--", alpha=0.4, axis="y")
    fig.tight_layout()
    return _fig_to_bytes(fig)


# ──────────────────────────────────────────────────────────────────────────────
#  MODEL OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────

def render_model_overview(model):
    st.markdown("## 🏗️ Model Overview")
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    has_grad = any(p.grad is not None for p in model.parameters())
    has_skips = any(s is not None for s in model.skips)
    n_attn_layers = sum(1 for b in model.blocks if _is_attention(b.spatial_mixer))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Layers", model.n_layer)
    c2.metric("Embed dim", model.embed_dim)
    c3.metric("Total params", f"{total_params:,}")
    c4.metric("Trainable", f"{train_params:,}")
    c5.metric("Has gradients", "✅" if has_grad else "❌")

    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("Vocab size", model.vocab_size)
    c7.metric("Max len", model.max_len)
    c8.metric("Attn layers", n_attn_layers)
    c9.metric("Skip conns", sum(s is not None for s in model.skips))
    c10.metric("ZeroSkip", "✅" if model.zeroskip else "❌")

    st.markdown("### Layer Graph")
    graph_html = _build_layer_graph_html(model)
    st.markdown(graph_html, unsafe_allow_html=True)

    if has_skips and model.skip_lambdas:
        st.markdown("### Skip-Connection Gate Values")
        img = plot_skip_lambdas(model)
        if img:
            st.image(img)

    if model.zeroskip:
        st.markdown("### ZeroSkip Coefficients")
        img = plot_zeroskip_params(model)
        if img:
            st.image(img)

    st.markdown("### Parameter L2 Norms")
    st.image(plot_param_norms(model))


def _build_layer_graph_html(model) -> str:
    rows = []
    rows.append('<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%;font-size:0.78rem">')
    rows.append("<tr><th style='padding:6px 10px;text-align:left;color:#818CF8'>Layer</th>"
                "<th style='padding:6px 10px;color:#818CF8'>d_model</th>"
                "<th style='padding:6px 10px;color:#818CF8'>FFN dim</th>"
                "<th style='padding:6px 10px;color:#818CF8'>Spatial mixer</th>"
                "<th style='padding:6px 10px;color:#818CF8'>Skip from</th>"
                "<th style='padding:6px 10px;color:#818CF8'>Skip to</th>"
                "<th style='padding:6px 10px;color:#818CF8'>Params</th></tr>")

    for i, block in enumerate(model.blocks):
        sm_name = type(block.spatial_mixer).__name__
        is_attn = _is_attention(block.spatial_mixer)
        sm_badge = f'<span class="badge badge-attn">{sm_name}</span>' if is_attn else f'<span class="badge badge-ffn">{sm_name}</span>'

        skip_from = ""
        skip_to = ""
        if model.skips[i] is not None:
            skip_to = f'<span class="badge badge-skip">→ L{model.skips[i]}</span>'
        for src, tgt in enumerate(model.skips):
            if tgt == i:
                skip_from = f'<span class="badge badge-skip">L{src} →</span>'

        n_params = sum(p.numel() for p in block.parameters())
        bg = "#13132A" if i % 2 == 0 else "#0D0D1A"
        rows.append(f"<tr style='background:{bg}'>"
                    f"<td style='padding:5px 10px;color:#A78BFA;font-weight:700'>Layer {i}</td>"
                    f"<td style='padding:5px 10px;text-align:center'>{model.d_models[i]}</td>"
                    f"<td style='padding:5px 10px;text-align:center'>{model.M_dims[i]}</td>"
                    f"<td style='padding:5px 10px'>{sm_badge}</td>"
                    f"<td style='padding:5px 10px'>{skip_from}</td>"
                    f"<td style='padding:5px 10px'>{skip_to}</td>"
                    f"<td style='padding:5px 10px'><span class='badge badge-param'>{n_params:,}</span></td>"
                    "</tr>")
    rows.append("</table></div>")
    return "".join(rows)


# ──────────────────────────────────────────────────────────────────────────────
#  LAYER INSPECTOR
# ──────────────────────────────────────────────────────────────────────────────

def render_layer_inspector(model):
    st.markdown("## 🔍 Layer Inspector")
    layer_idx = st.selectbox("Select layer", list(range(model.n_layer)),
                             format_func=lambda i: f"Layer {i}  —  {type(model.blocks[i].spatial_mixer).__name__}")
    block = model.blocks[layer_idx]
    sm = block.spatial_mixer
    cm = block.channel_mixer

    tags = []
    if _is_attention(sm):
        tags.append('<span class="badge badge-attn">Attention</span>')
    if model.skips[layer_idx] is not None:
        tags.append(f'<span class="badge badge-skip">→ skip L{model.skips[layer_idx]}</span>')
    for src, tgt in enumerate(model.skips):
        if tgt == layer_idx:
            tags.append(f'<span class="badge badge-skip">skip from L{src}</span>')
    if model.zeroskip:
        zval = float(model.zeroskip_params[layer_idx].detach())
        tags.append(f'<span class="badge badge-zero">zeroskip α={zval:.4f}</span>')
    st.markdown(" ".join(tags), unsafe_allow_html=True)

    comp_choice = st.radio("Component", ["Spatial Mixer (SM)", "Channel Mixer (FFN)", "Both"], horizontal=True,
                           key="layer_comp_choice")

    if comp_choice == "Spatial Mixer (SM)":
        selected_modules = {"spatial": sm}
    elif comp_choice == "Channel Mixer (FFN)":
        selected_modules = {"channel": cm}
    else:
        selected_modules = {"spatial": sm, "channel": cm}

    all_params: Dict[str, torch.Tensor] = {}
    for prefix, module in selected_modules.items():
        for name, p in module.named_parameters():
            all_params[f"{prefix}.{name}"] = p

    if not all_params:
        st.info("No parameters found in this component.")
        return

    tabs = st.tabs(
        ["📊 Distributions", "📉 SVD Spectrum", "🌊 Gradients", "🌊 Gradient SVD", "👁️ Attention Weights", "📋 Stats Table"])

    with tabs[0]:
        st.image(plot_weight_distribution(all_params))
    with tabs[1]:
        log_s = st.toggle("Log scale", value=True, key=f"svd_log_{layer_idx}")
        img = plot_svd_spectrum(all_params, log_scale=log_s)
        if img:
            st.image(img)
        else:
            st.info("No 2-D parameters to plot SVD for.")
    with tabs[2]:
        if not any(p.grad is not None for p in all_params.values()):
            st.warning("No gradients available. Run a forward+backward pass first.")
        else:
            img = plot_gradient_distribution(all_params)
            if img: st.image(img)
    with tabs[3]:
        if not any(p.grad is not None for p in all_params.values()):
            st.warning("No gradients available.")
        else:
            log_gs = st.toggle("Log scale", value=True, key=f"gsvd_log_{layer_idx}")
            img = plot_gradient_svd(all_params, log_scale=log_gs)
            if img:
                st.image(img)
            else:
                st.info("Gradients are all 1-D or unavailable.")
    with tabs[4]:
        if not _is_attention(sm):
            st.info("Spatial mixer is not an attention layer.")
        else:
            st.caption("Provide a sample input to visualise attention patterns.")
            seq_len = st.slider("Sequence length (dummy input)", 8, 256, 32, key=f"attn_seq_{layer_idx}")
            batch_s = 1
            d = model.d_models[layer_idx]
            if st.button("Compute attention weights", key=f"attn_btn_{layer_idx}"):
                dummy_x = torch.randn(batch_s, seq_len, d)
                sm.eval()
                w = _get_attention_weights(sm, dummy_x)
                if w is None:
                    st.error("Could not extract attention weights for this mixer type.")
                else:
                    st.image(plot_attention_weights(w))
                    st.caption(f"Shape: {w.shape}  |  max={w.max():.4f}  min={w.min():.4f}")
    with tabs[5]:
        rows = []
        for pname, p in all_params.items():
            s = _tensor_stats(p)
            rows.append({
                "Parameter": pname, "Shape": str(s["shape"]), "Numel": f"{s['numel']:,}",
                "Mean": f"{s['mean']:.4e}", "Std": f"{s['std']:.4e}", "Min": f"{s['min']:.4e}",
                "Max": f"{s['max']:.4e}", "L2": f"{s['l2']:.4e}", "Sparsity": f"{s['sparsity']:.2%}",
                "Has grad": "✅" if p.grad is not None else "❌",
            })
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
#  MULTI-LAYER COMPARISON
# ──────────────────────────────────────────────────────────────────────────────

def render_comparison(model):
    st.markdown("## ⚖️ Multi-Layer Comparison")
    layers_sel = st.multiselect("Select layers to compare", list(range(model.n_layer)),
                                default=list(range(min(model.n_layer, 4))), format_func=lambda i: f"Layer {i}")
    if len(layers_sel) < 2:
        st.info("Select at least 2 layers.")
        return

    layer_data: Dict[str, Dict[str, torch.Tensor]] = {}
    all_param_keys: set = set()
    for i in layers_sel:
        block = model.blocks[i]
        params = {}
        for prefix, mod in [("sm", block.spatial_mixer), ("ffn", block.channel_mixer)]:
            for name, p in mod.named_parameters():
                params[f"{prefix}.{name}"] = p
        layer_data[f"L{i}"] = params
        all_param_keys.update(params.keys())

    common_keys = [k for k in sorted(all_param_keys) if all(k in layer_data[f"L{i}"] for i in layers_sel)]
    if not common_keys:
        st.warning("No common parameter names across selected layers.")
        return

    param_key = st.selectbox("Parameter to compare", common_keys)
    mode = st.radio("Compare by", ["SVD Spectrum", "Distribution"], horizontal=True, key="compare_mode")
    log_s = st.toggle("Log scale", value=True, key="cmp_log")

    if mode == "SVD Spectrum":
        img = plot_comparison_svd(layer_data, param_key, log_scale=log_s)
    else:
        img = plot_comparison_hist(layer_data, param_key)
    st.image(img)

    st.markdown("#### Per-layer stats for selected parameter")
    import pandas as pd
    rows = []
    for lname, params in layer_data.items():
        if param_key not in params: continue
        s = _tensor_stats(params[param_key])
        rows.append({"Layer": lname, **{k: v for k, v in s.items() if k != "shape"}})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
#  ATTENTION EXPLORER (CON FFN MANIFOLD)
# ──────────────────────────────────────────────────────────────────────────────

def render_attention_explorer(model):
    st.markdown("## 🧠 Manifold Explorer")

    attn_layers = [i for i, b in enumerate(model.blocks) if _is_attention(b.spatial_mixer)]
    if not attn_layers:
        st.info("No attention layers found in this model.")
        return

    layer_idx = st.selectbox("Seleziona Layer", attn_layers,
                             format_func=lambda i: f"Layer {i} ({type(model.blocks[i].spatial_mixer).__name__})")

    block = model.blocks[layer_idx]
    sm = block.spatial_mixer
    cm = block.channel_mixer
    d = model.d_models[layer_idx]

    st.markdown(f"**Mixer type:** `{type(sm).__name__}`  |  "
                f"**n_heads:** `{getattr(sm, 'n_heads', '?')}`  |  "
                f"**head_dim:** `{getattr(sm, 'head_dim', '?')}`  |  "
                f"**causal:** `{getattr(sm, 'causal', '?')}`")

    input_mode = st.radio("Strategia di Test",
                          ["Pattern Sintetico (Dummy)", "Testo Reale (HuggingFace Tokenizer)"],
                          horizontal=True)

    if input_mode == "Pattern Sintetico (Dummy)":
        seq_len = st.slider("Sequence length (dummy)", 8, 512, 64)
        pattern = st.selectbox("Input pattern", ["random", "ones", "zeros", "eye"])
        if st.button("Analizza Pattern Sintetico"):
            if pattern == "random":
                x = torch.randn(1, seq_len, d)
            elif pattern == "ones":
                x = torch.ones(1, seq_len, d)
            elif pattern == "zeros":
                x = torch.zeros(1, seq_len, d)
            else:
                x = torch.zeros(1, seq_len, d)
                for j in range(min(seq_len, d)): x[0, j, j] = 1.0

            sm.eval()
            cm.eval()

            # Cache the computation to survive interactive Streamlit reruns
            st.session_state["ffn_explorer_cache"] = {
                "layer_idx": layer_idx, "spatial_x": x, "channel_x": x,
                "tokens": None, "mode": input_mode, "d_model": d
            }

    elif input_mode == "Testo Reale (HuggingFace Tokenizer)":
        if not HAS_TRANSFORMERS:
            st.error("Libreria `transformers` non trovata. Esegui `pip install transformers` nel tuo ambiente.")
            return

        st.info(
            "Esegue il calcolo catturando l'attivazione spaziale e di canale ($X^{(l)}$) tramite un PyTorch Forward Hook.",
            icon="ℹ️")

        c1, c2 = st.columns([1, 2])
        tokenizer_id = c1.text_input("Tokenizer HF Repository", "gpt2", help="E.g. gpt2, bert-base-uncased")
        use_bos = c1.checkbox("Aggiungi BOS token automaticamente", value=True)
        user_prompt = c2.text_area("Prompt (Testo)",
                                   "The mathematical structure of a transformer is an associative memory.")

        if st.button("Analizza Attenzione e FFN (Hook)"):
            with st.spinner("Tokenizzazione e Forward Pass (Hooking) in corso..."):
                try:
                    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_id)
                    full_prompt = (tokenizer.bos_token if use_bos and tokenizer.bos_token else "") + user_prompt
                    inputs = tokenizer(full_prompt, return_tensors="pt")
                    try:
                        device = next(model.parameters()).device
                    except StopIteration:
                        device = torch.device("cpu")

                    input_ids = inputs["input_ids"].to(device)
                    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                except Exception as e:
                    st.error(f"Errore caricamento Tokenizer HF: `{e}`")
                    return

                captured = {"spatial_x": None, "channel_x": None}

                def hook_spatial(m, i):
                    captured["spatial_x"] = i[0].detach()

                def hook_channel(m, i):
                    captured["channel_x"] = i[0].detach()

                h1 = sm.register_forward_pre_hook(hook_spatial)
                h2 = cm.register_forward_pre_hook(hook_channel)

                try:
                    model.eval()
                    with torch.no_grad():
                        model(input_ids)
                except Exception as e:
                    st.error(
                        f"Errore durante il Forward Pass del modello: `{e}`. Verifica che `model(input_ids)` sia supportato.")
                    h1.remove()
                    h2.remove()
                    return
                finally:
                    h1.remove()
                    h2.remove()

                if captured["spatial_x"] is None or captured["channel_x"] is None:
                    st.error("Il Forward Hook non ha catturato l'attivazione.")
                    return

                # Cache the computation to survive interactive Streamlit reruns
                st.session_state["ffn_explorer_cache"] = {
                    "layer_idx": layer_idx, "spatial_x": captured["spatial_x"], "channel_x": captured["channel_x"],
                    "tokens": tokens, "mode": input_mode, "d_model": d
                }

    # --- RENDER UI FROM CACHE ---
    # Questo approccio garantisce che la UI interattiva (slider scala) funzioni senza ricalcolare la forward pass
    cache = st.session_state.get("ffn_explorer_cache")
    if cache and cache["layer_idx"] == layer_idx and cache["mode"] == input_mode:
        tab_attn, tab_ffn = st.tabs(["👁️ Spatial Attention (QK)", "💎 Channel Expansion (XW)"])

        with tab_attn:
            X_manifold = cache["spatial_x"]
            if cache["tokens"] is not None:
                st.caption(
                    f"Dimensione dell'attivazione catturata $X^{{(l)}}$: `{list(X_manifold.shape)}` | Token effettivi: {len(cache['tokens'])}")
            w_attn = _get_attention_weights(sm, X_manifold)
            _render_attention_output(w_attn, tokens=cache["tokens"])

        with tab_ffn:
            st.markdown("### Impostazioni e Metriche Manifold FFN")

            c_scale, _ = st.columns([1, 2])
            default_temp = float(math.sqrt(cache["d_model"]))
            temp_scale = c_scale.slider("Scala / Temperatura ($\\tau$ per la Concept Attention)",
                                        min_value=0.1, max_value=max(100.0, default_temp * 2),
                                        value=default_temp, step=0.1)

            ffn_data = _get_ffn_activations(cm, cache["channel_x"], temp_scale)
            if ffn_data is not None:
                view_mode = st.radio(
                    "Seleziona Proiezione Matrice",
                    ["Attivazioni FFN $Act(X W_{expand})$",
                     "Logits Grezzi $X W_{expand}$",
                     "Concept Attention $\\text{Softmax}(X W_{expand} / \\tau)$"],
                    horizontal=True, key=f"view_mode_ffn"
                )

                if "Act" in view_mode:
                    mat = ffn_data["post_act"]
                    title = "FFN Activations $Act(X W_{expand})$"
                elif "Grezzi" in view_mode:
                    mat = ffn_data["pre_act"]
                    title = "Raw Concept Logits $X W_{expand}$"
                else:
                    mat = ffn_data["concept_attn"]
                    title = "Concept Attention $\\text{Softmax}(X W_{expand} / \\tau)$"

                # Metriche Matematiche del Manifold Selezionato
                sparsity = float((np.abs(mat) < 1e-6).mean())
                l2_norm = float(np.linalg.norm(mat))

                try:
                    # SVD su matrice 2D [L, M_dim]
                    svd_vals = np.linalg.svd(mat, compute_uv=False)
                    # Rango Approssimato: numero di autovalori > 0.1% dell'autovalore massimo
                    eff_rank = int(np.sum(svd_vals > svd_vals[0] * 1e-3))
                except np.linalg.LinAlgError:
                    svd_vals = None
                    eff_rank = "N/A"

                c_m1, c_m2, c_m3 = st.columns(3)
                c_m1.metric("Sparsità (Valori prossimi allo zero)", f"{sparsity:.2%}")
                c_m2.metric("Norma L2 Globale ($\|Y\|_2$)", f"{l2_norm:.2f}")
                c_m3.metric("Rango Effettivo (SVD > 0.1%)", eff_rank)

                # Rendering Matrice
                display_neurons = min(128, mat.shape[1])
                st.image(plot_heatmap(mat[:, :display_neurons],
                                      f"{title} (Primi {display_neurons} neuroni su {mat.shape[1]})",
                                      "Neurons (Hidden Features/Concepts)", "Tokens", cache["tokens"]))

                # ── NEW: Distribuzione Pre/Post Attivazione ──
                st.markdown("### Analisi del Regime di Saturazione e Geometria dell'Attivazione")
                st.info(
                    "Sovrapporre la PDF marginale $p(x)$ dei logits $x = X W_{exp}$ alla mappa non-lineare $f(x)$ "
                    "permette di diagnosticare il condition number locale della Jacobiana. "
                    "Un eccesso di massa asintotica può portare ad annullamento del gradiente, mentre (per SiLU/GeLU) "
                    "masse localizzate sui minimi locali suggeriscono la presenza di strong gating attivi sulla feature."
                )

                # Cerchiamo di localizzare l'attivazione nel channel mixer per proiettarla matematicamente
                act_fn = getattr(cm, "activation", None)
                img_act = plot_activation_distribution(ffn_data["pre_act"], ffn_data["post_act"], act_fn)
                if img_act:
                    st.image(img_act)

                # Rendering Spettro SVD
                if svd_vals is not None:
                    st.markdown("### Spettro Singolare del Manifold FFN")
                    fig_svd, ax_svd = plt.subplots(figsize=(6, 2.5))
                    ax_svd.plot(svd_vals, color=COLORS[1], linewidth=1.5, marker=".", markersize=3)
                    ax_svd.set_yscale("log")
                    ax_svd.set_title("Spettro dei Valori Singolari ($\Sigma$)", color="#A78BFA", fontsize=10)
                    ax_svd.set_xlabel("Indice $\sigma_i$", fontsize=8)
                    ax_svd.set_ylabel("Magnitudo (Log Scale)", fontsize=8)
                    ax_svd.grid(True, ls="--", alpha=0.4)
                    fig_svd.tight_layout()
                    st.image(_fig_to_bytes(fig_svd))
            else:
                st.warning("Impossibile estrarre attivazioni FFN.")


def _render_attention_output(w: np.ndarray, tokens: Optional[List[str]] = None):
    if w is None:
        st.error("Cannot compute weights for this mixer. Re-check the projection matrices shape.")
        return

    st.image(plot_attention_weights(w, tokens=tokens))

    entropy_per_head = []
    w_calc = w if w.ndim == 3 else w[np.newaxis]

    for h in range(w_calc.shape[0]):
        wh = w_calc[h] + 1e-12
        ent = -(wh * np.log(wh)).sum(axis=-1).mean()
        entropy_per_head.append(float(ent))

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(len(entropy_per_head)), entropy_per_head, color=COLORS)
    ax.set_xlabel("Head index")
    ax.set_ylabel("Avg Entropy (nats)")
    ax.set_title("Per-head Expected Shannon Entropy $\mathbb{E}[\mathcal{H}]$", color="#A78BFA")
    ax.grid(True, ls="--", alpha=0.4, axis="y")
    fig.tight_layout()
    st.image(_fig_to_bytes(fig))


# ──────────────────────────────────────────────────────────────────────────────
#  LOADER
# ──────────────────────────────────────────────────────────────────────────────

def _try_load_pt(raw: bytes):
    buf = io.BytesIO(raw)
    try:
        obj = torch.load(buf, map_location="cpu", weights_only=False)
    except Exception as e:
        return None, None, str(e)
    if isinstance(obj, nn.Module): return obj, "full_model", None
    if not isinstance(obj, dict): return None, None, f"Unexpected object type: {type(obj).__name__}"
    if "model_state_dict" in obj: return obj["model_state_dict"], "vathos_checkpoint", None
    for key in ("model", "module", "net", "network"):
        if key in obj and isinstance(obj[key], nn.Module): return obj[key], f"dict['{key}']", None
    if obj and all(isinstance(v, torch.Tensor) for v in obj.values()): return obj, "state_dict", None
    return None, None, ("Unrecognised checkpoint format.")


def _infer_config_from_state_dict(sd: dict) -> dict:
    cfg = {}
    block_indices = {int(k.split(".")[1]) for k in sd if k.startswith("blocks.") and k.split(".")[1].isdigit()}
    cfg["n_layers"] = max(block_indices) + 1 if block_indices else 0
    for k, v in sd.items():
        if "embedder.embedding.weight" in k:
            cfg["vocab_size"] = v.shape[0]
            cfg["embed_dim"] = v.shape[1]
            break
    d_models, M_dims = [], []
    for i in range(cfg.get("n_layers", 0)):
        exp_key = f"blocks.{i}.channel_mixer.expand.weight"
        con_key = f"blocks.{i}.channel_mixer.contract.weight"
        if exp_key in sd and con_key in sd:
            M_dims.append(sd[exp_key].shape[0])
            d_models.append(sd[con_key].shape[0])
        else:
            d_models.append(cfg.get("embed_dim", "?"))
            M_dims.append("?")
    cfg["d_models"] = d_models
    cfg["M_dims"] = M_dims
    cfg["has_zeroskip"] = any("zeroskip_params" in k for k in sd)
    cfg["skip_keys"] = [k for k in sd if "skip_lambdas" in k]
    for i in range(cfg.get("n_layers", 0)):
        qkv_key = f"blocks.{i}.spatial_mixer.qkv.weight"
        qk_key = f"blocks.{i}.spatial_mixer.qk.weight"
        d = d_models[i] if i < len(d_models) and isinstance(d_models[i], int) else cfg.get("embed_dim", 0)
        if qkv_key in sd and d:
            cfg["spatial_type"] = "MultiheadAttentionMixer"
            break
        elif qk_key in sd:
            cfg["spatial_type"] = "MultiheadAttentionMixerNOV"
            break
    else:
        cfg["spatial_type"] = "unknown"
    return cfg


def _render_state_dict_panel(sd: dict, fmt: str, source_name: str):
    is_vathos = (fmt == "vathos_checkpoint")
    if is_vathos:
        st.info(f"**Vathos checkpoint** detected (`{source_name}`).  \nWeights + training state found.", icon="ℹ️")
    else:
        st.warning(f"**Bare state dict** loaded (`{source_name}`).", icon="⚠️")

    cfg = _infer_config_from_state_dict(sd)
    st.markdown("### Inferred architecture")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Layers", cfg["n_layers"])
    c2.metric("Embed dim", cfg.get("embed_dim", "?"))
    c3.metric("Vocab size", cfg.get("vocab_size", "?"))
    c4.metric("Spatial type", cfg.get("spatial_type", "?"))

    st.markdown("**Per-layer dims** (inferred from weight shapes)")
    import pandas as pd
    rows = [{"layer": i, "d_model": d, "M_dim (FFN)": m, "Spatial Mixer": cfg.get("spatial_type", "?")} for i, (d, m) in
            enumerate(zip(cfg["d_models"], cfg["M_dims"]))]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    default_code = _build_default_snippet(cfg)
    code = st.text_area("Constructor code", value=default_code, height=280, key="reconstruct_code")
    label = "🔧 Reconstruct & load_checkpoint()" if is_vathos else "🔧 Reconstruct & load_state_dict()"
    if st.button(label, use_container_width=True, type="primary"):
        _reconstruct_from_code(code, sd, source_name, fmt)


def _build_default_snippet(cfg: dict) -> str:
    n = cfg["n_layers"]
    d = cfg.get("embed_dim", 128)
    v = cfg.get("vocab_size", 1000)
    ds = cfg["d_models"]
    ms = cfg["M_dims"]
    d_str = "[" + ", ".join(str(x) for x in ds) + "]"
    m_str = "[" + ", ".join(str(x) for x in ms) + "]"
    stype = cfg.get("spatial_mixer", "MultiheadAttentionMixer")
    if "NOV" in stype:
        spatial_import, spatial_class = "MultiheadAttentionMixerNOV", "MultiheadAttentionMixerNOV", "MultiheadAttentionMixer"
    else:
        spatial_import, spatial_class = "GroupedQueryAttention", "GroupedQueryAttention"
    spatial_import = spatial_class = stype
    return f"""import sys
# sys.path.insert(0, "/path/to/your/project")  # uncomment if needed
from Vathos.blocks import *
spatials = [Builder({spatial_class}, n_heads=8)] * {n}
skips    = [None] * {n}

model = ModdedFormer(
    vocab_size={v},
    embed_dim={d},
    d_models={d_str},
    spatials=spatials,
    M_dims={m_str},
    norm=RMSNorm,
    weights_tying=True,
    skips=skips,
)"""


def _reconstruct_from_code(code: str, sd: dict, source_name: str, fmt: str):
    ns = {}
    try:
        exec(compile(code, "<inspector>", "exec"), ns)
    except Exception as e:
        st.error(f"❌ Constructor error:\n```\n{e}\n```")
        return
    model = ns.get("model")
    if not isinstance(model, nn.Module):
        st.error("❌ Code must assign an `nn.Module` to a variable named `model`.")
        return

    if fmt == "vathos_checkpoint":
        import os
        path = st.session_state.get("_pending_source_name", "")
        if os.path.isfile(path):
            try:
                model.load_checkpoint(path)
            except Exception as e:
                st.error(f"❌ load_checkpoint failed:\n```\n{e}\n```")
                return
            st.success("✅ `model.load_checkpoint()` succeeded.")
        else:
            st.warning("File was uploaded (bytes only), falling back to load_state_dict().")
            _apply_state_dict(model, sd)
    else:
        _apply_state_dict(model, sd)

    st.session_state["model"] = model
    st.session_state["_load_fmt"] = fmt
    st.session_state["_load_name"] = source_name
    st.session_state.pop("_pending_state_dict", None)
    st.session_state.pop("_pending_source_name", None)
    st.rerun()


def _apply_state_dict(model: nn.Module, sd: dict):
    try:
        missing, unexpected = model.load_state_dict(sd, strict=False)
    except Exception as e:
        st.error(f"❌ load_state_dict failed:\n```\n{e}\n```")
        return
    if missing: st.warning(f"Missing keys ({len(missing)})")
    if unexpected: st.warning(f"Unexpected keys ({len(unexpected)})")
    st.success("✅ Weights loaded.")


def _load_raw(raw: bytes, name: str):
    import os
    obj, fmt, err = _try_load_pt(raw)
    if err:
        st.sidebar.error(f"❌ {err}")
        return
    if isinstance(obj, nn.Module):
        st.session_state["model"] = obj
        st.session_state["_load_fmt"] = fmt
        st.session_state["_load_name"] = name
        st.sidebar.success(f"✅ Loaded `{os.path.basename(name)}` ({fmt})")
    elif isinstance(obj, dict):
        st.session_state["_pending_state_dict"] = obj
        st.session_state["_pending_source_name"] = name
        st.session_state["_load_fmt"] = fmt
        st.session_state.pop("model", None)
        st.sidebar.info(f"Checkpoint loaded (`{fmt}`).")
    else:
        st.sidebar.error("❌ Could not interpret checkpoint contents.")


def _render_loader_sidebar() -> Optional[nn.Module]:
    st.sidebar.markdown("## 📂 Load Model")
    source = st.sidebar.radio("Source", ["Upload .pt / .pth file", "Local file path"], label_visibility="collapsed")
    if source == "Upload .pt / .pth file":
        uploaded = st.sidebar.file_uploader("Drop your checkpoint here", type=["pt", "pth", "bin"])
        if uploaded is not None:
            if st.session_state.get("_last_uploaded") != uploaded.name:
                st.session_state["_last_uploaded"] = uploaded.name
                _load_raw(uploaded.read(), uploaded.name)
                st.rerun()
    elif source == "Local file path":
        path_input = st.sidebar.text_input("Absolute path to .pt / .pth file", key="local_path_input")
        if st.sidebar.button("Load from path", use_container_width=True):
            import os
            if not path_input:
                st.sidebar.warning("Enter a path first.")
            elif not os.path.isfile(path_input):
                st.sidebar.error(f"File not found: `{path_input}`")
            else:
                with open(path_input, "rb") as fh:
                    _load_raw(fh.read(), path_input)
                st.rerun()

    model = st.session_state.get("model", None)
    if model is not None:
        fmt = st.session_state.get("_load_fmt", "unknown")
        name = st.session_state.get("_load_name", type(model).__name__)
        total = sum(p.numel() for p in model.parameters())
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Loaded model**")
        st.sidebar.markdown(f"- `{name}`\n- format: `{fmt}`\n- type: `{type(model).__name__}`\n- params: `{total:,}`")
        if st.sidebar.button("🗑️ Unload", use_container_width=True):
            for k in (
            "model", "_last_uploaded", "_load_fmt", "_load_name", "_pending_state_dict", "_pending_source_name"):
                st.session_state.pop(k, None)
            st.rerun()
    return model


# ──────────────────────────────────────────────────────────────────────────────
#  TRAINING HISTORY (Completamente Ripristinato)
# ──────────────────────────────────────────────────────────────────────────────

def render_training_history(model):
    """Training History page — mirrors VathosModel.plot_losses / plot_metrics."""
    st.markdown("## Training History")

    has_losses = hasattr(model, "_losses_dict") and len(getattr(model, "_losses_dict", {})) > 0
    has_epochs = hasattr(model, "_losses_per_epoch_dict") and len(getattr(model, "_losses_per_epoch_dict", {})) > 0
    has_metrics = hasattr(model, "_metrics_per_epoch") and len(getattr(model, "_metrics_per_epoch", {})) > 0

    if not has_losses and not has_metrics:
        st.info(
            "No training history found on this model.  \n"
            "Training history is only available when the model was loaded from a **Vathos checkpoint** "
            "(`model.save_checkpoint()`) that includes at least one `register_loss()` call.",
            icon="ℹ️",
        )
        return

    steps = getattr(model, "steps", "–")
    epochs = getattr(model, "epochs", "–")
    best_loss = getattr(model, "best_loss", float("inf"))
    checkpoints = getattr(model, "checkpoints", "–")
    spe = getattr(model, "steps_per_epoch", "–")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Steps", steps)
    c2.metric("Epochs", epochs)
    c3.metric("Best loss", f"{best_loss:.6g}" if isinstance(best_loss, float) else best_loss)
    c4.metric("Checkpoints", checkpoints)
    c5.metric("Steps/epoch", spe)

    st.markdown("---")

    # ── helpers ───────────────────────────────────────────────────────────────
    DARK = "#0e1117"
    GRID = "#2a2a2a"
    FG = "#e0e0e0"
    C1 = "#4c9be8"  # step-level colour
    C2 = "#f5a623"  # epoch-level colour
    CMAP = ["#4c9be8", "#f5a623", "#50c87a", "#e85d7a", "#a78bfa", "#fb923c"]

    def _base_fig(title, xlabel, ylabel):
        fig, ax = plt.subplots(figsize=(10, 3.8), facecolor=DARK)
        ax.set_facecolor(DARK)
        ax.set_title(title, color=FG, fontsize=13)
        ax.set_xlabel(xlabel, color=FG, fontsize=10)
        ax.set_ylabel(ylabel, color=FG, fontsize=10)
        ax.tick_params(colors=FG)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(True, color=GRID, linewidth=0.7, alpha=0.6)
        return fig, ax

    tab_names = ["Loss Curves"]
    if has_metrics:
        tab_names += [f"Metric: {m}" for m in getattr(model, "_metrics_per_epoch", {})]
    tab_names.append("Raw tables")

    tabs = st.tabs(tab_names)

    # ── Tab 0: Loss curves ────────────────────────────────────────────────────
    with tabs[0]:
        losses_dict = getattr(model, "_losses_dict", {})
        losses_ep_dict = getattr(model, "_losses_per_epoch_dict", {})

        win = 1
        if len(losses_dict) > 20:
            win = st.slider("Smoothing window (step-level)", 1, max(2, len(losses_dict) // 20), 1, key="loss_smooth")

        fig, ax = _base_fig("Training Loss", "Steps", "Loss")

        if losses_dict:
            xs = list(losses_dict.keys())
            ys = list(losses_dict.values())
            if win > 1:
                import numpy as np
                kernel = np.ones(win) / win
                ys_sm = np.convolve(ys, kernel, mode="valid")
                xs_sm = xs[win - 1:]
                ax.plot(xs, ys, color=C1, linewidth=0.6, alpha=0.25, label="Loss (raw)")
                ax.plot(xs_sm, ys_sm, color=C1, linewidth=1.4, label=f"Loss (smooth w={win})")
            else:
                ax.plot(xs, ys, color=C1, linewidth=0.8, alpha=0.55, label="Loss (step)")

        if losses_ep_dict:
            ex = list(losses_ep_dict.keys())
            ey = list(losses_ep_dict.values())
            ax.plot(ex, ey, color=C2, linewidth=2.2, marker="o", markersize=4, label="Loss per epoch")
            best = getattr(model, "best_loss", None)
            if best is not None and isinstance(best, float) and not (best == float("inf")):
                ax.axhline(best, color="#e85d7a", linewidth=1, linestyle="--", alpha=0.7, label=f"Best loss {best:.5g}")

        ax.legend(facecolor=DARK, edgecolor=GRID, labelcolor=FG, fontsize=9)
        st.image(_fig_to_bytes(fig), use_container_width=True)
        plt.close(fig)

        if losses_ep_dict:
            fig2, ax2 = _base_fig("Loss per Epoch (bar)", "Epoch", "Mean Loss")
            ey_list = list(losses_ep_dict.values())
            colors = [("#e85d7a" if v == min(ey_list) else C2) for v in ey_list]
            ax2.bar(range(len(ey_list)), ey_list, color=colors, edgecolor=GRID, linewidth=0.5)
            ax2.set_xticks(range(len(ey_list)))
            st.image(_fig_to_bytes(fig2), use_container_width=True)
            plt.close(fig2)

        losses_this = getattr(model, "_losses_this_epoch", [])
        if losses_this:
            st.caption(
                f"**In-progress epoch** — {len(losses_this)} steps logged, mean loss {sum(losses_this) / len(losses_this):.6g}")
            fig3, ax3 = _base_fig("Current Epoch Loss", "Step (this epoch)", "Loss")
            ax3.plot(losses_this, color=C1, linewidth=0.9)
            st.image(_fig_to_bytes(fig3), use_container_width=True)
            plt.close(fig3)

    # ── Metric tabs ───────────────────────────────────────────────────────────
    metrics_ep = getattr(model, "_metrics_per_epoch", {})
    metrics_all = getattr(model, "_metrics", {})

    for t_idx, metric_name in enumerate(metrics_ep, start=1):
        with tabs[t_idx]:
            ep_vals = metrics_ep.get(metric_name, [])
            step_vals = metrics_all.get(metric_name, [])

            col = CMAP[t_idx % len(CMAP)]
            fig, ax = _base_fig(f"Metric: {metric_name}", "Steps / Epochs", metric_name)

            if step_vals:
                ax.plot(range(len(step_vals)), step_vals,
                        color=col, linewidth=0.8, alpha=0.45, label=f"{metric_name} (step)")
            if ep_vals:
                scale = len(step_vals) / len(ep_vals) if step_vals and ep_vals else 1
                ep_xs = [i * scale for i in range(len(ep_vals))]
                ax.plot(ep_xs, ep_vals,
                        color=col, linewidth=2.2, marker="o", markersize=5,
                        label=f"{metric_name} per epoch")

            ax.legend(facecolor=DARK, edgecolor=GRID, labelcolor=FG, fontsize=9)
            st.image(_fig_to_bytes(fig), use_container_width=True)
            plt.close(fig)

            if ep_vals:
                fig2, ax2 = _base_fig(f"{metric_name} per Epoch (bar)", "Epoch", metric_name)
                best_idx = ep_vals.index(max(ep_vals))
                bar_colors = [(col if i != best_idx else "#50c87a") for i in range(len(ep_vals))]
                ax2.bar(range(len(ep_vals)), ep_vals, color=bar_colors, edgecolor=GRID, linewidth=0.5)
                ax2.set_xticks(range(len(ep_vals)))
                st.image(_fig_to_bytes(fig2), use_container_width=True)
                plt.close(fig2)

            metrics_this = getattr(model, "_metrics_this_epoch", {})
            this_vals = metrics_this.get(metric_name, [])
            if this_vals:
                st.caption(f"In-progress epoch: {len(this_vals)} steps, mean {sum(this_vals) / len(this_vals):.6g}")

    # ── Raw tables tab ────────────────────────────────────────────────────────
    with tabs[-1]:
        import pandas as pd
        st.markdown("#### Step-level losses")
        if losses_dict:
            df_loss = pd.DataFrame({"step": list(losses_dict.keys()), "loss": list(losses_dict.values())})
            st.dataframe(df_loss, use_container_width=True, hide_index=True)
        else:
            st.caption("No step-level loss data.")

        st.markdown("#### Epoch-level losses")
        if losses_ep_dict:
            df_ep = pd.DataFrame({"step": list(losses_ep_dict.keys()), "mean_loss": list(losses_ep_dict.values())})
            df_ep.insert(0, "epoch", range(len(df_ep)))
            st.dataframe(df_ep, use_container_width=True, hide_index=True)
        else:
            st.caption("No epoch-level loss data.")

        for metric_name, vals in metrics_ep.items():
            st.markdown(f"#### Metric: `{metric_name}` (per epoch)")
            df_m = pd.DataFrame({"epoch": range(len(vals)), metric_name: vals})
            st.dataframe(df_m, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN APP & ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def _render_welcome():
    st.markdown("## Welcome")
    st.info("No model loaded yet. Use the **sidebar** to load one.", icon="👈")
    st.markdown("""
### How to load a model

#### Option 1 — Save from your script, load via sidebar (recommended)

In your training or eval script, just save the model to disk:

```python
# saves full model object (architecture + weights) — easiest to reload
torch.save(model, "my_model.pt")

# or use the Vathos native checkpoint format
model.save_checkpoint("my_model-checkpoint.pt")
```

Then in the inspector sidebar choose **Local file path** or **Upload** to open it.

---

#### Option 2 — Use `run_inspector` in your Python script

```python
from vathos_inspector import run_inspector
run_inspector(model)
```
""")


def _render_app():
    st.title("🔬 Vathos ModdedFormer Inspector v2.1")
    with st.sidebar:
        model = _render_loader_sidebar()
        if model is not None:
            st.markdown("---")
            st.markdown("### Navigate")
            page = st.radio("", ["Model Overview", "Training History", "Layer Inspector", "Multi-Layer Comparison",
                                 "Attention Explorer"], label_visibility="collapsed", key="nav_page")

    pending_sd = st.session_state.get("_pending_state_dict", None)
    if pending_sd is not None and model is None:
        _render_state_dict_panel(sd=pending_sd, fmt=st.session_state.get("_load_fmt", "state_dict"),
                                 source_name=st.session_state.get("_pending_source_name", "checkpoint"))
        return

    if model is None:
        _render_welcome()
        return

    if page == "Model Overview":
        render_model_overview(model)
    elif page == "Training History":
        render_training_history(model)
    elif page == "Layer Inspector":
        render_layer_inspector(model)
    elif page == "Multi-Layer Comparison":
        render_comparison(model)
    elif page == "Attention Explorer":
        render_attention_explorer(model)


_PRELOADED_MODEL = None
if _PRELOADED_MODEL is not None and "model" not in st.session_state:
    st.session_state["model"] = _PRELOADED_MODEL
    st.session_state["_load_fmt"] = "preloaded"
    st.session_state["_load_name"] = type(_PRELOADED_MODEL).__name__


def run_inspector(model_to_inspect=None):
    """
    Utility function to run the inspector dynamically from another script.
    Usage:
        from vathos_inspector import run_inspector
        run_inspector(model)
    """
    import sys
    from streamlit.web import cli as stcli
    if model_to_inspect is not None:
        global _PRELOADED_MODEL
        _PRELOADED_MODEL = model_to_inspect
    sys.argv = ["streamlit", "run", __file__]
    sys.exit(stcli.main())


if __name__ == '__main__':
    _render_app()