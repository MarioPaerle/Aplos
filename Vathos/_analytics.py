"""
Vathos Analytics & Interpretability Toolkit
===========================================
Modulo standalone per l'analisi spettrale, la topologia dell'attenzione e
l'interpretazione semantica dei concetti (FFN) nei modelli ModdedFormer.

Progettato per Jupyter Notebooks e script Python off-Streamlit.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple, Any
import math


# ──────────────────────────────────────────────────────────────────────────────
#  1. ANALISI MATEMATICA E SPETTRALE (Core)
# ──────────────────────────────────────────────────────────────────────────────

def compute_tensor_stats(t: torch.Tensor) -> dict:
    """Calcola i momenti statistici e la sparsità di un tensore o manifold."""
    f = t.detach().float().cpu()
    v = f.numpy().flatten()
    return {
        "shape": tuple(t.shape),
        "numel": t.numel(),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "min": float(v.min()),
        "max": float(v.max()),
        "l2_norm": float(np.linalg.norm(v)),
        "sparsity": float((np.abs(v) < 1e-6).mean())
    }


def compute_svd_spectrum(t: torch.Tensor) -> Optional[np.ndarray]:
    """
    Decomposizione in Valori Singolari (SVD) per studiare il collasso dimensionale.
    Ritorna i valori singolari sigma. Se il tensore è 3D (es. Head), media gli spettri.
    """
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


def compute_shannon_entropy(attn_weights: np.ndarray) -> List[float]:
    """
    Calcola l'Entropia di Shannon E[H] lungo l'asse delle Key per ogni Head.
    H = - sum(p * log(p)). Valori bassi indicano forte 'certezza' (es. Induction Heads).
    """
    w_calc = attn_weights if attn_weights.ndim == 3 else attn_weights[np.newaxis]
    entropies = []
    for h in range(w_calc.shape[0]):
        wh = w_calc[h] + 1e-12  # epsilon for numerical stability in the log
        ent = -(wh * np.log(wh)).sum(axis=-1).mean()
        entropies.append(float(ent))
    return entropies


# ──────────────────────────────────────────────────────────────────────────────
#  2. ESTRAZIONE TOPOLOGICA (Hooks & Forward Passes)
# ──────────────────────────────────────────────────────────────────────────────

def get_attention_matrix(mixer: nn.Module, x: torch.Tensor) -> Optional[np.ndarray]:
    """Ricalcola rigorosamente Softmax(QK^T / sqrt(d)) per estrarre le matrici d'attenzione."""
    if hasattr(mixer, "get_attention_weights"):
        with torch.no_grad():
            return mixer.get_attention_weights(x).detach().float().cpu().numpy()

    try:
        with torch.no_grad():
            B, L, D = x.shape
            if hasattr(mixer, "qkv"):
                proj = mixer.qkv
                n_heads = mixer.n_heads
                head_dim = mixer.head_dim
                qkv = proj(x).view(B, L, 3, n_heads, head_dim)
                q, k, _ = qkv.unbind(dim=2)
                q, k = q.transpose(1, 2).float(), k.transpose(1, 2).float()
                scale = math.sqrt(head_dim)
                scores = torch.matmul(q, k.transpose(-2, -1)) / scale
                if getattr(mixer, "causal", True):
                    mask = torch.tril(torch.ones(L, L, device=x.device)).bool()
                    scores = scores.masked_fill(~mask, float("-inf"))
                return torch.softmax(scores, dim=-1)[0].cpu().numpy()
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
                return torch.softmax(scores, dim=-1)[0].cpu().numpy()
    except Exception as e:
        print(f"Errore estrazione QK: {e}")
    return None


def get_ffn_manifolds(cm: nn.Module, x: torch.Tensor, tau: float = None) -> Optional[Dict[str, np.ndarray]]:
    """
    Tratta il Channel Mixer come Memoria Associativa.
    Ritorna Logits (XW), Attivazioni (Act(XW)) e Concept Attention.
    """
    if tau is None:
        tau = math.sqrt(x.shape[-1])  # Default temperature = sqrt(d_model)

    try:
        with torch.no_grad():
            pre_act = cm.expand(x)  # Y = XW_{expand}
            post_act = cm.act(pre_act) if hasattr(cm, "act") else pre_act
            concept_attn = torch.softmax(pre_act / tau, dim=-1)

            return {
                "pre_act": pre_act[0].cpu().numpy(),  # [L, M_dim]
                "post_act": post_act[0].cpu().numpy(),  # [L, M_dim]
                "concept_attn": concept_attn[0].cpu().numpy()  # [L, M_dim]
            }
    except Exception as e:
        print(f"Errore estrazione FFN Manifold: {e}")
    return None


def capture_layer_manifolds(model: nn.Module, input_ids: torch.Tensor, layer_idx: int) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """
    Usa i forward hooks per intercettare i manifold esatti in ingresso
    ai mixer Spaziali e di Canale per un dato strato L.
    """
    block = model.blocks[layer_idx]
    sm = block.spatial_mixer
    cm = block.channel_mixer

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
    finally:
        h1.remove()
        h2.remove()

    return captured["spatial_x"], captured["channel_x"]


# ──────────────────────────────────────────────────────────────────────────────
#  3. FUNZIONI DI PLOTTING (Matplotlib Figure Generators)
# ──────────────────────────────────────────────────────────────────────────────

def set_dark_style():
    """Opzionale: Applica il tema scuro (stile Vathos) globale a Matplotlib."""
    plt.rcParams.update({
        "figure.facecolor": "#0D0D1A", "axes.facecolor": "#0D0D1A", "axes.edgecolor": "#252550",
        "axes.labelcolor": "#C0C0E0", "xtick.color": "#9CA3AF", "ytick.color": "#9CA3AF",
        "text.color": "#D8D8F0", "grid.color": "#1E1E3A", "grid.alpha": 0.6
    })


def plot_weight_distributions(tensors_dict: Dict[str, torch.Tensor], bins: int = 80) -> Tuple[plt.Figure, np.ndarray]:
    n = len(tensors_dict)
    cols = min(n, 3)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    colors = ["#A78BFA", "#60A5FA", "#34D399", "#F472B6", "#FB923C", "#FBBF24"]

    for idx, (name, t) in enumerate(tensors_dict.items()):
        ax = axes[idx // cols][idx % cols]
        v = t.detach().float().cpu().numpy().flatten()
        stats = compute_tensor_stats(t)
        ax.hist(v, bins=bins, color=colors[idx % len(colors)], alpha=0.8, linewidth=0)
        ax.set_title(f"{name}\n$\mu$={stats['mean']:.3e}  $\sigma$={stats['std']:.3e}", fontsize=9)
        ax.grid(True, ls="--", alpha=0.4)

    for idx in range(n, rows * cols): axes[idx // cols][idx % cols].set_visible(False)
    fig.suptitle("Distribuzioni di Densità", fontsize=12)
    fig.tight_layout()
    return fig, axes


def plot_svd_spectra(tensors_dict: Dict[str, torch.Tensor], log_scale: bool = True) -> Tuple[plt.Figure, np.ndarray]:
    eligible = {k: v for k, v in tensors_dict.items() if v.ndim >= 2}
    n = len(eligible)
    if n == 0: raise ValueError("Nessun tensore 2D+ fornito per la SVD.")

    cols = min(n, 3)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    colors = ["#A78BFA", "#60A5FA", "#34D399", "#F472B6", "#FB923C", "#FBBF24"]

    for idx, (name, t) in enumerate(eligible.items()):
        ax = axes[idx // cols][idx % cols]
        sv = compute_svd_spectrum(t)
        if sv is None: continue

        color = colors[idx % len(colors)]
        if sv.ndim == 2:
            for h in range(sv.shape[0]):
                ax.plot(sv[h], color=color, alpha=0.2, linewidth=0.8)
            ax.plot(sv.mean(0), color=color, linewidth=2, label="Mean $\Sigma$")
            ax.legend()
        else:
            ax.plot(sv, color=color, linewidth=1.5, marker=".", markersize=3)

        cond = sv.flatten()[0] / (sv.flatten()[-1] + 1e-12)
        r_eff = int((sv.flatten() > sv.flatten()[0] * 1e-3).sum())
        ax.set_title(f"{name}\n$\kappa={cond:.1f}$ | Rango Effettivo $\\approx {r_eff}$", fontsize=9)
        if log_scale: ax.set_yscale("log")
        ax.grid(True, ls="--", which="both", alpha=0.4)

    for idx in range(n, rows * cols): axes[idx // cols][idx % cols].set_visible(False)
    fig.tight_layout()
    return fig, axes


def plot_attention_topology(attn_weights: np.ndarray, tokens: Optional[List[str]] = None) -> Tuple[
    plt.Figure, np.ndarray]:
    """Plotta la matrice $L \times L$ per le varie Head d'attenzione."""
    if attn_weights.ndim == 2: attn_weights = attn_weights[np.newaxis]
    H, L_q, L_k = attn_weights.shape
    cols = min(H, 4)
    rows = math.ceil(H / cols)

    cell_size = max(0.2, min(0.6, 12.0 / L_q)) if tokens else 0.5
    fig_w = max(4 * cols, cell_size * L_q * cols)
    fig_h = max(3.5 * rows, cell_size * L_q * rows + 1)

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.array(axes).flatten() if H > 1 else [axes]

    clean_tokens = [t.replace('Ġ', '').replace(' ', '') for t in tokens] if tokens else None

    for h in range(H):
        ax = axes[h]
        w = attn_weights[h]
        im = ax.imshow(w, aspect="auto", cmap="magma", vmin=0, vmax=w.max() + 1e-9)
        ax.set_title(f"Head {h}", fontsize=10)

        if clean_tokens and len(clean_tokens) == L_q:
            ax.set_xticks(range(L_k))
            ax.set_yticks(range(L_q))
            fs = max(5, min(10, int(300 / L_q)))
            ax.set_xticklabels(clean_tokens, rotation=90, fontsize=fs)
            ax.set_yticklabels(clean_tokens, fontsize=fs)
        else:
            ax.set_xlabel("Key Position (Source)")
            ax.set_ylabel("Query Position (Target)")

        fig.colorbar(im, ax=ax, shrink=0.8)

    for h in range(H, len(axes)): axes[h].set_visible(False)
    fig.tight_layout()
    return fig, axes


def plot_ffn_concept_manifold(matrix: np.ndarray, title: str, tokens: Optional[List[str]] = None,
                              max_neurons: int = 128) -> Tuple[plt.Figure, plt.Axes]:
    """Plotta la mappa $L \times M$ di attivazione dei concetti per il Channel Mixer."""
    mat_to_plot = matrix[:, :max_neurons]
    rows_c, cols_c = mat_to_plot.shape

    fig_w = max(8, cols_c * 0.15) if tokens else 10
    fig_h = max(6, rows_c * 0.3) if tokens else 6

    fig, ax = plt.subplots(figsize=(min(fig_w, 20), min(fig_h, 15)))
    im = ax.imshow(mat_to_plot, aspect="auto", cmap="magma")

    ax.set_title(f"{title} (Primi {cols_c} neuroni su {matrix.shape[1]})", pad=15)
    ax.set_ylabel("Tokens (Queries)")
    ax.set_xlabel("FFN Neurons (Latent Concepts)")

    if tokens and len(tokens) == rows_c:
        clean_tokens = [t.replace('Ġ', '').replace(' ', '') for t in tokens]
        fs = max(5, min(10, int(400 / rows_c)))
        ax.set_yticks(range(rows_c))
        ax.set_yticklabels(clean_tokens, fontsize=fs)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig, ax


def plot_architecture_scalars(model: nn.Module) -> Tuple[Optional[plt.Figure], Optional[plt.Figure]]:
    """Genera i plot per gli scalari strutturali: ZeroSkip e Skip-connection Lambdas."""
    fig_skip, fig_zero = None, None

    # 1. Skip Lambdas
    if hasattr(model, "skip_lambdas") and model.skip_lambdas:
        lambdas = {k: float(v.detach()) for k, v in model.skip_lambdas.items()}
        fig_skip, ax = plt.subplots(figsize=(max(6, len(lambdas) * 1.4), 4))
        keys, vals = list(lambdas.keys()), list(lambdas.values())
        ax.bar(range(len(keys)), vals, alpha=0.85, color="#60A5FA")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([k.replace("route_", "").replace("_to_", "→") for k in keys], rotation=30, ha="right")
        ax.axhline(0, color="gray", linewidth=0.8, ls="--")
        ax.set_title("Skip-connection Routing Gates ($\lambda$)")
        fig_skip.tight_layout()

    # 2. ZeroSkip Parameters
    if hasattr(model, "zeroskip_params") and model.zeroskip:
        vals = [float(p.detach()) for p in model.zeroskip_params]
        fig_zero, ax = plt.subplots(figsize=(max(6, len(vals) * 0.7), 4))
        ax.plot(vals, marker="o", color="#FBBF24", linewidth=1.5)
        ax.axhline(0, color="gray", linewidth=0.8, ls="--")
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("ZeroSkip $\\alpha$")
        ax.set_title("ZeroSkip Coefficients per Layer")
        ax.grid(True, ls="--", alpha=0.4)
        fig_zero.tight_layout()

    return fig_skip, fig_zero


# ──────────────────────────────────────────────────────────────────────────────
#  4. HIGH-LEVEL API PER JUPYTER (Sintesi per Speedrun & ARC)
# ──────────────────────────────────────────────────────────────────────────────

def analyze_layer_semantics(model: nn.Module, tokenizer: Any, prompt: str, layer_idx: int,
                            add_bos: bool = True, ffn_tau: float = None,
                            show_plots: bool = True) -> Dict[str, Any]:
    """
    Funzione master per i Notebook. Esegue l'analisi completa del layer indicato
    su un prompt specifico, restituendo statistiche e plottando i grafici.

    Uso:
        results = analyze_layer_semantics(model, tokenizer, "Test sequence", layer_idx=4)
        print("Sparsità FFN:", results["ffn_sparsity"])
    """
    device = next(model.parameters()).device
    full_prompt = (tokenizer.bos_token if add_bos and tokenizer.bos_token else "") + prompt
    inputs = tokenizer(full_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # 1. Forward Pass con intercettazione Manifold
    sm_x, cm_x = capture_layer_manifolds(model, input_ids, layer_idx)
    if sm_x is None:
        raise RuntimeError(f"Hook fallito. Il layer {layer_idx} non è stato eseguito.")

    block = model.blocks[layer_idx]
    sm = block.spatial_mixer
    cm = block.channel_mixer

    d_model = model.d_models[layer_idx] if hasattr(model, "d_models") else sm_x.shape[-1]
    if ffn_tau is None: ffn_tau = math.sqrt(d_model)

    # 2. Estrazione Matrici
    attn_w = get_attention_matrix(sm, sm_x)
    ffn_manifolds = get_ffn_manifolds(cm, cm_x, tau=ffn_tau)

    results = {
        "tokens": tokens,
        "X_spatial": sm_x,
        "X_channel": cm_x,
        "attention_matrix": attn_w,
        "ffn_manifolds": ffn_manifolds,
    }

    # 3. Calcoli Spettrali e Statistiche (Information Theory)
    if attn_w is not None:
        results["attention_entropy"] = compute_shannon_entropy(attn_w)

    if ffn_manifolds is not None:
        Y_act = ffn_manifolds["post_act"]
        stats = compute_tensor_stats(torch.tensor(Y_act))
        results["ffn_sparsity"] = stats["sparsity"]
        results["ffn_l2_norm"] = stats["l2_norm"]

        try:
            svd_vals = np.linalg.svd(Y_act, compute_uv=False)
            results["ffn_eff_rank"] = int(np.sum(svd_vals > svd_vals[0] * 1e-3))
            results["ffn_svd_spectrum"] = svd_vals
        except np.linalg.LinAlgError:
            pass

    # 4. Rendering Opzionale
    if show_plots:
        print(f"--- ANALISI DEL MANIFOLD | LAYER {layer_idx} ---")

        if attn_w is not None:
            print("\n>> Topologia Spaziale dell'Attenzione (QK)")
            fig_attn, _ = plot_attention_topology(attn_w, tokens)
            plt.show()
            print(f"   E[H] (Entropia per Head): {[f'{e:.3f}' for e in results['attention_entropy']]}")

        if ffn_manifolds is not None:
            print("\n>> Channel Expansion (Memory Concepts)")
            print(f"   Sparsità (Soglia < 1e-6): {results.get('ffn_sparsity', 0):.2%}")
            print(f"   Rango Effettivo (SVD): {results.get('ffn_eff_rank', 'N/A')} su {Y_act.shape[1]} M_dim")

            fig_ffn, _ = plot_ffn_concept_manifold(ffn_manifolds["post_act"], "FFN $Act(X W_{expand})$", tokens)
            plt.show()

            if "ffn_svd_spectrum" in results:
                fig_svd, ax_svd = plt.subplots(figsize=(6, 3))
                ax_svd.plot(results["ffn_svd_spectrum"], color="red", marker=".")
                ax_svd.set_yscale("log")
                ax_svd.set_title("Spettro $\Sigma$ del Manifold dei Concetti (FFN)")
                ax_svd.grid(True, ls="--", alpha=0.5)
                plt.show()

    return results