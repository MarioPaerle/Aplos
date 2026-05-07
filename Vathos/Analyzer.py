"""
Vathos ModdedFormer Analyzer API
=================================
A pure-Python/Jupyter alternative to the Streamlit inspector.
Designed for rigorous, programmatic topological and statistical inspection
of Transformer parameters, gradients, and activation manifolds.

Usage in a Jupyter Notebook:
----------------------------
    import torch
    from vathos_analyzer import VathosAnalyzer

    # 1. Initialize
    analyzer = VathosAnalyzer(model)

    # 2. Plot Architecture Overview & Parameter distributions
    analyzer.plot_overview()
    analyzer.plot_weight_distributions(layer_idx=0, component="both")

    # 3. Manifold Extraction via Forward Hooks
    x = torch.randint(0, model.vocab_size, (1, 64))
    analyzer.capture_forward(x) # Installs hooks, runs forward, saves state

    # 4. Explore Layer Topology (Attention & FFN mapping f(x))
    analyzer.plot_attention_weights(layer_idx=0)
    analyzer.plot_ffn_manifold(layer_idx=0)

    # 5. Training History
    analyzer.plot_training_history()

    # --- OR USE THE ONE-LINER MACRO ---
    # from vathos_analyzer import study
    # analyzer = study(model)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Union

try:
    from IPython.display import display
    import pandas as pd

    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

# ──────────────────────────────────────────────────────────────────────────────
#  DARK THEME STYLING
# ──────────────────────────────────────────────────────────────────────────────
DARK_THEME = {
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
COLORS = ["#A78BFA", "#60A5FA", "#34D399", "#F472B6", "#FB923C",
          "#FBBF24", "#38BDF8", "#A3E635", "#E879F9", "#F87171"]


def apply_theme():
    plt.rcParams.update(DARK_THEME)


# ──────────────────────────────────────────────────────────────────────────────
#  CORE ANALYZER CLASS
# ──────────────────────────────────────────────────────────────────────────────
class VathosAnalyzer:
    def __init__(self, model: nn.Module, dark_mode: bool = True):
        """
        Initializes the analyzer.
        :param model: The Vathos ModdedFormer instance.
        """
        self.model = model
        self.captured_activations: Dict[int, Dict[str, torch.Tensor]] = {}
        if dark_mode:
            apply_theme()

    # ─── EXTRACTION LOGIC ─────────────────────────────────────────────────────

    def _is_attention(self, mixer) -> bool:
        cls_name = type(mixer).__name__.lower()
        return "attention" in cls_name or "attn" in cls_name

    def _tensor_stats(self, t: torch.Tensor) -> dict:
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

    def _svd_values(self, t: torch.Tensor) -> Optional[np.ndarray]:
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

    def _get_attention_weights(self, mixer, x: torch.Tensor) -> Optional[np.ndarray]:
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

    def get_layer_params(self, layer_idx: int, component: str = "both") -> Dict[str, torch.Tensor]:
        block = self.model.blocks[layer_idx]
        sm = block.spatial_mixer
        cm = block.channel_mixer

        selected_modules = {}
        if component in ["spatial", "both"]:
            selected_modules["spatial"] = sm
        if component in ["channel", "both"]:
            selected_modules["channel"] = cm

        all_params = {}
        for prefix, module in selected_modules.items():
            for name, p in module.named_parameters():
                all_params[f"{prefix}.{name}"] = p
        return all_params

    # ─── FORWARD PASS HOOKING (MANIFOLD CAPTURE) ──────────────────────────────

    def capture_forward(self, x: torch.Tensor, clear_previous: bool = True):
        """
        Runs a forward pass while attaching hooks to capture inputs/outputs of
        Spatial Mixers, and the strict pre/post activations of Channel Mixers.
        """
        if clear_previous:
            self.captured_activations.clear()

        handles = []

        for i, block in enumerate(self.model.blocks):
            self.captured_activations[i] = {}

            # Hook 1: Input to Spatial Mixer (For Q/K/V computations)
            def make_spatial_hook(layer_idx):
                def hook(m, inp):
                    self.captured_activations[layer_idx]["spatial_in"] = inp[0].detach()

                return hook

            handles.append(block.spatial_mixer.register_forward_pre_hook(make_spatial_hook(i)))

            cm = block.channel_mixer

            # Hook 2 & 3: Channel Mixer Expansion / Contraction (Pre and Post Activation)
            if hasattr(cm, 'expand') and hasattr(cm, 'contract'):
                def make_expand_hook(layer_idx):
                    def hook(m, inp, out):
                        self.captured_activations[layer_idx]["channel_pre_act"] = out.detach()

                    return hook

                def make_contract_hook(layer_idx):
                    def hook(m, inp):
                        self.captured_activations[layer_idx]["channel_post_act"] = inp[0].detach()

                    return hook

                handles.append(cm.expand.register_forward_hook(make_expand_hook(i)))
                handles.append(cm.contract.register_forward_pre_hook(make_contract_hook(i)))
            else:
                # Fallback per block custom
                def make_cm_hook(layer_idx):
                    def hook(m, inp):
                        self.captured_activations[layer_idx]["channel_in"] = inp[0].detach()

                    return hook

                handles.append(cm.register_forward_pre_hook(make_cm_hook(i)))

        # Run forward pass
        self.model.eval()
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        with torch.no_grad():
            self.model(x.to(device))

        # Cleanup hooks
        for h in handles:
            h.remove()

        print(f"✅ Manifolds captured for {len(self.model.blocks)} layers.")

    # ─── PLOTTING ROUTINES ────────────────────────────────────────────────────

    def show_fig(self, fig: plt.Figure):
        """Helper to show a figure inline or wait for script end."""
        if HAS_IPYTHON:
            display(fig)
            plt.close(fig)
        else:
            fig.show()

    def plot_overview(self):
        """Plots topological params: param norms, skip lambdas, zeroskips."""
        print("=== 🏗️ Model Topological Overview ===")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total Params: {total_params:,} | Layers: {self.model.n_layer} | Embed Dim: {self.model.embed_dim}")

        # 1. Param Norms
        layer_names, l2_norms = [], []
        for i, block in enumerate(self.model.blocks):
            for name, p in block.named_parameters():
                layer_names.append(f"L{i}.{name}")
                l2_norms.append(float(p.detach().norm(2)))

        fig, ax = plt.subplots(figsize=(max(8, len(layer_names) * 0.35), 4))
        ax.bar(range(len(layer_names)), l2_norms, color="#60A5FA", alpha=0.8)
        ax.set_xticks(range(len(layer_names)))
        ax.set_xticklabels(layer_names, rotation=90, fontsize=6)
        ax.set_ylabel("L2 norm")
        ax.set_title("Per-parameter L2 Norms", color="#A78BFA")
        ax.grid(True, ls="--", alpha=0.4, axis="y")
        fig.tight_layout()
        self.show_fig(fig)

        # 2. Skip Lambdas
        if hasattr(self.model, "skip_lambdas") and self.model.skip_lambdas:
            lambdas = {k: float(v.detach()) for k, v in self.model.skip_lambdas.items()}
            fig2, ax2 = plt.subplots(figsize=(max(6, len(lambdas) * 1.4), 3.5))
            keys = list(lambdas.keys())
            vals = [lambdas[k] for k in keys]
            bars = ax2.bar(range(len(keys)), vals, color=COLORS[:len(keys)], alpha=0.85)
            ax2.set_xticks(range(len(keys)))
            ax2.set_xticklabels([k.replace("route_", "").replace("_to_", "→") for k in keys], rotation=30, ha="right",
                                fontsize=8)
            ax2.axhline(0, color="#9CA3AF", linewidth=0.8, ls="--")
            ax2.set_title("Skip-connection Gate Values (λ)", color="#A78BFA")
            for bar, v in zip(bars, vals):
                ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.003, f"{v:.4f}", ha="center", va="bottom", fontsize=8,
                         color="#D8D8F0")
            fig2.tight_layout()
            self.show_fig(fig2)

        # 3. Zeroskip Params
        if hasattr(self.model, "zeroskip") and self.model.zeroskip:
            vals = [float(p.detach()) for p in self.model.zeroskip_params]
            fig3, ax3 = plt.subplots(figsize=(max(6, len(vals) * 0.7), 3.5))
            ax3.plot(vals, marker="o", color="#FBBF24", linewidth=1.5, markersize=6)
            ax3.axhline(0, color="#9CA3AF", linewidth=0.8, ls="--")
            ax3.set_xlabel("layer index")
            ax3.set_ylabel("zeroskip α")
            ax3.set_title("ZeroSkip Parameters (x₀ coefficient)", color="#A78BFA")
            ax3.grid(True, ls="--", alpha=0.4)
            fig3.tight_layout()
            self.show_fig(fig3)

    def print_layer_stats(self, layer_idx: int, component: str = "both"):
        """Prints a statistical dataframe of parameters for a given layer."""
        params = self.get_layer_params(layer_idx, component)
        rows = []
        for pname, p in params.items():
            s = self._tensor_stats(p)
            rows.append({
                "Parameter": pname, "Shape": str(s["shape"]), "Numel": s['numel'],
                "Mean": s['mean'], "Std": s['std'], "Min": s['min'],
                "Max": s['max'], "L2": s['l2'], "Sparsity": s['sparsity'],
                "Grad": "Yes" if p.grad is not None else "No",
            })
        if HAS_IPYTHON:
            df = pd.DataFrame(rows)
            display(df)
        else:
            for r in rows:
                print(r)

    def plot_weight_distributions(self, layer_idx: int, component: str = "both", plot_gradients: bool = False,
                                  bins: int = 80):
        params = self.get_layer_params(layer_idx, component)
        if plot_gradients:
            params = {k: v.grad for k, v in params.items() if v.grad is not None}
            if not params:
                print("No gradients found. Run loss.backward() first.")
                return

        n = len(params)
        if n == 0: return
        cols = min(n, 3)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
        if n == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)

        for idx, (name, t) in enumerate(params.items()):
            ax = axes[idx // cols][idx % cols]
            v = t.detach().float().cpu().numpy().flatten()
            st_val = self._tensor_stats(t)
            ax.hist(v, bins=bins, color=COLORS[idx % len(COLORS)], alpha=0.8, density=True)
            ax.set_title(f"{name}\nμ={st_val['mean']:.3e}  σ={st_val['std']:.3e}", fontsize=9)
            ax.grid(True, ls="--", alpha=0.4)

        for idx in range(n, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        fig.suptitle(f"Layer {layer_idx} - {'Gradient' if plot_gradients else 'Weight'} Distributions", color="#A78BFA",
                     fontsize=12)
        fig.tight_layout()
        self.show_fig(fig)

    def plot_svd_spectrum(self, layer_idx: int, component: str = "both", plot_gradients: bool = False,
                          log_scale: bool = True):
        params = self.get_layer_params(layer_idx, component)
        if plot_gradients:
            params = {k: v.grad for k, v in params.items() if v.grad is not None}

        eligible = {k: v for k, v in params.items() if v.ndim >= 2}
        if not eligible:
            print("No 2D parameters found for SVD.")
            return

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
            sv = self._svd_values(t)
            if sv is None: continue

            if sv.ndim == 2:
                for h in range(sv.shape[0]):
                    ax.plot(sv[h], color=COLORS[idx % len(COLORS)], alpha=0.3, linewidth=0.8)
                ax.plot(sv.mean(0), color=COLORS[idx % len(COLORS)], linewidth=2, label="mean")
                ax.legend(fontsize=8)
            else:
                ax.plot(sv, color=COLORS[idx % len(COLORS)], linewidth=1.5, marker=".", markersize=3)

            cond = sv.flatten()[0] / (sv.flatten()[-1] + 1e-12)
            ax.set_title(f"{name} | cond={cond:.1f}", fontsize=9)
            if log_scale: ax.set_yscale("log")
            ax.grid(True, ls="--", alpha=0.4)

        for idx in range(n, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        fig.suptitle(f"Layer {layer_idx} - Singular Value Spectra ($\Sigma$)", color="#A78BFA", fontsize=12)
        fig.tight_layout()
        self.show_fig(fig)

    # ─── MANIFOLD EXPLORATION (ATTENTION E FFN) ───────────────────────────────

    def plot_attention_weights(self, layer_idx: int, tokens: Optional[List[str]] = None):
        """Plots the attention map of the layer using the captured manifold."""
        if layer_idx not in self.captured_activations or "spatial_in" not in self.captured_activations[layer_idx]:
            print(f"No captured input for layer {layer_idx}. Run `capture_forward(x)` first.")
            return

        sm = self.model.blocks[layer_idx].spatial_mixer
        if not self._is_attention(sm):
            print(f"Layer {layer_idx} spatial mixer is not Attention (is {type(sm).__name__}).")
            return

        x = self.captured_activations[layer_idx]["spatial_in"]
        w = self._get_attention_weights(sm, x)
        if w is None:
            print("Could not extract attention weights.")
            return

        if w.ndim == 2: w = w[np.newaxis]
        H, L_q, L_k = w.shape
        cols = min(H, 4)
        rows = math.ceil(H / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
        axes = np.array(axes).flatten() if H > 1 else [axes]

        for h in range(H):
            ax = axes[h]
            wh = w[h]
            im = ax.imshow(wh, aspect="auto", cmap="magma", vmin=0, vmax=wh.max() + 1e-9)
            ax.set_title(f"Head {h}", fontsize=10, color="#C4B5FD")
            if tokens and len(tokens) == L_q:
                clean_tokens = [t.replace('Ġ', ' ') for t in tokens]
                fs = max(4, min(10, int(300 / L_q)))
                ax.set_xticks(range(L_k))
                ax.set_yticks(range(L_q))
                ax.set_xticklabels(clean_tokens, rotation=90, fontsize=fs, color="#9CA3AF")
                ax.set_yticklabels(clean_tokens, fontsize=fs, color="#9CA3AF")
            fig.colorbar(im, ax=ax, shrink=0.8)

        for h in range(H, len(axes)):
            axes[h].set_visible(False)

        fig.suptitle(f"Layer {layer_idx} - Attention Topology Analysis", color="#A78BFA", fontsize=12)
        fig.tight_layout()
        self.show_fig(fig)

    def plot_ffn_manifold(self, layer_idx: int):
        """
        Traccia la densità (PDF empirica) delle attivazioni prima della non linearità (XW_exp),
        sovrapposta alla mappa non-lineare $f(x)$. Questo rivela immediatamente condizioni
        di annullamento gradiente (dead neurons) o eccessiva saturazione.
        Dipende dai Forward Hooks eseguiti tramite `capture_forward()`.
        """
        if layer_idx not in self.captured_activations:
            print(f"No manifold captured. Run `capture_forward(x)` first.")
            return

        cap = self.captured_activations[layer_idx]
        cm = self.model.blocks[layer_idx].channel_mixer

        if "channel_pre_act" in cap and "channel_post_act" in cap:
            pre_act = cap["channel_pre_act"][0].cpu().numpy()
            post_act = cap["channel_post_act"][0].cpu().numpy()
        elif "channel_in" in cap:
            # Fallback se non ci sono expand/contract espliciti
            x = cap["channel_in"]
            data = self._get_ffn_activations(cm, x, temp=1.0)
            if not data:
                print("Could not compute FFN manifold.")
                return
            pre_act = data["pre_act"]
            post_act = data["post_act"]
        else:
            print("Missing necessary hooks data for FFN.")
            return

        # Estrazione act_fn per mapping analitico
        act_fn = getattr(cm, "activation", getattr(cm, "act", getattr(cm, "act_fn", None)))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        pre_flat = pre_act.flatten()
        post_flat = post_act.flatten()

        # ── Subplot 1: Pre-act vs f(x) ──
        ax1 = axes[0]
        ax1_twin = ax1.twinx()

        # Dominio assoluto
        true_min = float(pre_flat.min())
        true_max = float(pre_flat.max())
        x_min = min(-5.0, true_min - 1.0)
        x_max = max(5.0, true_max + 1.0)
        x_vals = np.linspace(x_min, x_max, 1000)

        y_vals = None
        is_empirical = False

        # 1. Tentativo analitico
        if act_fn is not None:
            try:
                device = next(cm.parameters()).device if list(cm.parameters()) else "cpu"
                dtype = next(cm.parameters()).dtype if list(cm.parameters()) else torch.float32
                with torch.no_grad():
                    xt = torch.tensor(x_vals, dtype=dtype, device=device).view(1, 1, -1)
                    yt = act_fn(xt)
                    y_vals = yt.cpu().float().numpy().flatten()
            except Exception:
                pass

        # 2. Empirico / Fallback
        if (y_vals is None or np.array_equal(y_vals, x_vals)) and pre_flat.shape == post_flat.shape:
            if not np.allclose(pre_flat, post_flat):
                sort_idx = np.argsort(pre_flat)
                x_vals = pre_flat[sort_idx]
                y_vals = post_flat[sort_idx]
                is_empirical = True
            else:
                y_vals = x_vals
        elif y_vals is None:
            y_vals = x_vals

        label_map = "$f(x)$ map (Empirica)" if is_empirical else "$f(x)$ map"

        # Curve
        ax1.plot(x_vals, y_vals, color="#38BDF8", lw=3.0, label=label_map, zorder=4)
        ax1.axvline(0, color="#4B5563", lw=1.5, ls="--", zorder=1)
        ax1.axhline(0, color="#4B5563", lw=1.5, ls="--", zorder=1)

        # Min / Max assoluti (Individuazione leak instabili)
        ax1.plot(true_min, 0, marker='v', color='#EF4444', markersize=7, zorder=5)
        ax1.plot(true_max, 0, marker='v', color='#EF4444', markersize=7, zorder=5)
        ax1.text(true_min, 0.05, f"Min:\n{true_min:.1f}", color='#EF4444', ha='center', va='bottom', fontsize=9,
                 transform=ax1.get_xaxis_transform())
        ax1.text(true_max, 0.05, f"Max:\n{true_max:.1f}", color='#EF4444', ha='center', va='bottom', fontsize=9,
                 transform=ax1.get_xaxis_transform())

        ax1.set_xlabel("Pre-activation $x = (X W_{exp})$", fontsize=10)
        ax1.set_ylabel("Activation Output $f(x)$", color="#38BDF8", fontsize=10)
        ax1.tick_params(axis='y', labelcolor="#38BDF8")

        # PDF pre-attivazione
        ax1_twin.hist(pre_flat, bins=150, range=(x_min, x_max), color="#F472B6", alpha=0.4, density=True, zorder=2)
        ax1_twin.set_ylabel("Empirical PDF $p(x)$", color="#F472B6", fontsize=10)
        ax1_twin.tick_params(axis='y', labelcolor="#F472B6")

        ax1.set_title(f"L{layer_idx} - Overlay: Activation $f(x)$ & Input Distribution", color="#A78BFA", fontsize=11)
        ax1.legend(loc="upper left", fontsize=9)

        # ── Subplot 2: Post-act distribution (Sparsity check) ──
        ax2 = axes[1]
        post_min, post_max = np.percentile(post_flat, [0.0, 99.9])
        ax2.hist(post_flat, bins=120, range=(post_min, max(post_max, 1e-3)), color="#F472B6", alpha=0.6, density=True)
        ax2.axvline(0, color="#9CA3AF", lw=1, ls="--", alpha=0.5)
        ax2.set_xlabel("Post-activation $Act(X W_{exp})$", fontsize=10)
        ax2.set_ylabel("Density", fontsize=10)
        ax2.set_title("Post-Activation Distribution (Sparsity check)", color="#A78BFA", fontsize=11)
        ax2.grid(True, ls="--", alpha=0.3)

        fig.tight_layout()
        self.show_fig(fig)

    # ─── TRAINING HISTORY ─────────────────────────────────────────────────────

    def plot_training_history(self):
        """Plots losses and metrics matching the Vathos logic."""
        losses_dict = getattr(self.model, "_losses_dict", {})
        losses_ep = getattr(self.model, "_losses_per_epoch_dict", {})
        metrics_ep = getattr(self.model, "_metrics_per_epoch", {})

        if not losses_dict and not metrics_ep:
            print("No training history found in this model.")
            return

        print("=== 📈 Training History ===")
        print(
            f"Steps: {getattr(self.model, 'steps', '-')} | Epochs: {getattr(self.model, 'epochs', '-')} | Best Loss: {getattr(self.model, 'best_loss', '-')}")

        if losses_dict or losses_ep:
            fig, ax = plt.subplots(figsize=(10, 4))
            if losses_dict:
                xs = list(losses_dict.keys())
                ys = list(losses_dict.values())
                ax.plot(xs, ys, color="#4c9be8", linewidth=0.8, alpha=0.4, label="Loss (step)")
                # Smooth curve
                if len(ys) > 10:
                    win = max(2, len(ys) // 20)
                    ys_sm = np.convolve(ys, np.ones(win) / win, mode="valid")
                    ax.plot(xs[win - 1:], ys_sm, color="#4c9be8", linewidth=1.5, label=f"Smoothed (w={win})")

            if losses_ep:
                ex = list(losses_ep.keys())
                ey = list(losses_ep.values())
                ax.plot(ex, ey, color="#f5a623", linewidth=2.2, marker="o", markersize=5, label="Loss per epoch")

            ax.set_title("Training Loss", color="#A78BFA")
            ax.set_xlabel("Steps / Epochs")
            ax.set_ylabel("Loss")
            ax.grid(True, ls="--", alpha=0.4)
            ax.legend(fontsize=9)
            fig.tight_layout()
            self.show_fig(fig)

        # Plot Metrics
        for t_idx, (m_name, m_vals) in enumerate(metrics_ep.items()):
            fig, ax = plt.subplots(figsize=(10, 3.5))
            col = COLORS[t_idx % len(COLORS)]
            ax.plot(range(len(m_vals)), m_vals, color=col, linewidth=2.2, marker="o", markersize=5,
                    label=f"{m_name} per epoch")
            ax.set_title(f"Metric: {m_name}", color="#A78BFA")
            ax.set_xlabel("Epochs")
            ax.set_ylabel(m_name)
            ax.grid(True, ls="--", alpha=0.4)
            ax.legend(fontsize=9)
            fig.tight_layout()
            self.show_fig(fig)

    def plot_all(self, layer_idx: int = 0, tokens: Optional[List[str]] = None):
        """
        Esegue un'analisi topologica e statistica completa, chiamando in sequenza
        tutte le routine grafiche. Genera un referto end-to-end del modello.
        """
        print(f"\n{'=' * 70}\n🔬 INITIATING FULL TOPOLOGICAL ANALYSIS (Focus Layer {layer_idx})\n{'=' * 70}")
        self.plot_overview()

        print(f"\n[1/4] Weight Distributions (Layer {layer_idx})")
        self.plot_weight_distributions(layer_idx=layer_idx)

        print(f"\n[2/4] SVD Spectrum & Condition Numbers (Layer {layer_idx})")
        self.plot_svd_spectrum(layer_idx=layer_idx)

        if not self.captured_activations:
            print("\n⚠️ Nessun forward pass catturato nello stato. Salto l'analisi del manifold (Attention/FFN).")
            print("    Chiama `capture_forward(x)` prima di `plot_all()` per abilitarla.")
        else:
            print(f"\n[3/4] Attention Topology (Layer {layer_idx})")
            self.plot_attention_weights(layer_idx=layer_idx, tokens=tokens)

            print(f"\n[4/4] FFN Manifold & Activation Geometry f(x) (Layer {layer_idx})")
            self.plot_ffn_manifold(layer_idx=layer_idx)

        print("\n--- Training History ---")
        self.plot_training_history()
        print(f"\n{'=' * 70}\n✅ ANALYSIS COMPLETE\n{'=' * 70}")


# ──────────────────────────────────────────────────────────────────────────────
#  TOP-LEVEL MACRO
# ──────────────────────────────────────────────────────────────────────────────

def study(model: nn.Module, layer_idx: int = 0, dummy_seq_len: int = 64,
          device: Optional[Union[str, torch.device]] = None) -> VathosAnalyzer:
    """
    Macro globale per profilare immediatamente un modello in un Notebook o script nudo.
    1. Assicura il corretto posizionamento e deduce il device dei tensori.
    2. Istanzia il VathosAnalyzer.
    3. Genera un input sintetico coerente col dominio.
    4. Cattura il manifold (iniettando i Forward Hooks estrae i pre-logits).
    5. Esegue tutta la suite grafica (plot_all).

    Ritorna:
        L'istanza di VathosAnalyzer configurata per ulteriori ispezioni interattive.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
            model.to(device)

    print(f"🔍 Starting Rigorous Study on {type(model).__name__} (Device: {device})")

    analyzer = VathosAnalyzer(model)

    # Tentiamo di dedurre il vocab_size rigorosamente, con fallback
    vocab_size = getattr(model, "vocab_size", 1000)
    if not hasattr(model, "vocab_size") and hasattr(model, "embedder") and hasattr(model.embedder, "embedding"):
        vocab_size = getattr(model.embedder.embedding, "num_embeddings", 1000)

    # Dummy input per forzare il passaggio nei manifold locali ed estrarre la Jacobiana implicita
    dummy_x = torch.randint(0, vocab_size, (1, dummy_seq_len), device=device)

    print(f"📸 Capturing activation manifold with dummy sequence (L={dummy_seq_len})...")
    analyzer.capture_forward(dummy_x)

    # Lancia l'analisi completa concentrandosi sul layer indicato (0 di default)
    analyzer.plot_all(layer_idx=layer_idx)

    return analyzer