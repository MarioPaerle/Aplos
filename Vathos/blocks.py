"""
All the layers included here aim to be already optimized by torch itself, without requiring triton/cuda kernels (explicitly),
in fact they try to only use matmul and Linear operations which are arguably already optimized implicitly in torch.
"""
import time

import torch

from Vathos._basics import *
from typing import Tuple, Optional, Union, List
import re
from Vathos.complexity import combine_big_o_product, combine_big_o_sum
import math
import torch.nn.functional as F
from tqdm import tqdm
from Vathos._spatials import *


class Block1d(Layer):
    def __init__(self, d_model, channel_mixer: Layer, spatial_mixer: Layer, norm=nn.LayerNorm):
        super().__init__()
        self.spatial_mixer = spatial_mixer
        self.channel_mixer = channel_mixer
        self.norm1 = norm(spatial_mixer.d_model)
        self.norm2 = norm(spatial_mixer.d_model)

    def forward(self, x: torch.Tensor, ve=None):
        if ve is not None:
            x = x + self.spatial_mixer(self.norm1(x), ve)
        else:
            x = x + self.spatial_mixer(self.norm1(x))

        x = x + self.channel_mixer(self.norm2(x))
        return x

    def generate(self, x: torch.Tensor, ve=None):
        h = self.norm1(x)
        if self.spatial_mixer.has_custom_generate():
            spatial_out = _call_with_ve(self.spatial_mixer.generate, h, ve)
        else:
            spatial_out = _call_with_ve(self.spatial_mixer, h, ve)
        x = x + spatial_out

        if self.channel_mixer.has_custom_generate():
            x = x + self.channel_mixer.generate(self.norm2(x))
        else:
            x = x + self.channel_mixer(self.norm2(x))

        return x


def _call_with_ve(fn, h, ve):
    """Call fn(h, ve=ve) if it accepts ve, otherwise fn(h). Used by Block.generate
    so that spatial mixers without ve support stay drop-in."""
    if ve is None:
        return fn(h)
    try:
        return fn(h, ve=ve)
    except TypeError:
        return fn(h)


class CBlock1d(Layer):
    def __init__(self, d_model, channel_mixer: Layer, spatial_mixer: Layer, norm=nn.LayerNorm):
        super().__init__()
        self.spatial_mixer = spatial_mixer
        self.channel_mixer = channel_mixer
        self.norm1 = norm(spatial_mixer.d_model)
        self.norm2 = norm(spatial_mixer.d_model)
        self.l = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0]))

    def forward(self, x: torch.Tensor, ve=None):
        if ve is not None:
            x = x * self.l[0] + self.spatial_mixer(self.norm1(x), ve) * self.l[1]
        else:
            x = x * self.l[0] + self.spatial_mixer(self.norm1(x)) * self.l[1]
        x = x * self.l[2] + self.channel_mixer(self.norm2(x)) * self.l[3]
        return x

    def generate(self, x: torch.Tensor, ve=None):
        h = self.norm1(x)
        if self.spatial_mixer.has_custom_generate():
            spatial_out = _call_with_ve(self.spatial_mixer.generate, h, ve)
        else:
            spatial_out = _call_with_ve(self.spatial_mixer, h, ve)
        x = x * self.l[0] + spatial_out * self.l[1]
        if self.channel_mixer.has_custom_generate():
            x = x * self.l[2] + self.channel_mixer.generate(self.norm2(x)) * self.l[3]
        else:
            x = x * self.l[2] + self.channel_mixer(self.norm2(x)) * self.l[3]
        return x


# TODO: Smear
class SmearBlock1d(Layer):
    def __init__(self, d_model, channel_mixer: Layer, spatial_mixer: Layer, norm=nn.LayerNorm):
        super().__init__()
        self.spatial_mixer = spatial_mixer
        self.channel_mixer = channel_mixer
        self.norm1 = norm(spatial_mixer.d_model)
        self.norm2 = norm(spatial_mixer.d_model)

    def forward(self, x: torch.Tensor):
        x = x + self.spatial_mixer(self.norm1(x))
        x = x + self.channel_mixer(self.norm2(x))
        return x

    def generate(self, x: torch.Tensor):
        # Use generate if available, otherwise forward
        if self.spatial_mixer.has_custom_generate():
            x = x + self.spatial_mixer.generate(self.norm1(x))
        else:
            x = x + self.spatial_mixer(self.norm1(x))

        if self.channel_mixer.has_custom_generate():
            x = x + self.channel_mixer.generate(self.norm2(x))
        else:
            x = x + self.channel_mixer(self.norm2(x))

        return x


class JBlock1d(Layer):
    def __init__(self, d_model, channel_mixer: nn.Module, spatial_mixer: nn.Module, norm=nn.LayerNorm):
        super().__init__()
        self.channel_mixer = channel_mixer
        self.spatial_mixer = spatial_mixer
        self.norm = norm(d_model)

    def forward(self, x: torch.Tensor, ve=None):
        h = self.norm(x)
        if ve is not None:
            return x + self.spatial_mixer(h, ve) + self.channel_mixer(h)
        return x + self.spatial_mixer(h) + self.channel_mixer(h)

    def generate(self, x: torch.Tensor, ve=None):
        h = self.norm(x)
        sm_fn = self.spatial_mixer.generate if (hasattr(self.spatial_mixer, 'has_custom_generate') and self.spatial_mixer.has_custom_generate()) else self.spatial_mixer
        cm_fn = self.channel_mixer.generate if (hasattr(self.channel_mixer, 'has_custom_generate') and self.channel_mixer.has_custom_generate()) else self.channel_mixer
        spatial_out = _call_with_ve(sm_fn, h, ve)
        channel_out = cm_fn(h)
        return x + spatial_out + channel_out


class ExpandingBlock1d(Layer):
    def __init__(self, d_model, channel_mixer: Layer, spatial_mixer: Layer, expand=1):
        super().__init__()
        self.spatial_mixer = spatial_mixer
        self.channel_mixer = channel_mixer
        self.expander = nn.Linear(d_model, d_model * expand, bias=False)
        self.contractor = nn.Linear(d_model * expand, d_model, bias=False)
        self.norm1 = nn.LayerNorm(spatial_mixer.d_model)
        self.norm2 = nn.LayerNorm(spatial_mixer.d_model)

    def forward(self, x: torch.Tensor):
        x = x + self.contractor(self.spatial_mixer(self.norm1(self.expander(x))))
        x = x + self.channel_mixer(self.norm2(x))
        return x


class Renamer:
    def __init__(self, constructor, renames: dict):
        super().__init__()
        self.constructor = constructor
        self.renames = renames

    def __call__(self, *args, **kwargs):
        renamed_kwargs = {}
        for key, value in kwargs.items():
            if key in self.renames:
                renamed_kwargs[self.renames[key]] = value
            else:
                renamed_kwargs[key] = value
        return self.constructor(*args, renamed_kwargs)


class BlockStack(Layer):
    def __init__(self, blocks: Tuple[Block1d | Layer]):
        super().__init__()
        self.blocks = blocks
        self.stack = nn.ModuleList(blocks)

    def forward(self, x: torch):
        for block in self.stack:
            x = block(x)
        return x


class DepthwiseCausalConv1d(Layer):
    __name__ = "CausalConv1d"
    __complexity__ = "O(L d k)"

    def __init__(self, d_model, k=3):
        super().__init__()
        self.d_model = d_model
        self.k = k
        self.pad = k - 1

        self.K = nn.Parameter(torch.randn(d_model, k) / (k ** 0.5))

    def forward(self, x):
        b, L, d = x.shape
        x = x.transpose(1, 2)
        x_pad = F.pad(x, (self.pad, 0))
        out = F.conv1d(x_pad, self.K.unsqueeze(1), groups=d)
        return out.transpose(1, 2)


class CausalConv1d(Layer):
    __name__ = "CausalConv1d"
    __complexity__ = "O(L d k)"

    def __init__(self, d_model, k=3, groups=None, outproj=False):
        super().__init__()
        self.d_model = d_model
        self.k = k
        self.pad = k - 1
        self.groups = groups if groups is not None else d_model
        self.outproj = nn.Linear(groups, d_model, bias=False) if outproj else Identity()
        if self.groups != d_model and not outproj:
            flag(
                "Output channel numbers will be the number of groups, use outproj=True if you want to enable a linear projection to go back to d_model")

        self.K = nn.Parameter(torch.randn(d_model, k) / (k ** 0.5))

    def forward(self, x):
        x = x.transpose(1, 2)
        x_pad = F.pad(x, (self.pad, 0))
        out = F.conv1d(x_pad, self.K.unsqueeze(1), groups=self.groups)
        return out.transpose(1, 2)


class LSTM(Layer):
    __name__ = "LSTM"
    __complexity__ = "O(L d^2)"

    def __init__(self, d_model, d_hidden, bidirectional=False, dropout=0.1, n_layer=1):
        super().__init__()
        self.bidirectional = bidirectional
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.projout = nn.Linear(d_hidden, d_model) if d_model != d_hidden else Identity()
        self.LSTM = nn.LSTM(d_model, d_hidden, bidirectional=bidirectional, batch_first=True, num_layers=n_layer)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.LSTM(x)
        out = self.dropout(out)
        return self.projout(out)


class GRU(Layer):
    __name__ = "GRU"
    __complexity__ = "O(L d^2)"

    def __init__(self, d_model, d_hidden, bidirectional=False, dropout=0.1, n_layer=1):
        super().__init__()
        self.bidirectional = bidirectional
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.projout = nn.Linear(d_hidden, d_model) if d_model != d_hidden else Identity()
        self.GRU = nn.GRU(d_model, d_hidden, bidirectional=bidirectional, batch_first=True, num_layers=n_layer)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.GRU(x)
        out = self.dropout(out)
        return self.projout(out)


class LinearMixer(Layer):
    __name__ = "Linear Mixer"
    __complexity__ = "O(L^2 d)"

    def __init__(self, d_model, max_len=1000, causal=True):
        super().__init__()
        self.causal = causal
        self.d_model = d_model
        self.W = nn.Parameter(torch.randn(max_len, max_len) * 0.01)

    def forward(self, x):
        x = x / math.sqrt(self.d_model)
        if self.causal:
            return torch.tril(self.W[:x.shape[1], :x.shape[1]]) @ x
        else:
            return self.W[:x.shape[1], :x.shape[1]] @ x


class MLPMixer(Layer):
    __name__ = "Linear Mixer"
    __complexity__ = "O(L^2 d)"

    def __init__(self, d_model, max_len=1000, L_expand=1, causal=False, activation=nn.GELU):
        super().__init__()
        if causal:
            if L_expand < 1:
                raise ValueError("For the MLPMixer to be causal, L_expand must be greater than 1")
            if L_expand > 1:
                raise NotImplementedError(
                    "Causal MLP Mixer with L_expand is currently not implemented")  # TODO: Causal Lex
        self.causal = causal
        self.d_model = d_model
        self.W1 = nn.Parameter(torch.randn(max_len * L_expand, max_len) / max_len)
        self.W2 = nn.Parameter(torch.randn(max_len, max_len * L_expand) / max_len)
        self.activation = activation()

    def forward(self, x):
        x = x / math.sqrt(self.d_model)

        return self.W2 @ self.activation(self.W1 @ x)


class ShortConvGatedMixer(Layer):
    def __init__(self, d_model, mixer, mixer_params, k=4, activation=nn.Sigmoid, k1=None, k2=None):
        super().__init__()
        if k1 is None:
            k1 = k2 = k

        self.mixer = mixer(mixer_params)
        self.conv_1 = DepthwiseCausalConv1d(d_model, k=k1)
        self.conv_2 = DepthwiseCausalConv1d(d_model, k=k2)
        self.activation = activation()

    def forward(self, x):
        g1 = self.conv_1(x)
        g2 = self.activation(self.conv_2(x))
        return self.mixer(g1) * g2 + (1 - g2) * x


class Embedder(Layer):
    __name__ = "SymbolicEmbedder"
    __complexity__ = "O(L d)"

    def __init__(self, vocab_size, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.frozen = False
        self.embedding = nn.Embedding(vocab_size, d_model)

    def freeze(self):
        if self.frozen:
            flag("Trying to freeze an already frozen Embedder")
        else:
            self.embedding.weight.requires_grad = False
            self.frozen = True

    def unfreeze(self):
        if self.frozen:
            self.embedding.weight.requires_grad = True
            self.frozen = False
        else:
            flag("Trying to unfreeze an already unfrozen Embedder")

    def forward(self, x):
        return self.embedding(x)

    def finetune(self):
        self.freeze()


class EasyEmbedder(Layer):
    __name__ = "SymbolicEmbedder"
    __complexity__ = "O(L d)"

    def __init__(self, vocab_size, d_model: int, dropout=0.13):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.embedding(x))


class HybridAttentionBlock1d(Layer):
    def __init__(self, d_model, sec_mixer, sec_params, attn_params, n_attn=1, n_sec=3,
                 channel_mixer=MLP, channel_params=None):
        super().__init__()
        self.n_sec = n_sec
        self.n_attn = n_attn
        self.d_model = d_model
        self.attn_params = attn_params
        self.sec_params = sec_params
        self.sec_mixer = sec_mixer
        self.channel_mixer = channel_mixer
        self.channel_params = channel_params

        if channel_params is None and isinstance(channel_mixer, MLP):
            self.channel_params = {'expand': 2}
        else:
            self.channel_params = channel_params
        self.attn_blocks = nn.ModuleList([
            Block1d(
                channel_mixer=self.channel_mixer(d_model=d_model, **self.channel_params),
                spatial_mixer=MultiheadAttentionMixer(d_model=d_model, **self.attn_params)
            )
            for _ in range(self.n_attn)
        ])
        self.sec_blocks = nn.ModuleList([
            Block1d(
                channel_mixer=self.channel_mixer(d_model=d_model, **self.channel_params),
                spatial_mixer=self.sec_mixer(d_model=d_model, **self.sec_params)
            )
            for _ in range(self.n_sec)
        ])

    def forward(self, x):
        for sec in self.sec_blocks:
            x = sec(x)
        for attn in self.attn_blocks:
            x = attn(x)
        return x


########################################################################################################################
#   VISION
########################################################################################################################


class PatchEmbedder(Layer):
    __name__ = "PatchEmbedder"

    def __init__(
            self,
            vocab_size=None,
            d_model: int = 768,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            flatten: bool = True,
            norm_layer: Optional[Layer] = None,
            cls: bool = False,
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = d_model
        self.flatten = flatten
        self.frozen = False
        self.norm = norm_layer if norm_layer is not None else None
        self.use_cls = cls  # ← NEW

        self.proj = nn.Conv2d(in_chans, d_model, kernel_size=patch_size, stride=patch_size)

        if self.use_cls:
            assert flatten, "CLS token requires flatten=True"
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))  # ← NEW

        self.grid_size = None
        if img_size is not None:
            self.grid_size = (math.ceil(img_size[0] / patch_size[0]),
                              math.ceil(img_size[1] / patch_size[1]))

        self._num_patches = None
        if self.grid_size is not None:
            self._num_patches = self.grid_size[0] * self.grid_size[1]

    def freeze(self):
        if self.frozen:
            flag("Trying to freeze an already frozen PatchEmbedder")
            return
        for p in self.proj.parameters():
            p.requires_grad = False
        if self.norm is not None:
            for p in self.norm.parameters():
                p.requires_grad = False
        if self.use_cls:  # ← FREEZE CLS TOKEN TOO
            self.cls_token.requires_grad = False
        self.frozen = True

    def unfreeze(self):
        if not self.frozen:
            flag("Trying to unfreeze an already unfrozen PatchEmbedder")
            return
        for p in self.proj.parameters():
            p.requires_grad = True
        if self.norm is not None:
            for p in self.norm.parameters():
                p.requires_grad = True
        if self.use_cls:
            self.cls_token.requires_grad = True
        self.frozen = False

    def num_patches(self, H: Optional[int] = None, W: Optional[int] = None) -> int:
        if H is None or W is None:
            if self.img_size is not None:
                H, W = self.img_size
            else:
                raise ValueError("Must provide H and W or set img_size during init.")
        ph, pw = self.patch_size
        Hp = math.ceil(H / ph)
        Wp = math.ceil(W / pw)
        return Hp * Wp

    def _pad_to_patch_multiple(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        ph, pw = self.patch_size
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        if pad_h == 0 and pad_w == 0:
            return x
        return nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, "Input must be 4D (B, C, H, W)"
        assert x.dtype in (torch.float32, torch.float16, torch.bfloat16), "Input dtype must be float"
        B, C, H, W = x.shape
        assert C == self.in_chans, f"Expected {self.in_chans} channels, got {C}"

        x = self._pad_to_patch_multiple(x)

        x = self.proj(x)  # (B, embed_dim, H_p, W_p)
        H_p, W_p = x.shape[2], x.shape[3]
        self._num_patches = H_p * W_p

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)

            if self.use_cls:
                cls_tok = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
                x = torch.cat((cls_tok, x), dim=1)  # (B, 1+N, D)

            if self.norm is not None:
                x = self.norm(x)
        else:
            if self.norm is not None:
                x = self.norm(x)

        return x


class MeanClassificationHead(Layer):
    __name__ = "MeanClassificationHead"

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = x.mean(dim=1)
        return self.proj(x)


class ClsHead(Layer):
    __name__ = "Cls Head"

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = x[:, -1:, :]
        return self.proj(x)[:, 0, :]


class MultiHeadUnembedder(Layer):
    __name__ = "UnbiasedLinear"
    __complexity__ = "O(L d^2)"

    def __init__(self, d_model, vocab_size, k=4):
        super(MultiHeadUnembedder, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size * k, bias=False)

    def forward(self, x):
        return self.linear(x)


class VathosModel(Layer):
    __name__ = "VathosModel"

    def __init__(self):
        super().__init__()

        self._losses = []
        self._losses_dict = {}
        self._losses_per_epoch = []
        self._losses_per_epoch_dict = {}
        self._losses_this_epoch = []
        self._metrics = dict()
        self._metrics_this_epoch = dict()
        self._metrics_per_epoch = dict()
        self.autosave = True
        self.autosave_overwrite = True
        self.best_loss = float('inf')
        self.checkpoints = 0
        self.steps = 0
        self.steps_per_epoch = 0
        self.epochs = 0
        self.finetuning = False
        self.name = 'VathosModel_defaultName'

    def flag_not_training(self):
        if not self.training:
            flag("You are not training")

    def register_loss(self, loss: float):
        self._losses_dict[self.steps] = loss
        self._losses.append(loss)
        self._losses_this_epoch.append(loss)
        self.steps += 1

    def get_last_loss(self):
        return self._losses[-1]

    def get_mean_loss(self, epoch=True):
        if epoch:
            return np.mean(self._losses_this_epoch)
        else:
            return np.mean(self._losses)

    def register_metrics(self, metrics: dict):
        for metric in metrics:
            if metric in self._metrics:
                self._metrics[metric].append(metrics[metric])
                self._metrics_this_epoch[metric].append(metrics[metric])
            else:
                flag(f"Registering a new metric {metric}")
                self._metrics[metric] = [metrics[metric]]
                self._metrics_this_epoch[metric] = [metrics[metric]]

    def register_epoch(self):
        self.epochs += 1
        self._losses_per_epoch.append(np.mean(self._losses_this_epoch))
        self._losses_per_epoch_dict[self.steps] = np.mean(self._losses_this_epoch)
        self._losses_this_epoch = []

        for metric in self._metrics_this_epoch:
            if metric in self._metrics_per_epoch:
                self._metrics_per_epoch[metric].append(np.mean(self._metrics_this_epoch[metric]))
            else:
                self._metrics_per_epoch[metric] = [np.mean(self._metrics_this_epoch[metric])]
            self._metrics_this_epoch[metric] = []

        if self._losses_per_epoch[-1] < self.best_loss:
            self.best_loss = self._losses_per_epoch[-1]
            if self.autosave:
                self.checkpoints += 1
                if self.autosave_overwrite:
                    self.save_checkpoint(f'{self.name}-checkpoint.pt')
                else:
                    self.save_checkpoint(f'{self.name}-checkpoint-{self.checkpoints}.pt')

    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)

    def plot_losses(self):
        print(self.steps_per_epoch)
        plt.plot(
            list(self._losses_dict.keys()),
            list(self._losses_dict.values()),
            label="Losses", linewidth=1)
        plt.plot(
            list(self._losses_per_epoch_dict.keys()),
            list(self._losses_per_epoch_dict.values()),
            label="Losses Per Epoch", linewidth=2)

        plt.xlabel("steps")
        plt.ylabel("loss")
        plt.title("Model Losses per Steps")
        plt.show()

    def plot_metrics(self, figsize=(12, 8)):
        n_metrics = len(self._metrics_per_epoch)
        n_plots = 1 + n_metrics

        n_cols = 2
        n_rows = (n_plots + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        if n_plots == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()

        ax = axes[0]
        ax.plot(
            list(self._losses_dict.keys()),
            list(self._losses_dict.values()),
            label="Losses", linewidth=1, alpha=0.6)
        ax.plot(
            list(self._losses_per_epoch_dict.keys()),
            list(self._losses_per_epoch_dict.values()),
            label="Losses Per Epoch", linewidth=2)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        for idx, (metric_name, metric_values) in enumerate(self._metrics_per_epoch.items(), start=1):
            ax = axes[idx]

            if metric_name in self._metrics and len(self._metrics[metric_name]) > 0:
                step_indices = list(range(len(self._metrics[metric_name])))
                ax.plot(step_indices, self._metrics[metric_name],
                        label=f"{metric_name}", linewidth=1, alpha=0.6)

            epoch_indices = list(range(len(metric_values)))
            ax.plot(epoch_indices, metric_values,
                    label=f"{metric_name} Per Epoch", linewidth=2, marker='o')

            ax.set_xlabel("Steps/Epochs")
            ax.set_ylabel(metric_name)
            ax.set_title(f"Metric: {metric_name}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide any unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.show()

    def save_checkpoint(self, path):
        """Save complete model checkpoint including training state"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'losses': self._losses,
            'losses_dict': self._losses_dict,
            'losses_per_epoch': self._losses_per_epoch,
            'losses_per_epoch_dict': self._losses_per_epoch_dict,
            'losses_this_epoch': self._losses_this_epoch,
            'metrics': self._metrics,
            'metrics_this_epoch': self._metrics_this_epoch,
            'metrics_per_epoch': self._metrics_per_epoch,
            'best_loss': self.best_loss,
            'checkpoints': self.checkpoints,
            'steps': self.steps,
            'steps_per_epoch': self.steps_per_epoch,
            'epochs': self.epochs,
            'autosave': self.autosave,
            'autosave_overwrite': self.autosave_overwrite,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Load complete model checkpoint including training state"""
        checkpoint = torch.load(path, weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'])

        # Restore training state
        self._losses = checkpoint['losses']
        self._losses_dict = checkpoint['losses_dict']
        self._losses_per_epoch = checkpoint['losses_per_epoch']
        self._losses_per_epoch_dict = checkpoint['losses_per_epoch_dict']
        self._losses_this_epoch = checkpoint['losses_this_epoch']
        self._metrics = checkpoint['metrics']
        self._metrics_this_epoch = checkpoint['metrics_this_epoch']
        self._metrics_per_epoch = checkpoint['metrics_per_epoch']
        self.best_loss = checkpoint['best_loss']
        self.checkpoints = checkpoint['checkpoints']
        self.steps = checkpoint['steps']
        self.steps_per_epoch = checkpoint['steps_per_epoch']
        self.epochs = checkpoint['epochs']
        self.autosave = checkpoint['autosave']
        self.autosave_overwrite = checkpoint['autosave_overwrite']

    def finetune(self):
        self.finetuning = True

    def train(self, *args, **kwargs):
        self.finetuning = False
        super().train(*args, **kwargs)

    def eval(self):
        self.finetuning = False
        super().eval()


########################################################################################################################
#   Assemblers
########################################################################################################################

class SequenceModel(VathosModel):
    __name__ = "SequenceModel"

    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 max_len=1024,
                 pos_encoder: bool | None | Layer | nn.Module = None,
                 embedder=Embedder,
                 embedder_args: dict = None,
                 unembedder=UnbiasedLinear,
                 unembedder_args=None,
                 channel_mixer=MLP,
                 spatial_mixer: Layer | nn.Module = MultiheadAttentionMixer,
                 channel_args: dict = None,
                 spatial_args: dict = None,
                 name='',
                 pad='none',
                 baseblock=Block1d,
                 baseblock_args=None,
                 dropout=0.1,
                 weight_tying=False,
                 norm=nn.LayerNorm,
                 d_modifiers: List | None = None,
                 unet_skips=False,
                 unet_weights=False
                 ):
        super().__init__()
        self.pad = pad
        flag(
            "If you need to use any RoPE or alternative positional encodings which operate directly in the spatial mixer, be sure to call activate it in the spatial_args (e.g. rope=True)",
            2)

        if channel_args is None and channel_mixer is MLP:
            channel_args = {"expand": 2, "activation": nn.GELU, "depth": 2}
        if spatial_args is None:
            spatial_args = {}
        if channel_args is None:
            channel_args = {}
        if embedder_args is None:
            embedder_args = {}
        if unembedder_args is None:
            unembedder_args = {'input_features': d_model, 'output_features': vocab_size}
        if baseblock_args is None:
            baseblock_args = {}
        if d_modifiers is None:
            d_modifiers = [1 for _ in range(d_model)]
        else:
            print(
                f"{NUM} Initialized SequenceModel with d_model structure: {[int(d_model * d) for d in d_modifiers]}{RES}")

        self.pipe = {}
        self.name = name
        self.baseblock = baseblock
        self.spatial_mixer = spatial_mixer
        self.channel_mixer = channel_mixer
        self.baseblock_args = baseblock_args

        self.spatial_args = spatial_args
        self.channel_args = channel_args
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.unet_skips = unet_skips

        if self.unet_skips:
            self.skip_weights = nn.Parameter(torch.zeros(self.n_layers // 2), requires_grad=unet_weights)
            if d_modifiers is not None:
                flag("Unet Skips detected, assure d_modifier are symmetrical to make it work!")

        self.embedder = embedder(vocab_size=vocab_size, d_model=d_model, **embedder_args)

        self.pos_encoder = pos_encoder(d_model, max_len=max_len) if pos_encoder not in (True, False, None) else \
            (SinusoidalPositionalEncoding(d_model, max_len=max_len) if pos_encoder is True else nn.Identity())

        self.blocks = nn.ModuleList([
            self.baseblock(
                d_model=int(d_model * d_modifiers[i]),
                channel_mixer=channel_mixer(d_model=int(d_model * d_modifiers[i]), **channel_args),
                spatial_mixer=spatial_mixer(d_model=int(d_model * d_modifiers[i]), **spatial_args),
                norm=norm,
                **baseblock_args
            )
            for i in range(n_layers)
        ])
        self._piped_blocks = None

        self.norm = nn.LayerNorm(d_model)
        self.unembedder = unembedder(**unembedder_args)

        if weight_tying and hasattr(self.embedder, 'embedding') and hasattr(self.unembedder, 'linear'):
            self.unembedder.linear.weight = self.embedder.embedding.weight

        elif weight_tying:
            raise TypeError(
                "Automatic weight tying is only possible if the the Embedder has a 'embedding' attribute and Unembedder has a linear attribute"
                "You should manually do weight tying if you aim to use specific layer:"
                "\n e.g. model.unembedder.yourmodule.weight = model.embedder.youembeddings.weight is an auto weight tying example")

        self.embedder_complexity = embedder.__complexity__ if hasattr(embedder, "__complexity__") else "O(L d)"
        self.unembedder_complexity = unembedder.__complexity__ if hasattr(unembedder, "__complexity__") else "O(L d)"
        self.spatial_complextiy = spatial_mixer.__complexity__ if hasattr(spatial_mixer, "__complexity__") else "O(L d)"
        self.channel_complexity = channel_mixer.__complexity__ if hasattr(channel_mixer, "__complexity__") else "O(L d)"
        self.runned = False
        self.embed_scale = math.sqrt(d_model)
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights following GPT-style conventions:
        - Small std for embeddings
        - Xavier/Kaiming for linear layers
        - Scaled initialization for residual projections
        """
        std_embed = 0.02
        try:
            nn.init.normal_(self.embedder.embedding.weight, mean=0.0, std=std_embed)
        except:
            pass
        try:
            nn.init.normal_(self.unembedder.weight, mean=0.0, std=std_embed)
        except:
            pass

        for module in self.modules():
            if isinstance(module, nn.Linear):
                std_init = 0.02
                nn.init.normal_(module.weight, mean=0.0, std=std_init)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        for block_idx, block in enumerate(self.blocks):
            for name, module in block.named_modules():
                if isinstance(module, nn.Linear):
                    depth_scale = (2.0 * self.n_layers) ** -0.5
                    if 'out' in name.lower() or 'proj' in name.lower() or '.l2' in name or '.g2' in name:
                        with torch.no_grad():
                            module.weight.data *= depth_scale

    def forward(self, x: torch.LongTensor, unembed=True):
        # B, L = x.size(0), x.size(1)
        # x = self.embedder(x) * self.embed_scale
        x = self.pos_encoder(x)

        skips = []
        half_layers = self.n_layers // 2

        for i, block in enumerate(self.blocks):
            if self.unet_skips:
                if i < half_layers:
                    skips.append(x)
                elif i >= self.n_layers - half_layers:
                    skip_idx = self.n_layers - 1 - i
                    x = x + self.skip_weights[skip_idx] * skips[skip_idx]

            x = block(x)

        if unembed:
            x = self.norm(self.unembedder(x))

        return x

    def insert_block(self, idx, module):
        self.blocks.insert(idx, module)

    def append(self, module):
        self.blocks.append(module)

    @torch.no_grad()
    def _clear_all_caches(self):
        """Clear KV caches in all attention layers"""
        for block in self.blocks:
            for module in block.modules():
                if hasattr(module, 'clear_cache'):
                    module.clear_cache()

    def forward(self, x: torch.LongTensor, unembed=True):
        B, L = x.size(0), x.size(1)
        x = self.embedder(x) * self.embed_scale
        x = self.pos_encoder(x)

        if self.pad == 'sqrt':
            n = int((x.shape[1] ** 0.5) + 0.999999)
            x = F.pad(x, (0, 0, 0, n ** 2 - L), mode="constant", value=0)

        for block in self.blocks:
            x = block(x)

        if unembed:
            x = self.unembedder(x)

        return x[:, :L, :]

    def _sample_token(self, logits, temperature=1.0, top_p=1.0, top_k=None):
        logits = logits / (temperature + 1e-8)

        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def generate(self, *args, **kwargs):
        if not 'custom_generate' in kwargs:
            kwargs['custom_generate'] = False
        if kwargs['custom_generate']:
            del kwargs['custom_generate']
            return self.custom_generate(*args, **kwargs)
        else:
            del kwargs['custom_generate']
            return self.simple_generate(*args, **kwargs)

    @torch.no_grad()
    def simple_generate(self, prompt: torch.Tensor, max_len=100, temperature=1.0,
                        top_p=1.0, top_k=50, token_end=None, repetition_penalty=1.0):
        self.eval()
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)

        generated = prompt.clone()
        pbar = tqdm(range(max_len), desc="Simple Gen")

        for _ in pbar:
            logits = self.forward(generated, unembed=True)
            next_token_logits = logits[:, -1, :]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated, repetition_penalty
                )

            next_token = self._sample_token(next_token_logits, temperature, top_p, top_k)
            generated = torch.cat([generated, next_token], dim=1)

            if token_end is not None and (next_token == token_end).all():
                break

        return generated

    @torch.no_grad()
    def custom_generate(self, prompt: torch.Tensor, max_len=100, temperature=1.0,
                        top_p=1.0, top_k=50, token_end=None, repetition_penalty=1.0):
        self.eval()
        self._clear_all_caches()

        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)

        x = self.embedder(prompt) * self.embed_scale
        if self.pos_encoder is not None and not isinstance(self.pos_encoder, nn.Identity):
            x = self.pos_encoder(x)

        for block in self.blocks:
            if block.has_custom_generate():
                x = block.generate(x)
            else:
                x = block(x)

        generated = prompt.clone()
        logits = self.unembedder(x)
        next_token_logits = logits[:, -1, :]

        # Apply repetition penalty
        if repetition_penalty != 1.0:
            next_token_logits = self._apply_repetition_penalty(
                next_token_logits, generated, repetition_penalty
            )

        next_token = self._sample_token(next_token_logits, temperature, top_p, top_k)
        generated = torch.cat([generated, next_token], dim=1)

        pbar = tqdm(range(max_len - 1), desc="Fast Gen")
        for _ in pbar:
            x_t = self.embedder(next_token) * self.embed_scale
            current_pos = generated.shape[1] - 1

            if isinstance(self.pos_encoder, SinusoidalPositionalEncoding):
                pe_slice = self.pos_encoder.pe[current_pos: current_pos + 1].unsqueeze(0)
                x_t = x_t + pe_slice

            for block in self.blocks:
                if block.has_custom_generate():
                    x_t = block.generate(x_t)
                else:
                    x_t = block(x_t)

            logits = self.unembedder(x_t)
            next_token_logits = logits[:, -1, :]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated, repetition_penalty
                )

            next_token = self._sample_token(next_token_logits, temperature, top_p, top_k)
            generated = torch.cat([generated, next_token], dim=1)

            if token_end is not None and (next_token == token_end).all():
                break

        self._clear_all_caches()
        return generated

    def _apply_repetition_penalty(self, logits: torch.Tensor,
                                  generated: torch.Tensor,
                                  repetition_penalty: float) -> torch.Tensor:
        """
        Apply repetition penalty to logits based on previously generated tokens.

        Args:
            logits: Shape (batch_size, vocab_size)
            generated: Shape (batch_size, seq_len) - previously generated tokens
            repetition_penalty: Penalty factor (> 1.0 discourages repetition)

        Returns:
            Modified logits with repetition penalty applied
        """
        batch_size = logits.shape[0]

        for i in range(batch_size):
            # Get unique tokens in the generated sequence for this batch item
            unique_tokens = generated[i].unique()

            # Apply penalty: divide logits by penalty if positive, multiply if negative
            for token_id in unique_tokens:
                if logits[i, token_id] > 0:
                    logits[i, token_id] /= repetition_penalty
                else:
                    logits[i, token_id] *= repetition_penalty

        return logits

    def summary(self):
        complexities = [self.channel_complexity, self.spatial_complextiy, self.embedder_complexity,
                        self.unembedder_complexity]
        print(f'{NUM}VATHOS{RES} {self.name} Summary:')
        print(f"{NUM}SequenceModel{RES}(d_model={NUM}{self.d_model}{RES}, n_layer={NUM}{self.n_layers}{RES})")
        print(f"\t - {NUM}VOCAB_SIZE:{RES}: {NUM}{self.vocab_size}{RES}")
        print(f"\t - {NUM}D_MODEL:{RES}: {NUM}{self.d_model}{RES}")
        print(f"\t - {NUM}N_LAYERS:{RES}: {NUM}{self.n_layers}{RES}")
        print("")
        print(f"\t - {NUM}Embedder{RES}: {getname(self.embedder)} - {NUM}{self.embedder_complexity}{RES}")
        print(f"\t - {NUM}Unembedder{RES}: {getname(self.unembedder)} - {NUM}{self.unembedder_complexity}{RES}")
        print(
            f"\t - {NUM}Spatial Mixer{RES}: {getname(self.spatial_mixer)}({self.spatial_args}) - {NUM}{self.spatial_complextiy}{RES}")
        print(
            f"\t - {NUM}Channel Mixer{RES}: {getname(self.channel_mixer)}({self.channel_args}) - {NUM}{self.channel_complexity}{RES}")
        print(f"Num Parameters: {NUM}{sum([p.numel() for p in self.parameters()]):_}{RES}")
        print(f"Num Trainable Parameters: {NUM}{sum([p.numel() for p in self.parameters() if p.requires_grad]):_}{RES}")
        print(f"Total Complexity: {NUM}{combine_big_o_sum(complexities)}{RES}")

    def finetune(self):
        flag(
            "Finetune simply checks for finetune() methods in spatial mixers, channel mixers, embedder and unembedder, "
            "if a finetune method is not available, the module/Layer will be left as it is")
        super().finetune()

        if hasattr(self.embedder, "finetune"):
            self.embedder.finetune()
        if hasattr(self.unembedder, "finetune"):
            self.unembedder.finetune()
        for block in self.blocks:
            if hasattr(block, "finetune"):
                block.channel_mixer.finetune()
                block.spatial_mixer.finetune()


class ModdedFormer(VathosModel):
    def __init__(self, vocab_size: int, embed_dim: int, d_models: List[int], spatials: List[Builder], expand=3,
                 M_dims: List[int] | None = None, weights_tying=True,
                 baseblock=Block1d, norm=RMSNorm, ffn_act=ReLU2, unet_skips=False, max_len=2400, zeroskip=False,
                 UDLP=VariableUDLP,
                 skips: List[int | None] | None = None, value_embeddings: List[int | None] | None = None,
                 ve_type: str = 'scalar', ve_gate_dim: int | None = None,
                 learnable_pe: bool = False, input_projections: bool | int=False, smear_gate: None | nn.Module | Layer=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        assert d_models[0] == embed_dim
        assert max(d_models) == min(d_models), "Variables d_models is WIP and not working"
        self.d_models = d_models
        n_layers = self.n_layer = len(d_models)
        self.input_projections = input_projections
        self.smear_gate = smear_gate
        if M_dims is not None:
            self.M_dims = M_dims
        else:
            self.M_dims = [d * expand for d in d_models]

        self.embedder = Embedder(vocab_size, embed_dim)
        self.unembedder = UnbiasedLinear(d_models[-1], vocab_size)
        self.spatials = spatials
        self.norm = norm
        self.unet = unet_skips
        self.max_len = max_len
        self.zeroskip = zeroskip

        self.skips = skips if skips is not None else [None] * n_layers
        assert len(self.skips) == n_layers

        self.skip_lambdas = nn.ParameterDict()
        for source_idx, target_idx in enumerate(self.skips):
            if target_idx is not None:
                assert target_idx > source_idx
                assert target_idx < n_layers
                self.skip_lambdas[f"route_{source_idx}_to_{target_idx}"] = nn.Parameter(torch.zeros(1))

        if weights_tying:
            assert d_models[-1] == embed_dim
            self.unembedder.linear.weight = self.embedder.embedding.weight

        blocks = []
        for i in range(n_layers):
            in_dim = embed_dim if i == 0 else self.d_models[i - 1]
            out_dim = self.d_models[i]
            blocks.append(
                baseblock(
                    self.d_models[i],
                    UDLP(in_dim, d_output=out_dim, M=self.M_dims[i], activation=ffn_act),
                    self.spatials[i](self.d_models[i]),
                    norm=norm
                ))

        self.blocks = nn.ModuleList(blocks)

        if zeroskip:
            self.zeroskip_params = nn.ParameterList([nn.Parameter(torch.tensor([0.0])) for _ in range(n_layers)])
        if input_projections > 0:
            if not zeroskip:
                raise ValueError("Zeroskip must be activated when using input_projections")
            self.inputs_projection_linears = nn.Linear(in_features=d_models[0], out_features=d_models[0]*input_projections, bias=False)
            self.inputs_projection_params = nn.ParameterList([nn.Parameter(torch.tensor([0.0])) for _ in range(n_layers) for i in range(input_projections)])


        assert ve_type in ('scalar', 'gate'), f"ve_type must be 'scalar' or 'gate', got {ve_type}"
        self.ve_type = ve_type
        self.ve_gate_dim = ve_gate_dim if ve_gate_dim is not None else embed_dim
        self.value_embeddings_cfg = value_embeddings

        if value_embeddings is not None:
            assert len(value_embeddings) == n_layers, "value_embeddings length must equal n_layers"
            unique_groups = sorted(set(v for v in value_embeddings if v is not None))
            self.ve_embeddings = nn.ModuleDict({
                str(g): nn.Embedding(vocab_size, embed_dim)
                for g in unique_groups
            })
            if ve_type == 'scalar':
                self.ve_scales = nn.ParameterDict({
                    str(i): nn.Parameter(torch.zeros(1))
                    for i, v in enumerate(value_embeddings) if v is not None
                })
            else:  # gate
                self.ve_gates = nn.ModuleDict({
                    str(i): nn.Linear(self.ve_gate_dim, embed_dim, bias=False)
                    for i, v in enumerate(value_embeddings) if v is not None
                })
        else:
            self.ve_embeddings = nn.ModuleDict()

        self.learnable_pe = learnable_pe
        if self.learnable_pe:
            self.pos_emb = nn.Parameter(torch.zeros(1, max_len, embed_dim))

        self.final_norm = RMSNorm(d_models[-1])
        self._init_weights()

    def _apply_ve_weight(self, ve: torch.Tensor, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if self.ve_type == 'scalar':
            return ve * self.ve_scales[str(layer_idx)]
        else:  # gate: nanogpt-style, 2*sigmoid(linear(x)) elementwise on ve
            gate = 2.0 * torch.sigmoid(self.ve_gates[str(layer_idx)](x[..., :self.ve_gate_dim]))
            return ve * gate

    def forward(self, x):
        input_ids = x
        seq_len = x.size(1)

        x0 = self.embedder(x)

        if self.input_projections > 0:
            xs = self.inputs_projection_linears(x0).chunk(self.input_projections, dim=-1)

        if self.learnable_pe:
            x0 = x0 + self.pos_emb[:, :seq_len, :]

        if self.smear_gate:
            x0 = self.smear_gate(x0)

        x = x0

        active_skips = {}

        for i, block in enumerate(self.blocks):
            ve = None
            if self.value_embeddings_cfg is not None and self.value_embeddings_cfg[i] is not None:
                group_id = self.value_embeddings_cfg[i]
                ve = self.ve_embeddings[str(group_id)](input_ids)
                ve = self._apply_ve_weight(ve, x, i)
            if self.zeroskip and self.input_projections > 0:
                x = block(x, ve=ve) + x0 * self.zeroskip_params[i] + sum([x * p for x, p in zip(xs,self.inputs_projection_params[i * self.input_projections: (i + 1) * self.input_projections])])
            elif self.zeroskip:
                x = block(x, ve=ve) + x0 * self.zeroskip_params[i]
            else:
                x = block(x, ve=ve)

            for source_idx, target_idx in enumerate(self.skips):
                if target_idx == i:
                    gate = self.skip_lambdas[f"route_{source_idx}_to_{target_idx}"]
                    x = x + gate * active_skips[source_idx]
                    del active_skips[source_idx]

            if self.skips[i] is not None:
                active_skips[i] = x

        x = self.unembedder(self.final_norm(x))
        return x

    @torch.no_grad()
    def _clear_all_caches(self):
        for block in self.blocks:
            for module in block.modules():
                if hasattr(module, 'clear_cache'):
                    module.clear_cache()

    def _embed_input(self, ids: torch.Tensor, pos_start: int) -> torch.Tensor:
        x0 = self.embedder(ids)
        if self.learnable_pe:
            L = ids.size(1)
            x0 = x0 + self.pos_emb[:, pos_start:pos_start + L, :]
        if self.smear_gate:
            x0 = self.smear_gate(x0)
        return x0

    def _blocks_generate(self, x: torch.Tensor, x0: torch.Tensor,
                        xs, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Mirror of forward()'s block loop, but calls block.generate(...) so
        that attention mixers populate / consume their KV cache.
        Works for both prefill (L>1) and single-token increments (L=1).
        """
        active_skips = {}
        for i, block in enumerate(self.blocks):
            ve = None
            if self.value_embeddings_cfg is not None and self.value_embeddings_cfg[i] is not None:
                group_id = self.value_embeddings_cfg[i]
                ve = self.ve_embeddings[str(group_id)](input_ids)
                ve = self._apply_ve_weight(ve, x, i)

            if self.zeroskip and self.input_projections > 0:
                proj_params = self.inputs_projection_params[
                    i * self.input_projections: (i + 1) * self.input_projections
                ]
                x = block.generate(x, ve=ve) + x0 * self.zeroskip_params[i] + sum(
                    [xj * p for xj, p in zip(xs, proj_params)]
                )
            elif self.zeroskip:
                x = block.generate(x, ve=ve) + x0 * self.zeroskip_params[i]
            else:
                x = block.generate(x, ve=ve)

            for source_idx, target_idx in enumerate(self.skips):
                if target_idx == i:
                    gate = self.skip_lambdas[f"route_{source_idx}_to_{target_idx}"]
                    x = x + gate * active_skips[source_idx]
                    del active_skips[source_idx]

            if self.skips[i] is not None:
                active_skips[i] = x
        return x

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_len: int = 100,
                 temperature: float = 1.0, top_k: int | None = None,
                 top_p: float = 1.0, repetition_penalty: float = 1.0,
                 token_end=None, simple: bool = False):
        """
        KV-cached AR generation. Prefills the prompt in one shot (every spatial
        mixer's .generate accepts L>1 and populates its kv_cache), then walks
        token-by-token reusing the cached keys/values.

        simple=True falls back to simple_ar_generate (O(L^2)) — useful for
        regression-checking the cached path.
        """
        if simple:
            return simple_ar_generate(
                self, prompt, max_len=max_len, temperature=temperature,
                top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                token_end=token_end,
            )

        self.eval()
        self._clear_all_caches()

        if self.value_embeddings_cfg is not None:
            flag("ModdedFormer.generate: value_embeddings only work with spatial mixers "
                 "whose .generate accepts a 've=' kwarg (e.g. MultiheadAttentionMixer). "
                 "On other mixers ve is silently dropped during generation.", 2)

        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)
        device = next(self.parameters()).device
        prompt = prompt.to(device)

        L = prompt.size(1)
        x0 = self._embed_input(prompt, pos_start=0)
        xs = (self.inputs_projection_linears(x0).chunk(self.input_projections, dim=-1)
              if self.input_projections > 0 else None)

        x = self._blocks_generate(x0, x0, xs, prompt)
        logits = self.unembedder(self.final_norm(x))[:, -1, :]

        if repetition_penalty != 1.0:
            logits = apply_repetition_penalty(logits, prompt, repetition_penalty)
        next_token = sample_next_token(logits, temperature, top_k, top_p)
        generated = torch.cat([prompt, next_token], dim=1)

        pos = L
        pbar = tqdm(range(max_len - 1), desc="ModdedFormer fast gen")
        for _ in pbar:
            x0_t = self._embed_input(next_token, pos_start=pos)
            xs_t = (self.inputs_projection_linears(x0_t).chunk(self.input_projections, dim=-1)
                    if self.input_projections > 0 else None)

            x_t = self._blocks_generate(x0_t, x0_t, xs_t, next_token)
            logits = self.unembedder(self.final_norm(x_t))[:, -1, :]

            if repetition_penalty != 1.0:
                logits = apply_repetition_penalty(logits, generated, repetition_penalty)
            next_token = sample_next_token(logits, temperature, top_k, top_p)
            generated = torch.cat([generated, next_token], dim=1)
            pos += 1

            if token_end is not None and (next_token == token_end).all():
                break

        self._clear_all_caches()
        return generated

    def _init_weights(self):
        std_embed = 0.02
        try:
            nn.init.normal_(self.embedder.embedding.weight, mean=0.0, std=std_embed)
        except:
            pass
        try:
            nn.init.normal_(self.unembedder.weight, mean=0.0, std=std_embed)
        except:
            pass

        if self.learnable_pe:
            nn.init.normal_(self.pos_emb, mean=0.0, std=std_embed)

        for block_idx, block in enumerate(self.blocks):
            block.channel_mixer._init_weights()
            try:
                block.spatial_mixer._init_weights()
            except:
                pass

    def summary(self):
        print(f"Vathos {NUM}ModdedFormer{RES} Summary")
        print(f"Embedding dim: {self.embed_dim}")
        print(f"Learnable PE: {self.learnable_pe}")
        print(f"Skips: {self.skips}")
        for i in range(self.n_layer):
            if i == 0:
                in_dim = self.embed_dim
            else:
                in_dim = self.d_models[i - 1]
            out_dim = self.d_models[i]
            M = self.M_dims[i]
            print(f"Layer {i}: {in_dim} -> {out_dim}")
            print(f"Block: {i}: {self.blocks[i]}")
            print(f"\tFFN Dimension: {M}")
            print(f"\tFFN Activation: {self.blocks[i].channel_mixer.activation}")

        print(f"Num Parameters: {NUM}{sum([p.numel() for p in self.parameters()]):_}{RES}")
        print(f"Num Trainable Parameters: {NUM}{sum([p.numel() for p in self.parameters() if p.requires_grad]):_}{RES}")


#######################################################################################################################

"""def test_causality(module=MTransformer(8, 4, 2)):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = module
    x = torch.randn(2, 20, model.d_model, device=device)

    print(f"\nInput shape: {x.shape}")

    with torch.no_grad():
        output = model(x)

    model.eval()
    with torch.no_grad():
        full_output = model(x)

        for t in range(1, 20):
            prefix_output = model(x[:, :t, :])

            max_diff = (full_output[:, :t, :] - prefix_output).abs().max().item()

            if max_diff > 1e-5:
                print(f"Causality violated")
                break
        else:
            print("Causality verified")


def test_causality_symbolic(module=SequenceModel(128, 16, 4, 2, pos_encoder=True)):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = module
    x = torch.randint(0, model.vocab_size, (2, model.max_len), device=device)

    print(f"\nInput shape: {x.shape}")

    with torch.no_grad():
        output = model(x)

    model.eval()
    with torch.no_grad():
        full_output = model(x, unembed=False)

        for t in range(1, 20):
            prefix_output = model(x[:, :t], unembed=False)

            max_diff = (full_output[:, :t] - prefix_output).abs().max().item()

            if max_diff > 1e-5:
                print(f"Causality violated")
                break
        else:
            print("Causality verified")


def test_symbolic_model(model):
    vocab_size = model.vocab_size
    length = model.max_len
    x = torch.randint(0, vocab_size, (2, length))
    print(f"Input shape {x.shape}")
    x = model(x)
    print(f"Output shape {x.shape}")
    print(f"Bounds:  min:{x.min()}, max:{x.max()}, mean:{x.mean()}, std:{x.std()}, sum_of_a_vector: {x[0, 0, :].sum()}")"""

########################################################################################################################

NAMES = {
    "MLP": MLP,
    "Attention": MultiheadAttentionMixer,
    "MHA": MultiheadAttentionMixer,
    "Embed": Embedder,
    "EMBED": Embedder,
    "E": Embedder,
    "PE": SinusoidalPositionalEncoding,
    "PatchEmbedder": PatchEmbedder,
    "ClassificationHead": MeanClassificationHead,
    "MCH": MeanClassificationHead,
    "ClsHead": ClsHead,
}


def expand_architecture(arch_string):
    pattern = r"(\(.+?\))x(\d+)"

    def replacer(match):
        content = match.group(1)
        count = int(match.group(2))
        return " -> ".join([content] * count)

    return re.sub(pattern, replacer, arch_string)


def assemble(code, d_model=512):
    code = code.strip()
    if "x" in code:
        code = expand_architecture(code)
    divided = code.split("->")

    for arch in divided:
        arch = arch.strip()
        if arch.startswith("("):
            layers = []
            for subarch in arch[1:-1].split(","):
                subarch = subarch.strip()
                if subarch in NAMES:
                    layer = NAMES[subarch](d_model=d_model)
                else:
                    raise ValueError(f"Unknown Layer: {subarch}")
                layers.append(subarch)
            block = Block1d(*layers)

        else:
            if arch in NAMES:
                layer = NAMES[arch]
            elif hasattr(nn, arch):
                layer = getattr(nn, arch)
            else:
                raise ValueError(f"Unknown Layer: {arch}")


def wrap(module):
    return tWrapper(module)


def get_builder(layer, params):
    pass


########################################################################################################################
# Pre Built
########################################################################################################################


GQA = GroupedQueryAttention
GQANOO = GroupedQueryAttentionNOO
GQANOV = GroupedQueryAttentionNOV
Attention = MultiheadAttentionMixer
AttentionNOV = MultiheadAttentionMixerNOV
NOVLa2 = MultiheadAttentionMixerNOVLa2

if __name__ == "__main__":
    d_models = [64 for i in range(4)]
    m_dims = [128, 64, 32, 16]
    attn = Builder(GQA, n_heads=4, n_kv_heads=2)
    model = ModdedFormer(
        vocab_size=100,
        embed_dim=64,
        d_models=d_models,
        spatials=[attn for _ in range(len(d_models))],
        M_dims=m_dims,
        input_projections=3,
        zeroskip=True
    )
    model.summary()

    out = model(
        torch.randint(0, 99, (2, 128))
    )
    print(out.shape)

    model.profile()
