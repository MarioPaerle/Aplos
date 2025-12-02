import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
import math
from Vathos.Utils import *
from typing import Tuple, Optional, Union, List
import re
from Vathos.complexity import combine_big_o
from timeit import default_timer as timer

ACTIVS = {
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'elu': nn.ELU,
    'lrelu': nn.LeakyReLU,
    'leaky_relu': nn.LeakyReLU,
}
PROFILE = True
PROFILE_BATCHED = True


class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()
        self.complexity = "O(1)"
        self.__name__ = "BasicLayer"
        self._timer_unbatched = not PROFILE_BATCHED
        self._tstart = 0
        self._tend = 0
        self._time = 0
        self._times = []
        self._sublayers = None

    def start_timer(self):
        self._tstart = timer()

    def end_timer(self, batch_size):
        self._tend = timer()
        self._time = (self._tend - self._tstart) / batch_size
        self._times.append(self._time)

    def get_mean_execution_time(self):
        mean = np.mean(self._times) if len(self._times) > 0 else None
        return f"{mean:.4f}s" if mean is not None else "no time recorded"

    def get_last_execution_time(self):
        return self._time

    def register_sublayers(self):
        if self._sublayers is None:
            self._sublayers = dict()

        def get_unique_name(base_name, existing_names):
            if base_name not in existing_names:
                return base_name
            counter = 1
            while f"{base_name}_{counter}" in existing_names:
                counter += 1
            return f"{base_name}_{counter}"

        def collect_layers(module, prefix="", level=0):
            layers = []
            for name, child in module.named_children():
                if isinstance(child, Layer):
                    layers.append((name, child, level))
                    layers.extend(collect_layers(child, prefix=f"{name}.", level=level + 1))
                elif isinstance(child, nn.ModuleList):
                    for i, item in enumerate(child):
                        if isinstance(item, Layer):
                            layers.append((f"{name}[{i}]", item, level))
                            layers.extend(collect_layers(item, prefix=f"{name}[{i}].", level=level + 1))
                elif isinstance(child, nn.Module):
                    layers.extend(collect_layers(child, prefix=f"{name}.", level=level))
            return layers

        all_layers = collect_layers(self)

        for original_name, layer, level in all_layers:
            class_name = type(layer).__name__
            unique_name = get_unique_name(class_name, self._sublayers.keys())
            self._sublayers[unique_name] = {'layer': layer, 'level': level}

    def __call__(self, *args, **kwargs):
        if self._sublayers is None:
            self.register_sublayers()
        if kwargs.get("profile") or PROFILE:
            if kwargs.get("profile"):
                del kwargs["profile"]
            self.start_timer()
            rets = self.forward(*args, **kwargs)
            if not self._timer_unbatched:
                self.end_timer(batch_size=args[0].shape[0])
            else:
                self.end_timer(1)
            return rets
        else:
            return self.forward(*args, **kwargs)

    def profile(self):
        """a Basic Layer level profiler operation"""
        batched = not self._timer_unbatched
        print(
            f"Layer {NUM}{type(self).__name__}{RES} Times Profile (batched: {GOOD if batched else BAD}{batched}{RES}):")

        for sublayer_name, sublayer_info in self._sublayers.items():
            layer = sublayer_info['layer']
            level = sublayer_info['level']
            indent = "    " * level  # 4 spaces per level
            print(f"{indent}- {NUM}{sublayer_name}{RES}: {layer.get_mean_execution_time()}")

    @staticmethod
    def register_exectution_time(fn, *args, **kwargs):
        start_time = timer()
        rets = fn(*args, **kwargs)
        elapsed = timer() - start_time
        return elapsed, rets


class tWrapper(Layer):
    __name__ = "tWrapper"

    def __init__(self, module: nn.Module):
        super(tWrapper, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        self.module(*args, **kwargs)


class Identity(Layer):
    __name__ = "Identity"
    __complexity__ = "O(1)"

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ConvResBlock(Layer):
    __complexity__ = "O(L k^2 in out)"

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=nn.ReLU):
        super(ConvResBlock, self).__init__()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same')
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same')
        self.convout = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.activation = activation()

    def forward(self, x):
        res = x
        x = self.bn(x)
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        x = x + res
        return self.convout(x)


class UnbiasedLinear(Layer):
    __name__ = "UnbiasedLinear"
    __complexity__ = "O(L d^2)"

    def __init__(self, input_features, output_features):
        super(UnbiasedLinear, self).__init__()
        self.linear = nn.Linear(input_features, output_features, bias=False)

    def forward(self, x):
        return self.linear(x)


class SwiGLU(Layer):
    gated = True
    __name__ = "SwiGLU"
    __complexity__ = "O(L)"

    def forward(self, x: torch.Tensor):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


class MLP(Layer):
    __name__ = "MLP"
    __complexity__ = "O(depth L d)"

    def __init__(self, d_model: int, depth: int, expand: int, activation: Callable):
        super().__init__()
        hidden_dim = d_model * expand
        self.d_model = d_model
        self.depth = depth
        self.expand = expand
        self.activation = activation

        layers = []
        for i in range(depth):
            if i == 0:
                in_dim = d_model
                out_dim = hidden_dim
            elif i == depth - 1:
                in_dim = hidden_dim
                out_dim = d_model
            else:
                in_dim = hidden_dim
                out_dim = hidden_dim

            if hasattr(activation, 'gated') and i < depth - 1:
                out_dim = out_dim * 2

            layers.append(nn.Linear(in_dim, out_dim, bias=True))
            if i < depth - 1:
                layers.append(activation())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class ResMLPBlock(Layer):
    def __init__(self, d_model, expand=2, norm=True, activation: Callable = nn.GELU):
        super().__init__()
        self.activation1 = activation()
        self.activation2 = activation()
        self.norm = norm
        self.d_model = d_model
        self.expand = expand
        self.l1 = nn.Linear(d_model, d_model * expand, bias=True)
        self.l2 = nn.Linear(d_model * expand, d_model, bias=True)

        self.g1 = nn.Linear(d_model, d_model * expand, bias=True)
        self.g2 = nn.Linear(d_model * expand, d_model, bias=True)

        self.norm = nn.LayerNorm(d_model) if self.norm else nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.l2(self.activation1(self.l1(x)))
        x = self.norm(x) + x
        x = self.g2(self.activation2(self.g1(x)))
        return x


class ResMLP(Layer):
    def __init__(self, d_model: int, depth: int, expand: int, activation: Callable):
        super().__init__()
        self.d_model = d_model
        self.depth = depth
        self.expand = expand
        self.activation = activation

        layers = []
        for i in range(depth):
            layers.append(ResMLPBlock(d_model, expand=expand, activation=activation))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class Block1d(Layer):
    def __init__(self, channel_mixer: Layer, spatial_mixer: Layer):
        super().__init__()
        self.spatial_mixer = spatial_mixer
        self.channel_mixer = channel_mixer
        self.norm1 = nn.LayerNorm(spatial_mixer.d_model)
        self.norm2 = nn.LayerNorm(spatial_mixer.d_model)

    def forward(self, x: torch.Tensor):
        x = x + self.spatial_mixer(self.norm1(x))
        x = x + self.channel_mixer(self.norm2(x))
        return x


class BlockStack(Layer):
    def __init__(self, blocks: Tuple[Block1d | Layer]):
        super().__init__()
        self.blocks = blocks
        self.stack = nn.ModuleList(blocks)

    def forward(self, x: torch):
        for block in self.stack:
            x = block(x)
        return x


class CausalConv1d(Layer):
    __name__ = "CausalConv1d"
    __complexity__ = "O(L d k)"

    def __init__(self, d, k=3):
        super().__init__()
        self.d = d
        self.k = k
        self.pad = k - 1

        self.K = nn.Parameter(torch.randn(d, k) / (k ** 0.5))

    def forward(self, x):
        b, L, d = x.shape
        x = x.transpose(1, 2)
        x_pad = F.pad(x, (self.pad, 0))
        out = F.conv1d(x_pad, self.K.unsqueeze(1), groups=d)
        return out.transpose(1, 2)


class LSTM(Layer):
    __name__ = "LSTM"
    __complexity__ = "O(L d)"

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
    __complexity__ = "O(L d)"

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
        self.W = nn.Parameter(torch.randn(max_len, max_len) / max_len)

    def forward(self, x):
        x = x / math.sqrt(self.d_model)
        if self.causal:
            return torch.tril(self.W[:x.shape[1], :x.shape[1]]) @ x
        else:
            self.W[:x.shape[1], :x.shape[1]] @ x


class MLPMixer(Layer):
    __name__ = "Linear Mixer"
    __complexity__ = "O(L^2 d)"

    def __init__(self, d_model, max_len=1000, L_expand=1, causal=False, activation=nn.GELU):
        super().__init__()
        if causal:
            if L_expand < 1:
                raise ValueError("For the MLPMixer to be causal, L_expand must be greater than 1")
            if L_expand > 1:
                raise NotImplementedError("Causal MLP Mixer with L_expand is currently not implemented") # TODO: Causal Lex
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
        self.conv_1 = CausalConv1d(d_model, k=k1)
        self.conv_2 = CausalConv1d(d_model, k=k2)
        self.activation = activation()

    def forward(self, x):
        g1 = self.conv_1(x)
        g2 = self.activation(self.conv_2(x))
        return self.mixer(g1) * g2 + (1 - g2) * x


########################################################################################################################
#   TRANSFORMERS
########################################################################################################################

# TODO: Needs fixes!
class RoPE(Layer):
    __name__ = "RoPE"

    def __init__(self, dim: int, max_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_len = max_len

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0

    def _update_cache(self, seq_len: int, dtype: torch.dtype, device: torch.device):
        if seq_len > self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq)

            emb = torch.cat([freqs, freqs], dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()

    def _apply_rotary_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-3] if x.ndim == 4 else x.shape[-2]

        cos = cos[:seq_len].unsqueeze(0).unsqueeze(-2 if x.ndim == 4 else 0)  # [1, L, 1, D] or [1, L, D]
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(-2 if x.ndim == 4 else 0)

        x1, x2 = x.chunk(2, dim=-1)  # each: [..., D//2]

        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor = None):
        assert q.shape[-1] == self.dim, f"Last dim of q must be {self.dim}, got {q.shape[-1]}"
        if k is not None:
            assert k.shape == q.shape, "k must have same shape as q"

        self._update_cache(q.shape[-3] if q.ndim == 4 else q.shape[-2], q.dtype, q.device)

        cos = self._cos_cached
        sin = self._sin_cached

        q_rope = self._apply_rotary_emb(q, cos, sin)
        k_rope = self._apply_rotary_emb(k, cos, sin) if k is not None else None

        return (q_rope, k_rope) if k_rope is not None else q_rope


class SinusoidalPositionalEncoding(Layer):
    __name__ = "SinusoidalPositionalEncoding"
    __complexity__ = "O(L^2 d^2)"

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        return x + self.pe[:L]


class MultiheadAttentionMixer(Layer):
    __name__ = "MultiheadAttentionMixer"
    __complexity__ = "O(L^2 d^2)"

    def __init__(self, d_model: int, n_heads: int, causal: bool, rope=False):
        super().__init__()
        self.causal = causal
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        if rope:
            self.rope = RoPE(self.head_dim)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        if self.rope is not None:
            q, k = self.rope(q, k)

        attn_mask = None
        """if self.causal:
            attn_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)"""

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        attn = attn.transpose(1, 2).reshape(B, L, D)

        return self.out(attn)


class CausalMultiheadAttentionMixer(Layer):
    __name__ = "CausalMultiheadAttentionMixer"

    def __init__(self, d_model: int, n_heads: int, causal=True):
        super().__init__()
        assert causal, \
            ("CausalMultiheadAttentionMixLayer only supports causal=True, "
             "if you meant to create a non Causal Attention use the MultiheadAttentionMixer")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.rope = RoPE(self.head_dim)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        q, k = self.rope(q, k)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(B, L, D)

        return self.out(attn)


class MTransformer(Layer):
    def __init__(self, d_model: int, n_layers: int, n_heads: int = 8, mlp_expand: int = 4, causal: bool = True):
        super().__init__()
        self.d_model = d_model

        self.blocks = nn.ModuleList([
            Block1d(
                channel_mixer=MLP(d_model, depth=2, expand=mlp_expand, activation=SwiGLU),
                spatial_mixer=CausalMultiheadAttentionMixer(d_model, n_heads, causal=causal)
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class _MTemplate(Layer):
    def __init__(self, d_model: int, n_layers: int, n_heads: int = 8, mlp_expand: int = 4, causal: bool = True,
                 channel_mixer=MLP, spatial_mixer=MultiheadAttentionMixer):
        super().__init__()
        self.d_model = d_model

        self.blocks = nn.ModuleList([
            Block1d(
                channel_mixer=channel_mixer(d_model, depth=2, expand=mlp_expand, activation=nn.GELU),
                spatial_mixer=spatial_mixer(d_model, n_heads, causal=causal)
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


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
        assert x.dtype == torch.long and x.min() >= 0 and x.max() <= self.vocab_size, ("either dtype is not long, or x "
                                                                                       "is not in [0, vocab_size]")
        return self.embedding(x)


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
                spatial_mixer=self.sec_mixer(d_model=d_model, **self.attn_params)
            )
            for _ in range(self.n_sec)
        ])

    def forward(self, x):
        for sec in self.sec_blocks:
            x = sec(x)
        for attn in self.attn_mixers:
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


class VathosModel(Layer):
    __name__ = "VathosModel"

    def __init__(self):
        super().__init__()

    def forward(self):
        pass

    def summary(self):
        pass

    def computecomplexity(self):
        pass


class SequenceModel(Layer):
    __name__ = "SequenceModel"

    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 max_len=1024,
                 pos_encoder: bool | None | Layer | nn.Module = None,
                 embedder=Embedder,
                 embedder_args: dict = None,
                 unembedder=UnbiasedLinear,
                 channel_mixer=MLP,
                 spatial_mixer: Layer | nn.Module = MultiheadAttentionMixer,
                 channel_args: dict = None,
                 spatial_args: dict = None,
                 rope=False,
                 name=''
                 ):
        super().__init__()
        if rope and spatial_mixer is not MultiheadAttentionMixer:
            flag("rope=True works only with MultiheadAttentionMixer, which you seem not to be using right?")
        if channel_args is None and channel_mixer is MLP:
            channel_args = {"expand": 2, "activation": nn.GELU, "depth": 2}
        if spatial_args is None and spatial_mixer is MultiheadAttentionMixer:
            spatial_args = {"causal": True, "n_heads": 8, "rope": rope}
        if spatial_args is None:
            spatial_args = {}
        if channel_args is None:
            channel_args = {}
        if embedder_args is None:
            embedder_args = {}

        self.name = name
        self.spatial_mixer = spatial_mixer
        self.channel_mixer = channel_mixer

        self.spatial_args = spatial_args
        self.channel_args = channel_args
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.n_layers = n_layers

        self.embedder = embedder(vocab_size=vocab_size, d_model=d_model, **embedder_args)

        self.pos_encoder = pos_encoder(d_model, max_len=max_len) if pos_encoder not in (True, False, None) else \
            (SinusoidalPositionalEncoding(d_model, max_len=max_len) if pos_encoder is True else nn.Identity())

        self.blocks = nn.ModuleList([
            Block1d(
                channel_mixer=channel_mixer(d_model=d_model, **channel_args),
                spatial_mixer=spatial_mixer(d_model=d_model, **spatial_args)
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.unembedder = unembedder(d_model, vocab_size)

        self.embedder_complexity = embedder.__complexity__ if hasattr(embedder, "__complexity__") else "O(L d)"
        self.unembedder_complexity = unembedder.__complexity__ if hasattr(unembedder, "__complexity__") else "O(L d)"
        self.spatial_complextiy = spatial_mixer.__complexity__ if hasattr(spatial_mixer, "__complexity__") else "O(L d)"
        self.channel_complexity = channel_mixer.__complexity__ if hasattr(channel_mixer, "__complexity__") else "O(L d)"

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
        x = self.embedder(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        if unembed:
            x = self.unembedder(x)
        return x

    @torch.no_grad()
    def generate(self, prompt, max_length=512, temperature=0.8, top_p=0.9,
                 device='cpu'):
        self.eval()
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)

        generated = prompt.clone().to(device)

        for _ in range(max_length):
            logits_uncond = self.forward(generated)

            logits = logits_uncond
            next_token_logits = logits[:, -1, :] / temperature

            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == 0:
                break

        return generated

    def summary(self):
        complexities = [self.channel_complexity, self.spatial_complextiy, self.embedder_complexity,
                        self.unembedder_complexity]
        print(f'{NUM}VATHOS{RES} Model Summary:')
        print(f"{NUM}Symbolic1dSeq2SeqModel{RES}(d_model={NUM}{self.d_model}{RES}, n_layer={NUM}{self.n_layers}{RES})")
        print(f"\t - {NUM}Embedder{RES}: {getname(self.embedder)} - {NUM}{self.embedder_complexity}{RES}")
        print(f"\t - {NUM}Unembedder{RES}: {getname(self.unembedder)} - {NUM}{self.unembedder_complexity}{RES}")
        print(
            f"\t - {NUM}Spatial Mixer{RES}: {getname(self.spatial_mixer)}({self.spatial_args}) - {NUM}{self.spatial_complextiy}{RES}")
        print(
            f"\t - {NUM}Channel Mixer{RES}: {getname(self.channel_mixer)}({self.channel_args}) - {NUM}{self.channel_complexity}{RES}")
        print(f"Num Parameters: {NUM}{sum([p.numel() for p in self.parameters()]):_}{RES}")
        print(f"Num Trainable Parameters: {NUM}{sum([p.numel() for p in self.parameters() if p.requires_grad]):_}{RES}")
        print(f"Total Complexity: {NUM}{combine_big_o(complexities)}{RES}")


#######################################################################################################################

def test_causality(module=MTransformer(8, 4, 2)):
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
    print(f"Bounds:  min:{x.min()}, max:{x.max()}, mean:{x.mean()}, std:{x.std()}, sum_of_a_vector: {x[0, 0, :].sum()}")


#######################################################################################################ù

NAMES = {
    "MLP": MLP,
    "Attention": MultiheadAttentionMixer,
    "MHA": MultiheadAttentionMixer,
    "CMHA": CausalMultiheadAttentionMixer,
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


########################################################################################################################
# Pre Built
########################################################################################################################


if __name__ == "__main__":
    """pe = PatchEmbedder(img_size=224, patch_size=16, embed_dim=768, cls=True)
    unem = ClsHead(768, 10)"""
    x = torch.randint(0, 9, size=(2, 64))
    """y = pe(img)
    print(y.shape)
    print(unem(y).shape)"""
    # ViT = Symbolic1dSeq2SeqModel(10, 16, 4, 200, pos_encoder=True,
    #                                                embedder=PatchEmbedder, unembedder=ClsHead,
    #                                embedder_args={'img_size': 32, 'patch_size': 4})
    spatial_mixer = HybridAttentionBlock1d
    attn_params = {
        'n_heads': 8,
        'causal': True
    }
    sec_mixer = CausalConv1d
    sec_params = {
        'k': 3
    }
    spatial_params = {
        'sec_mixer': sec_mixer,
        'sec_params': sec_params,
        'attn_params': attn_params
    }
    model = SequenceModel(10, 256, 6, 256, pos_encoder=True,
                          embedder=Embedder, unembedder=UnbiasedLinear, rope=False,
                          spatial_mixer=LinearMixer,
                          spatial_args={'max_len': 256})
    model.summary()
    out = model(x)
    model.profile()
    """"EMBED -> (Attention, MLP)x4 -> UNEMBED"
    "EMBED -> (Attention, MLP)x4 -> UNEMBED"""
