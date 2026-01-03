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


class Block1d(Layer):
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


########################################################################################################################
#   TRANSFORMERS
########################################################################################################################

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

    def _apply_rotary_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                          start_pos: int = 0) -> torch.Tensor:
        """Apply rotary embeddings starting from start_pos"""
        seq_len = x.shape[-3] if x.ndim == 4 else x.shape[-2]

        cos = cos[start_pos:start_pos + seq_len]
        sin = sin[start_pos:start_pos + seq_len]

        cos = cos.unsqueeze(0).unsqueeze(-2 if x.ndim == 4 else 0)
        sin = sin.unsqueeze(0).unsqueeze(-2 if x.ndim == 4 else 0)

        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor = None, start_pos: int = 0):
        """
        Args:
            q: Query tensor
            k: Key tensor (optional)
            start_pos: Starting position for RoPE (used during generation)
        """
        assert q.shape[-1] == self.dim, f"Last dim of q must be {self.dim}, got {q.shape[-1]}"
        if k is not None:
            assert k.shape == q.shape, "k must have same shape as q"

        seq_len = q.shape[-3] if q.ndim == 4 else q.shape[-2]
        self._update_cache(start_pos + seq_len, q.dtype, q.device)

        cos = self._cos_cached
        sin = self._sin_cached

        q_rope = self._apply_rotary_emb(q, cos, sin, start_pos)
        k_rope = self._apply_rotary_emb(k, cos, sin, start_pos) if k is not None else None

        return (q_rope, k_rope) if k_rope is not None else q_rope


class MultiheadAttentionMixer(Layer):
    __name__ = "MultiheadAttentionMixer"
    __complexity__ = "O(L^2 d +  L d^2)"

    def __init__(self, d_model: int, n_heads: int, causal: bool, rope=False, dropout=0.05):
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

        self.dropout = nn.Dropout(dropout)

        self.kv_cache = None

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        if self.rope is not None:
            q, k = self.rope(q, k, start_pos=0)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        attn = attn.transpose(1, 2).reshape(B, L, D)
        attn = self.dropout(attn)
        return self.out(attn)

    def generate(self, x: torch.Tensor):
        B, L, D = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        # 1. Handle Cache Offset
        if self.kv_cache is not None:
            k_cache, v_cache = self.kv_cache
            pos_offset = k_cache.shape[2]
        else:
            pos_offset = 0
            k_cache, v_cache = None, None

        # 2. Apply RoPE (Rotates the *new* q and k relative to history)
        if self.rope is not None:
            q, k = self.rope(q, k, start_pos=pos_offset)

        # 3. Update Cache
        if k_cache is not None:
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        # Save updated cache
        self.kv_cache = (k, v)

        # 4. Attention with Dynamic Masking
        # If L > 1, we are processing a prompt (Prefill), so we MUST be causal.
        # If L == 1, we are generating a token step-by-step, attending to the whole history.
        use_causal = self.causal and (L > 1)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=use_causal)

        attn = attn.transpose(1, 2).reshape(B, L, D)
        attn = self.dropout(attn)
        return self.out(attn)

    def clear_cache(self):
        self.kv_cache = None

    def finetune(self):
        self.qkv.weight.data.requires_grad = False
        self.out.weight.data.requires_grad = False


class MultiheadAttentionMixerNOV(Layer):
    __name__ = "MultiheadAttentionMixer"
    __complexity__ = "O(L^2 d +  L d^2)"

    def __init__(self, d_model: int, n_heads: int, causal: bool, rope=False, dropout=0.05):
        super().__init__()
        self.causal = causal
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qk = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        if rope:
            self.rope = RoPE(self.head_dim)
        else:
            self.rope = None

        self.dropout = nn.Dropout(dropout)

        self.kv_cache = None

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape

        qk = self.qk(x).reshape(B, L, 2, self.n_heads, self.head_dim)
        q, k = qk.permute(2, 0, 3, 1, 4)
        v = x.view(B, L, 1, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)[0]

        if self.rope is not None:
            q, k = self.rope(q, k, start_pos=0)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        attn = attn.transpose(1, 2).reshape(B, L, D)
        attn = self.dropout(attn)
        return self.out(attn)

    def generate(self, x: torch.Tensor):
        B, L, D = x.shape

        qk = self.qk(x).reshape(B, L, 2, self.n_heads, self.head_dim)
        q, k = qk.permute(2, 0, 3, 1, 4)
        v = x.view(B, L, 1, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)[0]

        if self.kv_cache is not None:
            k_cache, v_cache = self.kv_cache
            pos_offset = k_cache.shape[2]
        else:
            pos_offset = 0
            k_cache, v_cache = None, None

        if self.rope is not None:
            q, k = self.rope(q, k, start_pos=pos_offset)

        if k_cache is not None:
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        self.kv_cache = (k, v)
        use_causal = self.causal and (L > 1)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=use_causal)

        attn = attn.transpose(1, 2).reshape(B, L, D)
        attn = self.dropout(attn)
        return self.out(attn)

    def clear_cache(self):
        self.kv_cache = None

    def finetune(self):
        self.qk.weight.data.requires_grad = False
        self.out.weight.data.requires_grad = False


class FFMultiheadAttentionMixerNOV(Layer):
    __name__ = "MultiheadAttentionMixer"
    __complexity__ = "O(L^2 d +  L d^2)"

    def __init__(self, d_model: int, n_heads: int, causal: bool, rope=False, dropout=0.05):
        super().__init__()
        self.causal = causal
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        if rope:
            self.rope = RoPE(self.head_dim)
        else:
            self.rope = None

        self.dropout = nn.Dropout(dropout)

        self.kv_cache = None

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape

        q = self.q(x).reshape(B, L, self.n_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        k = x.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = x.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.rope is not None:
            q, k = self.rope(q, k, start_pos=0)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        attn = attn.transpose(1, 2).reshape(B, L, D)
        attn = self.dropout(attn)
        return self.out(attn)

    def generate(self, x: torch.Tensor):
        B, L, D = x.shape

        qk = self.qk(x).reshape(B, L, 2, self.n_heads, self.head_dim)
        q, k = qk.permute(2, 0, 3, 1, 4)
        v = x.view(B, L, 1, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)[0]

        if self.kv_cache is not None:
            k_cache, v_cache = self.kv_cache
            pos_offset = k_cache.shape[2]
        else:
            pos_offset = 0
            k_cache, v_cache = None, None

        if self.rope is not None:
            q, k = self.rope(q, k, start_pos=pos_offset)

        if k_cache is not None:
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        self.kv_cache = (k, v)
        use_causal = self.causal and (L > 1)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=use_causal)

        attn = attn.transpose(1, 2).reshape(B, L, D)
        attn = self.dropout(attn)
        return self.out(attn)

    def clear_cache(self):
        self.kv_cache = None

    def finetune(self):
        self.qk.weight.data.requires_grad = False
        self.out.weight.data.requires_grad = False


class FFFMultiheadAttentionMixer(Layer):
    __name__ = "MultiheadAttentionMixer"
    __complexity__ = "O(L^2 d +  L d^2)"

    def __init__(self, d_model: int, n_heads: int, causal: bool, rope=False, dropout=0.05):
        super().__init__()
        self.causal = causal
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q = LowRankLinear(d_model, d_model, rank=math.isqrt(d_model), bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        if rope:
            self.rope = RoPE(self.head_dim)
        else:
            self.rope = None

        self.dropout = nn.Dropout(dropout)

        self.kv_cache = None

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape

        q = self.q(x).reshape(B, L, self.n_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        k = x.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = x.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.rope is not None:
            q, k = self.rope(q, k, start_pos=0)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        attn = attn.transpose(1, 2).reshape(B, L, D)
        attn = self.dropout(attn)
        return self.out(attn)

    def generate(self, x: torch.Tensor):
        B, L, D = x.shape

        qk = self.qk(x).reshape(B, L, 2, self.n_heads, self.head_dim)
        q, k = qk.permute(2, 0, 3, 1, 4)
        v = x.view(B, L, 1, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)[0]

        if self.kv_cache is not None:
            k_cache, v_cache = self.kv_cache
            pos_offset = k_cache.shape[2]
        else:
            pos_offset = 0
            k_cache, v_cache = None, None

        if self.rope is not None:
            q, k = self.rope(q, k, start_pos=pos_offset)

        if k_cache is not None:
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        self.kv_cache = (k, v)
        use_causal = self.causal and (L > 1)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=use_causal)

        attn = attn.transpose(1, 2).reshape(B, L, D)
        attn = self.dropout(attn)
        return self.out(attn)

    def clear_cache(self):
        self.kv_cache = None

    def finetune(self):
        self.qk.weight.data.requires_grad = False
        self.out.weight.data.requires_grad = False


class GroupedQueryAttention(Layer):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            n_kv_heads: int,
            dropout: float = 0.0,
            bias: bool = False,
            causal: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = n_heads
        self.num_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.dropout_prob = dropout
        self.causal = causal

        self.n_rep = self.num_heads // self.num_kv_heads

        if self.d_model % self.num_heads != 0:
            raise ValueError(f"embed_dim ({d_model}) must be divisible by num_heads ({n_heads})")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads ({n_heads}) must be divisible by num_kv_heads ({n_kv_heads})")

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)

        self.kv_proj = nn.Linear(d_model, n_kv_heads * self.head_dim * 2, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
            nn.init.constant_(self.kv_proj.bias, 0)
            nn.init.constant_(self.o_proj.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x)
        kv = self.kv_proj(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        kv = kv.view(B, T, self.num_kv_heads, 2, self.head_dim)
        k, v = kv.unbind(dim=3)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.n_rep > 1:
            k = k[:, :, None, :, :].expand(B, self.num_kv_heads, self.n_rep, T, self.head_dim)
            v = v[:, :, None, :, :].expand(B, self.num_kv_heads, self.n_rep, T, self.head_dim)
            k = k.reshape(B, self.num_heads, T, self.head_dim)
            v = v.reshape(B, self.num_heads, T, self.head_dim)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=self.causal
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(attn_output)


class GroupedQueryAttentionNOV(Layer):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            n_kv_heads: int,
            dropout: float = 0.0,
            bias: bool = False,
            causal: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = n_heads
        self.num_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.dropout_prob = dropout
        self.causal = causal

        self.n_rep = self.num_heads // self.num_kv_heads

        if self.d_model % self.num_heads != 0:
            raise ValueError(f"embed_dim ({d_model}) must be divisible by num_heads ({n_heads})")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads ({n_heads}) must be divisible by num_kv_heads ({n_kv_heads})")

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
            nn.init.constant_(self.k_proj.bias, 0)
            nn.init.constant_(self.o_proj.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if self.n_rep > 1:
            k = k[:, :, None, :, :].expand(B, self.num_kv_heads, self.n_rep, T, self.head_dim)
            k = k.reshape(B, self.num_heads, T, self.head_dim)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=self.causal
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(attn_output)


class MultiheadLatentAttentionMixer(Layer):  # Changed Layer to nn.Module for standard torch
    __name__ = "MultiheadLatentAttentionMixer"
    # Adjusted complexity notation
    __complexity__ = "O(L^2 d + L d * d_kv_lora)"

    def __init__(self, d_model: int, n_heads: int, d_kv_lora: int, causal: bool, rope=False, dropout=0.1):
        super().__init__()
        self.causal = causal
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_kv_lora = d_kv_lora
        self.rope_enabled = rope

        self.q = nn.Linear(d_model, d_model, bias=False)

        self.kv_compress = nn.Linear(d_model, d_kv_lora, bias=False)
        self.k_content_decompress = nn.Linear(d_kv_lora, d_model, bias=False)
        self.v_decompress = nn.Linear(d_kv_lora, d_model, bias=False)

        self.out = nn.Linear(d_model, d_model, bias=False)

        if self.rope_enabled:
            self.rope = RoPE(self.head_dim)

        self.dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.out.weight)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        H = self.n_heads
        HD = self.head_dim

        q = self.q(x).view(B, L, H, HD).transpose(1, 2)

        c_kv = self.kv_compress(x)
        k = self.k_content_decompress(c_kv).view(B, L, H, HD).transpose(1, 2)
        v = self.v_decompress(c_kv).view(B, L, H, HD).transpose(1, 2)
        if self.rope_enabled:
            q, k = self.rope(q, k)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=self.causal
        )

        attn = attn.transpose(1, 2).contiguous().view(B, L, D)

        return self.out(attn)


class MultiheadDecoupledAttention(Layer):
    __name__ = "DecoupledSelfAttention"
    __complexity__ = "O(L^2 * d_qk + L * d_model * (d_qk + d_v))"

    def __init__(self, d_model: int, n_heads: int, qk_dim: int, causal: bool, rope=False, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.qk_dim = qk_dim
        self.v_dim = d_model // n_heads
        self.causal = causal
        self.rope_enabled = rope

        self.qk_proj = nn.Linear(d_model, n_heads * 2 * qk_dim, bias=False)

        self.v_proj = nn.Linear(d_model, n_heads * self.v_dim, bias=False)

        self.out_proj = nn.Linear(n_heads * self.v_dim, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        if self.rope_enabled:
            self.rope = RoPE(qk_dim)

        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor):
        B, L, _ = x.shape
        H = self.n_heads

        q, k = self.qk_proj(x).view(B, L, H, 2 * self.qk_dim).transpose(1, 2).chunk(2, -1)

        v = self.v_proj(x).view(B, L, H, self.v_dim).transpose(1, 2)

        if self.rope_enabled:
            q, k = self.rope(q, k)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=self.causal
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, H * self.v_dim)

        return self.out_proj(attn_out)


class MultiheadDecoupledAttentionNOV(Layer):
    __name__ = "DecoupledSelfAttention"
    __complexity__ = "O(L^2 * d_qk + L * d_model * (d_qk + d_v))"

    def __init__(self, d_model: int, n_heads: int, qk_dim: int, causal: bool, rope=False, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.qk_dim = qk_dim
        self.v_dim = d_model // n_heads
        self.causal = causal
        self.rope_enabled = rope

        self.qk_proj = nn.Linear(d_model, n_heads * 2 * qk_dim, bias=False)

        self.out_proj = nn.Linear(n_heads * self.v_dim, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        if self.rope_enabled:
            self.rope = RoPE(qk_dim)

        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor):
        B, L, _ = x.shape
        H = self.n_heads

        q, k = self.qk_proj(x).view(B, L, H, 2 * self.qk_dim).transpose(1, 2).chunk(2, -1)

        v = x.view(B, L, H, self.v_dim).transpose(1, 2)

        if self.rope_enabled:
            q, k = self.rope(q, k)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=self.causal
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, H * self.v_dim)

        return self.out_proj(attn_out)


class CausalMultiheadAttentionMixer(Layer):
    __name__ = "CausalMultiheadAttentionMixer"
    __complexity__ = "O(L^2 d +  L d^2)"

    def __init__(self, d_model: int, n_heads: int, causal=True, rope=False, dropout=0.1):
        super().__init__()
        assert causal, \
            ("CausalMultiheadAttentionMixLayer only supports causal=True, "
             "if you meant to create a non Causal Attention use the MultiheadAttentionMixer")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        if rope:
            self.rope = RoPE(self.head_dim)
        else:
            self.rope = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        if self.rope is not None:
            q, k = self.rope(q, k)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        attn = attn.transpose(1, 2).reshape(B, L, D)
        attn = self.dropout(attn)

        return self.out(attn)


class MTransformer(Layer):
    def __init__(self, d_model: int, n_layers: int, n_heads: int = 8, mlp_expand: int = 4, causal: bool = True):
        super().__init__()
        self.d_model = d_model

        self.blocks = nn.ModuleList([
            Block1d(
                d_model=d_model,
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

        # LOSSES and METRICS
        self._losses = []
        self._losses_dict = {}
        self._losses_per_epoch = []
        self._losses_per_epoch_dict = {}
        self._losses_this_epoch = []
        self._metrics = dict()
        self._metrics_this_epoch = dict()
        self._metrics_per_epoch = dict()
        self.autosave = True
        self.autosave_overwrite = 0
        self.best_loss = float('inf')
        self.checkpoints = 0
        self.steps = 0
        self.steps_per_epoch = 0
        self.epochs = 0
        self.finetuning = False

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
        """Plot all losses and metrics in subplots"""
        # Count how many plots we need: 1 for losses + number of metrics
        n_metrics = len(self._metrics_per_epoch)
        n_plots = 1 + n_metrics

        # Calculate grid dimensions
        n_cols = 2
        n_rows = (n_plots + 1) // 2  # Ceiling division

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        # Flatten axes array for easier indexing
        if n_plots == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()

        # Plot losses in first subplot
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

        # Plot each metric in subsequent subplots
        for idx, (metric_name, metric_values) in enumerate(self._metrics_per_epoch.items(), start=1):
            ax = axes[idx]

            # Plot per-step values if available
            if metric_name in self._metrics and len(self._metrics[metric_name]) > 0:
                # Create step indices for metrics (assuming they align with training steps)
                step_indices = list(range(len(self._metrics[metric_name])))
                ax.plot(step_indices, self._metrics[metric_name],
                        label=f"{metric_name}", linewidth=1, alpha=0.6)

            # Plot per-epoch values
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
                 rope=False,
                 name='',
                 pad='none',
                 baseblock=Block1d,
                 baseblock_args=None,
                 dropout=0.1,
                 weight_tying=False,
                 norm=nn.LayerNorm,
                 d_modifiers: List | None = None
                 ):
        super().__init__()
        self.pad = pad
        if rope and spatial_mixer not in (
        MultiheadAttentionMixer, MultiheadAttentionMixerNOV, CausalMultiheadAttentionMixer, GroupedQueryAttention):
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
        if unembedder_args is None:
            unembedder_args = {'input_features': d_model, 'output_features': vocab_size}
        if baseblock_args is None:
            baseblock_args = {}
        if d_modifiers is None:
            d_modifiers = [1 for _ in range(d_model)]

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
        B, L = x.size(0), x.size(1)
        x = self.embedder(x) * self.embed_scale
        x = self.pos_encoder(x)

        if self.pad == 'sqrt':
            n = int((x.shape[1] ** 0.5) + 0.999999)
            x = F.pad(x, (0, 0, 0, n ** 2 - L), mode="constant", value=0)
        else:
            pass

        for block in self.blocks:
            x = block(x)

        if unembed:
            x = self.norm(self.unembedder(x))

        return x[:, :L, :]

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


########################################################################################################################

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


def get_builder(layer, params):
    pass


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
    sec_mixer = DepthwiseCausalConv1d
    sec_params = {
        'k': 3
    }
    spatial_params = {
        'sec_mixer': sec_mixer,
        'sec_params': sec_params,
        'attn_params': attn_params
    }
    model = SequenceModel(
        vocab_size=100,
        d_model=128,
        n_layers=6,
        max_len=2116,
        pos_encoder=True,
        embedder=EasyEmbedder,
        unembedder=MultiHeadUnembedder,
        unembedder_args={'d_model': 128, 'vocab_size': 100, 'k': 4},
        channel_mixer=F_UDLPSwiGLU,
        channel_args={'expand': 2},
        rope=True,
        spatial_mixer=GroupedQueryAttentionNOV,
        spatial_args={'n_heads': 8, 'causal': True, 'n_kv_heads': 2}
    )
    # set_vathos_mode("debug")
    # model = torch.compile(model)
    out = model(x).detach()
    print(out.shape)
    exit()
    model.summary()
    model.profile()
    model.autosave = False
    model.generate(torch.tensor([0]), 1000, temperature=1, token_end=None, custom_generate=True, repetition_penalty=1.2)
    model.profile(avg=True, plot=True)

    model.save_checkpoint('caroler.pt')
    model.load_checkpoint('caroler.pt')
    model.plot_losses()
    model.profile(avg=True, plot=True)
