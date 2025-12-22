"""
FlexAttention implementation for Vathos library
Provides local causal attention using PyTorch's flex_attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from Vathos.blocks import *

flag("This module is WIP, LocalCausalFlexAttention is broken")  # TODO
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    flag("torch.nn.attention.flex_attention not found, consider, installing it")
    FLEX_ATTENTION_AVAILABLE = False
    flex_attention = None
    create_block_mask = None

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from functools import lru_cache, partial


# -----------------------------------------------------------------------------
# 1. Define the Mask Modality at Global Scope
# -----------------------------------------------------------------------------
def _sliding_window_mask_logic(b, h, q_idx, kv_idx, window_size):
    """
    Pure logic for Sliding Window Attention.
    (q_idx >= kv_idx)  -> Causal
    (q_idx - kv_idx < window_size) -> Local
    """
    return (q_idx >= kv_idx) & ((q_idx - kv_idx) < window_size)


# -----------------------------------------------------------------------------
# 2. Production Level Layer
# -----------------------------------------------------------------------------
class LocalCausalFlexAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            window_size: int = 512,
            rope: bool = False,
            dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.dropout_p = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        self.rope = RoPE(self.head_dim) if rope else nn.Identity()
        self.use_rope = rope

    @lru_cache(maxsize=32)
    def _get_block_mask(self, L, device):
        mask_mod = partial(_sliding_window_mask_logic, window_size=self.window_size)

        block_mask = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=L,
            KV_LEN=L,
            device=device
        )
        return block_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim)

        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_rope:
            q, k = self.rope(q, k, start_pos=0)

        block_mask = self._get_block_mask(L, q.device)
        out = flex_attention(
            q, k, v,
            block_mask=block_mask,
            scale=1.0 / (self.head_dim ** 0.5),
            enable_gqa=False
        )

        out = out.transpose(1, 2).reshape(B, L, self.d_model)
        return self.out(out)

class HybridLocalAttn(Layer):
    __name__ = "HybridLocalAttn"
    __complexity__ = "O(L d^2 + L * w * d)"

    def __init__(
            self,
            d_model: int,
            secondary_mixer: Layer,
            sec_params: dict,
            window_size: int = 256,
            n_heads: int = 8,
            rope: bool = False,
            dropout: float = 0.1
    ):
        super().__init__()
        self.sec_mixer = secondary_mixer(d_model=d_model, **sec_params)
        self.attn = LocalCausalFlexAttention(
            d_model=d_model,
            n_heads=n_heads,
            window_size=window_size,
            rope=rope,
            dropout=dropout,
            causal=True
        )

    def has_custom_generate(self):
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sec_mixer(x)
        x = self.attn(x)

        return x

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.sec_mixer, 'generate') and hasattr(self.sec_mixer,
                                                           'has_custom_generate') and self.sec_mixer.has_custom_generate():
            x = self.sec_mixer.generate(x)
        else:
            x = self.sec_mixer(x)

        x = self.attn.generate(x)

        return x

    def clear_cache(self):
        self.attn.clear_cache()
        if hasattr(self.sec_mixer, 'clear_cache'):
            self.sec_mixer.clear_cache()


if __name__ == '__main__':
    x = torch.randint(0, 9, size=(2, 64))
    """y = pe(img)
    print(y.shape)
    print(unem(y).shape)"""
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
        unembedder=UnbiasedLinear,
        channel_mixer=DLPSwiGLU,
        channel_args={'expand': 2},
        rope=True,
        spatial_mixer=LocalCausalFlexAttention,
        spatial_args={'n_heads': 8, 'causal': True}
    )
    out = model(x).detach()

    model.summary()
    model.autosave = False
    model.generate(torch.tensor([0]), 1000, temperature=1, token_end=None, custom_generate=True)
    model.profile()
