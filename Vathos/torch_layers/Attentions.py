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


def local_causal_mod(q_idx, kv_idx, window_size):
    """
    The mask modality function.
    Note: We pass window_size via partial or closure in the mask creation,
    but keeping the logic pure helps the compiler.
    """
    return (kv_idx <= q_idx) & ((q_idx - kv_idx) < window_size)


class LocalCausalFlexAttention(Layer):
    __name__ = "LocalCausalFlexAttention"
    __complexity__ = "O(L * w * d)"

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            window_size: int = 512,
            rope: bool = False,
            dropout: float = 0.1,
            causal: bool = True
    ):
        super().__init__()
        assert causal, "LocalCausalFlexAttention only supports causal=True"
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.use_rope = rope

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        self.rope = RoPE(self.head_dim) if rope else None
        self.dropout = nn.Dropout(dropout)

        self.use_flex_attention = FLEX_ATTENTION_AVAILABLE

        self._cached_block_mask = None
        self._cached_seq_len = -1

        self.kv_cache = None
        self.decoding_pos = 0
        self.debug = False

    def has_custom_generate(self):
        return True

    def _get_block_mask(self, L, device):
        if self._cached_block_mask is None or self._cached_seq_len != L:
            if self.debug: print(f"[FlexAttn] Compiling mask for L={L}")

            def bound_local_causal(b, h, q_idx, kv_idx):
                return local_causal_mod(b, h, q_idx, kv_idx, self.window_size)

            self._cached_block_mask = create_block_mask(
                bound_local_causal,
                B=None, H=None, Q_LEN=L, KV_LEN=L,
                device=device
            )
            self._cached_seq_len = L

        return self._cached_block_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard Forward Pass.
        """
        B, L, D = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)

        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_rope:
            q, k = self.rope(q, k, start_pos=0)

        attn_out = self._apply_attention(q, k, v)

        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        return self.out(attn_out)

    def _apply_attention(self, q, k, v):
        """
        Internal attention router: Flex vs SDPA fallback
        """
        B, H, L, D = q.shape

        if self.use_flex_attention:
            block_mask = self._get_block_mask(L, q.device)

            return flex_attention(q, k, v, block_mask=block_mask)
        else:
            return self._fallback_attention(q, k, v)

    def generate(self, x: torch.Tensor) -> torch.Tensor:

        B, L, D = x.shape

        # 1. Project QKV once
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. Apply RoPE
        if self.use_rope:
            # Note: start_pos determines the rotation for the new tokens
            q, k = self.rope(q, k, start_pos=self.decoding_pos)

        # 3. KV Cache Management
        if self.kv_cache is not None:
            k_cache, v_cache = self.kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        if k.size(2) > self.window_size:
            k = k[:, :, -self.window_size:, :]
            v = v[:, :, -self.window_size:, :]

        self.kv_cache = (k, v)
        self.decoding_pos += L

        if L > 1:
            if self.decoding_pos == L:  # Fresh prefill
                attn_output = self._apply_attention(q, k, v)
            else:
                attn_output = self._fallback_attention(q, k, v)
        else:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=False,
                dropout_p=0.0
            )

        attn_output = attn_output.transpose(1, 2).reshape(B, L, D)
        return self.out(attn_output)

    def _fallback_attention(self, q, k, v):
        """SDPA Fallback with memory-efficient mask creation"""
        L_q = q.size(2)
        L_kv = k.size(2)

        if L_q == 1:
            return F.scaled_dot_product_attention(q, k, v, is_causal=False)

        if L_q == L_kv:
            idx_q = torch.arange(L_q, device=q.device).unsqueeze(1)
            idx_k = torch.arange(L_kv, device=q.device).unsqueeze(0)

            mask = (idx_k <= idx_q) & ((idx_q - idx_k) < self.window_size)

            return F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0
            )
        else:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    def clear_cache(self):
        self.kv_cache = None
        self.decoding_pos = 0


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
