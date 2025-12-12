"""
FlexAttention implementation for Vathos library
Provides local causal attention using PyTorch's flex_attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from Vathos.blocks import *

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    flag("torch.nn.attention.flex_attention not found, consider, installing it")
    FLEX_ATTENTION_AVAILABLE = False
    flex_attention = None
    create_block_mask = None


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

        # Projection Layers
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        # Components
        self.rope = RoPE(self.head_dim) if rope else None
        self.dropout = nn.Dropout(dropout)

        # Flex Attention Setup
        self.use_flex_attention = FLEX_ATTENTION_AVAILABLE
        self._cached_block_mask = None
        self._cached_seq_len = 0

        # Internal State for SequenceModel generation
        self.kv_cache = None

    def has_custom_generate(self):
        """Signals to Block1d that this layer has a specialized generation method"""
        return True

    def _get_local_mask_mod(self):
        """Returns the mask function for create_block_mask"""
        w = self.window_size

        def local_causal(b, h, q_idx, kv_idx):
            return (kv_idx <= q_idx) & ((q_idx - kv_idx) < w)

        return local_causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard Forward Pass (Prefill / Training).
        Uses FlexAttention with Sparse Block Mask if available.
        """
        B, L, D = x.shape

        # (B, L, 3, H, D_head)
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)

        # FlexAttention requires (B, H, L, D_head) usually, but we permute to standard logic first
        # Layout: (3, B, H, L, D_head) -> (B, H, L, D_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).contiguous()

        # Apply RoPE (Absolute positions 0 to L)
        if self.use_rope:
            q, k = self.rope(q, k, start_pos=0)

        if self.use_flex_attention:
            # Efficient Block Mask Caching for fixed sized batches/seqs
            if self._cached_block_mask is None or self._cached_seq_len != L:
                self._cached_block_mask = create_block_mask(
                    self._get_local_mask_mod(),
                    B=None, H=None, Q_LEN=L, KV_LEN=L,
                    device=q.device
                )
                self._cached_seq_len = L

            # Pass 1: Flex Attention
            attn_output = flex_attention(q, k, v, block_mask=self._cached_block_mask)
        else:
            # Pass 2: Fallback (Heavy Memory Usage Warning)
            attn_output = self._fallback_attention(q, k, v)

        # Reshape output: (B, H, L, D_head) -> (B, L, D)
        attn_output = attn_output.transpose(1, 2).reshape(B, L, D)
        return self.out(self.dropout(attn_output))

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generation Step (Decode).
        Uses Standard SDPA + Rolling KV Cache (Internal State).
        """
        B, L, D = x.shape  # L is typically 1 here

        # 1. Project
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (B, H, 1, D_head)

        # 2. Manage Cache State
        if self.kv_cache is not None:
            k_cache, v_cache = self.kv_cache
            # Pos offset is the current length of cache (for RoPE)
            # Note: With rolling cache, absolute pos might drift from cache len,
            # but for local window RoPE, relative distance matters most.
            # If strict absolute position is needed, SequenceModel needs to pass `pos` in.
            # Vathos usually relies on standard RoPE appending.
            pos_offset = k_cache.shape[2]
        else:
            k_cache, v_cache = None, None
            pos_offset = 0

        # 3. Apply RoPE
        if self.use_rope:
            q, k = self.rope(q, k, start_pos=pos_offset)

        # 4. Update Rolling Cache
        if k_cache is not None:
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        # *** ROLLING WINDOW LOGIC ***
        # Physically delete old tokens from VRAM to keep O(1) memory during generation
        if k.size(2) > self.window_size:
            k = k[:, :, -self.window_size:, :]
            v = v[:, :, -self.window_size:, :]

        # Save to internal state
        self.kv_cache = (k, v)

        # 5. Attention (SDPA)
        # We don't need a mask because we physically trimmed the cache to the window.
        # We attend to everything currently in self.kv_cache
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=False,  # Cache is already past-only
            dropout_p=0.0
        )

        # 6. Output Projection
        attn_output = attn_output.transpose(1, 2).reshape(B, L, D)
        return self.out(attn_output)

    def _fallback_attention(self, q, k, v):
        """Fallback for when FlexAttention is missing. Uses dense mask."""
        L = q.size(2)
        # Create dense boolean mask (O(L^2) memory!)
        mask = torch.ones(L, L, device=q.device, dtype=torch.bool).tril()
        local_mask = torch.ones(L, L, device=q.device, dtype=torch.bool).triu(-self.window_size + 1)
        mask = mask & local_mask

        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )

    def clear_cache(self):
        """Called by SequenceModel before/after generation"""
        self.kv_cache = None


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
