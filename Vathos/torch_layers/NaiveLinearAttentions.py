"""
Linear Attentions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NaiveLinearAttention(nn.Module):
    """
    Naive Linear Attention: attention without softmax normalization.

    For causal=False: output = (Q @ K^T) @ V
    For causal=True:  output = ((Q @ K^T) * causal_mask) @ V

    This is useful for:
    - Benchmarking model quality vs softmax attention
    - Short sequences where O(L^2) is acceptable
    - Understanding the role of softmax normalization

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        causal: Whether to apply causal masking
        qk_norm: Whether to apply normalization to Q and K (recommended for stability)
        scale_output: Whether to scale output by 1/sqrt(L) for stability
        rope: Whether to use RoPE (requires RoPE module from the codebase)
    """
    __name__ = "NaiveLinearAttention"
    __complexity__ = "O(L^2 d)"

    def __init__(
            self,
            d_model: int,
            n_heads: int = 8,
            causal: bool = True,
            qk_norm: bool = True,
            scale_output: bool = True,
            rope: bool = False,
            dropout: float = 0.0
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.qk_norm = qk_norm
        self.scale_output = scale_output

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)
            self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)

        self.rope = None
        if rope:
            try:
                from Vathos.blocks import RoPE
                self.rope = RoPE(self.head_dim)
            except ImportError:
                print("Warning: RoPE not available, continuing without it")

        self.qk_scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, L, D)

        Returns:
            Output tensor of shape (B, L, D)
        """
        B, L, D = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.rope is not None:
            q, k = self.rope(q, k)

        q = q * self.qk_scale

        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        if self.causal:
            causal_mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
            attn_scores = attn_scores.masked_fill(~causal_mask, 0.0)

        attn_scores = self.dropout(attn_scores)
        out = torch.matmul(attn_scores, v)

        if self.scale_output:
            out = out / math.sqrt(L)

        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out(out)

        return out


class NaiveLinearAttentionV2(nn.Module):
    """
    Alternative implementation using explicit einsum operations.
    Can be more memory efficient for very long sequences in non-causal case.

    For non-causal attention, this can be computed as:
        output = Q @ (K^T @ V)
    which is O(L * d^2) instead of O(L^2 * d), but requires full context.
    """
    __name__ = "NaiveLinearAttentionV2"
    __complexity__ = "O(L^2 d) causal, O(L d^2) non-causal"

    def __init__(
            self,
            d_model: int,
            n_heads: int = 8,
            causal: bool = True,
            qk_norm: bool = True,
            use_efficient_noncausal: bool = True,
            rope: bool = False,
            dropout: float = 0.0
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.qk_norm = qk_norm
        self.use_efficient_noncausal = use_efficient_noncausal and not causal

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)
            self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)

        self.rope = None
        if rope:
            try:
                from Vathos.layers import RoPE
                self.rope = RoPE(self.head_dim)
            except ImportError:
                pass

        self.qk_scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.rope is not None:
            q, k = self.rope(q, k)

        q = q * self.qk_scale

        if self.use_efficient_noncausal:
            kv = torch.matmul(k.transpose(-2, -1), v)
            out = torch.matmul(q, kv)
            out = out / math.sqrt(L)
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1))

            if self.causal:
                causal_mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
                attn_scores = attn_scores.masked_fill(~causal_mask, 0.0)

            attn_scores = self.dropout(attn_scores)
            out = torch.matmul(attn_scores, v)
            out = out / math.sqrt(L)

        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out(out)

        return out
