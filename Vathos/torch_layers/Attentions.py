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

        # Projections
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        # Components
        self.rope = RoPE(self.head_dim) if rope else None
        self.dropout = nn.Dropout(dropout)

        # FlexAttention State
        self.use_flex_attention = FLEX_ATTENTION_AVAILABLE
        self._cached_block_mask = None
        self._cached_seq_len = 0

        # Generation / Cache State
        self.kv_cache = None
        self.decoding_pos = 0

        # Debugging
        self.debug = False

    def has_custom_generate(self):
        return True

    def _get_local_mask_mod(self):
        """Defines the sliding window logic for FlexAttention compiler"""
        w = self.window_size

        def local_causal(b, h, q_idx, kv_idx):
            return (kv_idx <= q_idx) & ((q_idx - kv_idx) < w)

        return local_causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard Forward Pass (Training / Prefill).
        Uses FlexAttention if available, otherwise SDPA with mask.
        """
        B, L, D = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).contiguous()

        if self.use_rope:
            q, k = self.rope(q, k, start_pos=0)

        if self.use_flex_attention:
            if self._cached_block_mask is None or self._cached_seq_len != L:
                if self.debug: print(f"[FlexAttn] Compiling mask for L={L}")
                self._cached_block_mask = create_block_mask(
                    self._get_local_mask_mod(),
                    B=None, H=None, Q_LEN=L, KV_LEN=L,
                    device=q.device
                )
                self._cached_seq_len = L

            # Run Fused Kernel
            attn_output = flex_attention(q, k, v, block_mask=self._cached_block_mask)
        else:
            # Fallback (Dense Mask)
            attn_output = self._fallback_attention(q, k, v)

        # 4. Output
        attn_output = attn_output.transpose(1, 2).reshape(B, L, D)
        return self.out(self.dropout(attn_output))

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generation Step. Handles Prefill (L>1) and Decode (L=1).
        """
        B, L, D = x.shape

        if L > 1:
            if self.debug: print(f"[Gen] Prefill L={L}")

            output = self.forward(x)

            with torch.no_grad():
                qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
                k = qkv[:, :, 1].transpose(1, 2)
                v = qkv[:, :, 2].transpose(1, 2)

                if self.use_rope:
                    _, k = self.rope(k, k, start_pos=self.decoding_pos)

                if self.kv_cache is not None:
                    k_old, v_old = self.kv_cache
                    k = torch.cat([k_old, k], dim=2)
                    v = torch.cat([v_old, v], dim=2)

                if k.size(2) > self.window_size:
                    k = k[:, :, -self.window_size:, :]
                    v = v[:, :, -self.window_size:, :]

                self.kv_cache = (k, v)
                self.decoding_pos += L

            return output

        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        if self.use_rope:
            q, k = self.rope(q, k, start_pos=self.decoding_pos)

        if self.kv_cache is not None:
            k_cache, v_cache = self.kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        if k.size(2) > self.window_size:
            k = k[:, :, -self.window_size:, :]
            v = v[:, :, -self.window_size:, :]

        self.kv_cache = (k, v)
        self.decoding_pos += L

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=False,
            dropout_p=0.0
        )

        if self.debug and self.decoding_pos % 100 == 0:
            print(f"[Gen] Step {self.decoding_pos} | Cache Size: {k.size(2)}")

        attn_output = attn_output.transpose(1, 2).reshape(B, L, D)
        return self.out(attn_output)

    def _fallback_attention(self, q, k, v):
        """Standard PyTorch SDPA with manually created local mask"""
        L = q.size(2)
        mask = torch.ones(L, L, device=q.device, dtype=torch.bool).tril()
        local_mask = torch.ones(L, L, device=q.device, dtype=torch.bool).triu(-self.window_size + 1)
        final_mask = mask & local_mask

        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=final_mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )

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
