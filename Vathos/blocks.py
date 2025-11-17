import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable


class RoPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        self.d_model = d_model
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cache(self, seq_len: int, device: torch.device):
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            self._cos_cached = freqs.cos()
            self._sin_cached = freqs.sin()

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        seq_len = q.shape[2]
        self._update_cache(seq_len, q.device)

        cos = self._cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self._sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)

        return (
            self._apply_rotary(q, cos, sin),
            self._apply_rotary(k, cos, sin)
        )

    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rotated = torch.stack([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin
        ], dim=-1).flatten(-2)
        return x_rotated


class MultiheadAttentionMixer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, causal: bool):
        super().__init__()
        self.causal = causal
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

        attn_mask = None
        if self.causal:
            attn_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=self.causal)
        attn = attn.transpose(1, 2).reshape(B, L, D)

        return self.out(attn)


class CausalMultiheadAttentionMixer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, causal=True):
        super().__init__()
        assert causal, \
            ("CausalMultiheadAttentionMixer Module only supports causal=True, "
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


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


class MLP(nn.Module):
    def __init__(self, d_model: int, depth: int, expand: int, activation: Callable):
        super().__init__()
        hidden_dim = d_model * expand

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

            if isinstance(activation, SwiGLU) and i < depth - 1:
                out_dim = out_dim * 2

            layers.append(nn.Linear(in_dim, out_dim, bias=True))
            if i < depth - 1:
                layers.append(activation)

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class Block1d(nn.Module):
    def __init__(self, channel_mixer: nn.Module, spatial_mixer: nn.Module):
        super().__init__()
        self.spatial_mixer = spatial_mixer
        self.channel_mixer = channel_mixer
        self.norm1 = nn.LayerNorm(spatial_mixer.d_model)
        self.norm2 = nn.LayerNorm(spatial_mixer.d_model)

    def forward(self, x: torch.Tensor):
        x = x + self.spatial_mixer(self.norm1(x))
        x = x + self.channel_mixer(self.norm2(x))
        return x


class MTransformer(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int = 8, mlp_expand: int = 4, causal: bool = True):
        super().__init__()
        self.d_model = d_model

        self.blocks = nn.ModuleList([
            Block1d(
                channel_mixer=MLP(d_model, depth=2, expand=mlp_expand, activation=SwiGLU()),
                spatial_mixer=CausalMultiheadAttentionMixer(d_model, n_heads, causal=causal)
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class _MTemplate(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int = 8, mlp_expand: int = 4, causal: bool = True,
                 channel_mixer=MLP, spatial_mixer=MultiheadAttentionMixer):
        super().__init__()
        self.d_model = d_model

        self.blocks = nn.ModuleList([
            Block1d(
                channel_mixer=channel_mixer(d_model, depth=2, expand=mlp_expand, activation=nn.GELU()),
                spatial_mixer=spatial_mixer(d_model, n_heads, causal=causal)
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


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


if __name__ == "__main__":
    test_causality(MTransformer(16, 4, 2))
