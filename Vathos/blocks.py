import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
import math
from Vathos.Utils import *


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


class SinusoidalPositionalEncoding(nn.Module):
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


class MultiheadAttentionMixer(nn.Module):
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
    gated = True

    def forward(self, x: torch.Tensor):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


class MLP(nn.Module):
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


class ResMLPBlock(nn.Module):
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


class ResMLP(nn.Module):
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


class _MTemplate(nn.Module):
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


class Embedder(nn.Module):
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


class Symbolic1dSeq2SeqModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 max_len=1024,
                 pos_encoder: bool | None | nn.Module = None,
                 embedder=Embedder,
                 channel_mixer=MLP,
                 spatial_mixer=MultiheadAttentionMixer,
                 channel_args: dict = None,
                 spatial_args: dict = None,
                 ):
        super().__init__()
        if channel_args is None and channel_mixer is MLP:
            channel_args = {"expand": 2, "activation": nn.GELU, "depth": 2}
        if spatial_args is None and spatial_mixer is MultiheadAttentionMixer:
            spatial_args = {"causal": True, "n_heads": 8}
        if spatial_args is None:
            spatial_args = {}
        if channel_args is None:
            channel_args = {}

        self.spatial_args = spatial_args
        self.channel_args = channel_args
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.n_layers = n_layers

        self.embedder = embedder(vocab_size=vocab_size, d_model=d_model)

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
        self.unembedder = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights following GPT-style conventions:
        - Small std for embeddings
        - Xavier/Kaiming for linear layers
        - Scaled initialization for residual projections
        """
        std_embed = 0.02
        nn.init.normal_(self.embedder.embedding.weight, mean=0.0, std=std_embed)

        nn.init.normal_(self.unembedder.weight, mean=0.0, std=std_embed)

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

    def forward(self, x: torch.LongTensor):
        x = self.embedder(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.unembedder(x)
        return x


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


def test_symbolic_model(model):
    vocab_size = model.vocab_size
    length = model.max_len
    x = torch.randint(0, vocab_size, (2, length))
    print(f"Input shape {x.shape}")
    x = model(x)
    print(f"Output shape {x.shape}")
    print(f"Bounds:  min:{x.min()}, max:{x.max()}, mean:{x.mean()}, std:{x.std()}, sum_of_a_vector: {x[0, 0, :].sum()}")


if __name__ == "__main__":
    test_causality(MTransformer(16, 4, 2))
    test_symbolic_model(Symbolic1dSeq2SeqModel(128, 16, 4, 2, pos_encoder=True))
