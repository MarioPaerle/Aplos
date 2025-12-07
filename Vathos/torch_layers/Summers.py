import torch.nn
from Vathos.blocks import *
from Vathos.functions import *
from Vathos.torch_layers.Kroneckers import KroneckerMixer1

class BaseSummerMixer(nn.Module):
    def __init__(self, d_model: int, causal=True, a=0.8):
        super().__init__()
        assert causal, "BaseSummerMixer Module only supports causal=True, "
        self.d_model = d_model
        self.a = a

    def forward(self, x: torch.Tensor):
        return power_weigthed_cumsum(x, a=self.a) + x


class BaseGatedSummerMixer(nn.Module):
    def __init__(self, d_model: int, causal=True, a=0.8):
        super().__init__()
        assert causal, "BaseSummerMixer Module only supports causal=True, "
        self.d_model = d_model
        self.gate = nn.Linear(d_model, d_model)
        self.a = a

    def forward(self, x: torch.Tensor):
        return power_weigthed_cumsum(x * self.gate(x), a=self.a) + x


class DSummer1(nn.Module):
    def __init__(self, d_model: int, causal=True):
        super().__init__()
        assert causal, "BaseSummerMixer Module only supports causal=True, "
        self.d_model = d_model

    def forward(self, x: torch.Tensor):
        return power_weigthed_cumsum(x, 1) + x


class DFullSummer1(nn.Module):
    def __init__(self, d_model: int, causal=True, a=0.8):
        super().__init__()
        assert causal, "BaseSummerMixer Module only supports causal=True, "
        self.d_model = d_model
        self.gate1 = nn.Linear(d_model, d_model)
        self.gate2 = nn.Linear(d_model, d_model)
        self.a = a

    def forward(self, x: torch.Tensor):
        return power_weigthed_cumsum(x * self.gate1(x), a=self.a) * self.gate2(x) + x


class LinAtt(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

    def feature(self, x):
        # return nn.functional.elu(x)
        return torch.nn.Identity()(x)

    def forward(self, x, a=0.95):
        Q = self.feature(self.q(x))
        K = self.feature(self.k(x))
        V = self.v(x)

        KV = power_weigthed_cumsum(K[..., :, None] * V[..., None, :], a=a)
        return torch.einsum("bld, bldd -> bld", Q, KV) + x


class LinAtt2(nn.Module):
    def __init__(self, d_model, expand):
        super().__init__()
        self.d_model = d_model
        self.q = nn.Linear(d_model, int(d_model * expand))
        self.k = nn.Linear(d_model, int(d_model * expand))
        self.v = nn.Linear(d_model, d_model)

    def feature(self, x):
        # return nn.functional.elu(x)
        return torch.nn.Identity()(x)

    def forward(self, x, a=0.95):
        Q = self.feature(self.q(x))
        K = self.feature(self.k(x))
        V = self.v(x)

        KV = power_weigthed_cumsum(K[..., :, None] * V[..., None, :], a=a)
        return torch.einsum("bld, blde -> ble", Q, KV) + x


class HolographicAttentionMixer(nn.Module):
    __name__ = "HolographicAttentionMixer"
    __complexity__ = "O(L d log d)"

    def __init__(self, d_model: int, n_heads: int, causal: bool, rope=False):
        super().__init__()
        self.causal = causal
        self.d_model = d_model

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)

        self.out = nn.Linear(d_model, d_model, bias=False)

        self.norm = RMSNorm(self.head_dim)

        if rope:
            self.rope = RoPE(self.head_dim)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).contiguous()

        if self.rope is not None:
            q, k = self.rope(q, k)
        T_vectors = holographic_binding(k, v)
        T_vectors = self.norm(T_vectors)

        if self.causal:
            S_vectors = torch.cumsum(T_vectors, dim=2)
        else:
            S_vectors = T_vectors.sum(dim=2, keepdim=True).repeat(1, 1, L, 1)

        S_vectors = self.norm(S_vectors)

        attn = holographic_binding(q, S_vectors)

        attn = attn.transpose(1, 2).reshape(B, L, D)

        return self.out(attn)


class HoloMultiQueryMixer(Layer):
    __name__ = "HolographicAttentionMixer"
    __complexity__ = "O(L sqrt(L) + L d log d)"

    def __init__(self, d_model: int, n_queries: int, rope=False, m_expand=1, max_len=2116):
        super().__init__()
        self.d_model = d_model
        self.n_queries = n_queries
        self.m_expand = m_expand
        self.head_dim = (d_model * m_expand) // n_queries
        self.rope_enabled = rope

        hidden_dim = d_model * m_expand
        self.qm_proj = nn.Linear(d_model, 2 * hidden_dim)
        self.summer = KroneckerMixer1(d_model=hidden_dim, max_len=max_len)

        #self.norm = nn.GroupNorm(n_queries, hidden_dim)
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(hidden_dim, d_model)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape

        projected = self.qm_proj(x)
        t, q = projected.chunk(2, dim=-1)

        # s = stable_power_weighted_cumsum(t, a=0.8, rescale=False)
        s = self.summer(t)

        s = s.view(B, L, self.n_queries, self.head_dim)
        q = q.view(B, L, self.n_queries, self.head_dim)

        o = holographic_binding(q, s).view(B, L, -1)
        # plot(t=t[0, :, 0, 0].detach(), s=s[0, :, 0, 0].detach(), o=o[0, :, 0, 0].detach())

        # o = self.norm(o.permute(0, 2, 1)).permute(0, 2, 1)
        o = self.norm(o)

        return self.out_proj(o)


if __name__ == '__main__':
    x = torch.randint(0, 9, size=(2, 2116))

    model = SequenceModel(
        vocab_size=100,
        d_model=128,
        n_layers=6,
        max_len=2116,
        pos_encoder=True,
        embedder=EasyEmbedder,
        unembedder=Identity,
        channel_mixer=MLP,
        channel_args={'expand': 2, 'activation': SwiGLU, 'depth': 2},
        rope=False,
        spatial_mixer=HoloMultiQueryMixer,
        spatial_args={'n_queries': 8}
    )
    out = model(x).detach()

    model.summary()
    model.profile()
    model.profile(avg=True, plot=True)
    model.autosave = False
    # model.generate(torch.tensor([0]), 1000, temperature=1)
    # model.plot_losses()