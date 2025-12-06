import torch.nn
from Vathos.blocks import *
from Vathos.functions import *


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
        T_vectors = batched_channelwise_conv1d(k, v)
        T_vectors = self.norm(T_vectors)

        if self.causal:
            S_vectors = torch.cumsum(T_vectors, dim=2)
        else:
            S_vectors = T_vectors.sum(dim=2, keepdim=True).repeat(1, 1, L, 1)

        S_vectors = self.norm(S_vectors)

        attn = batched_channelwise_conv1d(q, S_vectors)

        attn = attn.transpose(1, 2).reshape(B, L, D)

        return self.out(attn)