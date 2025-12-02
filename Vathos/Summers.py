import torch.nn
from Vathos.blocks import *
from Vathos.Utils import *


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






if __name__ == '__main__':
    from Mamba2 import Mamba2Mixer
    """x = torch.randn(2, 2116, 4)
    model1 = BaseGatedSummerMixer(4, a=0.95)
    model2 = BaseGatedSummerMixer(4, a=0.8)
    y1 = model1(x)
    y2 = model2(x)
    plot(x=x[0, :, 0].detach(), y1=y1[0, :, 0].detach(), y2=y2[0, :, 0].detach())"""
    # test_causality(LinAtt(8))
    model = SequenceModel(100, 256, 4,
                          channel_mixer=MLP,
                          channel_args={"depth": 2, "expand": 2, "activation": nn.ReLU},
                          spatial_mixer=MultiheadAttentionMixer,
                          spatial_args={"causal": True, "n_heads": 8})
    test_causality_symbolic(model)
    # print(model)
