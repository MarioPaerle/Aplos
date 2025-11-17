from blocks import *
from Utils import *


class BaseSummerMixer(nn.Module):
    def __init__(self, d_model: int, causal=True):
        super().__init__()
        assert causal, "BaseSummerMixer Module only supports causal=True, "
        self.d_model = d_model

    def forward(self, x: torch.Tensor):
        return power_weigthed_cumsum(x)


class BaseGatedSummerMixer(nn.Module):
    def __init__(self, d_model: int, causal=True):
        super().__init__()
        assert causal, "BaseSummerMixer Module only supports causal=True, "
        self.d_model = d_model
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        return power_weigthed_cumsum(x * self.gate(x))


class DSummer1(nn.Module):
    def __init__(self, d_model: int, causal=True):
        super().__init__()
        assert causal, "BaseSummerMixer Module only supports causal=True, "
        self.d_model = d_model

    def forward(self, x: torch.Tensor):
        return power_weigthed_cumsum(x, 1)


class DFullSummer1(nn.Module):
    def __init__(self, d_model: int, causal=True):
        super().__init__()
        assert causal, "BaseSummerMixer Module only supports causal=True, "
        self.d_model = d_model
        self.gate1 = nn.Linear(d_model, d_model)
        self.gate2 = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        return power_weigthed_cumsum(x * self.gate1(x), 0.9) + self.gate2(x)


test_causality(DFullSummer1(8))
