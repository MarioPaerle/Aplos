import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FastShiftConv1d_K3(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w0 = nn.Parameter(torch.ones(dim))
        self.w1 = nn.Parameter(torch.zeros(dim))
        self.w2 = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        x1 = F.pad(x[:, :-1, :], (0, 0, 1, 0))
        x2 = F.pad(x[:, :-2, :], (0, 0, 2, 0))

        return (x * self.w0.to(x.dtype)) + (x1 * self.w1.to(x.dtype)) + (x2 * self.w2.to(x.dtype))


class FastShiftConv1d_K4(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w0 = nn.Parameter(torch.ones(dim))
        self.w1 = nn.Parameter(torch.zeros(dim))
        self.w2 = nn.Parameter(torch.zeros(dim))
        self.w3 = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        x1 = F.pad(x[:, :-1, :], (0, 0, 1, 0))
        x2 = F.pad(x[:, :-2, :], (0, 0, 2, 0))
        x3 = F.pad(x[:, :-3, :], (0, 0, 3, 0))

        return (x * self.w0.to(x.dtype)) + \
            (x1 * self.w1.to(x.dtype)) + \
            (x2 * self.w2.to(x.dtype)) + \
            (x3 * self.w3.to(x.dtype))


class FastShiftConv1d_K2(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # Dirac/Identity Init: 1 for current token, 0 for past tokens
        self.w0 = nn.Parameter(torch.ones(dim))
        self.w1 = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        x1 = F.pad(x[:, :-1, :], (0, 0, 1, 0))

        return (x * self.w0.to(x.dtype)) + (x1 * self.w1.to(x.dtype))


class RWKVTimeMix(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.time_mix = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        x_prev = F.pad(x[:, :-1, :], (0, 0, 1, 0))
        mix = self.time_mix.to(x.dtype)

        return x * mix + x_prev * (1.0 - mix)
