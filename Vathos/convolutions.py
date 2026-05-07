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


class SmearGate2(nn.Module):
    def __init__(self, dim: int, gate_dim: int = 12):
        super().__init__()
        self.gate_dim = min(gate_dim, dim)
        self.W_g = nn.Parameter(torch.zeros(self.gate_dim, dim))
        self.smear_lambda = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        x_prev = F.pad(x[:, :-1, :], (0, 0, 1, 0))
        x_slice = x[..., :self.gate_dim]
        g = torch.sigmoid(torch.matmul(x_slice, self.W_g.to(x.dtype)))

        return x + self.smear_lambda * g * x_prev


class SmearGateLookback3(nn.Module):
    def __init__(self, dim: int, gate_dim: int = 12):
        super().__init__()
        self.gate_dim = min(gate_dim, dim)

        self.W_g = nn.Parameter(torch.zeros(self.gate_dim, dim * 3))

        self.smear_lambdas = nn.Parameter(torch.zeros(3))

    def forward(self, x: Tensor) -> Tensor:
        x_slice = x[..., :self.gate_dim]
        g_all = torch.sigmoid(torch.matmul(x_slice, self.W_g.to(x.dtype)))

        g1, g2, g3 = g_all.chunk(3, dim=-1)

        x1 = F.pad(x[:, :-1, :], (0, 0, 1, 0))
        x2 = F.pad(x[:, :-2, :], (0, 0, 2, 0))
        x3 = F.pad(x[:, :-3, :], (0, 0, 3, 0))

        return x + (self.smear_lambdas[0] * g1 * x1) \
            + (self.smear_lambdas[1] * g2 * x2) \
            + (self.smear_lambdas[2] * g3 * x3)


class ModdedSmearGate(nn.Module):
    def __init__(self, gate_dim: int = 12):
        super().__init__()
        self.gate_dim = gate_dim

        self.smear_gate = nn.Linear(gate_dim, 1, bias=False)

        with torch.no_grad():
            nn.init.zeros_(self.smear_gate.weight)

    def forward(self, x: Tensor) -> Tensor:
        x_prev = F.pad(x[:, :-1, :], (0, 0, 1, 0))
        x_slice = x[..., :self.gate_dim]

        g = torch.sigmoid(self.smear_gate(x_slice))
        return x + g * (x_prev - x)
