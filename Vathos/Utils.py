import torch
import matplotlib.pyplot as plt
import inspect


def plot(*args, **kwargs):
    for x_i in args:
        plt.plot(x_i)
    for x_i in kwargs:
        plt.plot(kwargs[x_i], label=str(x_i))

    plt.legend()
    plt.show()


def power_weigthed_cumsum(x, a=0.999):
    alpha_pow = torch.full([x.shape[1]], a, dtype=x.dtype).cumprod(dim=0).unsqueeze(0).unsqueeze(2)
    return torch.cumsum(x / alpha_pow, dim=1) * alpha_pow


def precompute_power_weigthed_cumsum(x, alpha_pow):
    return torch.cumsum(x / alpha_pow, dim=1) * alpha_pow
