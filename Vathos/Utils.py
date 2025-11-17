import torch
import matplotlib.pyplot as plt
import inspect

FLAG_PASS = 3

def plot(*args, **kwargs):
    for x_i in args:
        plt.plot(x_i)
    for x_i in kwargs:
        plt.plot(kwargs[x_i], label=str(x_i))

    plt.legend()
    plt.show()


def cumsum(x):
    return torch.cumsum(x, dim=1)


def power_weigthed_cumsum(x, a=0.999):
    if a-1 == 0:
        return cumsum(x)
    alpha_pow = torch.full([x.shape[1]], a, dtype=x.dtype).cumprod(dim=0).unsqueeze(0).unsqueeze(2)
    return torch.cumsum(x / alpha_pow, dim=1) * alpha_pow


def precompute_power_weigthed_cumsum(x, alpha_pow):
    return torch.cumsum(x / alpha_pow, dim=1) * alpha_pow


def flag(text, level=1):
    if level <= FLAG_PASS:
        print(f"||FLAG LV.{level}|| {text}")
