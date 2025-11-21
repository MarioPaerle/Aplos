import torch
import matplotlib.pyplot as plt

try:
    from colorama import Fore
    GOOD = Fore.GREEN
    BAD = Fore.RED
    RES = Fore.RESET
    SEC = Fore.LIGHTBLACK_EX
    NUM = Fore.BLUE
except:
    GOOD = ""
    BAD =  ""
    RES =  ""
    SEC =  ""
    NUM = ""


FLAG_PASS = 3

plt.style.use('ggplot')


def plot(*args, **kwargs):
    for x_i in args:
        plt.plot(x_i)
    for x_i in kwargs:
        plt.plot(kwargs[x_i], label=str(x_i))

    plt.legend()
    plt.show()


def cumsum(x):
    return torch.cumsum(x, dim=1)


def power_weigthed_cumsum(x, a=0.999, rescale=True):
    """
    a torch implementation of a power weighted cumsum,
    which can be applied to batched inputs of shapes [B, L, d] or [B, L, d1, d2], summing over L
    """
    if a - 1 == 0:
        return cumsum(x)
    if len(x.shape) == 3:
        alpha_pow = torch.full([x.shape[1]], a, dtype=x.dtype, device=x.device).cumprod(dim=0).unsqueeze(0).unsqueeze(2)
    elif len(x.shape) == 4:
        alpha_pow = torch.full([x.shape[1]], a, dtype=x.dtype, device=x.device).cumprod(dim=0).unsqueeze(0).unsqueeze(
            2).unsqueeze(2)
    else:
        raise RuntimeError("x shapes mus be of form [B, L, d] or [B, L, d, d]")

    z = x / alpha_pow
    z[:, alpha_pow[0, :, 0] < 0.01, :] = z  # TODO: This must be fixed
    # alpha_pow[alpha_pow < 0.0001] = 1
    return torch.cumsum(z, dim=1) * alpha_pow * (alpha_pow[:, 0, ...] if rescale else 1)


def precompute_power_weigthed_cumsum(x, alpha_pow):
    return torch.cumsum(x / alpha_pow, dim=1) * alpha_pow


def flag(text, level=1):
    if level <= FLAG_PASS:
        print(f"||FLAG LV.{level}|| {text}")


def getname(obj):
    return obj.__name__ if hasattr(obj, "__name__") else type(obj)
