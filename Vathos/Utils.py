import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

try:
    from colorama import Fore

    GOOD = Fore.GREEN
    BAD = Fore.RED
    RES = Fore.RESET
    SEC = Fore.LIGHTBLACK_EX
    HM = Fore.YELLOW
    NUM = Fore.BLUE
except:
    GOOD = ""
    BAD = ""
    RES = ""
    SEC = ""
    NUM = ""
    HM = ""

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


def batched_channelwise_conv1d(K, V):
    """
    Computes a convolution between two batched tensors
    K: [B, L, d] kernel
    V: [B, L, d] signal
    Output: [B, L, d]
    """
    B, L, d = V.shape
    V_reshaped = V.view(1, B * L, d)

    K_reshaped = K.view(B * L, 1, d).flip(-1)

    pad_total = d - 1
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    V_padded = F.pad(V_reshaped, (pad_left, pad_right))

    output = F.conv1d(V_padded, K_reshaped, groups=B * L)

    return output.view(B, L, d)


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


@torch.compile
def _kronecker_batch_matmul(A: torch.Tensor, B: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2-D square matrices")
    if X.ndim != 3:
        raise ValueError("X must be [B, L, d]")
    n = A.shape[0]
    if A.shape[1] != n or B.shape != (n, n):
        raise ValueError("A and B must be the same square size (n x n)")

    B_batch, L, d = X.shape
    if L != n * n:
        raise ValueError(f"L must equal n*n (got L={L}, n*n={n * n})")

    Xm = X.view(B_batch, n, n, d)

    Y = torch.einsum('ij,bjkf,lk->bilf', A, Xm, B)

    return Y.reshape(B_batch, L, d)


def flag(text, level=1):
    if level <= FLAG_PASS:
        print(f"{SEC}||{HM}FLAG LV.{level}{SEC}||{HM} {text}{RES}")


def getname(obj):
    return obj.__name__ if hasattr(obj, "__name__") else str(type(obj)).split('.')[-1]
