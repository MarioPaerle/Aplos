import torch
import torch.nn.functional as F
import time
import numpy as np

# matplotlib is only used by plot/benchmark utilities, never by the model code.
# Lazy + optional import to avoid breaking Vathos import in headless environments.
try:
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
except ImportError:
    plt = None  # le funzioni di plot solleveranno errore esplicito se chiamate

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


def _require_matplotlib():
    if plt is None:
        raise ImportError(
            "Funzione di plotting chiamata ma matplotlib non è installato. "
            "Installa con `pip install matplotlib` per usarla."
        )


def plot(*args, **kwargs):
    _require_matplotlib()
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


def holographic_binding(K, V):
    """
    Computes Batched Circular Convolution (Binding) using FFT.
    This is O(d log d) and numerically stable.

    K: [..., d] kernel
    V: [..., d] signal
    Output: [..., d]
    """
    V_f = torch.fft.rfft(V, dim=-1, norm='ortho')
    K_f = torch.fft.rfft(K, dim=-1, norm='ortho')

    output_f = V_f * K_f
    output = torch.fft.irfft(output_f, n=V.shape[-1], dim=-1, norm='ortho')

    return output


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

    return torch.cumsum(x / alpha_pow, dim=1) * alpha_pow * (alpha_pow[:, 0, ...] if rescale else 1)


def stable_power_weighted_cumsum(x: torch.Tensor, a: float = 0.999, rescale: bool = True) -> torch.Tensor:
    B, L = x.shape[:2]

    dtype = x.dtype
    if L > 1024 and x.dtype == torch.float32:
        x = x.to(torch.float64)
        a_tensor = torch.tensor(a, dtype=torch.float64, device=x.device)
    else:
        a_tensor = torch.tensor(a, dtype=x.dtype, device=x.device)

    t = torch.arange(1, L + 1, dtype=a_tensor.dtype, device=x.device)

    alpha_pow = torch.pow(a_tensor, t)

    reshape_dims = [1, L] + [1] * (len(x.shape) - 2)
    alpha_pow = alpha_pow.view(*reshape_dims)

    scaled_input = x / alpha_pow

    s_cumulative = torch.cumsum(scaled_input, dim=1)

    s = s_cumulative * alpha_pow

    if rescale:
        s = s * (1.0 - a_tensor)

    return s.to(dtype)


def precompute_power_weigthed_cumsum(x, alpha_pow):
    return torch.cumsum(x / alpha_pow, dim=1) * alpha_pow


def flag(text, level=1):
    if level <= FLAG_PASS:
        print(f"{SEC}||{HM}FLAG LV.{level}{SEC}||{HM} {text}{RES}")


def getname(obj):
    return obj.__name__ if hasattr(obj, "__name__") else str(type(obj)).split('.')[-1]


def benchmark_ar_symbolic_model(model, n, input_shape):
    """
    Benchmarks the autoregressive symbolic model on three tasks: forward pass,
    forward + backward, and forward + backward + optimizer step. Uses random
    generated data. Reports average times and shows a bar plot.

    Additionally, benchmarks the ratio of forward time / (forward + backward) time
    with increasing sequence length L, and plots it.

    Args:
        model: The model to benchmark (e.g., SequenceModel instance).
        n: Number of iterations to average over for each benchmark.
        input_shape: Tuple (batch_size, seq_len) for initial input shape.
    """
    device = next(model.parameters()).device
    is_cuda = device.type == 'cuda'
    model.train()  # Ensure model is in training mode for gradients
    optimizer = torch.optim.Adam(model.parameters())

    # Helper to sync if CUDA
    def sync():
        if is_cuda:
            torch.cuda.synchronize()

    # Part 1: Benchmark forward, forward+backward, forward+backward+opt
    times_fwd = []
    times_fb = []
    times_fbo = []

    for _ in range(n):
        x = torch.randint(0, model.vocab_size, input_shape, device=device, requires_grad=False)

        # Forward only
        sync()
        start = time.perf_counter()
        out = model(x)
        sync()
        end = time.perf_counter()
        times_fwd.append(end - start)

        # Forward + backward
        optimizer.zero_grad()
        sync()
        start = time.perf_counter()
        out = model(x)
        loss = out.sum()  # Dummy loss
        loss.backward()
        sync()
        end = time.perf_counter()
        times_fb.append(end - start)

        # Forward + backward + optimizer step
        optimizer.zero_grad()
        sync()
        start = time.perf_counter()
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        sync()
        end = time.perf_counter()
        times_fbo.append(end - start)

    avg_fwd = np.mean(times_fwd)
    avg_fb = np.mean(times_fb)
    avg_fbo = np.mean(times_fbo)

    print(f"Average Forward Time: {avg_fwd:.6f} s")
    print(f"Average Forward + Backward Time: {avg_fb:.6f} s")
    print(f"Average Forward + Backward + Opt Time: {avg_fbo:.6f} s")

    # Bar plot for the three tasks
    labels = ['Forward', 'Forward + Backward', 'Forward + Backward + Opt']
    times = [avg_fwd, avg_fb, avg_fbo]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, times, color=['blue', 'orange', 'green'])
    plt.ylabel('Average Time (s)')
    plt.title(f'Benchmark Averages over {n} Iterations')
    plt.show()

    # Part 2: Ratio of forward / (forward + backward) with increasing L
    batch_size = input_shape[0]
    Ls = [16, 32, 64, 128, 256, 512, 1024]  # Reasonable sequence lengths, adjust as needed
    ratios = []
    avg_times_fwd = []
    avg_times_fb = []
    inner_n = max(10, n // 10)  # Average over fewer iterations for speed, but at least 10

    for L in Ls:
        shape = (batch_size, L)
        x = torch.randint(0, model.vocab_size, shape, device=device, requires_grad=False)

        # Time forward (average over inner_n)
        fwd_times = []
        for _ in range(inner_n):
            sync()
            start = time.perf_counter()
            out = model(x)
            sync()
            end = time.perf_counter()
            fwd_times.append(end - start)
        time_fwd = np.mean(fwd_times)
        avg_times_fwd.append(time_fwd)

        # Time forward + backward (average over inner_n)
        fb_times = []
        for _ in range(inner_n):
            optimizer.zero_grad()
            sync()
            start = time.perf_counter()
            out = model(x)
            loss = out.sum()
            loss.backward()
            sync()
            end = time.perf_counter()
            fb_times.append(end - start)
        time_fb = np.mean(fb_times)
        avg_times_fb.append(time_fb)

        ratio = time_fwd / time_fb if time_fb > 0 else 0
        ratios.append(ratio)

    # Plot the ratio
    plt.figure(figsize=(8, 5))
    plt.plot(Ls, ratios, marker='o', color='red')
    plt.xlabel('Sequence Length L')
    plt.ylabel('Time Forward / Time (Forward + Backward)')
    plt.title('Efficiency Ratio vs Sequence Length')
    plt.grid(True)
    plt.show()

    # Optional: Plot absolute times for reference
    plt.figure(figsize=(8, 5))
    plt.plot(Ls, avg_times_fwd, marker='o', label='Forward', color='blue')
    plt.plot(Ls, avg_times_fb, marker='o', label='Forward + Backward', color='orange')
    plt.xlabel('Sequence Length L')
    plt.ylabel('Average Time (s)')
    plt.title('Absolute Times vs Sequence Length')
    plt.legend()
    plt.grid(True)
    plt.show()


def apply_repetition_penalty(logits: torch.Tensor, generated: torch.Tensor,
                             penalty: float) -> torch.Tensor:
    """
    Standard HF-style repetition penalty, vectorized.
    logits:    [B, V]   (last-position logits)
    generated: [B, L]   (token ids generated so far, including prompt)
    Divides logits>0 by penalty, multiplies logits<=0 by penalty, in-place-safe.
    """
    if penalty == 1.0:
        return logits
    score = torch.gather(logits, 1, generated)
    score = torch.where(score > 0, score / penalty, score * penalty)
    logits = logits.scatter(1, generated, score)
    return logits


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0,
                      top_k: int | None = None, top_p: float = 1.0) -> torch.Tensor:
    """
    Sample one token per row from [B, V] logits with temperature, top-k, top-p.
    Returns [B, 1] long tensor.
    """
    logits = logits / max(temperature, 1e-8)

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k)
        logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = 0
        mask = remove.scatter(1, sorted_idx, remove)
        logits = logits.masked_fill(mask, float('-inf'))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def simple_ar_generate(model, prompt, max_len=100, temperature=1.0,
                       top_k=None, top_p=1.0, repetition_penalty=1.0,
                       token_end=None, verbose=False):
    """
    Dumb autoregressive generation for debugging.

    Runs a full forward pass on the whole sequence each step, takes logits at the
    last position, samples, appends. O(L^2) total — useless for big models, but it
    works on *anything* whose forward(ids) returns [B, L, V] logits, with no
    assumptions about KV caches, custom generate(), pos encoders, ve, skips, etc.
    Good as a ground-truth reference when debugging more sophisticated generators.
    """
    flag("simple_ar_generate is O(L^2) and ignores KV caches: use it for debugging, "
         "not for real inference. If your model has a fast .generate(), prefer that.", 2)
    model.eval()
    if prompt.dim() == 1:
        prompt = prompt.unsqueeze(0)
    device = next(model.parameters()).device
    generated = prompt.clone().to(device)

    for _ in range(max_len):
        logits = model(generated)
        next_logits = logits[:, -1, :]

        if repetition_penalty != 1.0:
            next_logits = apply_repetition_penalty(next_logits, generated, repetition_penalty)

        next_token = sample_next_token(next_logits, temperature, top_k, top_p)
        generated = torch.cat([generated, next_token], dim=1)

        if verbose:
            print(next_token.flatten().tolist())

        if token_end is not None and (next_token == token_end).all():
            break

    return generated


def toeplitz_init(tensor: torch.Tensor, alpha: float, causal: bool = True, mul=0.1):
    """
    Initializes a square 2D tensor to a custom Toeplitz matrix with decaying kernels
    generated from alpha^t for t=1...L, where L is the side length of the matrix.

    If causal is True, the matrix is lower triangular (tril), containing values only
    where row index i >= column index j.

    Args:
        tensor (torch.Tensor): The square 2D tensor to initialize in place.
        alpha (float): The base for the exponential decay.
        causal (bool, optional): If True, makes the matrix lower triangular. Defaults to False.

    Returns:
        torch.Tensor: The initialized tensor (modified in place).
    """
    if tensor.dim() != 2 or tensor.size(0) != tensor.size(1):
        raise ValueError("Tensor must be a square 2D tensor.")

    L = tensor.size(0)
    i, j = torch.meshgrid(
        torch.arange(L, device=tensor.device, dtype=tensor.dtype),
        torch.arange(L, device=tensor.device, dtype=tensor.dtype),
        indexing='ij'
    )
    dist = torch.abs(i - j)
    t = dist + 1
    matrix = alpha ** t

    if causal:
        matrix = torch.where(i >= j, matrix, torch.tensor(0.0, dtype=tensor.dtype, device=tensor.device))

    with torch.no_grad():
        tensor.copy_(matrix)

    return tensor * mul


def zero_layer_mtp_loss( # TODO: Vibecoded for now
        logits: torch.Tensor,
        targets: torch.Tensor,
        mtp_weights: list[float],
        ignore_index: int = -100,
        softcap_val: float = 30.0
) -> torch.Tensor:
    """
    Production-grade 0-layer MTP loss.

    Optimizations:
    1. Computes LogSumExp exactly once per token.
    2. Uses index gathering for target logits to minimize memory bandwidth.
    3. Casts to FP32 for numerical stability before reduction.
    """
    assert sum(mtp_weights) == 1, "the sum of MTP weights must be 1.0"

    if softcap_val > 0.0:
        logits = softcap_val * torch.tanh(logits / softcap_val)

    lse = torch.logsumexp(logits, dim=-1).float()

    total_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

    for k, w in enumerate(mtp_weights):
        if w <= 0.0:
            continue

        if k == 0:
            valid_logits = logits
            valid_targets = targets
            valid_lse = lse
        else:
            valid_logits = logits[:, :-k]
            valid_targets = targets[:, k:]
            valid_lse = lse[:, :-k]

        mask = (valid_targets != ignore_index)

        safe_targets = torch.where(mask, valid_targets, torch.zeros_like(valid_targets))

        target_logits = valid_logits.gather(
            dim=-1,
            index=safe_targets.unsqueeze(-1)
        ).squeeze(-1).float()

        ce_loss = valid_lse - target_logits

        masked_loss = ce_loss * mask
        num_valid_tokens = mask.sum().clamp(min=1)
        loss_k = masked_loss.sum() / num_valid_tokens

        total_loss += w * loss_k

    return total_loss


def plot_tensor_diagnostics(
        name: str,
        tensor: torch.Tensor,
        on_grad: bool = False,
        top_k_svd: int | None = None,
        bins: int = 100
):

    has_grad = on_grad and hasattr(tensor, 'grad') and tensor.grad is not None
    rows = 2 if has_grad else 1
    fig, axes = plt.subplots(rows, 2, figsize=(12, 5 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)  # standardize indexing

    def _compute_and_plot(t: torch.Tensor, row: int, title_prefix: str):
        t_cpu = t.detach().float().cpu()

        if t_cpu.ndim == 1:
            mat = t_cpu.unsqueeze(0)
        else:
            mat = t_cpu.view(t_cpu.size(0), -1)

        s_vals = torch.linalg.svdvals(mat).numpy()
        if top_k_svd is not None:
            s_vals = s_vals[:top_k_svd]

        ax_svd = axes[row, 0]
        ax_svd.plot(s_vals, marker='.', linestyle='-', color='b', markersize=4)
        ax_svd.set_title(f"{title_prefix} Singular Values\nMax: {s_vals[0]:.4f}, Min: {s_vals[-1]:.4f}")
        ax_svd.set_xlabel("Index")
        ax_svd.set_ylabel("Singular Value $\sigma_i$")
        ax_svd.set_yscale('log')  # Log scale is crucial to see the long tail
        ax_svd.grid(True, which="both", ls="--", alpha=0.5)

        ax_hist = axes[row, 1]
        vals_flat = t_cpu.numpy().flatten()
        ax_hist.hist(vals_flat, bins=bins, color='g', alpha=0.7, log=True)
        mean, std = vals_flat.mean(), vals_flat.std()
        ax_hist.set_title(f"{title_prefix} Distribution\n$\mu$: {mean:.2e}, $\sigma$: {std:.2e}")
        ax_hist.set_xlabel("Value")
        ax_hist.set_ylabel("Count (Log Scale)")
        ax_hist.grid(True, ls="--", alpha=0.5)

    _compute_and_plot(tensor, row=0, title_prefix=f"Weight: {name}")

    if has_grad:
        _compute_and_plot(tensor.grad, row=1, title_prefix=f"Gradient: {name}")
    elif on_grad:
        print(f"Warning: on_grad=True for '{name}', but tensor.grad is None.")

    plt.tight_layout()
    plt.show()


import math

def compute_normalized_grad_norm(model: torch.nn.Module) -> float:
    """
    Returns the RMS gradient norm across all trainable parameters.

    RMS = sqrt( sum(g_i^2) / N )   where N = total number of scalar gradients.

    Because this *averages* rather than sums, the result is independent of
    model size: a 1 M-param and a 100 M-param model both live in [0, ~1]
    during healthy training, making cross-run comparisons meaningful.

    Returns 0.0 if no gradients are present yet.
    """
    total_sq = 0.0
    total_n  = 0

    for p in model.parameters():
        if p.grad is not None:
            total_sq += p.grad.detach().float().pow(2).sum().item()
            total_n  += p.grad.numel()

    if total_n == 0:
        return 0.0

    return math.sqrt(total_sq / total_n)


def compute_layer_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    """
    Returns a dict { layer_name -> RMS grad norm } for every named module
    that owns at least one parameter with a gradient.

    Same RMS normalisation as compute_normalized_grad_norm so per-layer
    values are still comparable across models of different sizes.
    Leaf modules only (skips container modules to avoid double-counting).
    """
    norms = {}

    for name, module in model.named_modules():
        # only leaf modules that directly own parameters
        own_params = list(module.parameters(recurse=False))
        if not own_params:
            continue

        sq_sum = 0.0
        n      = 0
        for p in own_params:
            if p.grad is not None:
                sq_sum += p.grad.detach().float().pow(2).sum().item()
                n      += p.grad.numel()

        if n > 0:
            norms[name] = math.sqrt(sq_sum / n)

    return norms