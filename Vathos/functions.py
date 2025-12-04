import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time

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

    return torch.cumsum(x / alpha_pow, dim=1) * alpha_pow * (alpha_pow[:, 0, ...] if rescale else 1)


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

