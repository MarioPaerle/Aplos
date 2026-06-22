"""
Triton-fused two-layer MLPs in the style of `VariableUDLP`:
    out = (act(x @ W1.T))^2 @ W2.T          (ReLU^2, LeakyReLU(0.5)^2)
    out = (silu(a) * b) @ W2.T               (SwiGLU)

The fused ReLU^2 and LeakyReLU^2 kernels are ports of:
  - modded-nanogpt   linear_relu_square        (Keller Jordan et al.)
  - parameter-golf   linear_leaky_relu_square  (samacqua / openai/parameter-golf)

Both use the Hopper TensorDescriptor (TMA) API, so they need:
  - triton >= 3.2
  - GPU compute capability >= 9.0 (H100 / GH200)
On older GPUs / when triton is missing, the wrapper modules transparently fall
back to a pure-torch path that is `torch.compile`-friendly.

The autograd.Function wrappers are accepted natively by `torch.compile` (>=2.0):
the compiler treats `.apply(...)` as an opaque op and traces around it.

API: each module mirrors `VariableUDLP(d_model, d_output, M, dropout)`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from Vathos._basics import Layer
from Vathos.functions import flag

try:
    import triton
    import triton.language as tl
    from triton.tools.tensor_descriptor import TensorDescriptor
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


def _hopper_available() -> bool:
    if not (_HAS_TRITON and torch.cuda.is_available()):
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


# =============================================================================
# Triton kernels
# =============================================================================

if _HAS_TRITON:

    @triton.jit
    def _selective_log_softmax_kernel(
        logits_ptr, index_ptr, out_ptr,
        N: tl.constexpr,
        V: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """One program per row: out[row] = logits[row, index] - logsumexp(row)."""
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_V)
        mask = offs < V
        vals = tl.load(logits_ptr + row * V + offs, mask=mask, other=-float("inf")).to(tl.float32)
        max_val = tl.max(vals, axis=0)
        lse = tl.log(tl.sum(tl.exp(vals - max_val), axis=0)) + max_val
        target = tl.load(index_ptr + row)
        chosen = tl.load(logits_ptr + row * V + target).to(tl.float32)
        tl.store(out_ptr + row, chosen - lse)

    @triton.jit
    def _linear_relu_square_kernel(
        a_desc, b_desc, c_desc, aux_desc,
        M, N, K,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        NUM_SMS: tl.constexpr,
        FORWARD: tl.constexpr,
    ):
        """Port of modded-nanogpt's linear_relu_square_kernel.

        FORWARD:  computes c = (x @ W1.T) and aux = relu(c)^2  (c is the pre-activation)
        BACKWARD: computes c = grad_pre = grad_post * 2 * relu(pre), reading pre from aux
                  (caller passes aux=pre, grad_post is the input matmul result)
        """
        dtype = tl.bfloat16
        start_pid = tl.program_id(axis=0)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_tiles = num_pid_m * num_pid_n
        k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

        for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for ki in range(k_tiles):
                offs_k = ki * BLOCK_SIZE_K
                a = a_desc.load([offs_am, offs_k])
                b = b_desc.load([offs_bn, offs_k])
                accumulator = tl.dot(a, b.T, accumulator)

            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)

            c0 = acc0.to(dtype)
            if not FORWARD:
                pre0 = aux_desc.load([offs_am, offs_bn])
                c0 = 2 * c0 * tl.where(pre0 > 0, pre0, 0)
            c_desc.store([offs_am, offs_bn], c0)
            if FORWARD:
                p0 = tl.maximum(c0, 0)
                aux_desc.store([offs_am, offs_bn], p0 * p0)

            c1 = acc1.to(dtype)
            if not FORWARD:
                pre1 = aux_desc.load([offs_am, offs_bn + BLOCK_SIZE_N // 2])
                c1 = 2 * c1 * tl.where(pre1 > 0, pre1, 0)
            c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)
            if FORWARD:
                p1 = tl.maximum(c1, 0)
                aux_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], p1 * p1)


    @triton.jit
    def _linear_leaky_relu_square_kernel(
        a_desc, b_desc, c_desc, aux_desc,
        M, N, K,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        NUM_SMS: tl.constexpr,
        FORWARD: tl.constexpr,
    ):
        """Port of parameter-golf's linear_leaky_relu_square_kernel.
        Uses LeakyReLU with negative slope 0.5, then squares.
        Forward: aux = lrelu(c)^2,    Backward: c = grad_post * 2 * (lrelu'(pre) * lrelu(pre))
                                                = grad_post * (where(pre>0, 2*pre, 0.5*pre))
        """
        dtype = tl.bfloat16
        start_pid = tl.program_id(axis=0)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_tiles = num_pid_m * num_pid_n
        k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

        for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for ki in range(k_tiles):
                offs_k = ki * BLOCK_SIZE_K
                a = a_desc.load([offs_am, offs_k])
                b = b_desc.load([offs_bn, offs_k])
                accumulator = tl.dot(a, b.T, accumulator)

            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)

            c0 = acc0.to(dtype)
            c1 = acc1.to(dtype)
            if not FORWARD:
                pre0 = aux_desc.load([offs_am, offs_bn])
                pre1 = aux_desc.load([offs_am, offs_bn + BLOCK_SIZE_N // 2])
                c0 = c0 * tl.where(pre0 > 0, 2.0 * pre0, 0.5 * pre0)
                c1 = c1 * tl.where(pre1 > 0, 2.0 * pre1, 0.5 * pre1)
            c_desc.store([offs_am, offs_bn], c0)
            c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)
            if FORWARD:
                aux0 = tl.where(c0 > 0, c0, 0.5 * c0)
                aux1 = tl.where(c1 > 0, c1, 0.5 * c1)
                aux_desc.store([offs_am, offs_bn], aux0 * aux0)
                aux_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], aux1 * aux1)


# -----------------------------------------------------------------------------
# T4-friendly LeakyReLU(0.5)^2 kernel (no TMA, fp16, smaller tiles).
# Designed for sm_75 (Turing, T4): pointer-based loads, BLOCK_M=BLOCK_N=64,
# BLOCK_K=32, fp16 inputs/outputs, fp32 accumulator. Same fwd/bwd convention
# as the Hopper kernels: forward stores c=pre and aux=post; backward reads
# pre from aux and writes grad_pre into c. Output dtype follows the inputs
# (cast to fp32 fails tl.dot on Turing — keep them fp16).
# -----------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _linear_lrelu2_kernel_t4(
        a_ptr, b_ptr, c_ptr, aux_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        FORWARD: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_remaining = K - k * BLOCK_K
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
            b_mask = (offs_n[:, None] < N) & (offs_k[None, :] < k_remaining)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc += tl.dot(a, b.T)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        out_dtype = c_ptr.dtype.element_ty
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        aux_ptrs = aux_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

        if FORWARD:
            # acc here is pre = x @ W1.T. Save pre into c, post = lrelu²(pre) into aux.
            tl.store(c_ptrs, acc.to(out_dtype), mask=mask)
            scale = tl.where(acc > 0, 1.0, 0.25)   # act² = z² * (1 if z>0 else 0.25)
            post = acc * acc * scale
            tl.store(aux_ptrs, post.to(out_dtype), mask=mask)
        else:
            # acc here is grad_post @ W2 (b=W2.T). Multiply elementwise by
            # d(lrelu²)/dpre = where(pre>0, 2*pre, 0.5*pre) using saved pre in aux.
            pre = tl.load(aux_ptrs, mask=mask, other=0.0).to(tl.float32)
            deriv = tl.where(pre > 0, 2.0 * pre, 0.5 * pre)
            tl.store(c_ptrs, (acc * deriv).to(out_dtype), mask=mask)


def _launch_lrelu2_t4(a: torch.Tensor, b: torch.Tensor, aux: torch.Tensor | None):
    """Pointer-based kernel launcher for T4. Returns (pre, post) in fwd, grad_pre in bwd."""
    M, K = a.shape
    N, K2 = b.shape
    assert K == K2

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    forward = aux is None
    if aux is None:
        aux = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _linear_lrelu2_kernel_t4[grid](
        a, b, c, aux,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=8,
        FORWARD=forward,
        num_warps=4,
        num_stages=2,
    )
    return (c, aux) if forward else c


class _FusedLinearLReLU2_T4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W1, W2):
        x_flat = x.reshape(-1, x.shape[-1])
        pre, post = _launch_lrelu2_t4(x_flat, W1, None)
        out = post @ W2.T
        ctx.save_for_backward(x, W1, W2, pre, post)
        return out.view(*x.shape[:-1], W2.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        x, W1, W2, pre, post = ctx.saved_tensors
        x_flat = x.reshape(-1, x.shape[-1])
        go = grad_out.reshape(-1, W2.shape[0])
        dW2 = go.T @ post
        dpre = _launch_lrelu2_t4(go, W2.T.contiguous(), pre)
        dW1 = dpre.T @ x_flat
        dx = dpre @ W1
        return dx.view_as(x), dW1, dW2


def _can_use_triton_t4(x) -> bool:
    """T4 (CC 7.5) requires fp16 inputs."""
    if not (_HAS_TRITON and torch.cuda.is_available()):
        return False
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (7, 5):
        return False
    return x.dtype == torch.float16


def _can_use_triton_ampere(x) -> bool:
    """A100/Ampere path for the pointer-based fused LeakyReLU² kernel."""
    if not (_HAS_TRITON and torch.cuda.is_available()):
        return False
    major, _ = torch.cuda.get_device_capability()
    if major != 8:
        return False
    return x.dtype in (torch.bfloat16, torch.float16)


# =============================================================================
# Host-side launchers (return both pre-activation and post-activation in fwd)
# =============================================================================

def _launch_fused_act_square(kernel_fn, a: torch.Tensor, b: torch.Tensor,
                             aux: torch.Tensor | None):
    """Shared launcher for both relu^2 and leaky_relu^2 kernels.
    Forward (aux=None): returns (pre, post) = (x@W.T, act(pre)^2)
    Backward (aux=pre): returns dpre = grad_post * d(act(pre)^2)/dpre
    """
    M, K = a.shape
    N, K2 = b.shape
    assert K == K2

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    forward = aux is None
    if aux is None:
        aux = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 256, 64
    num_stages = 4 if forward else 3

    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = TensorDescriptor.from_tensor(c, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])
    aux_desc = TensorDescriptor.from_tensor(aux, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])

    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    grid = lambda _meta: (
        min(num_sms, triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)),
    )
    kernel_fn[grid](
        a_desc, b_desc, c_desc, aux_desc,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        NUM_SMS=num_sms,
        FORWARD=forward,
        num_stages=num_stages,
        num_warps=8,
    )
    return (c, aux) if forward else c


# =============================================================================
# Autograd Functions — composes the fused kernel with the second matmul.
# Compatible with torch.compile (autograd.Function is opaque to dynamo).
# =============================================================================

class _FusedLinearReLU2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W1, W2):
        x_flat = x.reshape(-1, x.shape[-1])
        pre, post = _launch_fused_act_square(_linear_relu_square_kernel, x_flat, W1, None)
        out = post @ W2.T
        ctx.save_for_backward(x, W1, W2, pre, post)
        return out.view(*x.shape[:-1], W2.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        x, W1, W2, pre, post = ctx.saved_tensors
        x_flat = x.reshape(-1, x.shape[-1])
        go = grad_out.reshape(-1, W2.shape[0])
        dW2 = go.T @ post
        dpre = _launch_fused_act_square(_linear_relu_square_kernel, go, W2.T.contiguous(), pre)
        dW1 = dpre.T @ x_flat
        dx = dpre @ W1
        return dx.view_as(x), dW1, dW2


class _FusedLinearLReLU2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W1, W2):
        x_flat = x.reshape(-1, x.shape[-1])
        pre, post = _launch_fused_act_square(_linear_leaky_relu_square_kernel, x_flat, W1, None)
        out = post @ W2.T
        ctx.save_for_backward(x, W1, W2, pre, post)
        return out.view(*x.shape[:-1], W2.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        x, W1, W2, pre, post = ctx.saved_tensors
        x_flat = x.reshape(-1, x.shape[-1])
        go = grad_out.reshape(-1, W2.shape[0])
        dW2 = go.T @ post
        dpre = _launch_fused_act_square(_linear_leaky_relu_square_kernel, go, W2.T.contiguous(), pre)
        dW1 = dpre.T @ x_flat
        dx = dpre @ W1
        return dx.view_as(x), dW1, dW2


# =============================================================================
# Pure-torch fallbacks (torch.compile-friendly)
# =============================================================================

def _eager_relu2_mlp(x, W1, W2):
    return F.relu(F.linear(x, W1)).square() @ W2.T

def _eager_lrelu2_mlp(x, W1, W2):
    return F.leaky_relu(F.linear(x, W1), 0.5).square() @ W2.T


def _can_use_triton_fused(x, M, N) -> bool:
    """The TMA kernels require shapes divisible by their tile sizes and a Hopper GPU."""
    if not _hopper_available():
        return False
    if x.dtype not in (torch.bfloat16, torch.float16):
        return False
    return (M % 128 == 0) and (N % 256 == 0)


def triton_selective_log_softmax(logits: torch.Tensor,
                                 index: torch.Tensor) -> torch.Tensor | None:
    """Fused no-grad selective log-softmax.

    Computes ``logits[..., index] - logsumexp(logits, dim=-1)`` without launching
    separate reduction and gather kernels. This is intended for GRPO rollout
    logging under ``torch.no_grad()``. It deliberately returns ``None`` when grad
    is enabled so callers can keep the differentiable torch implementation for
    policy/reference scoring.
    """
    if not (_HAS_TRITON and logits.is_cuda):
        return None
    if torch.is_grad_enabled() or logits.requires_grad:
        return None
    if logits.dim() < 2:
        return None

    V = logits.shape[-1]
    if V <= 0 or V > 131072:
        return None

    logits_2d = logits.reshape(-1, V).contiguous()
    index_1d = index.reshape(-1).contiguous()
    if index_1d.numel() != logits_2d.shape[0]:
        raise ValueError("index leading shape must match logits leading shape")

    out = torch.empty((logits_2d.shape[0],), device=logits.device, dtype=torch.float32)
    block_v = triton.next_power_of_2(V)
    _selective_log_softmax_kernel[(logits_2d.shape[0],)](
        logits_2d,
        index_1d,
        out,
        N=logits_2d.shape[0],
        V=V,
        BLOCK_V=block_v,
        num_warps=8,
    )
    return out.view(index.shape)


# =============================================================================
# VariableUDLP-compatible nn.Modules
# =============================================================================

class _BaseTritonUDLP(Layer):
    """Shared structure: expand (d_model→M, no bias) -> fused act² -> contract (M→d_output, no bias).

    Matches `VariableUDLP(d_model, d_output, M, dropout)` so that it can be passed
    as the `UDLP=` argument of `ModdedFormer` without further changes. The
    `activation` parameter accepted by `VariableUDLP` is ignored here — the
    activation is fixed by the chosen kernel (subclass).
    """

    _autograd_fn = None
    _eager_fn = None
    _name = "TritonUDLP"

    def __init__(self, d_model: int, d_output: int, M: int,
                 activation=None, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_output = d_output
        self.M = M
        self.expand = nn.Linear(d_model, M, bias=False)
        self.contract = nn.Linear(M, d_output, bias=False)
        self.dropout = nn.Dropout(dropout)
        if activation is not None:
            flag(f"{self._name}: 'activation' arg ignored — kernel hardcodes the activation.", 3)

    def _init_weights(self):
        torch.nn.init.zeros_(self.contract.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W1 = self.expand.weight       # [M, d_model]
        W2 = self.contract.weight     # [d_output, M]
        if _can_use_triton_fused(x, self.M, self.d_output):
            out = self._autograd_fn.apply(x, W1, W2)
        else:
            out = self._eager_fn(x, W1, W2)
        return self.dropout(out)


class TritonReLU2UDLP(_BaseTritonUDLP):
    """Fused ReLU² UDLP. Drop-in for VariableUDLP with activation=ReLU2."""
    _autograd_fn = _FusedLinearReLU2 if _HAS_TRITON else None
    _eager_fn = staticmethod(_eager_relu2_mlp)
    _name = "TritonReLU2UDLP"


class TritonLReLU2UDLP(_BaseTritonUDLP):
    """Fused LeakyReLU(0.5)² UDLP. Drop-in for VariableUDLP with activation=LeakyReLU2."""
    _autograd_fn = _FusedLinearLReLU2 if _HAS_TRITON else None
    _eager_fn = staticmethod(_eager_lrelu2_mlp)
    _name = "TritonLReLU2UDLP"


class TritonLReLU2UDLP_T4(_BaseTritonUDLP):
    """LeakyReLU(0.5)² UDLP tuned for T4 / Turing (sm_75).

    Uses a non-TMA fp16 kernel with 64×64×32 tiles. Falls back to the eager
    torch path on non-T4 GPUs or when triton is missing.

    NOTE: weights are kept fp32 by Linear; cast inputs to fp16 before forward
    (e.g. `model = model.half()` or `with torch.autocast('cuda', dtype=torch.float16)`).
    """
    _autograd_fn = _FusedLinearLReLU2_T4 if _HAS_TRITON else None
    _eager_fn = staticmethod(_eager_lrelu2_mlp)
    _name = "TritonLReLU2UDLP_T4"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W1 = self.expand.weight
        W2 = self.contract.weight
        if _can_use_triton_t4(x):
            out = self._autograd_fn.apply(x, W1, W2)
        else:
            out = self._eager_fn(x, W1, W2)
        return self.dropout(out)


class TritonLReLU2UDLP_A100(_BaseTritonUDLP):
    """LeakyReLU(0.5)² UDLP tuned for A100 / Ampere (sm_80).

    Uses the pointer-based Triton kernel with arbitrary-shape masking, so it does
    not need Hopper TMA. Supports fp16 and bf16. Falls back to the eager torch
    path on non-Ampere GPUs or when Triton is missing.
    """
    _autograd_fn = _FusedLinearLReLU2_T4 if _HAS_TRITON else None
    _eager_fn = staticmethod(_eager_lrelu2_mlp)
    _name = "TritonLReLU2UDLP_A100"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W1 = self.expand.weight
        W2 = self.contract.weight
        if _can_use_triton_ampere(x):
            out = self._autograd_fn.apply(x, W1, W2)
        else:
            out = self._eager_fn(x, W1, W2)
        return self.dropout(out)


# -----------------------------------------------------------------------------
# SwiGLU — torch-only path (compile-friendly).
#
# SwiGLU is gated, so the expand projects to 2*M and the activation consumes
# both halves: silu(a) * b. There is no clean port of this in modded-nanogpt
# or parameter-golf; Liger-Kernel implements a full fused fwd+bwd for it but
# that's a separate dependency. The torch path below compiles cleanly and is
# usually within ~10% of a custom kernel for the typical M sizes we use.
# -----------------------------------------------------------------------------

class TritonSwiGLUUDLP(Layer):
    """SwiGLU UDLP, drop-in for VariableUDLP with activation=SwiGLU.
    Compile-friendly torch implementation — no custom kernel.
    `M` is the post-gate inner dimension (i.e. expand projects to 2*M).
    """

    def __init__(self, d_model: int, d_output: int, M: int,
                 activation=None, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_output = d_output
        self.M = M
        self.expand = nn.Linear(d_model, 2 * M, bias=False)
        self.contract = nn.Linear(M, d_output, bias=False)
        self.dropout = nn.Dropout(dropout)
        if activation is not None:
            flag("TritonSwiGLUUDLP: 'activation' arg ignored — gate is fixed to SiLU.", 3)

    def _init_weights(self):
        torch.nn.init.zeros_(self.contract.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.expand(x)
        a, b = h.chunk(2, dim=-1)
        return self.dropout(self.contract(F.silu(a) * b))


__all__ = [
    "TritonReLU2UDLP",
    "TritonLReLU2UDLP",
    "TritonLReLU2UDLP_A100",
    "TritonLReLU2UDLP_T4",
    "TritonSwiGLUUDLP",
    "triton_selective_log_softmax",
]
