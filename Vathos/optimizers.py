import random
import torch
import torch.optim as optim
from Vathos.functions import flag
from torch.optim import Optimizer, AdamW, SGD

try:
    from muon import Muon
except ImportError:
    flag("Muon library is not installed, usage of Muon dependent optimizer will not be possible", 2)


class ValueScheduler:
    def __init__(self):
        self.step = 0
        self.value = 0
        self.functions = {}
        self.values = []
        self.compiled = False

    def set(self, f, l=0, u=float('inf')):
        assert l < u
        self.functions[(l, u)] = f

    def compile(self, l, m):
        if self.compiled:
            flag("ValueScheduler Already Compiled, recompiling...")
        self.values = []
        for l, u in self.functions:
            i = l
            f = self.functions[(l, u)]
            while i <= u:
                self.values.append(f(i))

    def get(self, step):
        assert self.compiled
        return self.values[step]


class SignSGD(optim.Optimizer):

    def __init__(self, params, lr=0.01, rand_zero=True):
        defaults = dict(lr=lr, rand_zero=rand_zero)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            rand_zero = group['rand_zero']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = torch.sign(p.grad)

                if rand_zero:
                    zero_mask = (grad == 0)
                    if zero_mask.any():
                        grad[zero_mask] = torch.randint(
                            0, 2, (zero_mask.sum().item(),),
                            dtype=grad.dtype,
                            device=grad.device
                        ) * 2 - 1

                p.add_(grad, alpha=-lr)

        return loss


"""
stochastic_sign_muon.py
=======================
StochasticSignMuon: per-step stochastic interpolation between Muon and SignSGD
for 2-D hidden weight matrices, with AdamW for everything else.

Design principle
----------------
Both Muon and SignSGD operate on the *same* Nesterov momentum buffer:

    buf  ←  β·buf + (1-β)·grad                  # EMA (identical for both)
    pre  =  (1-β)·grad + β·buf                   # Nesterov blend (identical for both)

    Muon branch :  update = NS(pre) · scale      # orthogonalise
    Sign branch :  update = sign(pre) / √n       # sign-normalised

Because the momentum path is fully shared, τ can be freely scheduled — or even
changed every step — without resetting or corrupting any state.

Frobenius-norm alignment
------------------------
Muon's NS output has F-norm ≈ √m for an m×n matrix (m = rows).
sign(pre) has F-norm = √(m·n).
Dividing by √n makes the Sign branch produce the same F-norm as Muon,
so the learning rate carries the same meaning in both branches.

Usage (single device)
---------------------
    from stochastic_sign_muon import SingleDeviceStochasticSignMuon

    hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
    rest = (
        [p for p in model.body.parameters() if p.ndim < 2]
        + list(model.head.parameters())
        + list(model.embed.parameters())
    )

    optimizer = SingleDeviceStochasticSignMuon(
        [
            dict(params=hidden_weights, use_muon=True,  lr=0.02,  weight_decay=0.01),
            dict(params=rest,           use_muon=False, lr=3e-4,  betas=(0.9, 0.95),
                 weight_decay=0.01),
        ],
        tau=0.0,        # start as pure Muon; schedule upward to inject Sign noise
    )

    # Inside the training loop you can change tau at any time:
    optimizer.set_tau(0.3)   # 30 % of steps will now be Sign steps
"""

import torch
import torch.distributed as dist
try:
    from muon import zeropower_via_newtonschulz5, adam_update
except ImportError:
    flag("Unable to import muon, consider downloading it via pip install git+https://github.com/KellerJordan/Muon.git to use Muon backed optimizers")


# ---------------------------------------------------------------------------
# Shared update primitives
# ---------------------------------------------------------------------------

def _nesterov_momentum(
        grad: torch.Tensor,
        buf: torch.Tensor,
        beta: float,
) -> torch.Tensor:
    """
    EMA update followed by a Nesterov combination.

    Modifies `buf` and `grad` in-place (safe inside @torch.no_grad).
    Returns the Nesterov pre-update vector.
    """
    buf.lerp_(grad, 1.0 - beta)  # buf  ←  β·buf + (1-β)·grad
    return grad.lerp_(buf, beta)  # returns (1-β)·grad + β·buf  [Nesterov]


def _muon_branch(pre: torch.Tensor, ns_steps: int = 5) -> torch.Tensor:
    """
    Newton-Schulz orthogonalisation + anisotropic scaling.
    Output F-norm ≈ √m for an m×n pre-update.
    Returns a tensor of the same shape as `pre`.
    """
    u = pre.view(len(pre), -1) if pre.ndim == 4 else pre
    u = zeropower_via_newtonschulz5(u, steps=ns_steps)
    u = u * max(1.0, u.size(-2) / u.size(-1)) ** 0.5
    return u


def _sign_branch(pre: torch.Tensor, rand_zero: bool = True) -> torch.Tensor:
    """
    Signed update normalised to match Muon's output F-norm.

    sign(pre) has F-norm √(m·n). Dividing by √n yields √m, matching Muon.
    Zero entries are randomised to ±1 when rand_zero=True.
    Returns a tensor of the same shape as `pre`.
    """
    u = pre.view(len(pre), -1) if pre.ndim == 4 else pre

    s = torch.sign(u)

    if rand_zero:
        zero_mask = s == 0
        if zero_mask.any():
            s[zero_mask] = (
                    torch.randint(
                        0, 2,
                        (zero_mask.sum().item(),),
                        dtype=u.dtype,
                        device=u.device,
                    ) * 2 - 1
            )

    # Normalise: √(m·n) / √n = √m  →  same F-norm as the Muon branch
    s = s / (u.size(-1) ** 0.5)
    return s


# ---------------------------------------------------------------------------
# Single-device variant
# ---------------------------------------------------------------------------

class SingleDeviceStochasticSignMuon(torch.optim.Optimizer):
    """
    Non-distributed stochastic Muon ↔ SignSGD interpolation with AdamW aux.

    Parameters
    ----------
    param_groups : list[dict]
        Each group must contain ``use_muon`` (bool).

        use_muon=True  groups recognise:  lr, momentum, weight_decay
        use_muon=False groups recognise:  lr, betas, eps, weight_decay

    tau : float
        Probability of taking a Sign step instead of a Muon step. [0, 1].
        0.0 → pure Muon (default).  1.0 → pure SignSGD.
        Schedule via :meth:`set_tau`. No state reset required.

    sign_lr_scale : float
        Multiplicative downscale applied to lr when the Sign branch fires.
        Compensates for Sign's lower signal-to-noise ratio vs Muon at equal
        F-norm. Default 0.1; tune in [0.05, 0.2].
        Schedule via :meth:`set_sign_lr_scale`. No state reset required.
        Weight decay is always applied at the base group lr regardless.

    rand_zero : bool
        Randomise zero-gradient sign entries in the Sign branch (default True).

    nesterov : bool
        Use Nesterov momentum (default True, matching canonical Muon).

    ns_steps : int
        Newton-Schulz iteration count for the Muon branch (default 5).
    """

    def __init__(
            self,
            param_groups,
            tau: float = 0.0,
            sign_lr_scale: float = 0.1,
            rand_zero: bool = True,
            nesterov: bool = True,
            ns_steps: int = 5,
    ):
        assert 0.0 <= tau <= 1.0, f"tau must be in [0, 1], got {tau}"
        assert sign_lr_scale > 0.0, f"sign_lr_scale must be > 0, got {sign_lr_scale}"

        self.tau = tau
        self.sign_lr_scale = sign_lr_scale
        self.rand_zero = rand_zero
        self.nesterov = nesterov
        self.ns_steps = ns_steps

        for group in param_groups:
            assert "use_muon" in group, "Every param group must have a 'use_muon' key."
            if group["use_muon"]:
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0.0)
                allowed = {"params", "lr", "momentum", "weight_decay", "use_muon"}
                assert set(group.keys()) == allowed, (
                    f"Unexpected keys in Muon group: {set(group.keys()) - allowed}"
                )
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0.0)
                allowed = {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
                assert set(group.keys()) == allowed, (
                    f"Unexpected keys in Adam group: {set(group.keys()) - allowed}"
                )

        super().__init__(param_groups, {})

    # ------------------------------------------------------------------
    # Scheduling API
    # ------------------------------------------------------------------

    def set_tau(self, tau: float) -> None:
        """
        Set the Muon ↔ Sign mixing probability.
        Safe to call at any point; does not reset any optimizer state.
        """
        assert 0.0 <= tau <= 1.0, f"tau must be in [0, 1], got {tau}"
        self.tau = tau

    def set_sign_lr_scale(self, scale: float) -> None:
        """
        Set the lr multiplier for Sign steps.
        Safe to call at any point; does not reset any optimizer state.
        """
        assert scale > 0.0, f"sign_lr_scale must be > 0, got {scale}"
        self.sign_lr_scale = scale

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # One global Bernoulli draw: all 2-D params follow the same branch this step.
        use_sign: bool = (torch.rand(1).item() < self.tau)

        for group in self.param_groups:

            if group["use_muon"]:
                beta = group["momentum"]
                base_lr = group["lr"]
                effective_lr = base_lr * self.sign_lr_scale if use_sign else base_lr

                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)

                    state = self.state[p]
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    # Shared momentum step (identical regardless of branch)
                    pre = _nesterov_momentum(p.grad, state["momentum_buffer"], beta)

                    # Branch selection
                    if use_sign:
                        update = _sign_branch(pre, rand_zero=self.rand_zero)
                    else:
                        update = _muon_branch(pre, ns_steps=self.ns_steps)

                    # Weight decay at base lr; parameter update at effective lr
                    p.mul_(1.0 - base_lr * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-effective_lr)

            else:  # AdamW
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)

                    state = self.state[p]
                    if not state:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0

                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


# ---------------------------------------------------------------------------
# Distributed variant
# ---------------------------------------------------------------------------

class StochasticSignMuon(torch.optim.Optimizer):
    """
    Distributed stochastic Muon ↔ SignSGD interpolation with AdamW aux.

    Requires an initialised ``torch.distributed`` process group.
    Parameters are sorted by size and sharded across ranks exactly as in the
    original ``MuonWithAuxAdam``.

    The Bernoulli draw is made on rank 0 and broadcast to all ranks so that
    every GPU takes the same branch on every step.

    See ``SingleDeviceStochasticSignMuon`` for the full parameter docstring.
    """

    def __init__(
            self,
            param_groups,
            tau: float = 0.0,
            sign_lr_scale: float = 0.1,
            rand_zero: bool = True,
            nesterov: bool = True,
            ns_steps: int = 5,
    ):
        assert 0.0 <= tau <= 1.0, f"tau must be in [0, 1], got {tau}"
        assert sign_lr_scale > 0.0, f"sign_lr_scale must be > 0, got {sign_lr_scale}"

        self.tau = tau
        self.sign_lr_scale = sign_lr_scale
        self.rand_zero = rand_zero
        self.nesterov = nesterov
        self.ns_steps = ns_steps

        for group in param_groups:
            assert "use_muon" in group, "Every param group must have a 'use_muon' key."
            if group["use_muon"]:
                group["params"] = sorted(
                    group["params"], key=lambda x: x.size(), reverse=True
                )
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0.0)
                allowed = {"params", "lr", "momentum", "weight_decay", "use_muon"}
                assert set(group.keys()) == allowed
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0.0)
                allowed = {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
                assert set(group.keys()) == allowed

        super().__init__(param_groups, {})

    # ------------------------------------------------------------------
    # Scheduling API
    # ------------------------------------------------------------------

    def set_tau(self, tau: float) -> None:
        """Schedule the Muon ↔ Sign mixing probability. Safe from any rank."""
        assert 0.0 <= tau <= 1.0, f"tau must be in [0, 1], got {tau}"
        self.tau = tau

    def set_sign_lr_scale(self, scale: float) -> None:
        """Schedule the lr multiplier for Sign steps. Safe from any rank."""
        assert scale > 0.0, f"sign_lr_scale must be > 0, got {scale}"
        self.sign_lr_scale = scale

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # Broadcast the Bernoulli draw from rank 0 so every GPU takes the same branch.
        use_sign_t = torch.zeros(1, dtype=torch.float32,
                                 device=torch.device("cuda", rank))
        if rank == 0:
            use_sign_t[0] = 1.0 if torch.rand(1).item() < self.tau else 0.0
        dist.broadcast(use_sign_t, src=0)
        use_sign: bool = use_sign_t.item() > 0.5

        for group in self.param_groups:

            if group["use_muon"]:
                beta = group["momentum"]
                base_lr = group["lr"]
                effective_lr = base_lr * self.sign_lr_scale if use_sign else base_lr

                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (
                        world_size - len(params) % world_size
                )

                for base_i in range(0, len(params), world_size):
                    local_idx = base_i + rank
                    if local_idx < len(params):
                        p = params[local_idx]
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)

                        state = self.state[p]
                        if not state:
                            state["momentum_buffer"] = torch.zeros_like(p)

                        pre = _nesterov_momentum(
                            p.grad, state["momentum_buffer"], beta
                        )

                        if use_sign:
                            update = _sign_branch(pre, rand_zero=self.rand_zero)
                        else:
                            update = _muon_branch(pre, ns_steps=self.ns_steps)

                        # Weight decay at base lr; parameter update at effective lr
                        p.mul_(1.0 - base_lr * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-effective_lr)

                    dist.all_gather(
                        params_pad[base_i: base_i + world_size],
                        params_pad[base_i + rank],
                    )

            else:  # AdamW
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)

                    state = self.state[p]
                    if not state:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0

                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
