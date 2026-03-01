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


class StochasticMuonWithAuxAdam(Optimizer):
    def __init__(self, param_groups, alpha=0.0, muon_momentum=0.95, sgd_lr_scale=0.1):
        """
        Args:
            param_groups: Standard PyTorch param_groups dict list.
            alpha: Probability of taking an SGD step instead of MUON.
            muon_momentum: Shared momentum value.
            sgd_lr_scale: Multiplier for SGD's learning rate relative to MUON's scheduled LR.
        """
        self.alpha = alpha
        self.sgd_lr_scale = sgd_lr_scale

        muon_groups = []
        adam_groups = []

        # 1. Parse user groups
        for group in param_groups:
            group_copy = {k: v for k, v in group.items()}
            use_muon = group_copy.pop('use_muon', False)

            if use_muon:
                group_copy.setdefault('momentum', muon_momentum)
                muon_groups.append(group_copy)
            else:
                adam_groups.append(group_copy)

        # 2. Init wrapper optimizer
        defaults = dict(lr=1e-3)
        super().__init__(muon_groups + adam_groups, defaults)

        # 3. Re-slice references for the master scheduler
        self.muon_groups = self.param_groups[:len(muon_groups)]
        self.adam_groups = self.param_groups[len(muon_groups):]

        # 4. Initialize Internal Optimizers
        if self.muon_groups:
            # Safely check if Muon was imported or defined globally
            try:
                Muon
            except NameError:
                raise ImportError(
                    "The 'Muon' class is not defined. Please ensure the 'muon' package "
                    "is installed in your environment or the class is defined in your script."
                )

            # FIX: Muon's __init__ strictly asserts a list of parameter tensors, not dicts.
            # We extract them into a flat list to satisfy the assertion.
            flat_muon_params = []
            for g in self.muon_groups:
                flat_muon_params.extend(g['params'])

            init_lr = self.muon_groups[0].get('lr', 0.02)
            init_momentum = self.muon_groups[0].get('momentum', muon_momentum)
            init_wd = self.muon_groups[0].get('weight_decay', 0.0)

            self.muon = Muon(flat_muon_params, lr=init_lr, momentum=init_momentum, weight_decay=init_wd)

            # CRITICAL: Re-bind the param_groups to our master wrapper's groups.
            # This ensures that when LambdaLR updates the wrapper, Muon sees the exact same updated LRs!
            self.muon.param_groups = self.muon_groups

            # Give SGD its own distinct copy of the dictionaries.
            # We copy the keys, but the 'params' list still points to the EXACT SAME tensor objects.
            self.sgd_groups = [
                {'params': g['params'], **{k: v for k, v in g.items() if k != 'params'}}
                for g in self.muon_groups
            ]

            self.sgd = SGD(self.sgd_groups, lr=0.0)

            # Bind states together. Because the tensor objects are identical in memory,
            # they hash to the exact same momentum buffers in this master dictionary.
            self.muon.state = self.state
            self.sgd.state = self.state
        else:
            self.muon, self.sgd = None, None

        if self.adam_groups:
            self.adam = AdamW(self.adam_groups, lr=0.0)
            self.adam.state = self.state
        else:
            self.adam = None

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if self.adam is not None:
            self.adam.step()

        if self.muon is not None and self.sgd is not None:
            if random.random() < self.alpha:
                for mg, sg in zip(self.muon_groups, self.sgd_groups):
                    sg['lr'] = mg['lr'] * self.sgd_lr_scale
                self.sgd.step()
            else:
                self.muon.step()

        return loss

    def update_alpha(self, new_alpha):
        self.alpha = max(0.0, min(1.0, new_alpha))

    def update_sgd_scale(self, new_scale):
        """Optional: Dynamically adjust the SGD LR ratio during training."""
        self.sgd_lr_scale = max(0.0, new_scale)


import torch
from torch.optim import Optimizer


class InterpolatedMuon(Optimizer):
    """
    Ottimizzatore ibrido che interpola tra Muon (proiezione ortogonale) e SGD-Momentum.
    Implementa esplicitamente la curva spettrale D = beta * I + (1 - beta) * Sigma.
    """

    def __init__(self, params, lr=1e-3, momentum=0.9, beta_interp=0.8, weight_decay=0.01):
        if not 0.0 <= beta_interp <= 1.0:
            raise ValueError(f"beta_interp deve essere in [0, 1], ottenuto {beta_interp}")

        defaults = dict(lr=lr, momentum=momentum, beta_interp=beta_interp, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            beta = group['beta_interp']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)

                buf = state['momentum_buffer']

                if wd != 0:
                    p.mul_(1 - lr * wd)

                buf.mul_(momentum).add_(grad, alpha=1 - momentum)

                if p.ndim >= 2:
                    shape = buf.shape
                    M_2d = buf.view(shape[0], -1)

                    O_t = self._newton_schulz_accelerated(M_2d)
                    update_2d = beta * O_t + (1 - beta) * M_2d

                    update = update_2d.view(shape)
                else:
                    update = buf

                p.add_(update, alpha=-lr)

    @staticmethod
    @torch.compile
    def _newton_schulz_accelerated(G, steps=5):
        a, b, c = 3.4445, -4.7750, 2.0315

        X = G.to(torch.bfloat16) if G.dtype == torch.float16 else G.clone()

        X /= (X.norm() + 1e-7)

        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X

        return X.to(G.dtype)
