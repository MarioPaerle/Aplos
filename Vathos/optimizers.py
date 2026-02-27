import random
import torch
import torch.optim as optim
from Vathos.functions import flag
from torch.optim import Optimizer, AdamW, SGD


try:
    from muon import Muon
except ImportError:
    flag("Muon library is not installed, usage of Muon dependent optimizer will not be possible", 2)


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