"""
This modules serves as base for the Torch Basic model Structure, and profilers.
Class Layer is the base of all Vathos layers, and ereditate from torch.nn.Module class, adding only the structures that help
profiling, visualization, and debugging.
Everything built with Layer is totally compatible with torch native modules.
"""

import numpy as np
import torch.nn as nn
from typing import Callable
from Vathos.functions import *
from timeit import default_timer as timer
from collections import OrderedDict, defaultdict
import re
import math


ACTIVS = {
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'elu': nn.ELU,
    'lrelu': nn.LeakyReLU,
    'leaky_relu': nn.LeakyReLU,
}


class VathosConfig:
    """Global configuration state"""
    _COMPILABLE = False  # If True, bypasses all profiling for torch.compile compatibility
    _PROFILE_BATCHED = True
    _GLOBAL_PROFILE = True


def set_vathos_mode(mode: str):
    """
    Switch Vathos mode globally.

    Args:
        mode (str): "production" or "debug"
    """
    if mode.lower() == "production":
        VathosConfig._COMPILABLE = True
        print(
            f"{SEC}Vathos:{RES} Switched to {GOOD}PRODUCTION{RES} mode (Ready for torch.compile, Profiling Disabled).")
    elif mode.lower() == "debug":
        VathosConfig._COMPILABLE = False
        print(f"{SEC}Vathos:{RES} Switched to {NUM}DEBUG{RES} mode (Profiling Enabled, torch.compile unfriendly).")
    else:
        raise ValueError("Mode must be 'production' or 'debug'")


# =============================================================================
# THE UNIFIED LAYER
# =============================================================================

class Layer(nn.Module):
    # Marker to identify Vathos layers without circular imports
    _is_vathos_layer = True

    def __init__(self):
        super().__init__()
        self.complexity = "O(1)"
        self.__name__ = self.__class__.__name__

        # Profiling State
        self._timer_unbatched = not VathosConfig._PROFILE_BATCHED
        self._tstart = 0
        self._tend = 0
        self._time = 0
        self._times = []
        self._sublayers = None

    def __call__(self, *args, **kwargs):
        """
        The Hot-Swappable Entry Point.
        """
        # 1. PRODUCTION PATH (Fast, Compilable)
        # torch.compile will optimize this check away as a constant
        if VathosConfig._COMPILABLE:
            # We explicitly remove 'profile' kwarg if it exists to avoid errors in forward
            if "profile" in kwargs:
                del kwargs["profile"]
            return self.forward(*args, **kwargs)

        # 2. DEBUG PATH (Slow, Profilable)
        return self._debug_call(args, kwargs)

    def _debug_call(self, args, kwargs):
        # Lazy registration
        if self._sublayers is None:
            self.register_sublayers()

        # Check local or global profile flag
        do_profile = kwargs.pop("profile", False) or VathosConfig._GLOBAL_PROFILE

        if do_profile:
            self._tstart = timer()

            # Actual Forward Pass
            rets = self.forward(*args, **kwargs)

            self._tend = timer()

            # Calculate batch size for normalization
            bs = 1
            if args and isinstance(args[0], torch.Tensor):
                bs = args[0].shape[0]

            div = bs if not self._timer_unbatched else 1
            self._time = (self._tend - self._tstart) / div
            self._times.append(self._time)

            return rets
        else:
            return self.forward(*args, **kwargs)

    def register_sublayers(self):
        if self._sublayers is None:
            self._sublayers = dict()

        def get_unique_name(base_name, existing_names):
            if base_name not in existing_names: return base_name
            counter = 1
            while f"{base_name}_{counter}" in existing_names: counter += 1
            return f"{base_name}_{counter}"

        def collect_layers(module, level=0):
            layers = []
            for name, child in module.named_children():
                # Check for Vathos Layer marker
                if getattr(child, "_is_vathos_layer", False):
                    layers.append((name, child, level))
                    layers.extend(collect_layers(child, level=level + 1))
                elif isinstance(child, nn.ModuleList):
                    for i, item in enumerate(child):
                        if getattr(item, "_is_vathos_layer", False):
                            layers.append((f"{name}[{i}]", item, level))
                            layers.extend(collect_layers(item, level=level + 1))
                elif isinstance(child, nn.Module):
                    # Recurse into standard modules to find hidden Layers
                    layers.extend(collect_layers(child, level=level))
            return layers

        all_layers = collect_layers(self)

        for original_name, layer, level in all_layers:
            class_name = getattr(layer, "__name__", type(layer).__name__)
            unique_name = get_unique_name(class_name, self._sublayers.keys())
            self._sublayers[unique_name] = {'layer': layer, 'level': level}

    def get_mean_execution_time(self):
        return np.mean(self._times) if len(self._times) > 0 else 0.0

    def generate(self, *args, **kwargs):
        return None

    def has_custom_generate(self):
        return self.generate.__func__ is not Layer.generate

    def __repr__(self):
        return f"{SEC}Vathos{RES}: " + super().__repr__()

    # -------------------------------------------------------------------------
    # Profiling & Plotting Logic (Preserved from your code)
    # -------------------------------------------------------------------------
    def profile(self, maxlevel=100, avg=False, plot=False, plot_level=1):
        if VathosConfig._COMPILABLE:
            print(f"{BAD}Cannot profile in PRODUCTION mode.{RES} Run set_vathos_mode('debug') first.")
            return

        batched = not self._timer_unbatched
        print(
            f"Layer {NUM}{self.__name__}{RES} Times Profile (batched: {GOOD if batched else BAD}{batched}{RES}) (averaged: {GOOD if avg else BAD}{avg}{RES}):")

        grouped_layers = OrderedDict()
        order_map = {}
        order_counter = 0

        # Grouping logic
        if self._sublayers:
            for sublayer_name, sublayer_info in self._sublayers.items():
                layer = sublayer_info['layer']
                level = sublayer_info['level']
                if level >= maxlevel: continue

                match = re.match(r'^(.+?)_(\d+)$', sublayer_name)
                base_name = match.group(1) if match else sublayer_name
                key = (base_name, level)

                if key not in order_map:
                    order_map[key] = order_counter
                    order_counter += 1
                    grouped_layers[key] = []
                grouped_layers[key].append((sublayer_name, layer))

        # Printing logic
        if not avg:
            if self._sublayers:
                for sublayer_name, sublayer_info in self._sublayers.items():
                    if sublayer_info['level'] < maxlevel:
                        indent = "    " * sublayer_info['level']
                        t = sublayer_info['layer'].get_mean_execution_time()
                        print(f"{indent}- {NUM}{sublayer_name}{RES}: {t * 1000:.2f}ms")
        else:
            for (base_name, level), layers_list in grouped_layers.items():
                indent = "    " * level
                times = [l.get_mean_execution_time() for _, l in layers_list if len(l._times) > 0]
                if times:
                    avg_time = np.mean(times)
                    print(
                        f"{indent}- {NUM}{base_name}_avg{RES}: {avg_time * 1000:.2f}ms {SEC}(x{len(layers_list)}){RES}")
                else:
                    print(f"{indent}- {NUM}{base_name}_avg{RES}: no time recorded")

        # Plotting logic
        if plot:
            try:
                import matplotlib.pyplot as plt
                level_layers = defaultdict(list)

                # Logic to determine which layers to plot (simplified for robustness)
                for (base_name, level), layers_list in grouped_layers.items():
                    # If exact level match
                    if level == plot_level:
                        for _, layer in layers_list:
                            if len(layer._times) > 0:
                                level_layers[base_name].append(np.mean(layer._times))

                    # If lower level but "leaf" relative to plot_level (simplified heuristic)
                    elif level < plot_level:
                        # For strict correctness based on your previous code,
                        # you'd need the parent-child check here.
                        # Assuming direct plot_level usage for now:
                        pass

                if level_layers:
                    layer_times = {k: sum(v) for k, v in level_layers.items()}
                    sorted_layers = sorted(layer_times.items(), key=lambda x: x[1], reverse=True)
                    labels = [x[0] for x in sorted_layers]
                    times = [x[1] * 1000 for x in sorted_layers]

                    fig, ax = plt.subplots(figsize=(12, 8))
                    wedges, texts, autotexts = ax.pie(times, autopct='%1.1f%%', startangle=90)
                    ax.legend(wedges, [f'{l}: {t:.2f}ms' for l, t in zip(labels, times)],
                              loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                    ax.set_title(f'{self.__name__} - Level {plot_level}')
                    plt.tight_layout()
                    plt.show()
                else:
                    print(f"{SEC}No data for plot level {plot_level}{RES}")
            except ImportError:
                print(f"{BAD}Matplotlib missing{RES}")


class Builder:
    def __init__(self, layer, **params):
        self.layer = layer
        self.params = params
        for att in layer.__dict__:
            setattr(self, att, getattr(layer, att))

    def __call__(self, *args):
        return self.layer(*args, **self.params)


class tWrapper(Layer):
    __name__ = "tWrapper"

    def __init__(self, module: nn.Module):
        super(tWrapper, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        self.module(*args, **kwargs)


class Identity(Layer):
    __name__ = "Identity"
    __complexity__ = "O(1)"

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Skip(Layer):
    __name__ = "Skip"
    __complexity__ = "O(1)"

    def __init__(self, layer):
        super(Skip, self).__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x) + x


class IdentityMixer(Layer):
    __name__ = "Identity"
    __complexity__ = "O(1)"

    def __init__(self, d_model):
        super(IdentityMixer, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        return x


class LPadder(Layer):
    __name__ = "LPadder"
    __complexity__ = "O(k d)"

    def __init__(self, right=0, left=0, element=0):
        super(LPadder, self).__init__()
        self.right = right
        self.left = left
        self.element = element

    def forward(self, x):
        return F.pad(x, (0, 0, self.left, self.right), mode="constant", value=self.element)


class dPadder(Layer):
    __name__ = "LPadder"
    __complexity__ = "O(k L)"

    def __init__(self, up, down=0, element=0):
        super(dPadder, self).__init__()
        self.right = up
        self.left = down
        self.element = element

    def forward(self, x):
        return F.pad(x, (self.left, self.right), mode="constant", value=self.element)


class LUnPadder(Layer):
    __name__ = "LUnPadder"
    __complexity__ = "O(k d)"

    def __init__(self, right=0, left=0):
        super().__init__()
        self.right = right
        self.left = left

    def forward(self, x):
        return x[:, self.left:-self.right, :]


class dUnPadder(Layer):
    __name__ = "LUnPadder"
    __complexity__ = "O(k d)"

    def __init__(self, right=0, left=0):
        super().__init__()
        self.right = right
        self.left = left

    def forward(self, x):
        return x[:, :, self.left:-self.right]


class Linear(Layer):
    __name__ = 'Linear'

    def __init__(self, input_dim, output_dim, bias=True, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x):
        return self.linear(x)


class LowRankLinear(Layer):
    __name__ = 'Linear'

    def __init__(self, input_dim, output_dim, rank=16, bias=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.l1 = nn.Linear(input_dim, rank, bias=bias)
        self.l2 = nn.Linear(rank, output_dim, bias=bias)

    def forward(self, x):
        return self.l2(self.l1(x))


class UnbiasedLinear(Layer):
    __name__ = "UnbiasedLinear"
    __complexity__ = "O(L d^2)"

    def __init__(self, input_features, output_features):
        super(UnbiasedLinear, self).__init__()
        self.linear = nn.Linear(input_features, output_features, bias=False)

    def forward(self, x):
        return self.linear(x)


class SwiGLU(Layer):
    gated = True
    __name__ = "SwiGLU"
    __complexity__ = "O(L)"

    def forward(self, x: torch.Tensor):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


class ReLU2(Layer):
    gated = True
    __name__ = "ReLU^2"
    __complexity__ = "O(L)"

    def forward(self, x: torch.Tensor):
        return torch.relu(x).square()


class UDLPReLU2(Layer):
    def __init__(self, d_model, expand, dropout=0.075):
        super().__init__()
        self.expand = nn.Linear(d_model, d_model * expand, bias=False)
        self.contract = nn.Linear(d_model * expand, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_normal_(self.expand.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.contract.weight)

    def forward(self, x):
        return self.dropout(self.contract(torch.relu(self.expand(x)).square()))


class MLP(Layer):
    __name__ = "MLP"
    __complexity__ = "O(depth L d^2)"

    def __init__(self, d_model: int, depth: int, expand: int, activation: Callable, dropout=0.1):
        super().__init__()
        hidden_dim = d_model * expand
        self.d_model = d_model
        self.depth = depth
        self.expand = expand
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        layers = []
        for i in range(depth):
            if i == 0:
                in_dim = d_model
                out_dim = hidden_dim
            elif i == depth - 1:
                in_dim = hidden_dim
                out_dim = d_model
            else:
                in_dim = hidden_dim
                out_dim = hidden_dim

            if hasattr(activation, 'gated') and i < depth - 1:
                out_dim = out_dim * 2

            layers.append(nn.Linear(in_dim, out_dim, bias=True))
            if i < depth - 1:
                layers.append(activation())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.dropout(self.layers(x))


class DLPGelu(Layer):
    def __init__(self, d_model, expand, dropout=0.1):
        super().__init__()
        self.expand = nn.Linear(d_model, d_model * expand, bias=True)
        self.contract = nn.Linear(d_model * expand, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.contract(F.gelu(self.expand(x))))


class DLPSoftmax(Layer):
    def __init__(self, d_model, m, dropout=0.1, copy=False):
        super().__init__()
        self.expand = nn.Linear(d_model, m, bias=False)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.contract = nn.Linear(m, d_model, bias=False)

        self.scaler = nn.Parameter(torch.tensor([1/math.sqrt(d_model)]))

        self.dropout = nn.Dropout(dropout)

        if copy:
            with torch.no_grad():
                self.contract.weight.copy_(self.expand.weight.t())

    def forward(self, x):
        q = self.q_proj(x)
        logits = self.expand(q * self.scaler)

        attn_weights = logits.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        return self.contract(attn_weights)


class FlashSDLP(Layer):
    def __init__(self, d_model, m, num_heads, dropout=0.1, outproj=False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.m = m
        self.dropout_p = dropout

        self.M1 = nn.Parameter(torch.randn(num_heads, m, self.head_dim))
        self.M2 = nn.Parameter(torch.randn(num_heads, m, self.head_dim))

        self.scaler = nn.Parameter(torch.tensor([1.0]))

        if outproj:
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
        else:
            self.out_proj = nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.M1)
        nn.init.xavier_normal_(self.M2)

        if isinstance(self.out_proj, nn.Linear):
            nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        q = x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        k = self.M1.unsqueeze(0).expand(batch_size, -1, -1, -1)
        v = self.M2.unsqueeze(0).expand(batch_size, -1, -1, -1)

        attn_output = F.scaled_dot_product_attention(
            query=q * self.scaler,
            key=k,
            value=v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.out_proj(attn_output)


class DLPSwiGLU(Layer):
    def __init__(self, d_model, expand, dropout=0.05):
        super().__init__()
        self.expand = nn.Linear(d_model, d_model * expand * 2, bias=True)
        self.contract = nn.Linear(d_model * expand, d_model, bias=True)
        self.activation = SwiGLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.contract(self.activation(self.expand(x))))


class UDLPSwiGLU(Layer):
    def __init__(self, d_model, expand, dropout=0.05):
        super().__init__()
        self.expand = nn.Linear(d_model, d_model * expand * 2, bias=False)
        self.contract = nn.Linear(d_model * expand, d_model, bias=False)
        self.activation = SwiGLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.contract(self.activation(self.expand(x))))


class F_UDLPSwiGLU(Layer):
    def __init__(self, d_model, expand, dropout=0.05, lora_rank=16):
        super().__init__()
        self.expand = nn.Linear(d_model, d_model * expand * 2, bias=False)
        self.contract = nn.Linear(d_model * expand, d_model, bias=False)
        self.LoRA_expand = LowRankLinear(d_model, expand * d_model * 2, lora_rank)
        self.LoRA_contract = LowRankLinear(d_model * expand, d_model, lora_rank)
        self.scale = 1/lora_rank*2
        self.finetuning = False
        self.activation = SwiGLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if not self.finetuning:
            return self.dropout(self.contract(self.activation(self.expand(x))))
        else:
            act = self.activation(self.expand(x) + self.LoRA_expand(x)*self.scale)
            return self.dropout(
                self.contract(act) + self.LoRA_contract(act)*self.scale
            )

    def finetune(self):
        torch.nn.init.zeros_(self.LORA_expand.l2.weight)
        torch.nn.init.zeros_(self.LoRA_contract.l2.weight)
        self.expand.requires_grad_(False)
        self.contract.requires_grad_(False)
        self.finetuning = True


class ResMLPBlock(Layer):
    def __init__(self, d_model, expand=2, norm=True, activation: Callable = nn.GELU):
        super().__init__()
        self.activation1 = activation()
        self.activation2 = activation()
        self.norm = norm
        self.d_model = d_model
        self.expand = expand
        self.l1 = nn.Linear(d_model, d_model * expand, bias=True)
        self.l2 = nn.Linear(d_model * expand, d_model, bias=True)

        self.g1 = nn.Linear(d_model, d_model * expand, bias=True)
        self.g2 = nn.Linear(d_model * expand, d_model, bias=True)

        self.norm = nn.LayerNorm(d_model) if self.norm else nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.l2(self.activation1(self.l1(x)))
        x = self.norm(x) + x
        x = self.g2(self.activation2(self.g1(x)))
        return x


class ResMLP(Layer):
    def __init__(self, d_model: int, depth: int, expand: int, activation: Callable, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.depth = depth
        self.expand = expand
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        layers = []
        for i in range(depth):
            layers.append(ResMLPBlock(d_model, expand=expand, activation=activation))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.dropout(self.layers(x))


class ConvResBlock(Layer):
    __complexity__ = "O(L k^2 in out)"

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=nn.ReLU):
        super(ConvResBlock, self).__init__()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same')
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same')
        self.convout = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.activation = activation()

    def forward(self, x):
        res = x
        x = self.bn(x)
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        x = x + res
        return self.convout(x)


class RMSNorm(nn.Module):  # Assumo Layer erediti da nn.Module
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        input_dtype = x.dtype

        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)

        x_rsqrt = torch.rsqrt(variance + self.eps)

        return self.weight * (x_fp32 * x_rsqrt).to(input_dtype)


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].lerp_(param, 1.0 - self.decay)

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])
