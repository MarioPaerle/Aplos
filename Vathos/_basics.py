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
from torch import Tensor

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
    """Global configuration state."""
    _COMPILABLE = False  # True = production, torch.compile safe
    _PROFILE_BATCHED = True  # True = times divided by batch size
    _GLOBAL_PROFILE = True  # True = always profile in debug mode


def set_vathos_mode(mode: str):
    """
    Switch Vathos mode globally.

    'debug'      — profiling enabled, torch.compile unfriendly
    'production' — profiling disabled, torch.compile safe
    """
    if mode.lower() == "production":
        VathosConfig._COMPILABLE = True
        print(f"{SEC}Vathos:{RES} Switched to {GOOD}PRODUCTION{RES} mode.")
    elif mode.lower() == "debug":
        VathosConfig._COMPILABLE = False
        print(f"{SEC}Vathos:{RES} Switched to {NUM}DEBUG{RES} mode.")
    else:
        raise ValueError("Mode must be 'production' or 'debug'")


# =============================================================================
# BASE LAYER
# =============================================================================

class Layer(nn.Module):
    """
    ─────────────────────────────────────────────────────────────
    class MyBlock(Layer):
        def __init__(self, ...):
            super().__init__()
            # define sub-modules here
            self.lin = nn.Linear(...)
            self.attn = MultiheadAttentionMixer(...)

        def forward(self, x):
z            return self.lin(x)
    ─────────────────────────────────────────────────────────────

    DO NOT override __call__. Profiling is handled via native forward hooks,
    which are registered automatically in debug mode and completely absent
    in production mode — zero overhead, full torch.compile compatibility.

    IMPORTANT: gli hook sono registrati al __init__. Se il modello è costruito
    PRIMA di chiamare set_vathos_mode('production'), gli hook sopravvivono.
    Soluzioni:
      1) Chiama set_vathos_mode('production') PRIMA di istanziare moduli.
      2) Oppure chiama model.strip_profiling_hooks() dopo lo switch.
    """

    _is_vathos_layer = True

    def __init__(self):
        super().__init__()
        self.complexity = "O(1)"
        self.__name__ = self.__class__.__name__

        self._timer_unbatched = not VathosConfig._PROFILE_BATCHED
        self._tstart = 0.0
        self._tend = 0.0
        self._time = 0.0
        self._times = []

        self._sublayers = None

        # Hooks are registered ONCE at construction in debug mode.
        # In production mode, no hooks exist — no overhead whatsoever.
        if not VathosConfig._COMPILABLE and VathosConfig._GLOBAL_PROFILE:
            self._register_profiling_hooks()

    # -------------------------------------------------------------------------
    # Profiling hooks — native nn.Module mechanism, __call__ is never touched
    # -------------------------------------------------------------------------

    def _register_profiling_hooks(self):
        self.register_forward_pre_hook(self._pre_hook)
        self.register_forward_hook(self._post_hook)

    def _pre_hook(self, module, args):
        self._tstart = timer()

    def _post_hook(self, module, args, output):
        self._tend = timer()
        bs = 1
        if args and isinstance(args[0], torch.Tensor):
            bs = args[0].shape[0]
        div = bs if not self._timer_unbatched else 1
        self._time = (self._tend - self._tstart) / div
        self._times.append(self._time)

    # -------------------------------------------------------------------------
    # Sublayer registry — used by profile()
    # -------------------------------------------------------------------------

    def register_sublayers(self):
        if self._sublayers is None:
            self._sublayers = dict()

        def get_unique_name(base_name, existing_names):
            if base_name not in existing_names:
                return base_name
            counter = 1
            while f"{base_name}_{counter}" in existing_names:
                counter += 1
            return f"{base_name}_{counter}"

        def collect_layers(module, level=0):
            layers = []
            for name, child in module.named_children():
                if getattr(child, "_is_vathos_layer", False):
                    layers.append((name, child, level))
                    layers.extend(collect_layers(child, level=level + 1))
                elif isinstance(child, nn.ModuleList):
                    for i, item in enumerate(child):
                        if getattr(item, "_is_vathos_layer", False):
                            layers.append((f"{name}[{i}]", item, level))
                            layers.extend(collect_layers(item, level=level + 1))
                elif isinstance(child, nn.Module):
                    layers.extend(collect_layers(child, level=level))
            return layers

        for original_name, layer, level in collect_layers(self):
            class_name = getattr(layer, "__name__", type(layer).__name__)
            unique_name = get_unique_name(class_name, self._sublayers.keys())
            self._sublayers[unique_name] = {"layer": layer, "level": level}

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def get_mean_execution_time(self) -> float:
        return float(np.mean(self._times)) if self._times else 0.0

    def strip_profiling_hooks(self):
        """Rimuove gli hook profile da questo modulo e da tutti i sotto-Layer.

        Utile se il modello è stato istanziato in debug mode e poi si vuole
        passare a production senza ricostruirlo. Idempotente.
        """
        for module in self.modules():
            if isinstance(module, Layer):
                module._forward_pre_hooks.clear()
                module._forward_hooks.clear()
        return self

    def has_custom_generate(self) -> bool:
        return type(self).generate is not Layer.generate

    def generate(self, *args, **kwargs):
        """Override in subclasses that support autoregressive generation."""
        return None

    def clear_times(self):
        """Reset profiling history."""
        self._times.clear()

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

        if plot:
            try:
                import matplotlib.pyplot as plt
                level_layers = defaultdict(list)

                for (base_name, level), layers_list in grouped_layers.items():
                    if level == plot_level:
                        for _, layer in layers_list:
                            if len(layer._times) > 0:
                                level_layers[base_name].append(np.mean(layer._times))

                    elif level < plot_level:

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

    def __repr__(self):
        return f"{SEC}Vathos{RES}: " + super().__repr__()


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


class LearntSkip(Layer):
    __name__ = "Skip"
    __complexity__ = "O(1)"

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.w = nn.Parameter(torch.tensor([1.0, 1.0]), requires_grad=True)

    def forward(self, x):
        return self.layer(x) * self.w[0] + x * self.w[1]


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


class BottleneckRepeatedLinear(nn.Module):
    def __init__(self, in_features, out_features, num_repeats, bias=False, scale_variance=False):
        super().__init__()

        assert out_features % num_repeats == 0, "out_features must be strictly divisible by num_repeats"

        self.in_features = in_features
        self.out_features = out_features
        self.num_repeats = num_repeats
        self.bottleneck_dim = out_features // num_repeats
        self.scale_variance = scale_variance

        self.weight = nn.Parameter(torch.Tensor(self.bottleneck_dim, in_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.bottleneck_dim))
        else:
            self.register_parameter('bias', None)

        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)

        repeat_shape = [1] * (y.dim() - 1) + [self.num_repeats]

        out = y.repeat(*repeat_shape)

        if self.scale_variance:
            out = out / math.sqrt(self.num_repeats)

        return out


class ProductLinear(Layer):
    __name__ = 'Linear'

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight1 = nn.Parameter(0.1 * torch.randn(output_dim, input_dim))
        self.weight2 = nn.Parameter(torch.ones(output_dim, input_dim + 0.01 * torch.randn(output_dim, input_dim)))

    def forward(self, x):
        return F.linear(x, self.weight1 * self.weight2)


class DoubleLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight1 = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        nn.init.xavier_uniform_(self.weight1)

        self.weight2 = nn.Parameter(torch.empty((in_features, in_features), **factory_kwargs))
        nn.init.xavier_uniform_(self.weight2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_eff = self.weight1 @ self.weight2
        return F.linear(x, W_eff)


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
    gated = False
    __name__ = "ReLU^2"
    __complexity__ = "O(L)"

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pos = F.relu(x)
        return x_pos * x_pos


class LeakyReLU2(Layer):
    gated = False
    __name__ = "LeakyReLU^2"
    __complexity__ = "O(L)"

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pos = F.leaky_relu(x, 0.5)
        return x_pos * x_pos


class PSiLU2(Layer):
    gated = False
    __name__ = "SiLU^2"
    __complexity__ = "O(L)"

    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor([5.0, 2.0]), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pos = self.param[0] * F.silu(x / self.param[1])
        return x_pos * x_pos


class LeLU2(Layer):
    gated = False
    __name__ = "ReLU^2"
    __complexity__ = "O(L)"

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1) - 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pos = F.relu(x - self.a)
        return x_pos * x_pos


class PReLU2(Layer):
    def __init__(self, num_parameters=1):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(num_parameters))

    def forward(self, x: torch.Tensor):
        return self.alpha * torch.square(F.relu(x))


class UDLPReLU2(Layer):
    """Unbiased Dual Layer Perceptron, Relu^2 Activation"""

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

        self.scaler = nn.Parameter(torch.tensor([1 / math.sqrt(d_model)]))

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
    def __init__(self, d_model, expand, dropout=0.00):
        super().__init__()
        self.expand = nn.Linear(d_model, d_model * expand * 2, bias=True)
        self.contract = nn.Linear(d_model * expand, d_model, bias=True)
        self.activation = SwiGLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.contract(self.activation(self.expand(x))))


class LowRankGatedDLP(Layer):
    def __init__(self, d_model, expand, dropout=0.00, activation=nn.SiLU, rank=64, gate_activation=None):
        super().__init__()
        self.ga = gate_activation if gate_activation is not None else Identity()
        self.act = activation()
        self.M = d_model * expand
        self.rank = rank
        self.expand = nn.Linear(d_model, d_model * expand + rank, bias=False)
        self.contract = nn.Linear(d_model * expand, d_model, bias=False)
        self.rexp = nn.Linear(rank, d_model * expand, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out1, gate1 = self.expand(x).split([self.M, self.rank], dim=-1)
        return self.contract(self.act(out1 * self.ga(self.rexp(gate1))))


class UDLPSwiGLU(Layer):
    def __init__(self, d_model, expand, dropout=0.00):
        super().__init__()
        self.expand = nn.Linear(d_model, d_model * expand * 2, bias=False)
        self.contract = nn.Linear(d_model * expand, d_model, bias=False)
        self.activation = SwiGLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.contract(self.activation(self.expand(x))))


class VariableUDLP(Layer):
    def __init__(self, d_model, d_output, M, activation=ReLU2, dropout=0.00):
        super().__init__()
        self.expand = nn.Linear(d_model, M, bias=False)
        self.contract = nn.Linear(M, d_output, bias=False)
        self.activation = activation()
        self.dropout = nn.Dropout(dropout)

    def _init_weights(self):
        # Identity-at-init: expand orthogonal, contract → 0 ⇒ FFN contributo nullo.
        torch.nn.init.orthogonal_(self.expand.weight)
        torch.nn.init.zeros_(self.contract.weight)

    def forward(self, x):
        return self.dropout(self.contract(self.activation(self.expand(x))))


class VariableUDLP_Attention(Layer):
    def __init__(self, d_model, d_output, M, activation=ReLU2, dropout=0.00, use_qk_norm=True):
        super().__init__()
        self.d_model = d_model
        self.d_output = d_output
        self.half_dim = d_model // 2

        # Dimensione delle query/key. Per efficienza la tengo pari a half_dim
        self.d_k = self.half_dim

        # Q = X W_q
        self.W_q = nn.Linear(self.half_dim, self.d_k, bias=False)

        # K = W_k (Matrice di parametri appresi, forma: [M, d_k])
        self.W_k = nn.Parameter(torch.empty(M, self.d_k))
        torch.nn.init.normal_(self.W_k, std=self.d_k ** -0.5)

        # V = W_v corrisponde al tuo 'contract'
        self.contract = nn.Linear(M, d_output, bias=False)

        self.activation = activation()
        self.dropout = nn.Dropout(dropout)

        # QK Norm
        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(self.d_k)
            self.k_norm = nn.RMSNorm(self.d_k)

    def _init_weights(self):
        torch.nn.init.zeros_(self.contract.weight)

    def forward(self, x):
        # 1. Prendiamo solo mezza residual stream
        x_half = x[..., :self.half_dim]

        # 2. Calcoliamo le Query e prendiamo le Key
        q = self.W_q(x_half)
        k = self.W_k

        # 3. QK Norm
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 4. Attention scores: Q K^T
        # q is [..., d_k], k is [M, d_k]. F.linear(q, k) is exactly q @ k.T
        scores = F.linear(q, k)

        # 5. Generic activation (e.g. ReLU^2) and Dropout
        h = self.dropout(self.activation(scores))

        # 6. Proiezione V (contract) e padding automatico se d_output < d_model
        out = self.contract(h)

        if self.d_output < self.d_model:
            out = F.pad(out, (0, self.d_model - self.d_output))

        return out


class VariableUDLP_CrossGate(Layer):
    def __init__(self, d_model, d_output, M, activation=ReLU2, dropout=0.00):
        super().__init__()
        self.d_model = d_model
        self.d_output = d_output
        self.half_dim = d_model // 2

        self.expand_k = nn.Linear(self.half_dim, M, bias=False)

        self.expand_g = nn.Linear(d_model - self.half_dim, M, bias=False)

        self.contract = nn.Linear(M, d_output, bias=False)

        self.activation = activation()
        self.dropout = nn.Dropout(dropout)

    def _init_weights(self):
        torch.nn.init.zeros_(self.contract.weight)

    def forward(self, x):
        x_left = x[..., :self.half_dim]
        x_right = x[..., self.half_dim:]

        hk = self.expand_k(x_left)
        hg = self.expand_g(x_right)

        h = self.dropout(self.activation(hk) * hg)

        out = self.contract(h)

        return out


class VariableGatedUDLP(Layer):
    """UDLP + sparse output gate (parameter-golf / Modded-NanoGPT 1667 style).

    Sigmoid gate over the first `gate_input_dim` slice of the residual (sparse).
    Output: `contract(activation(expand(x))) * sigmoid(gate_proj(x[..., :gate_input_dim]))`.
    """
    __name__ = "VariableGatedUDLP"

    def __init__(self, d_model, d_output, M, activation=ReLU2, dropout=0.00, gate_input_dim=12):
        super().__init__()
        self.gate_input_dim = gate_input_dim
        self.expand = nn.Linear(d_model, M, bias=False)
        self.contract = nn.Linear(M, d_output, bias=False)
        self.gate_proj = nn.Linear(gate_input_dim, d_output, bias=False)
        self.activation = activation()
        self.dropout = nn.Dropout(dropout)

    def _init_weights(self):
        # Identity-at-init: expand + gate_proj orthogonal, contract → 0
        # (with contract=0 the branch is zero regardless of the gate).
        torch.nn.init.orthogonal_(self.expand.weight)
        torch.nn.init.zeros_(self.contract.weight)
        torch.nn.init.orthogonal_(self.gate_proj.weight)

    def forward(self, x):
        out = self.contract(self.activation(self.expand(x)))
        gate = torch.sigmoid(self.gate_proj(x[..., :self.gate_input_dim]))
        return self.dropout(out * gate)


class VariableGLU(Layer):
    """GLU-style channel mixer (à la SwiGLU/ReGLU): `contract(activation(expand(x)) * up(x))`.

    Default activation is `LeakyReLU2`, giving the "LeakyReGLU²" variant. Adds one extra
    projection (`up`, d_model→M) over `VariableUDLP` → ~+50% FFN params. Identity-at-init
    via `contract=0` (the branch is zero at init while gradients still flow through
    `expand` and `up`).

    Same `(d_model, d_output, M, activation, dropout)` signature as `VariableUDLP`, so it
    is a drop-in replacement in `Builder(...)`.
    """
    __name__ = "VariableGLU"

    def __init__(self, d_model, d_output, M, activation=LeakyReLU2, dropout=0.00):
        super().__init__()
        self.expand   = nn.Linear(d_model, M, bias=False)   # activation path (W in act(xW))
        self.up       = nn.Linear(d_model, M, bias=False)   # value path (V in · xV)
        self.contract = nn.Linear(M, d_output, bias=False)
        self.activation = activation()
        self.dropout = nn.Dropout(dropout)

    def _init_weights(self):
        # Identity-at-init: branch is null since contract = 0.
        torch.nn.init.orthogonal_(self.expand.weight)
        torch.nn.init.orthogonal_(self.up.weight)
        torch.nn.init.zeros_(self.contract.weight)

    def forward(self, x):
        return self.dropout(self.contract(self.activation(self.expand(x)) * self.up(x)))


class DoubleUDLP(Layer):
    def __init__(self, d_model, d_output, M, activation=ReLU2, dropout=0.00):
        super().__init__()
        self.expand = DoubleLinear(d_model, M)
        self.contract = nn.Linear(M, d_output)
        self.activation = activation()
        self.dropout = nn.Dropout(dropout)

    def _init_weights(self):
        torch.nn.init.zeros_(self.contract.weight)

    def forward(self, x):
        return self.dropout(self.contract(self.activation(self.expand(x))))


class M_UDLP(Layer):
    def __init__(self, d_model, d_output, M, activation=ReLU2, dropout=0.00):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model)
        self.expand = nn.Linear(d_model, M)
        self.contract = nn.Linear(M, d_output)
        self.activation = activation()
        self.dropout = nn.Dropout(dropout)

    def _init_weights(self):
        torch.nn.init.zeros_(self.contract.weight)

    def forward(self, x):
        return self.dropout(self.contract(self.activation(self.expand(x))))


class F_UDLPSwiGLU(Layer):
    def __init__(self, d_model, expand, dropout=0.05, lora_rank=16):
        super().__init__()
        self.expand = nn.Linear(d_model, d_model * expand * 2, bias=False)
        self.contract = nn.Linear(d_model * expand, d_model, bias=False)
        self.LoRA_expand = LowRankLinear(d_model, expand * d_model * 2, lora_rank)
        self.LoRA_contract = LowRankLinear(d_model * expand, d_model, lora_rank)
        self.scale = 1 / lora_rank * 2
        self.finetuning = False
        self.activation = SwiGLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if not self.finetuning:
            return self.dropout(self.contract(self.activation(self.expand(x))))
        else:
            act = self.activation(self.expand(x) + self.LoRA_expand(x) * self.scale)
            return self.dropout(
                self.contract(act) + self.LoRA_contract(act) * self.scale
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


class RMSNorm(nn.Module):
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


class Nova(Layer):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.7]))

    def forward(self, x):
        h = -self.beta * x
        return x * (torch.sigmoid(-h) - (1 / (1 + h ** 2)))


class MinusNova(Layer):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.7]))

    def forward(self, x):
        h = -self.beta * -x
        return x * (torch.sigmoid(-h) - (1 / (1 + h ** 2)))


class TaylorAct(nn.Module):
    def __init__(self, order=3):
        super().__init__()
        self.order = order
        self.coeffs = nn.Parameter(torch.randn(order + 1) * 0.02)

    def forward(self, x):
        device = x.device
        exponents = torch.arange(self.order + 1, device=device, dtype=x.dtype)
        x_pow = x.unsqueeze(-1).pow(exponents)

        return torch.matmul(x_pow, self.coeffs)

    def plot(self, bounds=(-4, 4), n_points=200):
        x = torch.linspace(bounds[0], bounds[1], n_points)
        with torch.no_grad():
            y = self.forward(x)

        plt.figure(figsize=(7, 4))
        plt.plot(x.numpy(), y.numpy(), label='TaylorAct', color='royalblue')
        plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        plt.axvline(0, color='gray', linewidth=0.8, linestyle='--')
        plt.title('TaylorAct Activation Function')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


class FastPoly3(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.a0 = nn.Parameter(torch.zeros(d_model))
        self.a1 = nn.Parameter(torch.ones(d_model))
        self.a2 = nn.Parameter(torch.zeros(d_model))
        self.a3 = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a0 + x * (self.a1 + x * (self.a2 + x * self.a3))


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


if __name__ == '__main__':
    act = TaylorAct()
    act.plot()  # default -4, 4
    act.plot(bounds=(-1, 1))  # custom bounds
    act.plot(bounds=(0, 1), n_points=100)
