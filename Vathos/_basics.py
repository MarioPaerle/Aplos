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

ACTIVS = {
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'elu': nn.ELU,
    'lrelu': nn.LeakyReLU,
    'leaky_relu': nn.LeakyReLU,
}
PROFILE = True
PROFILE_BATCHED = True


class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()
        self.complexity = "O(1)"
        self.__name__ = "BasicLayer"
        self._timer_unbatched = not PROFILE_BATCHED
        self._tstart = 0
        self._tend = 0
        self._time = 0
        self._times = []
        self._sublayers = None

    def start_timer(self):
        self._tstart = timer()

    def end_timer(self, batch_size):
        self._tend = timer()
        self._time = (self._tend - self._tstart) / batch_size
        self._times.append(self._time)

    def get_mean_execution_time(self):
        mean = np.mean(self._times) if len(self._times) > 0 else None
        return mean

    def get_last_execution_time(self):
        return self._time

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

        def collect_layers(module, prefix="", level=0):
            layers = []
            for name, child in module.named_children():
                if isinstance(child, Layer):
                    layers.append((name, child, level))
                    layers.extend(collect_layers(child, prefix=f"{name}.", level=level + 1))
                elif isinstance(child, nn.ModuleList):
                    for i, item in enumerate(child):
                        if isinstance(item, Layer):
                            layers.append((f"{name}[{i}]", item, level))
                            layers.extend(collect_layers(item, prefix=f"{name}[{i}].", level=level + 1))
                elif isinstance(child, nn.Module):
                    layers.extend(collect_layers(child, prefix=f"{name}.", level=level))
            return layers

        all_layers = collect_layers(self)

        for original_name, layer, level in all_layers:
            class_name = type(layer).__name__
            unique_name = get_unique_name(class_name, self._sublayers.keys())
            self._sublayers[unique_name] = {'layer': layer, 'level': level}

    def __call__(self, *args, **kwargs):
        if self._sublayers is None:
            self.register_sublayers()
        if kwargs.get("profile") or PROFILE:
            if kwargs.get("profile"):
                del kwargs["profile"]
            self.start_timer()
            rets = self.forward(*args, **kwargs)
            if not self._timer_unbatched:
                self.end_timer(batch_size=args[0].shape[0])
            else:
                self.end_timer(1)
            return rets
        else:
            return self.forward(*args, **kwargs)

    def __repr__(self):
        a = f"{SEC}Vathos{RES}: "
        return a + super().__repr__()

    def profile(self, maxlevel=100, avg=False, plot=False, plot_level=1):
        """a Basic Layer level profiler operation"""
        import re
        from collections import defaultdict, OrderedDict

        batched = not self._timer_unbatched
        print(
            f"Layer {NUM}{type(self).__name__}{RES} Times Profile (batched: {GOOD if batched else BAD}{batched}{RES}) (averaged: {GOOD if avg else BAD}{avg}{RES}):")

        grouped_layers = OrderedDict()

        order_map = {}
        order_counter = 0

        for sublayer_name, sublayer_info in self._sublayers.items():
            layer = sublayer_info['layer']
            level = sublayer_info['level']
            if level >= maxlevel:
                continue

            # Extract base name (remove trailing _N pattern)
            match = re.match(r'^(.+?)_(\d+)$', sublayer_name)
            if match:
                base_name = match.group(1)
            else:
                base_name = sublayer_name

            key = (base_name, level)

            # Track order of first appearance
            if key not in order_map:
                order_map[key] = order_counter
                order_counter += 1
                grouped_layers[key] = []

            grouped_layers[key].append((sublayer_name, layer))

        if not avg:
            for sublayer_name, sublayer_info in self._sublayers.items():
                layer = sublayer_info['layer']
                level = sublayer_info['level']
                if level >= maxlevel:
                    continue
                indent = "    " * level
                print(f"{indent}- {NUM}{sublayer_name}{RES}: {layer.get_mean_execution_time() * 1000:.2f}ms")
        else:
            # Print averaged groups in the order they were registered
            for (base_name, level), layers_list in grouped_layers.items():
                indent = "    " * level

                if len(layers_list) > 1:
                    # Calculate average time
                    times = []
                    for _, layer in layers_list:
                        if len(layer._times) > 0:
                            times.append(np.mean(layer._times))

                    if times:
                        avg_time = np.mean(times)
                        count = len(layers_list)
                        print(
                            f"{indent}- {NUM}{base_name}_avg{RES}: {avg_time * 1000:.2f}ms {SEC}(x{count}){RES}")
                    else:
                        print(f"{indent}- {NUM}{base_name}_avg{RES}: no time recorded (x{len(layers_list)})")
                else:
                    sublayer_name, layer = layers_list[0]
                    print(f"{indent}- {NUM}{sublayer_name}{RES}: {layer.get_mean_execution_time() * 1000:.2f}ms")

        if plot:
            try:
                import matplotlib.pyplot as plt

                # Collect layers at the specified plot_level
                # Also include layers at lower levels that have no children at the target level
                level_layers = defaultdict(list)

                layers_with_children = set()
                for (base_name, level), layers_list in grouped_layers.items():
                    if level == plot_level:
                        # Check parent layers
                        for sublayer_name, layer in layers_list:
                            # Find the parent in _sublayers
                            for parent_name, parent_info in self._sublayers.items():
                                parent_layer = parent_info['layer']
                                parent_level = parent_info['level']
                                # Check if this layer is a child of a parent at level < plot_level
                                if parent_level < plot_level:
                                    for child_name, child in parent_layer.named_children():
                                        if child is layer or (isinstance(child, nn.ModuleList) and layer in child):
                                            layers_with_children.add(parent_name)

                for (base_name, level), layers_list in grouped_layers.items():
                    if level == plot_level:
                        # Collect all instances at this level
                        for _, layer in layers_list:
                            if len(layer._times) > 0:
                                level_layers[base_name].append(np.mean(layer._times))
                    elif level < plot_level:
                        # Include layers at lower levels that have no children at target level
                        for sublayer_name, layer in layers_list:
                            if sublayer_name not in layers_with_children and base_name not in layers_with_children:
                                if len(layer._times) > 0:
                                    level_layers[base_name].append(np.mean(layer._times))

                if level_layers:
                    # Calculate total time for each layer type at this level
                    layer_times = {}
                    for base_name, times in level_layers.items():
                        # Total time = sum of all instances (each counted fully)
                        layer_times[base_name] = sum(times)

                    # Sort by time (descending) for better visualization
                    sorted_layers = sorted(layer_times.items(), key=lambda x: x[1], reverse=True)
                    labels = [name for name, _ in sorted_layers]
                    times = [time * 1000 for _, time in sorted_layers]  # Convert to ms

                    # Create pie chart with better label handling
                    fig, ax = plt.subplots(figsize=(12, 8))
                    colors = plt.cm.Set3(range(len(labels)))

                    # Custom autopct function to hide percentages for small slices
                    def autopct_format(pct):
                        return f'{pct:.1f}%' if pct > 2 else ''

                    # Use pctdistance to move percentages and remove labels from pie
                    wedges, texts, autotexts = ax.pie(
                        times,
                        labels=None,  # Don't show labels on pie
                        autopct=autopct_format,
                        startangle=90,
                        colors=colors,
                        pctdistance=0.85
                    )

                    # Enhance percentage text readability
                    for autotext in autotexts:
                        autotext.set_color('black')
                        autotext.set_fontweight('bold')
                        autotext.set_fontsize(9)

                    ax.set_title(f'{type(self).__name__} - Time Distribution (Level {plot_level})',
                                 fontsize=14, fontweight='bold', pad=20)

                    # Add legend with actual times and color patches
                    legend_labels = [f'{name}: {time:.2f}ms' for name, time in zip(labels, times)]
                    ax.legend(wedges, legend_labels,
                              loc='center left',
                              bbox_to_anchor=(1, 0, 0.5, 1),
                              fontsize=10)

                    plt.tight_layout()
                    plt.show()
                else:
                    print(f"{SEC}No timing data available at level {plot_level} for plotting.{RES}")

            except ImportError:
                print(f"{BAD}matplotlib not available. Install it to use plot feature.{RES}")

    @staticmethod
    def register_exectution_time(fn, *args, **kwargs):
        start_time = timer()
        rets = fn(*args, **kwargs)
        elapsed = timer() - start_time
        return elapsed, rets

    def generate(self, *args, **kwargs):
        return None

    def has_custom_generate(self):
        """Check if this layer has overridden the generate method"""
        return type(self).generate is not Layer.generate


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


class DLPSwiGLU(Layer):
    def __init__(self, d_model, expand, dropout=0.1):
        super().__init__()
        self.expand = nn.Linear(d_model, d_model * expand * 2, bias=True)
        self.contract = nn.Linear(d_model * expand, d_model, bias=True)
        self.activation = SwiGLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.contract(self.activation(self.expand(x))))


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


class RMSNorm(Layer):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)
