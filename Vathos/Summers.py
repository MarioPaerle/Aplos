from blocks import *
from Utils import *

class BaseSummer(nn.Module):
    def __init__(self, d_model: int, causal=True):
        super().__init__()
        assert causal, \
            ("CausalMultiheadAttentionMixer Module only supports causal=True, "
             "if you meant to create a non Causal Attention use the MultiheadAttentionMixer")
        self.d_model = d_model

    def forward(self, x: torch.Tensor):
        return power_weigthed_cumsum(x)
