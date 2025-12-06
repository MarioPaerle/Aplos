from Vathos.blocks import *


class FLAWrapper(nn.Module):
    def __init__(self, d_model, linatt, num_heads=8, expand_k=0.5, expand_v=1.0, mode='chunk'):
        super().__init__()
        self.d_model = d_model
        self.mode = mode
        self.gla = linatt(
            hidden_size=d_model,
            expand_k=expand_k,
            expand_v=expand_v,
            num_heads=num_heads
        )

    def forward(self, x):
        if not self.training:
            self.gla.past_key_values = None
        return self.gla(x, mode=self.mode, use_cache=False)[0]
