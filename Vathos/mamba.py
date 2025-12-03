from Vathos.blocks import *

try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError("If you're trying to use Mamba, please install mamba-ssm, via pip install mamba-ssm. \n"
                      "probably you'll need to also install the right torch version:\n"
                      "pip install -q -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126")


class MambaMixer(Layer):
    def __init__(self, d_model, expand=4, d_state=64, k=3):
        super(MambaMixer).__init__()
        self.expand = expand
        self.d_model = d_model
        self.d_state = d_state
        self.k = k
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_state,
            expand=expand,
        )

    def forward(self, x):
        x = self.mamba(x)
        return x
