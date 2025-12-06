# APLOS (Vathos and Eidos)
> [!NOTE]
> _version = alpha-0.0.2_  - - - - **unstable**

## Easy install it via pip:
`pip install -q git+https://github.com/MarioPaerle/Aplos.git`

Aditionally (its useful i swear) one can download colorama via 

`pip install colorama`

![logo.png](logo.png)


# Vathos
Vathos is a python library built on PyTorch whose aim is to accelerate the building of good level models for researchers, 
exploiting the repeating structure of Deep Learning architectures.

I'm making this library firstly for my self, since I felt I needed an easier way, to implement famous architectures, with a high grade
of customization, across all of my project, creating a sort of standard.

and APLOS will automatically become colorful!

> [!WARNING]
> This library is just born, I don't even started writing the documentation, and has plenty of bug, just wait until its **stable**.

---

### To Start
Here's a simple implementation of an AR (Causal) Transformer:

```python
from Vathos.blocks import *
model = SequenceModel(vocab_size=VOCAB_SIZE, 
                      d_model=D_MODEL, 
                      n_layers=6, 
                      max_len=2048,
                      pos_encoder=True, 
                      rope=False, 
                      spatial_mixer=MultiheadAttentionMixer, # This uses FlashAttention by default
                      spatial_args={'n_heads': 8, 'causal':True},
                      channel_mixer=MLP,
                      channel_args={"expand": 2, "activation": SwiGLU, "depth": 2}
                      ).to(device)
```
This will give you a Vathos Layer object, which is actually a nn.Module object, which will easily adapt to your existing code, since it 
behave exaclty as a torch object.
Note that in this example a specific channel mixer is provided 'MLP' and its params are passed via a dict.
Same thing could0ve been done for the spatial mixer, which by default is a MultiHeadAttention.
More details are in the DOCU.md file
> [!IMPORTANT] 
> Please Note that for now Vathos _SequenceModel_ is not ready for efficient inference algorithm, as its native 
> _generate()_ function is simply calling the forward. 
> KV Caching and other inference acceleration are in plan.

---
### Vathos Model class uses
Vathos Model class can be used also to track losses, metrics[WIP] and has a built-in profiling system.
Each Vathos Layer automatically keeps track of its timing internally, and can access to sublayers timers.
An example of Sequence Models summary obtained by 
```python
model.summary()
```
![summary_example.png](imgs%2Fsummary_example.png)

An example of Models Profiling Print and Plot obtained by 
```python
model.profile(avg=True, plot=True)
```
![profiling_example.png](imgs%2Fprofiling_example.png)
![profiler_plot.png](imgs%2Fprofiler_plot.png)

---
### FLA Integration Example

here's an example on how to create a Vathos Layer compatible mixer, by wrapping FLA library.
All the FLA library will work with Vathos Models. if wrapped this way.


```python
from Vathos.blocks import *
from fla.layers import GatedLinearAttention

class FLAWrapper(Layer):
    def __init__(self, d_model, num_heads=8, expand_k=0.5, expand_v=1.0, mode='chunk'):
        super().__init__()
        self.d_model = d_model
        self.mode = mode
        self.gla = GatedLinearAttention(
            # here we convert the hidden_size name to d_model since it is the Vathos standard
            hidden_size=d_model, 
            expand_k=expand_k,
            expand_v=expand_v,
            num_heads=num_heads,
            mode=mode
        )
    
    def forward(self, x):
        if not self.training:
            self.gla.past_key_values = None
        return self.gla(x, mode=self.mode, use_cache=False)[0]

# Now simply create the model by using FLAWrapper as spatial mixer
model = SequenceModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=6,
        max_len=1024,
        pos_encoder=True,
        embedder=EasyEmbedder,
        unembedder=UnbiasedLinear,
        channel_mixer=MLP,
        channel_args={'expand': 2, 'activation': SwiGLU, 'depth':2},
        spatial_mixer=FLAWrapper,  # FLA layer
        spatial_args={'num_heads': 8, 'mode': 'chunk'},
    )

```

