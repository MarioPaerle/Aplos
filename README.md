# APLOS 
> [!NOTE]
> _version = alpha-0.0.2_  - - - - **unstable**

Vathos is a python library built on PyTorch whose aim is to accelerate the building of good level models for researchers, 
exploiting the repeating structure of Deep Learning architectures.

I'm making this library firstly for my self, since I felt I needed an easier way, to implement famous architectures, with a high grade
of customization, across all of my project, creating a sort of standard.

![logo.png](logo.png)

> [!WARNING]
> This library is just born, I don't even started writing the documentation, and has plenty of bug, just wait until its **stable**.


I started building this collection of specific libraries to help myself with prototyping
new deep learning architettures.

## Easy install it via pip:
`pip install -q git+https://github.com/MarioPaerle/Aplos.git`

Aditionally (its useful i swear) one can download colorama via 

`pip install colorama`

and APLOS will automatically become colorful!

---

# To Start
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

---
# Aplos Modules:
just a draft...
### Vathos
To create, test, and experiment architettures I wanted to have a very similar implementation for every model,
both new ones and standards, mainly to make them easily comparable.


