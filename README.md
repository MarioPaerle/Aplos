# APLOS 
_version = alpha-0.0.2_  - - - - **unstable**

[WARNING] This library is just born, no documentation can be found, and has plenty of bug, just wait until its **stable**.


I started building this collection of specific libraries to help myself with prototyping
new deep learning architettures.

### Easy install it via pip:
`pip install -q git+https://github.com/MarioPaerle/Aplos.git`

Aditionally (its useful i swear) one can download colorama via 

`pip install colorama`

and APLOS will automatically become colorful!

---

# To Start
Here's a simple implementation of an AR (Causal) Transformer:
```from Vathos.blocks import *
channel_args = {"expand": 2, "activation": nn.GELU, "depth": 2}
model = SequenceModel(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=6, max_len=2048,
                               pos_encoder=True, rope=False, channel_args=channel_args).to(device)
```
This will give you a torch.nn.Module object, which will easily adapt to your existing code, since it 
behave exaclty as a torch object.

---
# Aplos Modules:
just a draft...
### Vathos
To create, test, and experiment architettures I wanted to have a very similar implementation for every model,
both new ones and standards, mainly to make them easily comparable.


