# APLOS (Vathos and Eidos)
> [!NOTE]
> _version = alpha-0.0.2_  - - - - **unstable**

_"Aplos from greek απλός (simple), Vathos from greek βάθος, for Deep, and Eidos εἶδος from (information, concept)"_
## Easy install it via pip:
`pip install -q git+https://github.com/MarioPaerle/Aplos.git`

Aditionally (its useful i swear) one can download colorama via 

`pip install colorama`

and APLOS will automatically become colorful!

![logo.png](logo.png)

---

# Vathos
Vathos is a python library built on PyTorch whose aim is to accelerate the building of good level models for researchers, 
exploiting the repeating structure of Deep Learning architectures.
It tries to preserve customizability while accelerating the coding experience. 

I'm making this library firstly for my self, since I felt I needed an easier way, to implement famous architectures, with a high grade
of customization, across all of my project, creating a sort of standard.

> [!WARNING]
> This library is just born, I don't even started writing the documentation, and has plenty of bug, just wait until its **stable**.

---

### To Start
Here's a simple implementation of an AR (Causal) Transformer completely compatible with torch:

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
This will give you a Vathos **Layer** object, which is actually a **nn.Module** object, which will easily adapt to your existing code, since it 
behave exaclty as a torch object.
Note that in this example a specific channel mixer is provided 'MLP' and its params are passed via a dict.
Same thing could've been done for the spatial mixer, which by default is a MultiHeadAttention.
More Details in the github wiki.

An Inference Example for this Auto-Regressive Transformer:
```python
model.generate(torch.tensor([0]), 1000, temperature=0.75, token_end=None, custom_generate=False)
```

> [!IMPORTANT] 
> Please Note that for now Vathos _SequenceModel_ is not completely ready for efficient inference algorithm.
> the SequenceModel.generate(... custom_generate=True) looks, into each spatial_mixer and channel_mixer, for a custom generate() method
> and fallbacks to the forward if it does not find any.

---
### Vathos Model class Uses
Vathos Model class can be used also to track losses, metrics[WIP] and has a built-in profiling system.
Each Vathos Layer automatically keeps track of its timing internally, and can access to sublayers timers.
An example of Sequence Models summary obtained by 
```python
model.summary()
```
![summary_example.png](imgs%2Fsummary_example.png)

An example of Models Profiling Print and Plot obtained by 
```python
model.register_sublayers() # builds the graph of model Layers
model.profile(avg=True, plot=True) # Shows metrics for nodes of the graph, and plots them.
```
![profiling_example.png](imgs%2Fprofiling_example.png)
![profiler_plot.png](imgs%2Fprofiler_plot.png)

## Saving and Loading VathosModels
```python
model.save_checkpoint('your_checkpoint_name.pt')
model.load_checkpoint('your_checkpoint_name.pt')
```

## Metrics Losses and training managment
```python
# in training pseudocode:

for epoch in range(num_epochs):
    ...
    for x in val_dataloader:
        loss = yourlossfunction(model(input), target)
        model.register_loss(loss.item())
        model.register_metrics({'metric1': metric1(model(input)), 'metric2': metric2(model(input), target)}) # be sure output is a number

    model.register_epoch() # This computes the average on the epoch step for both metrics and loss.

# After training
model.plot_losses() # shows the plot of losses
model.plot_metrics() # shows the plot of losses and all metrics [WIP]
```



> [!WARNING]
> `model.save_state_dict()` works as expected for the model parameters, but is not saving losses, metrics, and profiling attributes!



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

