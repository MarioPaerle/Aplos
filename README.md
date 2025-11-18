# APLOS 
_version = alpha-0.0.1_  - - - - **unstable**

[WARNING] This library is just born, no documentation can be found, and has plenty of bug, just wait until its **stable**.


I started building this collection of specific libraries to help myself with prototyping
new deep learning architettures.

### Easy install it via pip:
`!pip install -q git+https://github.com/MarioPaerle/Aplos.git`

# Aplos Modules:
just a draft...
### Vathos
To create, test, and experiment architettures I wanted to have a very similar implementation for every model,
both new ones and standards, mainly to make them easily comparable.
Vathos is defying two basic structures *Block1d* and *Block2d*, which are the foundamental building blocks for
1d sequence models (BERT Style models, GPT-2 Style models) and 2d sequence models (ViTs, CNNs, ...)
Blocks(1d/2d) are applying a spatial mixing (e.g. attention, convolution) and a channel mixing (MLP)
- x = spatial_mixing(norm1(x)) + x
- x = channel_mixing(norm2(x)) + x

Vathos also include other tools, functions, and Utils, to make my life easier when it comes to plots, especially for debugging.

