# A Sort of documentation initially for myself

## Vathos _blocks_ module
The only thing Vathos assumes by default is the existence of a _d_model_ variable which is common of
both spatial mixer, channel mixers, activations, embedder, and unembedder.

- MLP class has to receive a constructor as activation, not the function itself (e.g. `activation=nn.GELU` not `activation=nn.GELU()`)
- if MLP receives an activation with `gated=True` attribute then it will be assuming SwiGLU style channel mixing.
##### Building a Symbolic1dSeq2SeqModel
A Symbolic1dSeq2Seq model (_forward_) takes a tokenized input of shape [B, L] and LongTensor dtype.
Its structure its made of a channel mixer, a spatial mixer aggregated into a Block1d, and repeats blocks for
_n_layers_ times.
Must be specified:
- vocab_size
- d_model
- n_layers

If only these 3 values are specified, Symbolic1dSeq2SeqModel will construct a transformer, usinc MultiHeadAttentionMixer
and MLP.
To easily construct and test a Custom Symbolic1dSeq2SeqModel, its enough to specify a channel mixer constructor, a spatial mixer constructor,
and then pass the Symbolic1dSeq2SeqModel, the constructors arguments by (spatial/channel)_args={"arg1":1, "arg2:-, ..."}.

Here's an example:
```
model = Symbolic1dSeq2SeqModel(100, 16, 100,
                               channel_mixer=MLP,
                               channel_args={"depth": 2, "expand": 2, "activation": nn.GELU},
                               spatial_mixer=LinAtt2,
                               spatial_args={"expand": 2})
test_symbolic_model(model)
```
Additionally, _embedder_ and _pos_encoder_ classes can be also passes to Symbolic1dSeq2SeqModel init. pos_encoder can be also set to None -> Identity, False -> Identity, True -> Sinusoidal

