"""Basic Implementation of Famous Transformers Model using Vathos blocks SequenceModel.
serves firstly as an example on how to build with Vathos SequenceModel"""

from Vathos.blocks import *
import torch.nn as nn
from Vathos.functions import *


class GPTTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_len, ff_expand=2, activation='gelu', dropout=0.1,
                 pe='sinu'):
        super(GPTTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ff_expand = ff_expand
        self.activation = activation
        self.dropout = dropout
        if pe == 'sinu':
            self.pe = True
            self.rope = False
        elif pe == 'rope':
            self.pe = False
            self.rope = True
        elif self.pe == 'nope':
            self.pe = False
        else:
            raise ValueError('pe={} not recognized'.format(pe))

        self.model = SequenceModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            max_len=max_len,
            pos_encoder=self.pe,
            rope=self.rope,
            spatial_args={"causal": True, "n_heads": n_heads, "rope": self.rope},
            channel_args={"expand": ff_expand, "activation": ACTIVS[activation], "depth": 2}
        )

    def forward(self, x):
        return self.model(x)

    def __repr__(self):
        return self.model.__repr__()

    def __str__(self):
        return self.model.__str__()

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, *kwargs)

    def summary(self):
        self.model.summary()


class BERTTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_len, ff_expand=2, activation='gelu', dropout=0.1,
                 pe='sinu'):
        super(GPTTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ff_expand = ff_expand
        self.activation = activation
        self.dropout = dropout
        if pe == 'sinu':
            self.pe = True
            self.rope = False
        elif pe == 'rope':
            self.pe = False
            self.rope = True
        elif self.pe == 'nope':
            self.pe = False
        else:
            raise ValueError('pe={} not recognized'.format(pe))

        self.model = SequenceModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            max_len=max_len,
            pos_encoder=self.pe,
            rope=self.rope,
            spatial_args={"causal": False, "n_heads": n_heads, "rope": self.rope},
            channel_args={"expand": ff_expand, "activation": ACTIVS[activation], "depth": 2}
        )

    def forward(self, x):
        return self.model(x)

    def __repr__(self):
        return self.model.__repr__()

    def __str__(self):
        return self.model.__str__()

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        raise NotImplementedError("Yet to Implement")

    def summary(self):
        self.model.summary()


class DeiTTransformer(nn.Module):
    def __init__(self, n_class, img_size, patch_size, d_model, n_layers, n_heads, in_chans=3, ff_expand=2,
                 activation='gelu', dropout=0.1,
                 pe='sinu'):
        super().__init__()
        self.img_size = img_size
        self.n_class = n_class
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ff_expand = ff_expand
        self.activation = activation
        self.dropout = dropout
        if pe == 'sinu':
            self.pe = True
            self.rope = False
        elif pe == 'rope':
            self.pe = False
            self.rope = True
        elif self.pe == 'nope':
            self.pe = False
        else:
            raise ValueError('pe={} not recognized'.format(pe))

        self.model = SequenceModel(
            vocab_size=n_class,
            d_model=d_model,
            n_layers=n_layers,
            pos_encoder=self.pe,
            rope=self.rope,
            embedder=PatchEmbedder,
            embedder_args={"d_model": d_model, "img_size": img_size, "patch_size": patch_size, "in_chans": in_chans},
            unembedder=ClsHead,
            spatial_args={"causal": False, "n_heads": n_heads, "rope": self.rope},
            channel_args={"expand": ff_expand, "activation": ACTIVS[activation], "depth": 2}
        )

    def forward(self, x):
        return self.model(x)

    def __repr__(self):
        return self.model.__repr__()

    def __str__(self):
        return self.model.__str__()

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        raise NotImplementedError("Yet to Implement")

    def summary(self):
        self.model.summary()


class ViTTTransformer(nn.Module):
    def __init__(self, n_class, img_size, patch_size, d_model, n_layers, n_heads, in_chans=3, ff_expand=2,
                 activation='gelu', dropout=0.1,
                 pe='sinu'):
        super().__init__()
        self.img_size = img_size
        self.n_class = n_class
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ff_expand = ff_expand
        self.activation = activation
        self.dropout = dropout
        if pe == 'sinu':
            self.pe = True
            self.rope = False
        elif pe == 'rope':
            self.pe = False
            self.rope = True
        elif self.pe == 'nope':
            self.pe = False
        else:
            raise ValueError('pe={} not recognized'.format(pe))

        self.model = SequenceModel(
            vocab_size=n_class,
            d_model=d_model,
            n_layers=n_layers,
            pos_encoder=self.pe,
            rope=self.rope,
            embedder=PatchEmbedder,
            embedder_args={"d_model": d_model, "img_size": img_size, "patch_size": patch_size, "in_chans": in_chans},
            unembedder=MeanClassificationHead,
            spatial_args={"causal": False, "n_heads": n_heads, "rope": self.rope},
            channel_args={"expand": ff_expand, "activation": ACTIVS[activation], "depth": 2}
        )

    def forward(self, x):
        return self.model(x)

    def __repr__(self):
        return self.model.__repr__()

    def __str__(self):
        return self.model.__str__()

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        raise NotImplementedError("Yet to Implement")

    def summary(self):
        self.model.summary()


class DiTTransformer(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, ff_expand=2, activation='gelu', dropout=0.1,
                 pe='sinu'):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ff_expand = ff_expand
        self.activation = activation
        self.dropout = dropout
        if pe == 'sinu':
            self.pe = True
            self.rope = False
        elif pe == 'rope':
            self.pe = False
            self.rope = True
        elif self.pe == 'nope':
            self.pe = False
        else:
            raise ValueError('pe={} not recognized'.format(pe))

        self.model = SequenceModel(
            vocab_size=2,
            d_model=d_model,
            n_layers=n_layers,
            pos_encoder=self.pe,
            rope=self.rope,
            embedder=Identity,
            unembedder=Identity,
            spatial_args={"causal": False, "n_heads": n_heads, "rope": self.rope},
            channel_args={"expand": ff_expand, "activation": ACTIVS[activation], "depth": 2}
        )

    def forward(self, x):
        return self.model(x)

    def __repr__(self):
        return self.model.__repr__()

    def __str__(self):
        return self.model.__str__()

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        raise NotImplementedError("Yet to Implement")

    def summary(self):
        self.model.summary()
