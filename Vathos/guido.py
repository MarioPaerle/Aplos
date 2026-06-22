"""Vathos.guido — loader turnkey per i checkpoint Guido nel PiCOFormer di produzione.

I checkpoint Guido (modello `train_guido_small.PiCOFormerLM`, allenato nel vecchio PiCO/aplos)
caricano nel PiCOFormer di produzione di questo Aplos con:
  - attention 'gated_mha'  (= MultiheadGatedAttentionMixer, qkv fuso, gate per-head)
  - channel  'torch_leaky_reglu2'  (= VariableGLU + LeakyReLU², il FFN di Guido)
  - rope_base 1e6
  - converter pesi = strip del prefisso 'backbone.'
La config (d_model, n_layers, n_heads, d_ff, vocab, gate_input_dim, smear_gate) è AUTO-RILEVATA
dallo state_dict, quindi funziona per qualsiasi taglia Guido (200M/370M/0.5B/...).

Uso:
    from Vathos.guido import load_guido
    model = load_guido("step_138000.pt")                          # path locale
    model = load_guido("Paerle/Guido-0.5B")                       # HF repo (file step_*_final.pt o model.pt)
    model = load_guido("Paerle/Guido-0.5B", filename="model.pt")  # HF repo + file esplicito
"""
from __future__ import annotations
import os, re
import torch

from .picoformer import PiCOFormerConfig, build_picoformer, prepare_for_inference

HEAD_DIM = 64   # Guido: q_norm/k_norm hanno shape (head_dim,)


def _load_raw(source: str, filename: str | None) -> dict:
    """Ritorna lo state_dict (dict di tensori) da: path locale .pt, o repo HF."""
    if os.path.exists(source):
        blob = torch.load(source, map_location="cpu", weights_only=False)
        return blob.get("model", blob)
    # HF repo id
    from huggingface_hub import hf_hub_download, list_repo_files
    if filename is None:
        files = list_repo_files(source)
        cand = [f for f in files if f.endswith("_final.pt")] or \
               [f for f in files if f.endswith(".pt")] or \
               [f for f in files if f.endswith(".safetensors")]
        if not cand:
            raise FileNotFoundError(f"nessun checkpoint (.pt/.safetensors) in {source}: {files}")
        filename = sorted(cand)[-1]
    path = hf_hub_download(repo_id=source, filename=filename)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob.get("model", blob)


def infer_config(sd: dict, **overrides) -> PiCOFormerConfig:
    """Deduce la PiCOFormerConfig di Guido dallo state_dict (chiavi con prefisso 'backbone.')."""
    g = lambda k: sd[k] if k in sd else sd.get("backbone." + k)
    emb = g("embedder.embedding.weight")
    vocab, d_model = emb.shape
    n_layers = 1 + max(int(m.group(1)) for k in sd for m in [re.search(r"blocks\.(\d+)\.", k)] if m)
    d_ff = g("blocks.0.channel_mixer.expand.weight").shape[0]
    gate_w = g("blocks.0.spatial_mixer.gate_proj.weight")          # (n_heads, gate_input_dim)
    n_heads, gate_in = gate_w.shape
    assert d_model // n_heads == HEAD_DIM, f"head_dim atteso {HEAD_DIM}, ho {d_model//n_heads}"
    has_smear = any("smear_gate" in k for k in sd)
    smear_in = g("smear_gate.gate.weight").shape[1] if has_smear else 12
    cfg = dict(vocab_size=vocab, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
               n_kv_heads=n_heads, max_seq_len=8192, rope_base=1_000_000.0,
               attention="gated_mha", channel="torch_leaky_reglu2",
               ffn_multiplier=round(d_ff / d_model), qk_norm=True,
               attention_gate_input_dim=gate_in, logit_softcap=30.0, tied_embeddings=True,
               smear_gate=has_smear, smear_gate_input_dim=smear_in)
    cfg.update(overrides)
    return PiCOFormerConfig(**cfg)


def convert_state_dict(sd: dict) -> dict:
    """Guido → produzione: con attention='gated_mha' basta togliere il prefisso 'backbone.'."""
    return {(k[9:] if k.startswith("backbone.") else k): v for k, v in sd.items()}


def load_guido(source: str, *, filename: str | None = None, device: str = "cuda",
               dtype: torch.dtype = torch.bfloat16, channel: str = "torch_leaky_reglu2",
               for_inference: bool = True, strict: bool = True):
    """Carica un checkpoint Guido nel PiCOFormer di produzione. Ritorna il modello pronto.

    source: path locale .pt | repo HF id. filename: file esplicito nel repo HF (opzionale).
    channel: 'torch_leaky_reglu2' (default, corretto+compile-friendly) o 'triton_*' per velocità.
    """
    sd = _load_raw(source, filename)
    cfg = infer_config(sd, channel=channel)
    model = build_picoformer(cfg)
    miss, unexp = model.load_state_dict(convert_state_dict(sd), strict=False)
    if strict and (miss or unexp):
        raise RuntimeError(f"load impuro: missing={list(miss)[:4]} unexpected={list(unexp)[:4]}")
    if for_inference:
        model = prepare_for_inference(model, device=device, dtype=dtype)
    else:
        model = model.to(device=device, dtype=dtype)
    return model, cfg
