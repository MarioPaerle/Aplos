"""Vathos.guido — loader turnkey per i checkpoint Guido nel PiCOFormer di produzione.

I checkpoint Guido (modello `train_guido_small.PiCOFormerLM`, allenato nel vecchio PiCO/aplos)
caricano nel PiCOFormer di produzione di questo Aplos con:
  - attention 'gated_mha'  (= MultiheadGatedAttentionMixer, qkv fuso, gate per-head)
  - channel: dipende dall'attivazione del FFN del checkpoint (expand/up/contract sono IDENTICI
    tra SwiGLU e LeakyReGLU² → NON distinguibili dallo state_dict; conta solo l'attivazione):
      'torch_swiglu'        (VariableGLU + SiLU)        → **default**: i Guido SwiGLU attuali
                                                          (v5 / Guido-1-0.5B-Base). Carica 1:1,
                                                          nessuna fusione pesi, gira ovunque giri
                                                          PyTorch, compile-friendly.
      'triton_swiglu'       (SwiGLU fuso gate+up)       → stessa matematica, kernel Triton veloce
                                                          (fonde expand+up in [2*d_ff, d]).
      'torch_leaky_reglu2'  (VariableGLU + LeakyReLU²)  → SOLO per i Guido LEGACY con FFN LeakyReGLU²
                                                          (v4 e precedenti): passalo ESPLICITAMENTE.
  - rope_base 1e6
  - converter pesi = strip del prefisso 'backbone.' (+ fusione expand/up se channel='triton_swiglu')
La config (d_model, n_layers, n_heads, d_ff, vocab, gate_input_dim, smear_gate) è AUTO-RILEVATA
dallo state_dict, quindi funziona per qualsiasi taglia Guido (200M/370M/0.5B/...).

⚠️ ATTIVAZIONE FFN: lo state_dict NON dice se il FFN è SwiGLU(SiLU) o LeakyReGLU²(LeakyReLU²) —
le chiavi sono identiche. Con il channel SBAGLIATO il modello carica 0/0 ma calcola garbage
(LeakyReLU² su pesi SiLU ≈ 16× errore di norma → loop di ripetizione, accuracy crollata). Il
default 'torch_swiglu' è quello giusto per Guido-1-0.5B-Base; i checkpoint legacy passano
channel='torch_leaky_reglu2'.

Uso:
    from Vathos.guido import load_guido
    model, cfg = load_guido("Paerle/Guido-1-0.5B-Base")                        # default = torch_swiglu ✓
    model, cfg = load_guido("Paerle/Guido-1-0.5B-Base", channel="torch_swiglu")# esplicito (identico)
    model, cfg = load_guido("Paerle/Guido-1-0.5B-Base", channel="triton_swiglu")# path veloce (Triton)
    model, cfg = load_guido("step_138000.pt", channel="torch_leaky_reglu2")     # checkpoint LEGACY
"""
from __future__ import annotations
import os, re
import torch

from .picoformer import PiCOFormerConfig, build_picoformer, prepare_for_inference

HEAD_DIM = 64   # Guido: q_norm/k_norm hanno shape (head_dim,)

# Attivazione FFN di default per Guido: i checkpoint attuali (v5 / Guido-1-0.5B-Base) sono SwiGLU.
DEFAULT_CHANNEL = "torch_swiglu"


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
               attention="gated_mha", channel=DEFAULT_CHANNEL,
               ffn_multiplier=round(d_ff / d_model), qk_norm=True,
               attention_gate_input_dim=gate_in, logit_softcap=30.0, tied_embeddings=True,
               smear_gate=has_smear, smear_gate_input_dim=smear_in)
    cfg.update(overrides)
    return PiCOFormerConfig(**cfg)


def convert_state_dict(sd: dict, *, channel: str | None = None) -> dict:
    """Guido → produzione: strip del prefisso 'backbone.'.

    Con channel='triton_swiglu' fonde le proiezioni SwiGLU: i moduli torch GLU tengono
    expand/up come matrici SEPARATE, mentre TritonSwiGLUUDLP tiene un'unica expand di shape
    [2*d_ff, d] (metà SiLU, metà value). Concatenare preserva ESATTAMENTE la matematica SwiGLU
    adattandosi al layout del kernel veloce. torch_swiglu / torch_leaky_reglu2 le tengono separate.
    """
    out = {(k[9:] if k.startswith("backbone.") else k): v for k, v in sd.items()}
    if channel == "triton_swiglu":
        for key in list(out):
            if not key.endswith("channel_mixer.expand.weight"):
                continue
            up_key = key.replace("channel_mixer.expand.weight", "channel_mixer.up.weight")
            if up_key in out:
                out[key] = torch.cat([out[key], out.pop(up_key)], dim=0)
    return out


def load_guido(source: str, *, filename: str | None = None, device: str = "cuda",
               dtype: torch.dtype = torch.bfloat16, channel: str | None = None,
               for_inference: bool = True, strict: bool = True):
    """Carica un checkpoint Guido nel PiCOFormer di produzione. Ritorna (model, cfg) pronti.

    source: path locale .pt | repo HF id. filename: file esplicito nel repo HF (opzionale).
    channel: None → 'torch_swiglu' (DEFAULT, corretto per Guido-1-0.5B-Base / v5 SwiGLU, gira
             ovunque). Usa 'triton_swiglu' per il kernel veloce; usa 'torch_leaky_reglu2' SOLO
             per i checkpoint Guido LEGACY (v4 e precedenti) con FFN LeakyReGLU².
    """
    sd = _load_raw(source, filename)
    if channel is None:
        channel = DEFAULT_CHANNEL
    cfg = infer_config(sd, channel=channel)
    model = build_picoformer(cfg)
    miss, unexp = model.load_state_dict(convert_state_dict(sd, channel=channel), strict=False)
    if strict and (miss or unexp):
        raise RuntimeError(f"load impuro: missing={list(miss)[:4]} unexpected={list(unexp)[:4]}")
    if for_inference:
        model = prepare_for_inference(model, device=device, dtype=dtype)
    else:
        model = model.to(device=device, dtype=dtype)
    return model, cfg
