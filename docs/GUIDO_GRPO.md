# Caricare Guido nel PiCOFormer di produzione + GRPO

Guida per far girare i checkpoint **Guido** (modello `train_guido_small.py` / PiCOFormerLM,
allenato nel vecchio `PiCO/aplos`) dentro il **PiCOFormer di produzione** di questo Aplos
(`Vathos.picoformer`, fast-path A100 con KV-cache statica e `generate_grpo`).

Validato su 1 GPU (A100): load pulito, generazione corretta+boxed, `generate_grpo` funzionante.
Script di riferimento: [`debug_grpo.py`](../debug_grpo.py), [`grpo_rollout.py`](../grpo_rollout.py).

## Config per Guido

```python
from Vathos.picoformer import PiCOFormerConfig, build_picoformer, prepare_for_inference

cfg = PiCOFormerConfig(
    vocab_size=32768, d_model=1280, n_layers=24, n_heads=20, n_kv_heads=20,
    max_seq_len=8192, rope_base=1_000_000.0,   # Guido usa base 1e6 (NON il default 10000)
    attention="gated_mha",                     # = MultiheadGatedAttentionMixer (qkv fuso, gate per-head)
    channel="torch_leaky_reglu2",              # = VariableGLU + LeakyReLU² (il FFN di Guido)
    ffn_multiplier=3, qk_norm=True, attention_gate_input_dim=128,
    logit_softcap=30.0, tied_embeddings=True,
    smear_gate=True, smear_gate_input_dim=128,
)
model = build_picoformer(cfg)
```

## Converter dei pesi

Con `attention="gated_mha"` lo state_dict del PiCOFormer di produzione combacia **1:1** con il
checkpoint Guido (`qkv` fuso, `out`, `gate_proj`, `q/k_norm`, `channel_mixer.{expand,up,contract}`,
`x0_lambda`, `smear_gate`). L'unica differenza è il prefisso `backbone.`:

```python
import torch
raw = torch.load("step_138000.pt", map_location="cpu", weights_only=False)
sd = {(k[9:] if k.startswith("backbone.") else k): v for k, v in raw["model"].items()}
miss, unexp = model.load_state_dict(sd, strict=False)   # → missing=0 unexpected=0
model = prepare_for_inference(model, device="cuda", dtype=torch.bfloat16)
```

## GRPO

```python
from Vathos.picoformer import DecodeConfig, generate_grpo_rollouts, completion_logprobs
ids = torch.tensor(tok.encode(prompt, add_special_tokens=False), device="cuda")
decode = DecodeConfig(max_new_tokens=256, group_size=6, temperature=0.8,
                      eos_token_id=tok.eos_token_id, return_logprobs=True, compile_decode=False)
batch = generate_grpo_rollouts(model, ids, decode)     # batch.sequences [G, L]
logp  = completion_logprobs(model, batch)              # [G, T], gradabile
```

## Cosa è stato fixato in Aplos (per la compatibilità Guido)

L'architettura **NON mancava** — `torch_leaky_reglu2` (VariableGLU+LeakyReLU²) È il FFN di Guido,
e `gated_mha` è la sua esatta classe di attention. Servivano 3 cose:

1. **`rope_base` reso configurabile** in `PiCOFormerConfig` (default 10000) e passato alla `RoPE`
   in `_spatial_builder_for` (`picoformer.py`). Guido usa **1e6**; prima era hardcoded a 10000 →
   senza questo il forward esce errato (generazione vuota/garbage).
2. **Usare `attention="gated_mha"`** (NON `gqa_gated`): gated_mha→`MultiheadGatedAttentionMixer`
   (qkv fuso, MHA piena, gate per-head) = la classe del training. `gqa_gated`→`GQAGatedMixer`
   (q_proj/kv_proj separati) è un'altra classe → forward incompatibile.
3. **Converter pesi**: solo strip del prefisso `backbone.` (nessuno split/rename).

## Benchmark (A100 64GB, group=6, eager, `compile_decode=False`, `torch_leaky_reglu2`)

Problema: *"Determine the closed form S(n) for the sum of every pair number up to n. ex: S(6)=2+4+6"*

| framing | tok generati | tempo | tok/s |
|---|---|---|---|
| MATH-wrapped (`### MATH\n\nProblem: …\nSolution: `) | 1536 | 6.57s | **234** |
| RAW (nessun wrap) | 1498 | 5.57s | **269** |

Osservazione: col **wrap MATH** il modello produce CoT strutturato in formato-soluzione
(step numerati, LaTeX, `\boxed{}`); **senza wrap** divaga come testo web. Il modello è
format-sensitive → per il GRPO usare il framing MATH-wrapped.

## TODO (solo velocità, non correttezza)

- **Kernel Triton LeakyReGLU² GLU per A100** (`triton_lreglu2_a100`): manca. I channel Triton attuali
  sono UDLP (`triton_lrelu2*`) o SwiGLU (`triton_swiglu`), nessuno è GLU+LeakyReLU². Si può aggiungere
  adattando `TritonSwiGLUUDLP` con attivazione LeakyReLU². Ora il path corretto è `torch_leaky_reglu2`
  (compile-friendly). Con kernel Triton + `compile_decode=True` la velocità sale.
