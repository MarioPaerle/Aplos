"""debug_grpo.py — verifica che il checkpoint Guido carichi nel PiCOFormer di produzione (nuovo Aplos)
e che generate_grpo funzioni. 1 GPU. Converter pesi: strip 'backbone.', qkv→q_proj+kv_proj, out→o_proj.

Run: srun ... python debug_grpo.py
"""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

CKPT = os.environ.get("CKPT", "/leonardo_work/IscrC_YENDRI/paerle/PiCO/ckpts/guido_v4/step_138000.pt")
CHANNEL = os.environ.get("CHANNEL", "torch_leaky_reglu2")
os.environ.setdefault("HF_HOME", "/leonardo_scratch/fast/IscrC_YENDRI/mprignan/.cache/huggingface")

from Vathos.picoformer import (PiCOFormerConfig, DecodeConfig, build_picoformer,
                               configure_runtime, generate_grpo_rollouts,
                               completion_logprobs, prepare_for_inference)

def convert(sd):
    """attention='gated_mha' = STESSA classe del training (qkv fuso, out) → basta togliere 'backbone.'."""
    return {(k[len("backbone."):] if k.startswith("backbone.") else k): v for k, v in sd.items()}

def main():
    configure_runtime()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    cfg = PiCOFormerConfig(
        vocab_size=32768, d_model=1280, n_layers=24, n_heads=20, n_kv_heads=20,
        max_seq_len=8192, rope_base=1_000_000.0, attention="gated_mha",
        channel=CHANNEL, ffn_multiplier=3, qk_norm=True,
        attention_gate_input_dim=128, logit_softcap=30.0, tied_embeddings=True,
        smear_gate=True, smear_gate_input_dim=128,
    )
    print(f"[debug] channel={CHANNEL} rope_base={cfg.rope_base}")
    model = build_picoformer(cfg)

    raw = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = convert(raw["model"])
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[debug] load: missing={len(miss)} unexpected={len(unexp)}  (ATTESO 0/0)")
    if miss:   print("  missing[:5]:", list(miss)[:5])
    if unexp:  print("  unexpected[:5]:", list(unexp)[:5])

    model = prepare_for_inference(model, device=device, dtype=dtype)

    # --- sanity generativa su prompt MATH reale ---
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("mistralai/Mathstral-7B-v0.1", use_fast=True)
    prompt = "### MATH\n\nProblem: What is 7 multiplied by 6?\nSolution: "
    ids = torch.tensor(tok.encode(prompt, add_special_tokens=False), device=device)
    print(f"\n[debug] prompt: {prompt!r}  ({ids.numel()} tok)")
    decode = DecodeConfig(max_new_tokens=64, group_size=1, temperature=0.0,
                          eos_token_id=tok.eos_token_id, return_logprobs=False, compile_decode=False)
    batch = generate_grpo_rollouts(model, ids, decode)
    seq = batch.sequences[0].tolist()
    gen = tok.decode(seq[ids.numel():], skip_special_tokens=True)
    print(f"[debug] GREEDY GEN:\n{gen}\n")

    # --- generate_grpo group rollout (il path che serve per GRPO) ---
    decode_g = DecodeConfig(max_new_tokens=32, group_size=4, temperature=1.0,
                            eos_token_id=tok.eos_token_id, return_logprobs=True, compile_decode=False)
    bg = generate_grpo_rollouts(model, ids, decode_g)
    lp = completion_logprobs(model, bg)
    print(f"[debug] GRPO rollouts: sequences={tuple(bg.sequences.shape)} "
          f"logprobs={tuple(lp.shape)} mean_lp={lp.mean().item():.4f}")
    print("[debug] OK — il modello carica nel PiCOFormer di produzione e generate_grpo gira.")

if __name__ == "__main__":
    main()
