"""grpo_speed.py — throughput DECODE (autoregressivo) di generate_grpo: eager vs compile_decode,
a varie batch. Distinto dal throughput di TRAINING (pretrain_fwd_bwd) dei bench Aplos.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
os.environ.setdefault("HF_HOME", "/leonardo_scratch/fast/IscrC_YENDRI/mprignan/.cache/huggingface")
from Vathos.picoformer import (PiCOFormerConfig, DecodeConfig, build_picoformer,
                               configure_runtime, generate_grpo_rollouts, prepare_for_inference)

CKPT = "/leonardo_work/IscrC_YENDRI/paerle/PiCO/ckpts/guido_v4/step_138000.pt"
CHANNEL = os.environ.get("CHANNEL", "torch_leaky_reglu2")

def load(device, dtype):
    cfg = PiCOFormerConfig(vocab_size=32768, d_model=1280, n_layers=24, n_heads=20, n_kv_heads=20,
        max_seq_len=8192, rope_base=1e6, attention="gated_mha", channel=CHANNEL, ffn_multiplier=3,
        qk_norm=True, attention_gate_input_dim=128, logit_softcap=30.0, tied_embeddings=True,
        smear_gate=True, smear_gate_input_dim=128)
    m = build_picoformer(cfg)
    raw = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = {(k[9:] if k.startswith("backbone.") else k): v for k, v in raw["model"].items()}
    m.load_state_dict(sd, strict=False)
    return prepare_for_inference(m, device=device, dtype=dtype)

def bench(model, prompt_ids, group, max_new, compile_decode, eos):
    dc = DecodeConfig(max_new_tokens=max_new, group_size=group, temperature=1.0,
                      eos_token_id=None, return_logprobs=False, compile_decode=compile_decode)
    # warmup (compila se compile_decode=True)
    for _ in range(2):
        generate_grpo_rollouts(model, prompt_ids, dc); torch.cuda.synchronize()
    t0 = time.perf_counter()
    b = generate_grpo_rollouts(model, prompt_ids, dc); torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    gen = group * max_new                      # eos_token_id=None → tutti generano max_new
    return gen / dt, dt

def main():
    configure_runtime()
    dev = "cuda"; dt = torch.bfloat16
    print(f"[speed] channel={CHANNEL} dev={dev}")
    model = load(dev, dt)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("mistralai/Mathstral-7B-v0.1", use_fast=True)
    ids = torch.tensor(tok.encode("### MATH\n\nProblem: compute 2+2\nSolution: ", add_special_tokens=False), device=dev)
    print(f"\n{'mode':<18}{'group':>6}{'max_new':>8}{'tok/s (decode)':>16}{'s':>8}")
    for cd in (False, True):
        for group in (8, 32):
            try:
                tps, sec = bench(model, ids, group, 128, cd, tok.eos_token_id)
                print(f"{'compile' if cd else 'eager':<18}{group:>6}{128:>8}{tps:>16,.0f}{sec:>8.2f}", flush=True)
            except Exception as e:
                print(f"{'compile' if cd else 'eager':<18}{group:>6}  FAIL: {str(e)[:60]}", flush=True)

if __name__ == "__main__":
    main()
