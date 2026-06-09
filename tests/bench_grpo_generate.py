"""Benchmark: StaticKVCache generate_grpo vs dynamic-cat generate() on GPU.

Measures decode throughput (tokens/sec) and peak memory for the GRPO rollout
pattern (one prompt replicated to a batch of G rollouts), sweeping over batch size
and generation length. Run on an A100 via SLURM (see bench_grpo.slurm).

    PYTHONPATH=<torch>:<repo> python3 tests/bench_grpo_generate.py
"""

import time
import torch

from Vathos.blocks import PiCOFormer
from Vathos._spatials import GroupedQueryAttention, MultiheadAttentionMixer, RoPE
from Vathos._basics import Builder
import Vathos.blocks as blocks_mod
import Vathos.functions as fns_mod


def build(kind, d_model, n_layers, n_heads, n_kv_heads, vocab, device, dtype):
    head_dim = d_model // n_heads
    if kind == "gqa":
        sp = Builder(GroupedQueryAttention, n_heads=n_heads, n_kv_heads=n_kv_heads,
                     causal=True, pos_emb=RoPE(head_dim), qk_norm=True)
    else:
        sp = Builder(MultiheadAttentionMixer, n_heads=n_heads, causal=True,
                     pos_emb=RoPE(head_dim), qk_norm=True)
    m = PiCOFormer(vocab_size=vocab, d_model=d_model, n_layers=n_layers,
                   spatials=sp, logit_softcap=30.0)
    return m.to(device=device, dtype=dtype).eval()


def n_params(m):
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def dynamic_generate(model, prompt, max_new, batch):
    """Old path: model.generate() (dynamic torch.cat cache), looped per row would be
    slow, so we replicate the prompt to a batch and call the dynamic generate() once.
    generate() supports batched prompt via its prefill, so this is a fair decode-speed
    comparison of dynamic-cat vs static-cache at the same batch."""
    p = prompt.unsqueeze(0).expand(batch, -1).contiguous()
    return model.generate(p, max_len=max_new + 1, temperature=1.0, simple=False)


@torch.no_grad()
def static_generate(model, prompt, max_new, batch):
    return model.generate_grpo(prompt, max_new_tokens=max_new, group_size=batch,
                               temperature=1.0, eos_token_id=None,
                               return_logprobs=False)["sequences"]


def time_fn(fn, *args, warmup=1, iters=3, device="cuda"):
    for _ in range(warmup):
        fn(*args)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
    return dt, peak


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"device={device} dtype={dtype} torch={torch.__version__}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # tqdm in PiCOFormer.generate would spam; silence it.
    blocks_mod.tqdm = lambda x, **k: x

    # Representative small-LLM config (~0.5B class is heavy for the dbg queue; use a
    # mid config that still exercises the kernels meaningfully). Override via env if needed.
    cfg = dict(d_model=1024, n_layers=24, n_heads=16, n_kv_heads=4, vocab=49152)
    prompt_len = 64

    for kind in ("gqa", "mha"):
        m = build(kind, device=device, dtype=dtype, **cfg)
        P = n_params(m)
        print(f"\n=== {kind.upper()}  ~{P/1e6:.0f}M params  "
              f"d_model={cfg['d_model']} L={cfg['n_layers']} "
              f"heads={cfg['n_heads']}/{cfg['n_kv_heads']}kv ===")
        prompt = torch.randint(0, cfg["vocab"], (prompt_len,), device=device)

        print(f"{'batch':>6} {'newtok':>7} "
              f"{'dyn_tok/s':>11} {'stat_tok/s':>11} {'speedup':>8} "
              f"{'dyn_GB':>7} {'stat_GB':>8}")
        for batch in (8, 16, 32, 64):
            for max_new in (256, 512):
                try:
                    dt_d, gb_d = time_fn(dynamic_generate, m, prompt, max_new, batch,
                                         device=device)
                    dt_s, gb_s = time_fn(static_generate, m, prompt, max_new, batch,
                                         device=device)
                    tps_d = batch * max_new / dt_d
                    tps_s = batch * max_new / dt_s
                    print(f"{batch:>6} {max_new:>7} "
                          f"{tps_d:>11.0f} {tps_s:>11.0f} {tps_s/tps_d:>7.2f}x "
                          f"{gb_d:>7.1f} {gb_s:>8.1f}")
                except RuntimeError as e:
                    print(f"{batch:>6} {max_new:>7}  OOM/error: {str(e)[:50]}")
                    if device == "cuda":
                        torch.cuda.empty_cache()
        del m
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
