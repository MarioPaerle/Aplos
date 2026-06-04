"""Benchmark: generate_grpo(compile_decode=True) vs eager, on A100.

The compile_decode path attends over the full preallocated KV buffer with a
position mask (static shapes) and threads `pos` as a tensor, so
torch.compile(mode="reduce-overhead") can capture ONE CUDA graph and replay it
each decode step — removing the per-token Python + kernel-launch overhead that
made the naive static cache no faster than dynamic-cat.

    PYTHONPATH=<torch>:<repo> python3 tests/bench_graph_decode.py
"""

import time
import torch

from Vathos.blocks import PiCOFormer
from Vathos._spatials import GroupedQueryAttention, MultiheadAttentionMixer, RoPE
from Vathos._basics import Builder
import Vathos.blocks as blocks_mod


def build(kind, d_model=1024, n_layers=24, n_heads=16, n_kv_heads=4,
          vocab=49152, device="cuda", dtype=torch.bfloat16):
    head_dim = d_model // n_heads
    if kind == "gqa":
        sp = Builder(GroupedQueryAttention, n_heads=n_heads, n_kv_heads=n_kv_heads,
                     causal=True, pos_emb=RoPE(head_dim), qk_norm=True)
    else:
        sp = Builder(MultiheadAttentionMixer, n_heads=n_heads, causal=True,
                     pos_emb=RoPE(head_dim), qk_norm=True)
    m = PiCOFormer(vocab_size=vocab, d_model=d_model, n_layers=n_layers,
                   spatials=sp, logit_softcap=30.0)
    return m.to(device=device, dtype=dtype).eval(), vocab


def timed(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


@torch.no_grad()
def main():
    assert torch.cuda.is_available(), "need GPU"
    dev, dt = "cuda", torch.bfloat16
    print(f"torch={torch.__version__}  GPU={torch.cuda.get_device_name(0)}")
    blocks_mod.tqdm = lambda x, **k: x

    for kind in ("gqa", "mha"):
        m, vocab = build(kind, device=dev, dtype=dt)
        P = sum(p.numel() for p in m.parameters())
        print(f"\n=== {kind.upper()}  ~{P/1e6:.0f}M params ===")
        prompt = torch.randint(0, vocab, (64,), device=dev)

        # Correctness spot-check: greedy eager vs greedy compiled must match.
        import Vathos.functions as fns
        ob, of = blocks_mod.sample_next_token, fns.sample_next_token
        blocks_mod.sample_next_token = fns.sample_next_token = (
            lambda lg, temperature=1.0, top_k=None, top_p=1.0: lg.argmax(-1, keepdim=True))
        try:
            e = m.generate_grpo(prompt, 32, group_size=4, compile_decode=False)["completions"]
            c = m.generate_grpo(prompt, 32, group_size=4, compile_decode=True)["completions"]
            match = bool((e == c).all())
        finally:
            blocks_mod.sample_next_token, fns.sample_next_token = ob, of
        print(f"  greedy eager==compiled: {match}")

        print(f"  {'batch':>6} {'newtok':>7} {'eager_tok/s':>12} {'graph_tok/s':>12} {'speedup':>8}")
        for batch in (16, 32, 64):
            for T in (256, 512):
                def eager():
                    return m.generate_grpo(prompt, T, group_size=batch,
                                           compile_decode=False, return_logprobs=False)
                def graph():
                    return m.generate_grpo(prompt, T, group_size=batch,
                                           compile_decode=True, return_logprobs=False)
                try:
                    te = timed(eager, iters=3, warmup=1)
                    # extra warmup for compile (first call compiles + captures graph)
                    tg = timed(graph, iters=3, warmup=2)
                    tps_e, tps_g = batch * T / te, batch * T / tg
                    print(f"  {batch:>6} {T:>7} {tps_e:>12.0f} {tps_g:>12.0f} "
                          f"{tps_g/tps_e:>7.2f}x")
                except Exception as ex:
                    print(f"  {batch:>6} {T:>7}  error: {str(ex)[:60]}")
                    torch.cuda.empty_cache()
        del m
        torch.cuda.empty_cache()

    print("\n=== done ===")


if __name__ == "__main__":
    main()
