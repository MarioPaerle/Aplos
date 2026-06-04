"""Does torch.compile speed up the GRPO-relevant paths of PiCOFormer?

Two fronts, because GRPO cost has two parts:
  1. SCORING pass  — a single full-sequence forward (sequence_logprobs). Static
     shape per (batch, len) → compiles cleanly, can use reduce-overhead/CUDA graphs.
  2. DECODE loop   — token-by-token generation. The growing cache slice is a
     dynamic shape, hostile to CUDA graphs; we test torch.compile(dynamic=True)
     and the static-buffer + mask reformulation that IS graph-compatible.

Run on A100:
    PYTHONPATH=<torch>:<repo> python3 tests/bench_compile.py
"""

import time
import torch
import torch.nn.functional as F

from Vathos.blocks import PiCOFormer
from Vathos._spatials import GroupedQueryAttention, RoPE
from Vathos._basics import Builder
import Vathos.blocks as blocks_mod


def build(d_model=1024, n_layers=24, n_heads=16, n_kv_heads=4, vocab=49152,
          device="cuda", dtype=torch.bfloat16):
    head_dim = d_model // n_heads
    sp = Builder(GroupedQueryAttention, n_heads=n_heads, n_kv_heads=n_kv_heads,
                 causal=True, pos_emb=RoPE(head_dim), qk_norm=True)
    m = PiCOFormer(vocab_size=vocab, d_model=d_model, n_layers=n_layers,
                   spatials=sp, logit_softcap=30.0)
    return m.to(device=device, dtype=dtype).eval(), vocab


def timed(fn, iters=5, warmup=2):
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
    m, vocab = build(device=dev, dtype=dt)
    print(f"model ~{sum(p.numel() for p in m.parameters())/1e6:.0f}M params\n")

    # ---- Front 1: SCORING forward (sequence_logprobs) eager vs compiled --------
    print("=== Front 1: scoring forward (sequence_logprobs) ===")
    B, L = 64, 576                         # 64 rollouts, 64-prompt + 512-gen
    seq = torch.randint(0, vocab, (B, L), device=dev)

    def score_eager():
        return m.sequence_logprobs(seq)

    m_c = torch.compile(m, mode="reduce-overhead", fullgraph=False)
    def score_compiled():
        logits = m_c(seq)[:, :-1, :]
        from Vathos.functions import selective_log_softmax
        return selective_log_softmax(logits, seq[:, 1:])

    t_e = timed(score_eager)
    t_c = timed(score_compiled)
    print(f"  eager    : {t_e*1e3:8.2f} ms   ({B*L/t_e:,.0f} tok/s)")
    print(f"  compiled : {t_c*1e3:8.2f} ms   ({B*L/t_c:,.0f} tok/s)   "
          f"speedup {t_e/t_c:.2f}x\n")

    # ---- Front 2: DECODE loop eager vs compiled(dynamic) -----------------------
    print("=== Front 2: decode loop (generate_grpo) ===")
    prompt = torch.randint(0, vocab, (64,), device=dev)
    Tgen = 512

    def gen_eager():
        return m.generate_grpo(prompt, max_new_tokens=Tgen, group_size=1,
                               temperature=1.0, eos_token_id=None,
                               return_logprobs=False)

    # Compile the model's forward used inside block.generate? generate_grpo calls
    # block.generate directly, so compiling model.__call__ won't touch the loop.
    # Instead compile a standalone single-token decode closure with dynamic shapes.
    def decode_block_pass(model, x0_t):
        x_t = x0_t
        for blk in model.blocks:
            x_t = blk.generate(x_t, x0_t, ve=None)
        lg = model.unembedder(model.final_norm(x_t))[:, -1, :]
        if model.softcap > 0:
            lg = model.softcap * torch.tanh(lg / model.softcap)
        return lg

    decode_c = torch.compile(decode_block_pass, dynamic=True, fullgraph=False)

    @torch.no_grad()
    def gen_compiled_decode():
        m._clear_caches()
        p = prompt.unsqueeze(0)
        m._setup_static_caches(1, len(prompt) + Tgen, dev, dt)
        x0 = m.embedder(p)
        x = x0
        for blk in m.blocks:
            x = blk.generate(x, x0, ve=None)
        logits = m.unembedder(m.final_norm(x))[:, -1, :]
        if m.softcap > 0:
            logits = m.softcap * torch.tanh(logits / m.softcap)
        for _ in range(Tgen - 1):
            tok = logits.argmax(-1, keepdim=True)
            x0_t = m.embedder(tok)
            logits = decode_c(m, x0_t)
        m._clear_caches()

    t_e2 = timed(gen_eager, iters=3, warmup=1)
    try:
        t_c2 = timed(gen_compiled_decode, iters=3, warmup=2)  # extra warmup: compile
        print(f"  eager    : {t_e2*1e3:8.1f} ms   ({64*Tgen/t_e2:,.0f} tok/s)")
        print(f"  compiled : {t_c2*1e3:8.1f} ms   ({64*Tgen/t_c2:,.0f} tok/s)   "
              f"speedup {t_e2/t_c2:.2f}x")
    except Exception as e:
        print(f"  compiled decode failed: {str(e)[:120]}")

    print("\n=== done ===")


if __name__ == "__main__":
    main()
