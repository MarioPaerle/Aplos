"""Correctness tests for PiCOFormer.generate_grpo + StaticKVCache + sequence_logprobs.

Run (CPU is enough for correctness):
    PYTHONPATH=<torch>:<repo> python3 tests/test_grpo_generate.py

The central invariant (Test A): decoding with the preallocated StaticKVCache must
produce **numerically identical** per-position logits to a single full forward()
pass over the same sequence. If the cache were wrong (bad pos_offset, RoPE
start_pos, in-place write, or GQA expansion), mid-sequence logits would diverge.
"""

import sys
import torch
import torch.nn.functional as F

from Vathos.blocks import PiCOFormer, SmearGate
from Vathos._spatials import (MultiheadAttentionMixer, GroupedQueryAttention,
                              MultiheadGatedAttentionMixer, RoPE)
from Vathos._basics import Builder
from Vathos.functions import sequence_logprobs, selective_log_softmax, simple_ar_generate
import Vathos.blocks as blocks_mod


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def build_model(kind="mha", *, smear=False, vocab=80, d_model=64, n_layers=3, seed=0):
    """Build a small PiCOFormer. Weights randomized AWAY from identity-at-init so
    the test isn't vacuous (identity-init would make every block a near-no-op)."""
    torch.manual_seed(seed)
    head_dim = 16
    n_heads = d_model // head_dim
    if kind == "mha":
        sp = Builder(MultiheadAttentionMixer, n_heads=n_heads, causal=True,
                     pos_emb=RoPE(head_dim), qk_norm=True)
    elif kind == "gqa":
        sp = Builder(GroupedQueryAttention, n_heads=n_heads, n_kv_heads=2,
                     causal=True, pos_emb=RoPE(head_dim), qk_norm=True)
    elif kind == "gated":
        # Guido's spatial mixer: MHA + sparse per-head output gate, qk_norm, RoPE.
        sp = Builder(MultiheadGatedAttentionMixer, n_heads=n_heads, causal=True,
                     pos_emb=RoPE(head_dim), qk_norm=True, gate_input_dim=12)
    else:
        raise ValueError(kind)
    smear_gate = SmearGate(d_model) if smear else None
    m = PiCOFormer(vocab_size=vocab, d_model=d_model, n_layers=n_layers,
                   spatials=sp, logit_softcap=30.0,
                   smear_gate=smear_gate, smear_gate_lookback=(1 if smear else 0))
    # Randomize away from identity-at-init: perturb every parameter.
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.30 * torch.randn_like(p))
        if smear:
            # Make the smear actually active (lambda starts at 0 → no-op otherwise).
            m.smear_gate.smear_lambda.fill_(0.5)
    m.eval()
    return m


def forced_decode_logits(model, seq):
    """Replay a fixed sequence through the StaticKVCache decode path, capturing the
    next-token logits at every position. Returns [B, L, V] to compare with forward().

    This mirrors generate_grpo's stepping exactly (prefill 1 token, then feed the
    remaining tokens one at a time), but with *forced* tokens instead of sampling."""
    B, L = seq.shape
    device = seq.device
    dtype = next(model.parameters()).dtype
    model.eval()
    model._clear_caches()
    model._setup_static_caches(B, L, device, dtype)

    def block_pass(tokens, is_prefill):
        x0_raw = model.embedder(tokens)
        if model.smear_gate is not None:
            if model._smear_lookback > 0:
                if is_prefill:
                    x0 = model.smear_gate(x0_raw)
                    model._x_window = x0_raw[:, -model._smear_lookback:, :].clone()
                else:
                    combined = torch.cat([model._x_window, x0_raw], dim=1)
                    x0 = model.smear_gate(combined)[:, -1:, :]
                    model._x_window = combined[:, -model._smear_lookback:, :]
            else:
                x0 = model.smear_gate(x0_raw)
        else:
            x0 = x0_raw
        ves = model._compute_ves(tokens)
        x = x0
        for i, block in enumerate(model.blocks):
            x = block.generate(x, x0, ve=ves[i])
        lg = model.unembedder(model.final_norm(x))[:, -1, :]
        if model.softcap > 0:
            lg = model.softcap * torch.tanh(lg / model.softcap)
        return lg

    all_logits = []
    # Prefill on the first token, then feed the rest one by one.
    all_logits.append(block_pass(seq[:, :1], is_prefill=True))
    for t in range(1, L):
        all_logits.append(block_pass(seq[:, t:t + 1], is_prefill=False))
    model._clear_caches()
    return torch.stack(all_logits, dim=1)  # [B, L, V]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_A_cache_equals_forward():
    """StaticKVCache decode logits == full forward() logits (the cache invariant)."""
    print("\n[A] StaticKVCache decode == forward() — per-position logits")
    ok = True
    for kind in ("mha", "gqa", "gated"):
        for smear in (False, True):
            m = build_model(kind, smear=smear, seed=1)
            torch.manual_seed(42)
            seq = torch.randint(0, m.vocab_size, (3, 24))
            full = m(seq)                              # [B, L, V]
            cached = forced_decode_logits(m, seq)      # [B, L, V]
            max_diff = (full - cached).abs().max().item()
            # Sanity: logits must be non-trivial (not collapsed to ~0 by identity-init)
            spread = full.std().item()
            passed = max_diff < 1e-4 and spread > 0.5
            ok &= passed
            print(f"    {kind:4s} smear={smear!s:5s}  max|Δlogit|={max_diff:.2e}  "
                  f"logit_std={spread:.2f}  -> {'PASS' if passed else 'FAIL'}")
    return ok


def test_B_grpo_matches_reference():
    """Greedy generate_grpo (group=1) == greedy simple_ar_generate (no-cache ref)."""
    print("\n[B] generate_grpo == simple_ar_generate (greedy, token-exact)")
    ok = True

    # Patch sampling to deterministic greedy in BOTH namespaces: generate_grpo
    # resolves sample_next_token in Vathos.blocks, simple_ar_generate in
    # Vathos.functions. Patching only one leaves the other stochastic.
    import Vathos.functions as fns_mod
    orig_b, orig_f = blocks_mod.sample_next_token, fns_mod.sample_next_token
    def greedy(logits, temperature=1.0, top_k=None, top_p=1.0):
        return logits.argmax(dim=-1, keepdim=True)
    blocks_mod.sample_next_token = greedy
    fns_mod.sample_next_token = greedy
    try:
        for kind in ("mha", "gqa", "gated"):
            m = build_model(kind, seed=2)
            prompt = torch.tensor([3, 7, 1, 9, 2, 5])
            ref = simple_ar_generate(m, prompt, max_len=20, temperature=1.0,
                                     top_k=None, top_p=1.0, token_end=None)
            ref_comp = ref[0, len(prompt):]
            out = m.generate_grpo(prompt, max_new_tokens=19, group_size=1,
                                  temperature=1.0, eos_token_id=None)
            grpo_comp = out["completions"][0]
            # simple_ar_generate produces max_len-1 new tokens via its own loop;
            # compare the overlapping prefix.
            n = min(ref_comp.numel(), grpo_comp.numel())
            match = bool((ref_comp[:n] == grpo_comp[:n]).all())
            ok &= match
            print(f"    {kind:4s}  compared {n} tokens, identical={match} "
                  f"-> {'PASS' if match else 'FAIL'}")
    finally:
        blocks_mod.sample_next_token = orig_b
        fns_mod.sample_next_token = orig_f
    return ok


def test_C_group_batch():
    """group_size>1 near-greedy → all rollouts identical (batched decode correct)."""
    print("\n[C] batched group decode (greedy → identical rows from same prompt)")
    m = build_model("gqa", seed=3)
    # True greedy (argmax) — replicated prompt must give identical rollouts, which
    # proves the batched static cache has no cross-row contamination. (Avoid tiny
    # temperatures: logit/0.01 overflows softmax to nan.)
    import Vathos.functions as fns_mod
    orig_b, orig_f = blocks_mod.sample_next_token, fns_mod.sample_next_token
    def greedy(logits, temperature=1.0, top_k=None, top_p=1.0):
        return logits.argmax(dim=-1, keepdim=True)
    blocks_mod.sample_next_token = greedy
    fns_mod.sample_next_token = greedy
    try:
        prompt = torch.tensor([4, 8, 2, 6])
        out = m.generate_grpo(prompt, max_new_tokens=12, group_size=5,
                              temperature=1.0, eos_token_id=None)
    finally:
        blocks_mod.sample_next_token = orig_b
        fns_mod.sample_next_token = orig_f
    comps = out["completions"]                 # [5, 12]
    all_equal = bool((comps == comps[0:1]).all())
    print(f"    group_size=5  shape={tuple(comps.shape)}  "
          f"all rows identical={all_equal} -> {'PASS' if all_equal else 'FAIL'}")
    return all_equal


def test_D_eos_masking():
    """EOS semantics: mask is 1 up to & incl first EOS per row, 0 after; pad fill."""
    print("\n[D] per-sequence EOS masking")
    m = build_model("mha", seed=4)
    EOS, PAD = 0, 70   # both within vocab (80); finished rows are fed PAD
    # Script a deterministic token stream per row so EOS lands at a known step.
    # row0 emits EOS at step 2; row1 at step 4; row2 never.
    scripted = {
        0: [11, 12, EOS, 13, 14, 15, 16],
        1: [21, 22, 23, 24, EOS, 25, 26],
        2: [31, 32, 33, 34, 35, 36, 37],
    }
    step = {"t": 0}
    orig = blocks_mod.sample_next_token
    def scripted_sample(logits, temperature=1.0, top_k=None, top_p=1.0):
        t = step["t"]; step["t"] += 1
        return torch.tensor([[scripted[r][t]] for r in range(logits.shape[0])])
    blocks_mod.sample_next_token = scripted_sample
    try:
        prompt = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        out = m.generate_grpo(prompt, max_new_tokens=7, group_size=1,
                              temperature=1.0, eos_token_id=EOS, pad_token_id=PAD,
                              return_logprobs=True)
    finally:
        blocks_mod.sample_next_token = orig

    mask = out["completion_mask"]      # [3, 7]
    comp = out["completions"]          # [3, 7]
    lp = out["sampling_logprobs"]      # [3, 7]
    expected_mask = torch.tensor([
        [1, 1, 1, 0, 0, 0, 0],   # row0: EOS at t=2 → mask 1 through t=2, 0 after
        [1, 1, 1, 1, 1, 0, 0],   # row1: EOS at t=4
        [1, 1, 1, 1, 1, 1, 1],   # row2: never
    ], dtype=mask.dtype)
    mask_ok = bool((mask == expected_mask).all())
    # finished rows are pad-filled
    pad_ok = (comp[0, 3:] == PAD).all() and (comp[1, 5:] == PAD).all()
    # logprobs zeroed where masked
    lp_ok = bool((lp[expected_mask == 0] == 0).all())
    passed = mask_ok and bool(pad_ok) and lp_ok
    print(f"    mask_ok={mask_ok} pad_ok={bool(pad_ok)} logp_masked={lp_ok} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


def test_E_sequence_logprobs():
    """sequence_logprobs matches manual log_softmax+gather and is differentiable."""
    print("\n[E] sequence_logprobs correctness + grad flow")
    m = build_model("mha", seed=5)
    seq = torch.randint(0, m.vocab_size, (2, 14))

    lp = m.sequence_logprobs(seq)                     # [2, 13]
    # Manual reference
    logits = m(seq)[:, :-1, :]
    manual = F.log_softmax(logits.float(), dim=-1).gather(
        -1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)
    diff = (lp - manual).abs().max().item()
    num_ok = diff < 1e-5

    # Grad flows to parameters
    m.zero_grad(set_to_none=True)
    loss = -m.sequence_logprobs(seq).mean()
    loss.backward()
    grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    passed = num_ok and grad_ok
    print(f"    max|Δ vs manual|={diff:.2e}  grad_flows={grad_ok} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


def test_F_no_eos_full_length():
    """eos_token_id=None → mask all ones, full max_new_tokens generated."""
    print("\n[F] no-EOS run → full length, mask all ones")
    m = build_model("gqa", seed=6)
    out = m.generate_grpo(torch.tensor([1, 2, 3]), max_new_tokens=10,
                          group_size=2, temperature=1.0, eos_token_id=None)
    mask_ok = bool((out["completion_mask"] == 1).all())
    shape_ok = out["completions"].shape == (2, 10)
    passed = mask_ok and shape_ok
    print(f"    mask_all_ones={mask_ok} shape_ok={shape_ok} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


def test_G_early_break_no_garbage():
    """After an early all-finished break, sequences must contain only valid ids
    (pad-filled, not torch.empty garbage) so sequence_logprobs() doesn't crash."""
    print("\n[G] early-break leaves valid (pad) ids, sequence_logprobs survives")
    m = build_model("mha", seed=7)
    EOS, PAD = 0, 5
    # Script: every row emits EOS at step 1 → break at t=1, positions 2..T unwritten.
    orig = blocks_mod.sample_next_token
    def emit_eos_fast(logits, temperature=1.0, top_k=None, top_p=1.0):
        # step 0 -> some token, step 1 -> EOS for all rows
        emit_eos_fast.t += 1
        if emit_eos_fast.t == 1:
            return torch.full((logits.shape[0], 1), 11, dtype=torch.long)
        return torch.full((logits.shape[0], 1), EOS, dtype=torch.long)
    emit_eos_fast.t = 0
    blocks_mod.sample_next_token = emit_eos_fast
    try:
        out = m.generate_grpo(torch.tensor([1, 2, 3]), max_new_tokens=10,
                              group_size=3, eos_token_id=EOS, pad_token_id=PAD)
    finally:
        blocks_mod.sample_next_token = orig
    seq = out["sequences"]
    in_vocab = bool(((seq >= 0) & (seq < m.vocab_size)).all())
    # The real test: sequence_logprobs must run without an out-of-range embed crash.
    try:
        lp = m.sequence_logprobs(seq)
        scored_ok = lp.shape == (3, seq.shape[1] - 1)
    except Exception as e:
        scored_ok = False
        print(f"    sequence_logprobs crashed: {e}")
    passed = in_vocab and scored_ok
    print(f"    all ids in vocab={in_vocab}  sequence_logprobs ok={scored_ok} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


def graph_forced_decode_logits(model, seq):
    """Replay a fixed sequence through the CUDA-graph-compatible decode path
    (graph_generate: full-buffer masked SDPA + index_copy_ + rotate_at at a tensor
    position), eagerly (no torch.compile). Returns [B, L, V] to compare to forward()."""
    B, L = seq.shape
    device = seq.device
    dtype = next(model.parameters()).dtype
    model.eval()
    model._clear_caches()
    model._setup_static_caches(B, L, device, dtype)
    for blk in model.blocks:
        sm = blk.spatial_mixer
        if getattr(sm, "pos_emb", None) is not None and hasattr(sm.pos_emb, "prebuild"):
            sm.pos_emb.prebuild(L, device, dtype)
    arange = torch.arange(L, device=device)

    smear_on = (model.smear_gate is not None and model._smear_lookback > 0)

    outs = []
    # Prefill position 0 via the slice path (fills cache[0]); pos advances to 1.
    x0_raw = model.embedder(seq[:, :1])
    if smear_on:
        x0 = model.smear_gate(x0_raw)               # L==1 → no-op, but mirrors prefill
        model._x_window = x0_raw[:, -model._smear_lookback:, :].clone()
    else:
        x0 = x0_raw
    x = x0
    for blk in model.blocks:
        x = blk.generate(x, x0, ve=None)
    lg = model.unembedder(model.final_norm(x))[:, -1, :]
    if model.softcap > 0:
        lg = model.softcap * torch.tanh(lg / model.softcap)
    outs.append(lg)
    # Decode positions 1..L-1 via the graph path (mirrors generate_grpo._graph_decode).
    for t in range(1, L):
        pos = torch.tensor([t], device=device)
        mask = (arange <= pos).view(1, 1, 1, -1)
        x0t_raw = model.embedder(seq[:, t:t + 1])
        if smear_on:
            combined = torch.cat([model._x_window, x0t_raw], dim=1)
            x0t = model.smear_gate(combined)[:, -1:, :]
            model._x_window.copy_(combined[:, -model._smear_lookback:, :])
        else:
            x0t = x0t_raw
        xt = x0t
        for blk in model.blocks:
            xt = blk.graph_generate(xt, x0t, pos, mask)
        lg = model.unembedder(model.final_norm(xt))[:, -1, :]
        if model.softcap > 0:
            lg = model.softcap * torch.tanh(lg / model.softcap)
        outs.append(lg)
    model._clear_caches()
    return torch.stack(outs, dim=1)


def test_I_graph_path_equals_forward():
    """CUDA-graph-compatible decode (full-buffer mask + index_copy_ + tensor-pos
    RoPE) == full forward(). This is the correctness invariant for compile_decode."""
    print("\n[I] graph_generate decode (full-buffer mask) == forward()")
    ok = True
    for kind in ("mha", "gqa", "gated"):
        for smear in (False, True):
            m = build_model(kind, smear=smear, seed=11)
            torch.manual_seed(99)
            seq = torch.randint(0, m.vocab_size, (3, 28))
            full = m(seq)
            graph = graph_forced_decode_logits(m, seq)
            max_diff = (full - graph).abs().max().item()
            passed = max_diff < 1e-4 and full.std().item() > 0.5
            ok &= passed
            print(f"    {kind:5s} smear={smear!s:5s}  max|Δlogit|={max_diff:.2e} "
                  f"-> {'PASS' if passed else 'FAIL'}")
    return ok


def test_H_init_ratio_is_one():
    """At init pi_old (sampling_logprobs) == pi_theta (sequence_logprobs) on the
    completion tokens, for ANY sampling temperature → GRPO ratio == 1."""
    print("\n[H] pi_old == pi_theta at init (importance ratio == 1)")
    ok = True
    for temp in (1.0, 0.7, 1.3):
        m = build_model("gqa", seed=8)
        out = m.generate_grpo(torch.tensor([2, 4, 6, 8]), max_new_tokens=8,
                              group_size=2, temperature=temp, eos_token_id=None,
                              return_logprobs=True)
        P, T = out["prompt_len"], 8
        full_lp = m.sequence_logprobs(out["sequences"])        # [BG, L-1]
        pi_theta = full_lp[:, P - 1:P - 1 + T]                  # align to completion
        pi_old = out["sampling_logprobs"]                      # [BG, T]
        max_logratio = (pi_theta - pi_old).abs().max().item()
        passed = max_logratio < 1e-4
        ok &= passed
        print(f"    temp={temp}  max|log(pi_theta/pi_old)|={max_logratio:.2e} "
              f"-> {'PASS' if passed else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = {
        "A cache==forward":      test_A_cache_equals_forward(),
        "B grpo==reference":     test_B_grpo_matches_reference(),
        "C group batch":         test_C_group_batch(),
        "D eos masking":         test_D_eos_masking(),
        "E sequence_logprobs":   test_E_sequence_logprobs(),
        "F no-eos full length":  test_F_no_eos_full_length(),
        "G early-break no garbage": test_G_early_break_no_garbage(),
        "H init ratio == 1":     test_H_init_ratio_is_one(),
        "I graph-path==forward": test_I_graph_path_equals_forward(),
    }
    print("\n" + "=" * 60)
    n_pass = sum(results.values())
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n{n_pass}/{len(results)} tests passed.")
    sys.exit(0 if n_pass == len(results) else 1)
