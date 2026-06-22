"""Single-GPU A100 PiCOFormer benchmark.

This benchmark is intentionally self-contained and SLURM friendly. It measures:

* pretrain forward+backward throughput;
* torch.compile forward/backward throughput;
* short stability training with AdamW;
* GRPO/static-cache generation, eager and graph/compile decode;
* optional A100 Triton LeakyReLU² FFN channel.

Output is both human-readable and machine-parseable via lines prefixed with
``RESULT `` containing JSON.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from Vathos.picoformer import (
    DecodeConfig,
    PiCOFormerConfig,
    build_picoformer,
    configure_runtime,
    cut_cross_entropy_available,
    pico_cross_entropy_loss,
    prepare_for_inference,
    prebuild_rope_caches,
)


@dataclass(frozen=True)
class Preset:
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int
    train_batch: int
    prompt_len: int
    max_new_tokens: int
    group_size: int
    ffn_multiplier: int = 4


PRESETS = {
    "quick": Preset(
        d_model=768,
        n_layers=16,
        n_heads=12,
        n_kv_heads=4,
        vocab_size=32768,
        seq_len=384,
        train_batch=4,
        prompt_len=64,
        max_new_tokens=128,
        group_size=16,
    ),
    "base": Preset(
        d_model=1024,
        n_layers=24,
        n_heads=16,
        n_kv_heads=4,
        vocab_size=49152,
        seq_len=512,
        train_batch=4,
        prompt_len=64,
        max_new_tokens=256,
        group_size=32,
    ),
    "guido200m": Preset(
        d_model=768,
        n_layers=24,
        n_heads=12,
        n_kv_heads=4,
        vocab_size=32768,
        seq_len=2048,
        train_batch=16,
        prompt_len=64,
        max_new_tokens=128,
        group_size=16,
    ),
}


def sync():
    torch.cuda.synchronize()


def emit(kind: str, **payload):
    payload = {"kind": kind, **payload}
    print("RESULT " + json.dumps(payload, sort_keys=True), flush=True)


def peak_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9


def reset_peak():
    torch.cuda.reset_peak_memory_stats()


def make_tokens(batch: int, seq_len: int, vocab: int, device) -> torch.Tensor:
    return torch.randint(0, vocab, (batch, seq_len), device=device, dtype=torch.long)


def amp_context(dtype: torch.dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def logits_ce_loss(model, tokens: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    with amp_context(dtype):
        logits = model(tokens)
        logits = logits[:, :-1, :].contiguous()
        targets = tokens[:, 1:].contiguous()
        return F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            targets.view(-1),
        )


def hidden_ce_loss(model, tokens: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    with amp_context(dtype):
        return pico_cross_entropy_loss(
            model,
            tokens[:, :-1].contiguous(),
            tokens[:, 1:].contiguous(),
            use_cut_cross_entropy=True,
        )


def loss_fn(model, tokens: torch.Tensor, dtype: torch.dtype,
            loss_impl: str) -> torch.Tensor:
    if loss_impl == "logits":
        return logits_ce_loss(model, tokens, dtype)
    if loss_impl == "cce":
        return hidden_ce_loss(model, tokens, dtype)
    raise ValueError(f"unsupported loss implementation: {loss_impl}")


def grad_rms(model) -> float:
    total_sq = 0.0
    total_n = 0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach().float()
            total_sq += g.square().sum().item()
            total_n += g.numel()
    return (total_sq / max(total_n, 1)) ** 0.5


def timed(fn: Callable, *, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    sync()
    reset_peak()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters, peak_gb()


def build_cfg(preset: Preset, channel: str, max_seq_len: int) -> PiCOFormerConfig:
    return PiCOFormerConfig(
        vocab_size=preset.vocab_size,
        d_model=preset.d_model,
        n_layers=preset.n_layers,
        n_heads=preset.n_heads,
        n_kv_heads=preset.n_kv_heads,
        max_seq_len=max_seq_len,
        attention="gqa",
        channel=channel,
        ffn_multiplier=preset.ffn_multiplier,
        qk_norm=True,
        logit_softcap=30.0,
        production_mode=True,
    )


def apply_preset_overrides(args, preset: Preset) -> Preset:
    updates = {}
    for field in (
        "d_model",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "vocab_size",
        "seq_len",
        "train_batch",
        "prompt_len",
        "max_new_tokens",
        "group_size",
        "ffn_multiplier",
    ):
        value = getattr(args, field)
        if value is not None:
            updates[field] = value
    return Preset(**{**asdict(preset), **updates})


def build_model(preset: Preset, channel: str, device, dtype, compile_model: bool,
                compile_mode: str):
    max_seq_len = max(preset.seq_len, preset.prompt_len + preset.max_new_tokens)
    cfg = build_cfg(preset, channel, max_seq_len)
    torch.manual_seed(1234)
    model = build_picoformer(cfg)
    model = prepare_for_inference(model, device=device, dtype=dtype)
    if compile_model:
        prebuild_rope_caches(model, max_seq_len, device=device, dtype=dtype)
        torch._dynamo.reset()
        model = torch.compile(model, mode=compile_mode, fullgraph=False)
    return model, cfg


def bench_pretrain(args, preset: Preset, channel: str, loss_impl: str, device, dtype,
                   compile_model: bool):
    label = f"compiled_{args.compile_target}" if compile_model else "eager"
    compile_module = compile_model and args.compile_target == "model"
    model, cfg = build_model(
        preset, channel, device, dtype, compile_module, args.compile_mode,
    )
    model.train()
    if compile_model and args.compile_target == "loss":
        max_seq_len = max(preset.seq_len, preset.prompt_len + preset.max_new_tokens)
        prebuild_rope_caches(model, max_seq_len, device=device, dtype=dtype)
        torch._dynamo.reset()

        def raw_loss(batch_tokens):
            return loss_fn(model, batch_tokens, dtype, loss_impl)

        compiled_loss = torch.compile(
            raw_loss, mode=args.compile_mode, fullgraph=False,
        )
    else:
        compiled_loss = None
    tokens = make_tokens(preset.train_batch, preset.seq_len, preset.vocab_size, device)

    def step():
        model.zero_grad(set_to_none=True)
        loss = (
            compiled_loss(tokens)
            if compiled_loss is not None
            else loss_fn(model, tokens, dtype, loss_impl)
        )
        loss.backward()
        return loss

    try:
        dt, gb = timed(step, warmup=args.warmup, iters=args.train_iters)
        loss = float(step().detach().cpu())
        gnorm = grad_rms(model)
        tokens_per_s = preset.train_batch * preset.seq_len / dt
        emit(
            "pretrain_fwd_bwd",
            channel=channel,
            loss_impl=loss_impl,
            cut_cross_entropy=cut_cross_entropy_available(),
            mode=label,
            preset=args.preset,
            loss=loss,
            grad_rms=gnorm,
            seconds=dt,
            tokens_per_s=tokens_per_s,
            peak_gb=gb,
            config=asdict(cfg),
        )
    except RuntimeError as exc:
        emit(
            "pretrain_fwd_bwd_error",
            channel=channel,
            loss_impl=loss_impl,
            mode=label,
            preset=args.preset,
            error=str(exc)[:500],
        )
        torch.cuda.empty_cache()
    finally:
        del model, tokens
        gc.collect()
        torch.cuda.empty_cache()


def bench_stability(args, preset: Preset, channel: str, loss_impl: str, device, dtype):
    model, _ = build_model(preset, channel, device, dtype, False, args.compile_mode)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1, fused=True)
    losses = []
    finite = True
    reset_peak()
    t0 = time.perf_counter()
    try:
        for step in range(args.stability_steps):
            tokens = make_tokens(
                preset.train_batch, preset.seq_len, preset.vocab_size, device,
            )
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model, tokens, dtype, loss_impl)
            if not torch.isfinite(loss):
                finite = False
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        sync()
        dt = time.perf_counter() - t0
        emit(
            "stability_train",
            channel=channel,
            loss_impl=loss_impl,
            cut_cross_entropy=cut_cross_entropy_available(),
            preset=args.preset,
            finite=finite,
            steps=len(losses),
            first_loss=losses[0] if losses else None,
            last_loss=losses[-1] if losses else None,
            mean_loss=sum(losses) / max(len(losses), 1),
            tokens_per_s=(len(losses) * preset.train_batch * preset.seq_len / dt)
            if dt > 0 else 0.0,
            seconds=dt,
            peak_gb=peak_gb(),
        )
    except RuntimeError as exc:
        emit(
            "stability_train_error",
            channel=channel,
            loss_impl=loss_impl,
            preset=args.preset,
            error=str(exc)[:500],
        )
    finally:
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()


def patch_greedy_sampling():
    import Vathos.blocks as blocks_mod
    import Vathos.functions as fns_mod

    orig_b = blocks_mod.sample_next_token
    orig_f = fns_mod.sample_next_token

    def greedy(logits, temperature=1.0, top_k=None, top_p=1.0):
        return logits.argmax(dim=-1, keepdim=True)

    blocks_mod.sample_next_token = greedy
    fns_mod.sample_next_token = greedy
    return blocks_mod, fns_mod, orig_b, orig_f


def restore_sampling(state):
    blocks_mod, fns_mod, orig_b, orig_f = state
    blocks_mod.sample_next_token = orig_b
    fns_mod.sample_next_token = orig_f


@torch.no_grad()
def bench_generation(args, preset: Preset, channel: str, device, dtype,
                     compile_decode: bool, return_logprobs: bool):
    model, cfg = build_model(preset, channel, device, dtype, False, args.compile_mode)
    model.eval()
    if compile_decode:
        prebuild_rope_caches(
            model,
            preset.prompt_len + preset.max_new_tokens,
            device=device,
            dtype=dtype,
        )
    prompt = make_tokens(1, preset.prompt_len, preset.vocab_size, device)[0]
    decode = DecodeConfig(
        max_new_tokens=preset.max_new_tokens,
        group_size=preset.group_size,
        temperature=1.0,
        top_k=None,
        top_p=1.0,
        eos_token_id=None,
        pad_token_id=0,
        return_logprobs=return_logprobs,
        compile_decode=compile_decode,
    )

    sampling_state = patch_greedy_sampling() if args.greedy else None

    def run():
        return model.generate_grpo(prompt, **decode.kwargs())

    try:
        dt, gb = timed(
            run,
            warmup=args.gen_compile_warmup if compile_decode else args.warmup,
            iters=args.gen_iters,
        )
        emit(
            "generation",
            channel=channel,
            preset=args.preset,
            compile_decode=compile_decode,
            return_logprobs=return_logprobs,
            greedy=args.greedy,
            seconds=dt,
            tokens_per_s=preset.group_size * preset.max_new_tokens / dt,
            peak_gb=gb,
            config=asdict(cfg),
        )
    except RuntimeError as exc:
        emit(
            "generation_error",
            channel=channel,
            preset=args.preset,
            compile_decode=compile_decode,
            return_logprobs=return_logprobs,
            error=str(exc)[:500],
        )
        torch.cuda.empty_cache()
    finally:
        if sampling_state is not None:
            restore_sampling(sampling_state)
        del model, prompt
        gc.collect()
        torch.cuda.empty_cache()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    p.add_argument("--channels", default="torch_lrelu2,triton_lrelu2_a100")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--train-iters", type=int, default=4)
    p.add_argument("--gen-iters", type=int, default=3)
    p.add_argument("--gen-compile-warmup", type=int, default=2)
    p.add_argument("--stability-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--compile-mode", default="reduce-overhead")
    p.add_argument("--compile-target", choices=("model", "loss"), default="model")
    p.add_argument("--loss-impl", choices=("logits", "cce", "both"), default="logits")
    p.add_argument("--d-model", type=int)
    p.add_argument("--n-layers", type=int)
    p.add_argument("--n-heads", type=int)
    p.add_argument("--n-kv-heads", type=int)
    p.add_argument("--vocab-size", type=int)
    p.add_argument("--seq-len", type=int)
    p.add_argument("--train-batch", type=int)
    p.add_argument("--prompt-len", type=int)
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument("--group-size", type=int)
    p.add_argument("--ffn-multiplier", type=int)
    p.add_argument("--skip-compile-train", action="store_true")
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--skip-stability", action="store_true")
    p.add_argument("--greedy", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the A100 benchmark")

    random.seed(1234)
    torch.manual_seed(1234)
    configure_runtime()
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    device = torch.device("cuda")
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    preset = apply_preset_overrides(args, PRESETS[args.preset])
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]

    print("=== PiCOFormer A100 benchmark ===", flush=True)
    print(f"host={platform.node()} python={platform.python_version()}", flush=True)
    print(f"torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    try:
        import triton
        print(f"triton={triton.__version__}", flush=True)
    except Exception as exc:
        print(f"triton=unavailable ({exc})", flush=True)
    print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(f"preset={args.preset} {preset}", flush=True)
    print(f"channels={channels}", flush=True)
    print(f"loss_impl={args.loss_impl} cut_cross_entropy={cut_cross_entropy_available()}", flush=True)
    print(f"TORCHINDUCTOR_CACHE_DIR={os.environ.get('TORCHINDUCTOR_CACHE_DIR')}", flush=True)

    emit(
        "env",
        preset=args.preset,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        dtype=args.dtype,
        channels=channels,
        loss_impl=args.loss_impl,
        cut_cross_entropy=cut_cross_entropy_available(),
    )

    loss_impls = ["logits", "cce"] if args.loss_impl == "both" else [args.loss_impl]
    for channel in channels:
        for loss_impl in loss_impls:
            bench_pretrain(args, preset, channel, loss_impl, device, dtype, compile_model=False)
            if not args.skip_compile_train:
                bench_pretrain(args, preset, channel, loss_impl, device, dtype, compile_model=True)
            if not args.skip_stability:
                bench_stability(args, preset, channel, loss_impl, device, dtype)
        if not args.skip_generation:
            bench_generation(args, preset, channel, device, dtype,
                             compile_decode=False, return_logprobs=False)
            bench_generation(args, preset, channel, device, dtype,
                             compile_decode=True, return_logprobs=False)
            bench_generation(args, preset, channel, device, dtype,
                             compile_decode=False, return_logprobs=True)

    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()
