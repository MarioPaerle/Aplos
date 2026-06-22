#!/usr/bin/env python3
"""Fast A100 PiCOFormer pretraining entrypoint.

This is the bridge from the old Guido monolith to the newer Aplos library:
same practical training style, but model/loss/generation surfaces come from
``Vathos.picoformer`` so agents only need to change config flags instead of
forking architecture code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from Vathos.picoformer import (
    PiCOFormerConfig,
    build_picoformer,
    configure_picoformer_training_runtime,
    cut_cross_entropy_available,
    linear_lm_cross_entropy,
    pico_classifier_weight,
    pico_hidden_states,
    prebuild_rope_caches,
)


CORPUS_V2 = "/leonardo_scratch/fast/IscrC_YENDRI/mprignan/corpus_v2"
CORPUS_V4 = "/leonardo_scratch/large/userexternal/mprignan/corpus_v4"

MIXTURE_PRESETS = {
    "default": {
        "openmath_full": 0.34,
        "tinygsm": 0.14,
        "openmathreasoning": 0.10,
        "numina_15": 0.07,
        "numina_cot": 0.05,
        "fineweb_edu": 0.22,
        "cosmopedia": 0.08,
    },
    "v3_noloops": {
        "openmath_full": 0.37,
        "tinygsm": 0.15,
        "openmathreasoning": 0.11,
        "numina_15": 0.08,
        "numina_cot": 0.06,
        "fineweb_edu": 0.15,
        "cosmopedia": 0.08,
    },
    "fineweb_only": {"fineweb_edu": 1.0},
    "fineweb_v4_only": {"fineweb_edu_v4": 1.0},
}

MODEL_PRESETS = {
    "debug": dict(n_layers=2, d_model=128, n_heads=4, n_kv_heads=1),
    "small": dict(n_layers=12, d_model=512, n_heads=8, n_kv_heads=2),
    "guido95m": dict(n_layers=24, d_model=512, n_heads=8, n_kv_heads=8),
    "guido150m": dict(n_layers=24, d_model=640, n_heads=10, n_kv_heads=2),
    "guido200m": dict(n_layers=24, d_model=768, n_heads=12, n_kv_heads=3),
}


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_mixture(spec: str | None, preset: str) -> dict[str, float]:
    if not spec:
        if preset not in MIXTURE_PRESETS:
            raise ValueError(f"unknown mixture preset {preset!r}")
        return dict(MIXTURE_PRESETS[preset])
    out: dict[str, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, weight = item.split("=", 1)
        out[name.strip()] = float(weight)
    total = sum(out.values())
    if total <= 0:
        raise ValueError("mixture weights must sum to a positive value")
    return {k: v / total for k, v in out.items()}


def dtype_from_name(name: str) -> torch.dtype:
    lookup = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name not in lookup:
        raise ValueError(f"unsupported dtype {name!r}")
    return lookup[name]


def init_distributed() -> tuple[bool, int, int, int]:
    rank = int(os.environ.get("RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = rank >= 0 and world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    else:
        rank = 0
        world_size = 1
    return distributed, rank, local_rank, world_size


def rank0(rank: int, *parts: object) -> None:
    if rank == 0:
        print(*parts, flush=True)


def seed_everything(seed: int, rank: int) -> None:
    seed = seed + rank * 100_003
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def np_dtype(name: str) -> np.dtype:
    allowed = {"uint16": np.uint16, "uint32": np.uint32}
    if name not in allowed:
        raise ValueError(f"unsupported shard dtype {name!r}")
    return np.dtype(allowed[name])


class ShardDirectory:
    """Memory-mapped reader for Guido ``corpus_v2``/``corpus_v4`` shard dirs."""

    def __init__(self, shard_dir: str | Path):
        self.shard_dir = Path(shard_dir)
        with (self.shard_dir / "index.json").open("r", encoding="utf-8") as f:
            self.index = json.load(f)
        self.dtype = np_dtype(self.index.get("dtype", "uint32"))
        self.vocab_size = int(self.index["vocab_size"])
        self.eos_id = int(self.index["eos_id"])
        self.shards = list(self.index["shards"])
        self._cum_offsets = np.zeros(len(self.shards) + 1, dtype=np.int64)
        for i, shard in enumerate(self.shards):
            self._cum_offsets[i + 1] = self._cum_offsets[i] + int(shard["num_tokens"])
        self._mmaps: list[np.ndarray | None] = [None] * len(self.shards)

    @property
    def total_tokens(self) -> int:
        return int(self._cum_offsets[-1])

    def _get_mmap(self, shard_idx: int) -> np.ndarray:
        mmap = self._mmaps[shard_idx]
        if mmap is None:
            path = self.shard_dir / self.shards[shard_idx]["filename"]
            mmap = np.memmap(path, dtype=self.dtype, mode="r")
            self._mmaps[shard_idx] = mmap
        return mmap

    def read(self, start_global: int, length: int) -> np.ndarray:
        if start_global < 0 or start_global + length > self.total_tokens:
            raise IndexError(
                f"range [{start_global}, {start_global + length}) outside "
                f"{self.total_tokens} tokens for {self.shard_dir}"
            )
        shard_start = int(np.searchsorted(self._cum_offsets, start_global, side="right") - 1)
        local_start = start_global - int(self._cum_offsets[shard_start])
        mmap = self._get_mmap(shard_start)
        if local_start + length <= len(mmap):
            return mmap[local_start: local_start + length]

        out = np.empty(length, dtype=self.dtype)
        n_read = 0
        shard_idx = shard_start
        local = local_start
        while n_read < length:
            mmap = self._get_mmap(shard_idx)
            take = min(len(mmap) - local, length - n_read)
            out[n_read: n_read + take] = mmap[local: local + take]
            n_read += take
            shard_idx += 1
            local = 0
        return out


class ShuffledStream:
    """Non-overlapping fixed windows, shuffled without replacement per epoch."""

    def __init__(self, reader: ShardDirectory, rank: int, world_size: int,
                 seq_len: int, seed: int):
        self.reader = reader
        self.win = seq_len + 1
        self.n_chunks = reader.total_tokens // self.win
        if self.n_chunks <= 0:
            raise ValueError(f"{reader.shard_dir} is too small for seq_len={seq_len}")
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.ptr = 0
        self.my = np.empty(0, dtype=np.int64)
        self._reshuffle()

    def _reshuffle(self) -> None:
        rng = np.random.default_rng(self.seed + 104_729 * self.epoch)
        self.my = rng.permutation(self.n_chunks)[self.rank::self.world_size]
        self.ptr = 0
        self.epoch += 1

    def next_window(self) -> np.ndarray:
        if self.ptr >= len(self.my):
            self._reshuffle()
        chunk_idx = int(self.my[self.ptr])
        self.ptr += 1
        return self.reader.read(chunk_idx * self.win, self.win)


class MultiShardMixture:
    """Weighted dataset sampler over memory-mapped shard directories."""

    def __init__(self, mixture: dict[str, float], corpus_root: str | Path,
                 rank: int, world_size: int, seq_len: int, seed: int):
        self.corpus_root = Path(corpus_root)
        self.names = list(mixture)
        weights = np.asarray([mixture[name] for name in self.names], dtype=np.float64)
        self.weights = weights / weights.sum()
        self.streams: dict[str, ShuffledStream] = {}
        self.vocab_size: int | None = None
        self.eos_id: int | None = None
        for i, name in enumerate(self.names):
            reader = ShardDirectory(self.corpus_root / name)
            if self.vocab_size is None:
                self.vocab_size = reader.vocab_size
                self.eos_id = reader.eos_id
            if reader.vocab_size != self.vocab_size:
                raise ValueError(
                    f"vocab mismatch for {name}: {reader.vocab_size} != {self.vocab_size}"
                )
            self.streams[name] = ShuffledStream(
                reader, rank, world_size, seq_len, seed + i * 7_919,
            )
        self.rng = np.random.default_rng(seed * 31 + rank)
        self.seq_len = seq_len
        self.win = seq_len + 1

    def next_batch(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor, dict[str, int]]:
        picks = self.rng.choice(len(self.names), size=batch_size, p=self.weights)
        batch = np.empty((batch_size, self.win), dtype=np.int64)
        counts = {name: 0 for name in self.names}
        for row, pick in enumerate(picks):
            name = self.names[int(pick)]
            batch[row] = self.streams[name].next_window()
            counts[name] += 1
        tokens = torch.from_numpy(batch).to(device=device, non_blocking=True)
        return tokens[:, :-1], tokens[:, 1:], counts


def zeropower_via_newtonschulz5(grad: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Keller Jordan Muon quintic Newton-Schulz orthogonalizer."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = grad.bfloat16()
    x = x / (x.norm() + eps)
    transposed = grad.size(0) > grad.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * a_mat @ a_mat
        x = a * x + b_mat @ x
    return x.T if transposed else x


class Muon(torch.optim.Optimizer):
    """Small self-contained Muon optimizer used by Guido runs."""

    def __init__(self, params, lr: float, momentum: float,
                 backend_steps: int, nesterov: bool = True):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            backend_steps=backend_steps,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            total = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)
            cursor = 0
            for i, param in enumerate(params):
                if i % world_size == rank and param.grad is not None:
                    grad = param.grad
                    state = self.state[param]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    update_src = grad.add(buf, alpha=momentum) if nesterov else buf
                    update = zeropower_via_newtonschulz5(
                        update_src, steps=ns_steps,
                    )
                    update *= max(1.0, update.size(0) / update.size(1)) ** 0.5
                    updates_flat[cursor: cursor + param.numel()] = update.reshape(-1)
                cursor += param.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            cursor = 0
            for param in params:
                update = updates_flat[cursor: cursor + param.numel()].view_as(param)
                param.add_(update.to(param.dtype), alpha=-lr)
                cursor += param.numel()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    while isinstance(model, DDP) or hasattr(model, "_orig_mod"):
        if isinstance(model, DDP):
            model = model.module
        elif hasattr(model, "_orig_mod"):
            model = model._orig_mod
    return model


def split_optimizer_params(model: torch.nn.Module) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    core = unwrap_model(model)
    matrix_params: list[Tensor] = []
    embed_params: list[Tensor] = []
    scalar_params: list[Tensor] = []
    for name, param in core.named_parameters(remove_duplicate=True):
        if not param.requires_grad:
            continue
        if "embedder" in name or "unembedder" in name:
            embed_params.append(param)
        elif param.ndim == 2:
            matrix_params.append(param)
        else:
            scalar_params.append(param)
    return matrix_params, embed_params, scalar_params


def make_optimizers(args: argparse.Namespace, model: torch.nn.Module,
                    device: torch.device) -> tuple[list[torch.optim.Optimizer], list[str]]:
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.adam_lr,
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
            fused=device.type == "cuda",
        )
        for group in optimizer.param_groups:
            group["base_lr"] = args.adam_lr
        return [optimizer], ["adamw"]

    matrix_params, embed_params, scalar_params = split_optimizer_params(model)
    optimizers: list[torch.optim.Optimizer] = []
    names: list[str] = []
    opt_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in opt_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizers.append(opt_muon)
    names.append("muon")

    if embed_params:
        opt_embed = torch.optim.Adam(
            [{"params": embed_params, "lr": args.embed_lr, "base_lr": args.embed_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=device.type == "cuda",
        )
        optimizers.append(opt_embed)
        names.append("embed_adam")
    if scalar_params:
        opt_scalar = torch.optim.Adam(
            [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=device.type == "cuda",
        )
        optimizers.append(opt_scalar)
        names.append("scalar_adam")
    return optimizers, names


def lr_multiplier(step: int, args: argparse.Namespace) -> float:
    if args.warmup_steps > 0 and step < args.warmup_steps:
        return max(1e-8, (step + 1) / args.warmup_steps)
    if args.lr_decay_steps <= 0:
        return 1.0
    progress = min(1.0, max(0.0, (step - args.warmup_steps) / args.lr_decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine


def update_optimizer_schedules(optimizers: list[torch.optim.Optimizer],
                               step: int, args: argparse.Namespace) -> None:
    mult = lr_multiplier(step, args)
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = group.get("base_lr", group["lr"]) * mult
            if "momentum" in group and args.muon_warmup_steps > 0:
                t = min(1.0, (step + 1) / args.muon_warmup_steps)
                group["momentum"] = (
                    args.muon_momentum_warmup_start
                    + t * (args.muon_momentum - args.muon_momentum_warmup_start)
                )


def build_loss_fn(args: argparse.Namespace, model: torch.nn.Module):
    if args.loss == "logits":
        def logits_loss(input_ids: Tensor, targets: Tensor) -> Tensor:
            logits = model(input_ids)
            return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits_loss

    if dist.is_available() and dist.is_initialized():
        raise RuntimeError(
            f"loss={args.loss!r} is intentionally single-GPU in train_A100.py for now"
        )
    use_cce = args.loss in {"cce", "cce_auto"}
    if args.loss == "cce" and not cut_cross_entropy_available():
        raise RuntimeError(
            "loss='cce' requires the optional cut_cross_entropy package for this "
            "Torch environment. Use loss='cce_auto' to allow fallback or "
            "loss='hidden_logits' to benchmark the fallback explicitly."
        )

    core = unwrap_model(model)

    def hidden_loss(input_ids: Tensor, targets: Tensor) -> Tensor:
        hidden = pico_hidden_states(core, input_ids)
        weight = pico_classifier_weight(core)
        return linear_lm_cross_entropy(
            hidden,
            weight,
            targets,
            use_cut_cross_entropy=use_cce,
            softcap=core.softcap,
        )

    if args.compile_model:
        return torch.compile(hidden_loss, mode=args.compile_mode, fullgraph=False, dynamic=False)
    return hidden_loss


def save_checkpoint(path: Path, step: int, model: torch.nn.Module,
                    cfg: PiCOFormerConfig, args: argparse.Namespace,
                    optimizers: list[torch.optim.Optimizer],
                    optimizer_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    core = unwrap_model(model)
    torch.save(
        {
            "step": step,
            "model": core.state_dict(),
            "config": asdict(cfg),
            "args": vars(args),
            "optimizers": [opt.state_dict() for opt in optimizers],
            "optimizer_names": optimizer_names,
        },
        path,
    )


def run_generation_smoke(model: torch.nn.Module, args: argparse.Namespace,
                         device: torch.device, rank: int) -> None:
    if args.generate_smoke_tokens <= 0 or rank != 0:
        return
    core = unwrap_model(model)
    core.eval()
    prompt = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.generate_smoke_batch, args.generate_smoke_prompt),
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        out = core.generate_grpo(
            prompt,
            max_new_tokens=args.generate_smoke_tokens,
            group_size=args.generate_smoke_groups,
            temperature=1.0,
            top_k=None,
            top_p=1.0,
            eos_token_id=None,
            pad_token_id=0,
            return_logprobs=False,
            compile_decode=args.compile_decode_smoke,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(1e-9, time.perf_counter() - start)
    generated = int(out["completions"].numel())
    rank0(
        rank,
        f"generate_smoke: {generated / elapsed:,.0f} tok/s "
        f"({generated:,} tokens, compile_decode={args.compile_decode_smoke})",
    )
    core.train()


def positive_int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    out = int(value)
    return out if out > 0 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compiled A100 PiCOFormer trainer over Guido corpus shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--corpus-root", default=os.environ.get("CORPUS_ROOT", CORPUS_V2))
    parser.add_argument("--mixture-preset", default=os.environ.get("MIXTURE_PRESET", "default"),
                        choices=sorted(MIXTURE_PRESETS))
    parser.add_argument("--mixture", default=os.environ.get("MIXTURE"))
    parser.add_argument("--preset", default=os.environ.get("PICO_PRESET", "guido200m"),
                        choices=sorted(MODEL_PRESETS))
    parser.add_argument("--n-layers", type=positive_int_or_none,
                        default=positive_int_or_none(os.environ.get("N_LAYERS")))
    parser.add_argument("--d-model", type=positive_int_or_none,
                        default=positive_int_or_none(os.environ.get("D_MODEL")))
    parser.add_argument("--n-heads", type=positive_int_or_none,
                        default=positive_int_or_none(os.environ.get("N_HEADS")))
    parser.add_argument("--n-kv-heads", type=positive_int_or_none,
                        default=positive_int_or_none(os.environ.get("N_KV_HEADS")))
    parser.add_argument("--vocab-size", type=int, default=int(os.environ.get("VOCAB_SIZE", "0")))
    parser.add_argument("--seq-len", type=int, default=int(os.environ.get("SEQ_LEN", "2048")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BS_PER_DEV", "16")))
    parser.add_argument("--grad-accum", type=int, default=int(os.environ.get("GRAD_ACCUM", "1")))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("ITERATIONS", "1000")))

    parser.add_argument("--attention", default=os.environ.get("ATTENTION", "gqa_gated"),
                        choices=["gqa", "gqa_gated", "mha", "gated_mha", "xsa_mha"])
    parser.add_argument("--xsa-last-n", type=int, default=int(os.environ.get("XSA_LAST_N", "0")))
    parser.add_argument("--channel", default=os.environ.get("CHANNEL", "torch_lrelu2"),
                        choices=[
                            "torch_relu2",
                            "torch_lrelu2",
                            "torch_leaky_reglu2",
                            "triton_relu2",
                            "triton_lrelu2",
                            "triton_lrelu2_a100",
                            "triton_lrelu2_t4",
                            "triton_swiglu",
                        ])
    parser.add_argument("--ffn-multiplier", type=int,
                        default=int(os.environ.get("FFN_MULTIPLIER", "4")))
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction,
                        default=env_bool("QK_NORM", True))
    parser.add_argument("--smear-gate", action=argparse.BooleanOptionalAction,
                        default=env_bool("SMEAR_GATE", True))
    parser.add_argument("--smear-gate-lookback", type=int,
                        default=int(os.environ.get("SMEAR_GATE_LOOKBACK", "1")))
    parser.add_argument("--gate-input-dim", type=int,
                        default=int(os.environ.get("GATE_INPUT_DIM", "12")))
    parser.add_argument("--logit-softcap", type=float,
                        default=float(os.environ.get("LOGIT_SOFTCAP", "30.0")))

    parser.add_argument("--optimizer", default=os.environ.get("OPTIMIZER", "muon"),
                        choices=["muon", "adamw"])
    parser.add_argument("--matrix-lr", type=float, default=float(os.environ.get("MATRIX_LR", "0.01")))
    parser.add_argument("--embed-lr", type=float, default=float(os.environ.get("EMBED_LR", "0.005")))
    parser.add_argument("--scalar-lr", type=float, default=float(os.environ.get("SCALAR_LR", "0.04")))
    parser.add_argument("--adam-lr", type=float, default=float(os.environ.get("ADAM_LR", "3e-4")))
    parser.add_argument("--beta1", type=float, default=float(os.environ.get("BETA1", "0.9")))
    parser.add_argument("--beta2", type=float, default=float(os.environ.get("BETA2", "0.95")))
    parser.add_argument("--adam-eps", type=float, default=float(os.environ.get("ADAM_EPS", "1e-8")))
    parser.add_argument("--weight-decay", type=float, default=float(os.environ.get("WEIGHT_DECAY", "0.0")))
    parser.add_argument("--muon-momentum", type=float,
                        default=float(os.environ.get("MUON_MOMENTUM", "0.95")))
    parser.add_argument("--muon-momentum-warmup-start", type=float,
                        default=float(os.environ.get("MUON_MOMENTUM_WARMUP_START", "0.85")))
    parser.add_argument("--muon-warmup-steps", type=int,
                        default=int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", "200")))
    parser.add_argument("--muon-backend-steps", type=int,
                        default=int(os.environ.get("MUON_BACKEND_STEPS", "5")))
    parser.add_argument("--grad-clip", type=float, default=float(os.environ.get("GRAD_CLIP", "1.0")))
    parser.add_argument("--warmup-steps", type=int, default=int(os.environ.get("WARMUP_STEPS", "100")))
    parser.add_argument("--lr-decay-steps", type=int, default=int(os.environ.get("LR_DECAY_STEPS", "0")))
    parser.add_argument("--min-lr-ratio", type=float, default=float(os.environ.get("MIN_LR_RATIO", "0.1")))

    parser.add_argument("--dtype", default=os.environ.get("DTYPE", "bf16"),
                        choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--loss", default=os.environ.get("LOSS", "logits"),
                        choices=["logits", "hidden_logits", "cce", "cce_auto"],
                        help=(
                            "CE implementation: logits materializes model logits; "
                            "hidden_logits computes CE from hidden states with the torch fallback; "
                            "cce requires fused cut_cross_entropy; cce_auto uses it when present "
                            "and otherwise falls back to hidden_logits."
                        ))
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction,
                        default=env_bool("COMPILE_MODEL", True))
    parser.add_argument("--compile-mode", default=os.environ.get(
        "COMPILE_MODE", "max-autotune-no-cudagraphs"))
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction,
                        default=env_bool("ALLOW_TF32", True))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "1337")))
    parser.add_argument("--log-every", type=int, default=int(os.environ.get("LOG_EVERY", "20")))
    parser.add_argument("--save-every", type=int, default=int(os.environ.get("SAVE_EVERY", "1000")))
    parser.add_argument("--ckpt-dir", default=os.environ.get("CKPT_DIR", "ckpts/train_A100"))
    parser.add_argument("--resume", default=os.environ.get("RESUME"))
    parser.add_argument("--no-resume-optim", action="store_true")
    parser.add_argument("--loss-trace-csv", default=os.environ.get(
        "LOSS_TRACE_CSV", "logs/train_A100_loss_trace.csv"))

    parser.add_argument("--generate-smoke-tokens", type=int,
                        default=int(os.environ.get("GENERATE_SMOKE_TOKENS", "0")))
    parser.add_argument("--generate-smoke-batch", type=int,
                        default=int(os.environ.get("GENERATE_SMOKE_BATCH", "4")))
    parser.add_argument("--generate-smoke-groups", type=int,
                        default=int(os.environ.get("GENERATE_SMOKE_GROUPS", "4")))
    parser.add_argument("--generate-smoke-prompt", type=int,
                        default=int(os.environ.get("GENERATE_SMOKE_PROMPT", "128")))
    parser.add_argument("--compile-decode-smoke", action=argparse.BooleanOptionalAction,
                        default=env_bool("COMPILE_DECODE_SMOKE", False))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    preset = MODEL_PRESETS[args.preset]
    for key, value in preset.items():
        attr = key.replace("_", "_")
        if getattr(args, attr) is None:
            setattr(args, attr, value)
    if args.lr_decay_steps <= 0:
        args.lr_decay_steps = max(1, args.steps - args.warmup_steps)
    return args


def main() -> None:
    args = parse_args()
    distributed, rank, local_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed, rank)
    configure_picoformer_training_runtime(allow_tf32=args.allow_tf32)

    mixture = parse_mixture(args.mixture, args.mixture_preset)
    if args.mixture_preset == "fineweb_v4_only" and args.corpus_root == CORPUS_V2:
        args.corpus_root = CORPUS_V4
    loader = MultiShardMixture(
        mixture,
        args.corpus_root,
        rank,
        world_size,
        args.seq_len,
        args.seed,
    )
    if args.vocab_size <= 0:
        args.vocab_size = int(loader.vocab_size)
    elif args.vocab_size != int(loader.vocab_size):
        raise ValueError(f"--vocab-size={args.vocab_size} but data vocab={loader.vocab_size}")

    cfg = PiCOFormerConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        max_seq_len=args.seq_len + max(args.generate_smoke_tokens, 0) + args.generate_smoke_prompt,
        attention=args.attention,
        channel=args.channel,
        ffn_multiplier=args.ffn_multiplier,
        qk_norm=args.qk_norm,
        attention_gate_input_dim=args.gate_input_dim,
        xsa_last_n=args.xsa_last_n,
        logit_softcap=args.logit_softcap,
        tied_embeddings=True,
        smear_gate=args.smear_gate,
        smear_gate_input_dim=args.gate_input_dim,
        smear_gate_lookback=args.smear_gate_lookback,
        dropout=0.0,
        production_mode=True,
        torch_compile_forward=False,
    )
    model = build_picoformer(cfg).to(device=device, dtype=dtype_from_name(args.dtype))
    prebuild_rope_caches(model, cfg.max_seq_len, device=device, dtype=dtype_from_name(args.dtype))
    start_step = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        start_step = int(checkpoint.get("step", -1)) + 1
        rank0(rank, f"resumed model from {args.resume} at step {start_step}")

    if args.loss == "logits" and args.compile_model:
        model = torch.compile(model, mode=args.compile_mode, fullgraph=False, dynamic=False)

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )

    optimizers, optimizer_names = make_optimizers(args, model, device)
    if args.resume and not args.no_resume_optim:
        checkpoint = torch.load(args.resume, map_location=device)
        opt_states = checkpoint.get("optimizers")
        if opt_states and len(opt_states) == len(optimizers):
            for optimizer, state in zip(optimizers, opt_states):
                optimizer.load_state_dict(state)
            rank0(rank, "resumed optimizer states")

    loss_fn = build_loss_fn(args, model)
    n_params = sum(p.numel() for p in unwrap_model(model).parameters())
    matrix_params, embed_params, scalar_params = split_optimizer_params(model)
    rank0(rank, f"data: root={args.corpus_root}")
    rank0(rank, f"mixture={mixture}")
    for name, stream in loader.streams.items():
        rank0(
            rank,
            f"  stream {name:<20} chunks={stream.n_chunks:,} "
            f"per-rank={len(stream.my):,} weight={mixture[name]:.3f}",
        )
    rank0(
        rank,
        f"model={n_params / 1e6:.2f}M shape={args.n_layers}Lx{args.d_model} "
        f"heads={args.n_heads}/{args.n_kv_heads} seq={args.seq_len} "
        f"attention={args.attention} xsa_last_n={args.xsa_last_n} channel={args.channel}",
    )
    rank0(
        rank,
        f"opt={args.optimizer} matrix={sum(p.numel() for p in matrix_params) / 1e6:.2f}M "
        f"embed={sum(p.numel() for p in embed_params) / 1e6:.2f}M "
        f"scalar={sum(p.numel() for p in scalar_params) / 1e6:.2f}M "
        f"loss={args.loss} cce_available={cut_cross_entropy_available()}",
    )
    rank0(
        rank,
        f"batch/rank={args.batch_size} grad_accum={args.grad_accum} "
        f"global_batch={args.batch_size * world_size * args.grad_accum} "
        f"tok/step={args.batch_size * world_size * args.grad_accum * args.seq_len:,} "
        f"compile={args.compile_model} mode={args.compile_mode}",
    )

    if args.dry_run:
        x, y, counts = loader.next_batch(args.batch_size, device)
        with torch.autocast(device_type=device.type, dtype=dtype_from_name(args.dtype), enabled=device.type == "cuda"):
            loss = loss_fn(x, y)
        rank0(rank, f"dry_run loss={float(loss.detach()):.4f} counts={counts}")
        return

    if rank == 0:
        trace_path = Path(args.loss_trace_csv)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_file = trace_path.open("w", newline="", encoding="utf-8")
        trace_writer = csv.writer(trace_file)
        trace_writer.writerow(["step", "loss", "grad_norm", "lr", "math_frac", "tok_per_s"])
    else:
        trace_file = None
        trace_writer = None

    run_generation_smoke(model, args, device, rank)
    unwrap_model(model).train()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    interval_start = time.perf_counter()
    interval_tokens = 0
    last_step = start_step - 1

    for step in range(start_step, args.steps):
        last_step = step
        update_optimizer_schedules(optimizers, step, args)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)

        loss_accum = torch.zeros((), device=device)
        count_accum: dict[str, int] = {name: 0 for name in loader.names}
        for micro in range(args.grad_accum):
            x, y, counts = loader.next_batch(args.batch_size, device)
            for name, count in counts.items():
                count_accum[name] += count
            sync_context = nullcontext()
            if distributed and micro < args.grad_accum - 1:
                sync_context = model.no_sync()
            with sync_context:
                with torch.autocast(
                    device_type=device.type,
                    dtype=dtype_from_name(args.dtype),
                    enabled=device.type == "cuda" and dtype_from_name(args.dtype) != torch.float32,
                ):
                    loss = loss_fn(x, y) / args.grad_accum
                loss.backward()
                loss_accum += loss.detach()
            interval_tokens += args.batch_size * args.seq_len * world_size

        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                unwrap_model(model).parameters(), args.grad_clip,
            )
        else:
            grad_norm = torch.zeros((), device=device)
        for optimizer in optimizers:
            optimizer.step()

        bad = not torch.isfinite(loss_accum)
        if bool(bad):
            rank0(rank, f"non-finite loss at step {step}; aborting")
            break

        if (step + 1) % args.log_every == 0 or step == start_step:
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = max(1e-9, time.perf_counter() - interval_start)
            tok_per_s = interval_tokens / elapsed
            report = torch.tensor(
                [loss_accum.item(), float(grad_norm), tok_per_s],
                device=device,
                dtype=torch.float64,
            )
            if distributed:
                dist.all_reduce(report, op=dist.ReduceOp.AVG)
            nl_count = sum(
                count for name, count in count_accum.items()
                if "fineweb" in name or "cosmopedia" in name
            )
            total_count = max(1, sum(count_accum.values()))
            math_frac = 1.0 - nl_count / total_count
            lr = optimizers[0].param_groups[0]["lr"]
            peak_gb = (
                torch.cuda.max_memory_allocated(device) / 1e9
                if device.type == "cuda" else 0.0
            )
            rank0(
                rank,
                f"step {step + 1:>6d}/{args.steps} loss={report[0].item():.4f} "
                f"gnorm={report[1].item():.3f} lr={lr:.3e} "
                f"math_frac={math_frac:.2f} speed={report[2].item():,.0f} tok/s "
                f"peak={peak_gb:.1f}GB",
            )
            if trace_writer is not None:
                trace_writer.writerow([
                    step + 1,
                    f"{report[0].item():.6f}",
                    f"{report[1].item():.6f}",
                    f"{lr:.8e}",
                    f"{math_frac:.6f}",
                    f"{report[2].item():.2f}",
                ])
                trace_file.flush()
            interval_start = time.perf_counter()
            interval_tokens = 0

        if args.save_every > 0 and (step + 1) % args.save_every == 0 and rank == 0:
            save_checkpoint(
                Path(args.ckpt_dir) / f"step_{step + 1:07d}.pt",
                step,
                model,
                cfg,
                args,
                optimizers,
                optimizer_names,
            )

    if rank == 0:
        save_checkpoint(
            Path(args.ckpt_dir) / "latest.pt",
            max(0, last_step),
            model,
            cfg,
            args,
            optimizers,
            optimizer_names,
        )
        if trace_file is not None:
            trace_file.close()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
