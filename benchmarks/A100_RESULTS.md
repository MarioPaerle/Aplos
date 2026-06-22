# PiCOFormer A100 Benchmark Notes

Date: 2026-06-17  
Node/GPU: `lrdn1994`, NVIDIA A100-SXM-64GB  
Torch/Triton: `torch 2.11.0+cu129`, `triton 3.6.0`  
SLURM jobs:

- `47090163`: no compile-train, completed.
- `47090207`: compile-train with `compile_mode=default`, completed.
- `47089894`: initial compile-train with `compile_mode=reduce-overhead`, failed after producing partial results.
- `47090500`: compile-train after RoPE cache prebuild, completed.
- `47091068`: logits CE vs hidden CE fallback on torch 2.11/qwen env, completed.
- `47091121`: logits CE vs fused `cut_cross_entropy` on Guido venv, completed.
- `47091922`: `guido200m` single-A100 shape on torch 2.11/qwen env, completed.
- `47099227`: full `train_A100.py` loop on real `corpus_v2`, completed.
- `47102606`: full `train_A100.py` loop on the old 98.68M `v3_noloops` shape, completed.
- `47116262`: fused `cut_cross_entropy` full trainer loop on the 98.68M `v3_noloops` shape, completed.

## Benchmark Config

Preset: `quick`

- `d_model=768`
- `n_layers=16`
- `n_heads=12`
- `n_kv_heads=4`
- `vocab_size=32768`
- training: `batch=4`, `seq_len=384`, bf16
- generation: `prompt_len=64`, `max_new_tokens=128`, `group_size=16`

## Results

| Path | Channel | Throughput | Peak Mem | Notes |
|---|---:|---:|---:|---|
| pretrain fwd+bwd eager | `torch_lrelu2` | 37,721 tok/s | 2.10 GB | best training path |
| pretrain fwd+bwd eager | `triton_lrelu2_a100` | 31,685 tok/s | 1.95 GB | lower memory, slower |
| 8-step AdamW stability | `torch_lrelu2` | 27,365 tok/s | 2.61 GB | finite, loss 10.580 -> 10.542 |
| 8-step AdamW stability | `triton_lrelu2_a100` | 28,220 tok/s | 2.46 GB | finite, loss 10.580 -> 10.541 |
| generation static cache | `torch_lrelu2` | 1,129 tok/s | 0.40 GB | Python loop bottleneck |
| generation graph decode | `torch_lrelu2` | 9,406 tok/s | 0.40 GB | 8.3x faster |
| generation static cache + logprobs | `torch_lrelu2` | 1,072 tok/s | 0.39 GB | selective logprob overhead small |
| generation static cache | `triton_lrelu2_a100` | 957 tok/s | 0.40 GB | slower FFN path |
| generation graph decode | `triton_lrelu2_a100` | 8,619 tok/s | 0.40 GB | graph decode still dominant |
| generation static cache + logprobs | `triton_lrelu2_a100` | 956 tok/s | 0.40 GB | no-grad fused selective logsoftmax available |
| pretrain fwd+bwd compile default | `torch_lrelu2` | 132 tok/s | 1.71 GB | not viable |
| pretrain fwd+bwd compile default | `triton_lrelu2_a100` | 124 tok/s | 1.70 GB | not viable |
| pretrain fwd+bwd compiled model + prebuilt RoPE | `torch_lrelu2` | 99,068 tok/s | 1.71 GB | fixed compile path |
| pretrain fwd+bwd compiled model + prebuilt RoPE | `triton_lrelu2_a100` | 83,925 tok/s | 1.71 GB | slower than torch channel |

`compile_mode=reduce-overhead` for training failed with a CUDA Graph overwritten-output error in the RoPE/attention path. The pathological `132 tok/s` result was also a RoPE/lazy-cache compile artifact, not a real model-speed result. Calling `prebuild_rope_caches(...)` before `torch.compile(model)` fixes the default compile path on this quick shape.

## Loss Path Comparison

Same quick config, no generation or stability loop.

| Env | Loss | Channel | Mode | Throughput | Peak Mem | Notes |
|---|---|---:|---:|---:|---:|---|
| qwen torch 2.11/cu129 | logits CE | `torch_lrelu2` | eager | 37,565 tok/s | 2.10 GB | baseline |
| qwen torch 2.11/cu129 | logits CE | `torch_lrelu2` | compiled | 98,435 tok/s | 1.71 GB | fastest measured quick path |
| qwen torch 2.11/cu129 | hidden CE fallback | `torch_lrelu2` | compiled | 33,113 tok/s | 2.05 GB | `cut_cross_entropy` absent |
| Guido venv torch 2.6/cu124 | logits CE | `torch_lrelu2` | compiled | 90,039 tok/s | 1.70 GB | older compiler, still healthy |
| Guido venv torch 2.6/cu124 | fused `cut_cross_entropy` | `torch_lrelu2` | eager | 34,826 tok/s | 1.47 GB | memory win, not speed win here |
| Guido venv torch 2.6/cu124 | fused `cut_cross_entropy` | `torch_lrelu2` | compiled | 34,398 tok/s | 1.47 GB | compile does not help this custom op path |
| Guido venv torch 2.6/cu124 | logits CE | `triton_lrelu2_a100` | compiled | 83,842 tok/s | 1.70 GB | slower than torch channel |
| Guido venv torch 2.6/cu124 | fused `cut_cross_entropy` | `triton_lrelu2_a100` | compiled | 33,834 tok/s | 1.67 GB | lowest memory, slower |

Guido training logs are not directly comparable to this microbenchmark because they use a full training loop, 4 GPUs, long sequence length, much larger local batches, DDP, Muon, fused CE, and fixed-shape data loading. A representative `guido_small_42816539` log reports `global_bs=128`, `tok/step=262,144`, and late-run average throughput around `520k tok/s` global, which is roughly `130k tok/s/GPU`. That is consistent with the fixed compiled quick benchmark being in the right ballpark, and with the quick config underfilling the A100.

## Guido-Like Single-GPU Shape

Job `47091922` used the newer torch 2.11/qwen package path with `channel=torch_lrelu2`,
`loss_impl=logits`, no generation, and no stability loop:

- `d_model=768`
- `n_layers=24`
- `n_heads=12`
- `n_kv_heads=4`
- `vocab_size=32768`
- `seq_len=2048`
- `train_batch=16`

| Path | Throughput | Peak Mem | Notes |
|---|---:|---:|---|
| eager logits CE | 74,406 tok/s | 43.0 GB | large full-logits memory cost |
| compiled logits CE + prebuilt RoPE | 107,316 tok/s | 38.2 GB | current best single-A100 Guido-like result |

This validates that the newer Aplos + newer Torch stack is a real migration
target for Guido-style training. It does not replace fused CE for larger batches:
`batch=16` already uses `38 GB` peak with full logits.

## Full `train_A100.py` Loop

Job `47099227` ran the actual Guido-style trainer, including real shard mmap
loading, Muon, LR schedule, backward, optimizer step, CSV logging, and final
checkpoint save:

- Node/GPU: `lrdn2140`, NVIDIA A100-SXM-64GB
- Torch/Triton: `torch 2.11.0+cu129`, `triton 3.6.0`
- Data: `/leonardo_scratch/fast/IscrC_YENDRI/mprignan/corpus_v2`, `default` mixture
- Model: `176.21M`, `24L x 768`, heads `12/4`, `seq_len=2048`
- Training: `batch/rank=16`, `grad_accum=1`, `32,768 tok/step`, 160 steps
- Runtime: `attention=gqa_gated`, `channel=torch_lrelu2`, `loss=logits`, Muon,
  compiled model, `max-autotune-no-cudagraphs`

| Metric | Value | Notes |
|---|---:|---|
| SLURM wall time | 11m14s | includes import, Inductor autotune, training, final checkpoint |
| first logged step | 71 tok/s | compile/autotune polluted, not steady-state |
| steady-state average | 89,479 tok/s | excludes first compile-polluted CSV row |
| best interval | 89,686 tok/s | 10-step interval |
| peak CUDA memory | 36.5 GB | from trainer log |
| final loss | 2.9394 | short benchmark run, not a quality claim |

Compared with the `47091922` microbenchmark (`107,316 tok/s` compiled), the full
trainer costs about 17% throughput from Muon, data mixture sampling, logging, and
the full production loop around the model. The result is still a healthy
single-A100 baseline; four perfect replicas would be roughly `358k tok/s` global
before DDP communication and larger-batch effects.

## 98.68M No-Loops Shape

Job `47102606` matched the old `pico2_v3_noloops` family shape and data mix:

- Node/GPU: `lrdn0003`, NVIDIA A100-SXM-64GB
- Torch/Triton: `torch 2.11.0+cu129`, `triton 3.6.0`
- Data: `/leonardo_scratch/fast/IscrC_YENDRI/mprignan/corpus_v2`, `v3_noloops`
- Model: `98.68M`, `24L x 512`, heads `8/8`, `ffn_multiplier=3`, `seq_len=2048`
- Training: `batch/rank=32`, `grad_accum=1`, `65,536 tok/step`, 160 steps
- Runtime: `attention=gated_mha`, `channel=torch_leaky_reglu2`, `gate_input_dim=128`,
  `loss=logits`, Muon, compiled model, `max-autotune-no-cudagraphs`

| Metric | Value | Notes |
|---|---:|---|
| SLURM wall time | 13m41s | includes import, Inductor autotune, training, final checkpoint |
| first logged step | 116 tok/s | compile/autotune polluted, not steady-state |
| steady-state average | 134,360 tok/s | excludes first compile-polluted CSV row |
| best interval | 134,913 tok/s | 10-step interval |
| peak CUDA memory | 58.9 GB | full logits at batch 32 nearly fills an A100-64GB |
| final loss | 2.8689 | short benchmark run, not a quality claim |

The old `PiCO2_test/logs/v3_noloops_42948378.out` run was the same 98.68M shape
on 4xA100 with `batch/rank=32`. Its final cumulative throughput was `486,169 tok/s`
global (`121,542 tok/s/GPU`), while local 50-step intervals late in the run are
about `131k tok/s/GPU`. The new Aplos full loop is therefore slightly faster than
that no-loops baseline on a per-GPU basis. It is still below the older 92M
`loop_v2_42527514` pre-loop peak, which was about `162k tok/s/GPU`, and it has very
little memory headroom because logits CE uses `58.9 GB`.

### CE Implementation A/B

The trainer now exposes explicit CE modes through `--loss`:

- `logits`: materialize model logits and use PyTorch CE.
- `hidden_logits`: compute hidden states, then use the torch fallback linear CE.
- `cce`: require fused `cut_cross_entropy`; fail loudly if it is unavailable.
- `cce_auto`: use fused `cut_cross_entropy` when installed, otherwise fallback to
  hidden-state torch CE.

Job `47116262` used the same 98.68M shape as `47102606`, but switched to
`--loss cce` in the Guido torch 2.6/cu124 environment where `cut_cross_entropy`
is installed. The newer torch 2.11/qwen package path does not currently have CCE
installed, so this is not a pure CE-only comparison; rebuilding CCE for torch
2.11 is the next benchmark target.

| Job | Env | Loss | Throughput | Peak Mem | Notes |
|---|---|---|---:|---:|---|
| `47102606` | torch 2.11/cu129 | `logits` | 134,360 tok/s | 58.9 GB | fastest qwen-stack result so far, little memory headroom |
| `47116262` | torch 2.6/cu124 | `cce` | 144,297 tok/s | 34.4 GB | fused CCE available; +7.4% speed, -24.5 GB memory |

For this batch-32 shape, fused CCE is the better production path when available:
it is faster and dramatically safer on memory. The current blocker is packaging,
not the trainer surface.

## Conclusions

1. Use compiled logits CE as the current fastest quick-shape training path, but prebuild RoPE caches before compiling.
2. Use `compile_decode=True` for GRPO generation. It gives the largest immediate win.
3. Keep fused `cut_cross_entropy` available for large-vocab/large-batch memory headroom, but do not assume it improves raw speed on small batches.
4. The current A100 Triton FFN kernel reduces memory but loses throughput on this shape. It needs tile/autotune work before becoming the default.
5. For Guido migration, use the torch 2.11/qwen package path when possible, but rebuild or verify `cut_cross_entropy` before increasing batch above the measured full-logits memory envelope.
6. The remaining generation gap to vLLM-style throughput is mostly the per-token Python/control path around decode and sampling. Next work should push sampling/logprob/mask update into a more static compiled step and benchmark larger group sizes.
