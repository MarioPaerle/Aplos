# PiCOFormer Production Path

This is the supported fast path for training and GRPO rollout collection. The
older `SequenceModel` and `ModdedFormer` APIs remain useful for experiments, but
production jobs should start from `Vathos.picoformer`.

## Install

Minimal inference/training install:

```bash
pip install -e .
```

With Triton kernels:

```bash
pip install -e ".[triton]"
```

With tests:

```bash
pip install -e ".[dev,triton]"
```

## Build A Fast Model

```python
import torch
from Vathos.picoformer import (
    PiCOFormerConfig,
    DecodeConfig,
    build_picoformer,
    configure_runtime,
    generate_grpo_rollouts,
    completion_logprobs,
    prepare_for_inference,
)

configure_runtime()

cfg = PiCOFormerConfig(
    vocab_size=49152,
    d_model=1024,
    n_layers=24,
    n_heads=16,
    n_kv_heads=4,
    max_seq_len=2048,
    attention="gqa",
    channel="triton_lrelu2",
    ffn_multiplier=4,
)

model = build_picoformer(cfg)
model = prepare_for_inference(
    model,
    device="cuda",
    dtype=torch.bfloat16,
)
```

`channel="triton_lrelu2_a100"` uses the pointer-based custom Triton FFN path on
A100/Ampere and falls back to a compile-friendly torch implementation elsewhere.
`channel="triton_lrelu2"` is the Hopper/TMA variant. For T4 jobs use
`channel="triton_lrelu2_t4"` and run the model in fp16.

When Triton is installed, no-grad GRPO sampling-logprob collection also uses a
fused selective-log-softmax kernel. The differentiable scoring pass keeps the
torch implementation so gradients remain correct.

## Training Runtime

For fixed-shape pretraining jobs, call the training runtime helper before model
construction or compile:

```python
from Vathos.picoformer import (
    configure_picoformer_training_runtime,
    linear_lm_cross_entropy,
    prebuild_rope_caches,
)

configure_picoformer_training_runtime(inductor_cudagraphs=False)
```

`configure_picoformer_training_runtime` enables TF32, prefers Flash SDP, disables
the slower SDP fallbacks by default, and enables the Inductor knobs that helped
the Guido-style training benchmark. The default `RMSNorm` implementation already
uses `torch.nn.functional.rms_norm`, so external monkey patches are no longer
needed.

For monolithic trainers that already compute hidden states, use the stable CE
wrapper instead of importing accelerator packages directly:

```python
loss = linear_lm_cross_entropy(
    hidden.bfloat16(),
    classifier_weight.bfloat16(),
    targets,
    softcap=30.0,
)
```

When `cut_cross_entropy` is installed for the active Torch runtime this avoids
materializing full logits. Otherwise it falls back to standard PyTorch CE, which
is correct but may use much more memory for large vocabularies or batches.

Before compiling a PiCOFormer module, materialize RoPE tables:

```python
prebuild_rope_caches(model, max_seq_len, device="cuda", dtype=torch.bfloat16)
model = torch.compile(model, dynamic=False, fullgraph=False, mode="default")
```

This avoids the lazy RoPE cache compile failure/slow path that produced the
pathological `132 tok/s` training result.

## GRPO Rollouts

```python
decode = DecodeConfig(
    max_new_tokens=512,
    group_size=32,
    temperature=1.0,
    top_k=None,
    top_p=1.0,
    eos_token_id=None,
    pad_token_id=0,
    return_logprobs=True,
    compile_decode=True,
)

prompt = torch.randint(0, cfg.vocab_size, (64,), device="cuda")
batch = generate_grpo_rollouts(model, prompt, decode)

# Current policy scoring pass. Keep grad enabled here during training.
policy_logprobs = completion_logprobs(model, batch)

# Reference model scoring pass.
with torch.no_grad():
    ref_logprobs = completion_logprobs(ref_model, batch)
```

`generate_grpo_rollouts` uses the static KV cache path in `PiCOFormer`; it avoids
per-token `torch.cat` cache growth and supports batched prompt replication for
GRPO. `compile_decode=True` switches to the graph-friendly full-buffer masked
decode path.

## Save And Load

```python
from Vathos.picoformer import save_picoformer, load_picoformer

save_picoformer(model, "checkpoints/pico-fast", cfg)
model = load_picoformer(
    "checkpoints/pico-fast",
    device="cuda",
    dtype=torch.bfloat16,
)
```

## Validation And Benchmarks

CPU correctness smoke:

```bash
python3 tests/test_grpo_generate.py
python3 tests/test_picoformer_factory.py
```

GPU decode benchmarks:

```bash
python3 tests/bench_grpo_generate.py
python3 tests/bench_graph_decode.py
python3 tests/bench_compile.py
```

The key invariant is that cached decode logits must match a full forward pass.
Keep `tests/test_grpo_generate.py` passing before changing cache, RoPE, smear,
GQA expansion, or graph decode code.
