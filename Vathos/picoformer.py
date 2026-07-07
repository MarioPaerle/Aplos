"""Production-facing PiCOFormer scaffold.

This module keeps the research-heavy API in ``Vathos.blocks`` intact and exposes
one narrow, deployable surface for fast PiCOFormer training and GRPO rollout
generation:

* typed configs that can be saved with checkpoints;
* a factory that selects GQA/MHA/gated attention and optional Triton FFNs;
* runtime helpers for inference, GRPO rollouts, and completion scoring.

The custom Triton FFN modules degrade to a torch implementation when Triton or a
supported GPU is unavailable, so importing this module stays safe on CPU nodes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal
import json

import torch
import torch.nn.functional as F

from Vathos._basics import (
    Builder,
    LeakyReLU2,
    ReLU2,
    RMSNorm,
    VariableGLU,
    VariableUDLP,
    set_vathos_mode,
)
from Vathos._spatials import (
    GQAGatedMixer,
    GroupedQueryAttention,
    MultiheadAttentionMixerXSA,
    MultiheadAttentionMixer,
    MultiheadGatedAttentionMixer,
    RoPE,
)
from Vathos.blocks import PiCOFormer, SmearGate

try:  # Optional Apple cut-cross-entropy package used by Guido training runs.
    from cut_cross_entropy import linear_cross_entropy as _linear_cross_entropy
except Exception:  # pragma: no cover - optional accelerator dependency
    _linear_cross_entropy = None


AttentionKind = Literal["gqa", "gqa_gated", "mha", "gated_mha", "xsa_mha"]
ChannelKind = Literal[
    "torch_relu2",
    "torch_lrelu2",
    "torch_leaky_reglu2",
    "torch_swiglu",
    "triton_relu2",
    "triton_lrelu2",
    "triton_lrelu2_a100",
    "triton_lrelu2_t4",
    "triton_swiglu",
]


@dataclass(slots=True)
class PiCOFormerConfig:
    """Serializable model config for the fast PiCOFormer path."""

    vocab_size: int
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: int = 2
    max_seq_len: int = 2048
    rope_base: float = 10000.0
    attention: AttentionKind = "gqa"
    channel: ChannelKind = "torch_relu2"
    ffn_multiplier: int = 4
    qk_norm: bool = True
    attention_gate_input_dim: int = 12
    xsa_last_n: int = 0
    logit_softcap: float = 30.0
    tied_embeddings: bool = True
    smear_gate: bool = False
    smear_gate_input_dim: int = 12
    smear_gate_lookback: int = 1
    dropout: float = 0.0
    production_mode: bool = True
    torch_compile_forward: bool = False

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.d_model <= 0 or self.n_layers <= 0:
            raise ValueError("d_model and n_layers must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.attention in ("gqa", "gqa_gated"):
            if self.n_kv_heads <= 0:
                raise ValueError("n_kv_heads must be positive for GQA")
            if self.n_heads % self.n_kv_heads != 0:
                raise ValueError("n_heads must be divisible by n_kv_heads for GQA")
        if not 0 <= self.xsa_last_n <= self.n_layers:
            raise ValueError("xsa_last_n must be in [0, n_layers]")
        if self.attention == "xsa_mha" and self.xsa_last_n not in (0, self.n_layers):
            raise ValueError("attention='xsa_mha' already makes every layer XSA; leave xsa_last_n=0")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be positive")
        if self.smear_gate and self.smear_gate_lookback < 1:
            raise ValueError("smear_gate_lookback must be >= 1 when smear_gate=True")


@dataclass(slots=True)
class DecodeConfig:
    """Sampling/decode config for GRPO rollouts and inference."""

    max_new_tokens: int
    group_size: int = 1
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 1.0
    eos_token_id: int | None = None
    pad_token_id: int = 0
    return_logprobs: bool = True
    compile_decode: bool = False

    def kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GRPOBatch:
    """Typed wrapper around ``PiCOFormer.generate_grpo`` outputs."""

    sequences: torch.Tensor
    completions: torch.Tensor
    completion_mask: torch.Tensor
    sampling_logprobs: torch.Tensor | None
    prompt_len: int

    @classmethod
    def from_dict(cls, out: dict[str, Any]) -> "GRPOBatch":
        return cls(
            sequences=out["sequences"],
            completions=out["completions"],
            completion_mask=out["completion_mask"],
            sampling_logprobs=out["sampling_logprobs"],
            prompt_len=int(out["prompt_len"]),
        )


def configure_runtime(*, allow_tf32: bool = True,
                      matmul_precision: str = "high") -> None:
    """Set PyTorch runtime flags that matter for PiCOFormer throughput."""

    torch.set_float32_matmul_precision(matmul_precision)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32


def configure_picoformer_training_runtime(
    *,
    allow_tf32: bool = True,
    matmul_precision: str = "high",
    flash_sdp: bool = True,
    mem_efficient_sdp: bool = False,
    math_sdp: bool = False,
    cudnn_sdp: bool = False,
    inductor_coordinate_descent: bool = True,
    inductor_fx_graph_cache: bool = True,
    inductor_cudagraphs: bool | None = None,
) -> None:
    """Configure CUDA/PyTorch flags for fixed-shape PiCOFormer training jobs."""

    configure_runtime(
        allow_tf32=allow_tf32,
        matmul_precision=matmul_precision,
    )
    if not torch.cuda.is_available():
        return

    for name, enabled in (
        ("enable_flash_sdp", flash_sdp),
        ("enable_mem_efficient_sdp", mem_efficient_sdp),
        ("enable_math_sdp", math_sdp),
        ("enable_cudnn_sdp", cudnn_sdp),
    ):
        setter = getattr(torch.backends.cuda, name, None)
        if setter is not None:
            setter(enabled)

    try:
        import torch._inductor.config as inductor_config
    except Exception:
        return

    for name, value in (
        ("coordinate_descent_tuning", inductor_coordinate_descent),
        ("fx_graph_cache", inductor_fx_graph_cache),
    ):
        if hasattr(inductor_config, name):
            setattr(inductor_config, name, value)
    if inductor_cudagraphs is not None and hasattr(inductor_config, "triton"):
        triton_config = getattr(inductor_config, "triton")
        if hasattr(triton_config, "cudagraphs"):
            triton_config.cudagraphs = inductor_cudagraphs


def _spatial_builder_for(cfg: PiCOFormerConfig, attention: AttentionKind) -> Builder:
    head_dim = cfg.d_model // cfg.n_heads
    rope = RoPE(head_dim, max_len=cfg.max_seq_len, base=cfg.rope_base)
    common = {
        "causal": True,
        "pos_emb": rope,
        "dropout": cfg.dropout,
        "qk_norm": cfg.qk_norm,
    }

    if attention == "gqa":
        return Builder(
            GroupedQueryAttention,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            **common,
        )
    if attention == "gqa_gated":
        return Builder(
            GQAGatedMixer,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            gate_input_dim=cfg.attention_gate_input_dim,
            **common,
        )
    if attention == "mha":
        return Builder(MultiheadAttentionMixer, n_heads=cfg.n_heads, **common)
    if attention == "gated_mha":
        return Builder(
            MultiheadGatedAttentionMixer,
            n_heads=cfg.n_heads,
            gate_input_dim=cfg.attention_gate_input_dim,
            **common,
        )
    if attention == "xsa_mha":
        return Builder(MultiheadAttentionMixerXSA, n_heads=cfg.n_heads, **common)
    raise ValueError(f"unsupported attention kind: {attention}")


def _spatial_builders(cfg: PiCOFormerConfig) -> Builder | list[Builder]:
    if cfg.xsa_last_n <= 0:
        return _spatial_builder_for(cfg, cfg.attention)
    base = _spatial_builder_for(cfg, cfg.attention)
    xsa = _spatial_builder_for(cfg, "xsa_mha")
    split = cfg.n_layers - cfg.xsa_last_n
    return [base if i < split else xsa for i in range(cfg.n_layers)]


def _channel_builder(cfg: PiCOFormerConfig) -> Builder:
    hidden = cfg.ffn_multiplier * cfg.d_model
    common = {"d_output": cfg.d_model, "M": hidden, "dropout": cfg.dropout}

    if cfg.channel == "torch_relu2":
        return Builder(VariableUDLP, activation=ReLU2, **common)
    if cfg.channel == "torch_lrelu2":
        return Builder(VariableUDLP, activation=LeakyReLU2, **common)
    if cfg.channel == "torch_leaky_reglu2":
        return Builder(VariableGLU, activation=LeakyReLU2, **common)
    if cfg.channel == "torch_swiglu":
        # SwiGLU con proiezioni SEPARATE (expand/up/contract) + SiLU — layout identico al FFN
        # di training di Guido (`LeakyReGLU2_FFN(activation=nn.SiLU)`). Carica i checkpoint SwiGLU
        # 1:1 senza fondere i pesi (a differenza di `triton_swiglu`, che fonde gate+up in 2*d_ff).
        return Builder(VariableGLU, activation=torch.nn.SiLU, **common)

    from Vathos._triton import (
        TritonLReLU2UDLP,
        TritonLReLU2UDLP_A100,
        TritonLReLU2UDLP_T4,
        TritonReLU2UDLP,
        TritonSwiGLUUDLP,
    )

    if cfg.channel == "triton_relu2":
        return Builder(TritonReLU2UDLP, **common)
    if cfg.channel == "triton_lrelu2":
        return Builder(TritonLReLU2UDLP, **common)
    if cfg.channel == "triton_lrelu2_a100":
        return Builder(TritonLReLU2UDLP_A100, **common)
    if cfg.channel == "triton_lrelu2_t4":
        return Builder(TritonLReLU2UDLP_T4, **common)
    if cfg.channel == "triton_swiglu":
        return Builder(TritonSwiGLUUDLP, **common)
    raise ValueError(f"unsupported channel kind: {cfg.channel}")


def build_picoformer(cfg: PiCOFormerConfig) -> PiCOFormer:
    """Build a fast PiCOFormer from a serializable config."""

    cfg.validate()
    if cfg.production_mode:
        set_vathos_mode("production")

    smear_gate = (
        SmearGate(cfg.d_model, gate_input_dim=cfg.smear_gate_input_dim)
        if cfg.smear_gate else None
    )
    model = PiCOFormer(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        spatials=_spatial_builders(cfg),
        channel=_channel_builder(cfg),
        norm=RMSNorm,
        smear_gate=smear_gate,
        smear_gate_lookback=cfg.smear_gate_lookback if cfg.smear_gate else 0,
        logit_softcap=cfg.logit_softcap,
        tied_embeddings=cfg.tied_embeddings,
    )
    model.pico_config = cfg

    if cfg.torch_compile_forward:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    return model


def prepare_for_inference(model: torch.nn.Module, *,
                          device: str | torch.device | None = None,
                          dtype: torch.dtype | None = None,
                          compile_forward: bool = False) -> torch.nn.Module:
    """Move a model to device/dtype, strip profiling hooks, and switch to eval."""

    if hasattr(model, "strip_profiling_hooks"):
        model.strip_profiling_hooks()
    if device is not None:
        model = model.to(device=device)
    if dtype is not None:
        model = model.to(dtype=dtype)
    model.eval()
    if compile_forward:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    return model


def prebuild_rope_caches(model: torch.nn.Module, max_len: int,
                         device: str | torch.device | None = None,
                         dtype: torch.dtype | None = None) -> torch.nn.Module:
    """Materialize RoPE tables before torch.compile sees the model.

    Lazy cache creation mutates module state inside ``RoPE.forward``. That is fine
    in eager mode, but it is hostile to Dynamo/Inductor and especially CUDA graphs:
    the first compiled pass can capture cache tensors as graph outputs and later
    runs may hit overwritten-output errors or pathological recompiles.
    """

    raw_model = _unwrap_compiled(model)
    if device is None:
        try:
            device = next(raw_model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        try:
            dtype = next(raw_model.parameters()).dtype
        except StopIteration:
            dtype = torch.float32

    for module in raw_model.modules():
        pos_emb = getattr(module, "pos_emb", None)
        if pos_emb is not None and hasattr(pos_emb, "prebuild"):
            pos_emb.prebuild(max_len, device, dtype)
    return model


@torch.no_grad()
def generate_grpo_rollouts(model: PiCOFormer, prompt: torch.Tensor,
                           decode: DecodeConfig) -> GRPOBatch:
    """Run the static-cache GRPO rollout path and return a typed batch."""

    out = model.generate_grpo(prompt, **decode.kwargs())
    return GRPOBatch.from_dict(out)


def completion_logprobs(model: PiCOFormer, batch: GRPOBatch) -> torch.Tensor:
    """Return policy logprobs aligned to completion tokens.

    The output is ``[B, T]`` and is masked to zero after EOS/pad using
    ``batch.completion_mask``. Call this with gradients enabled for the current
    policy and under ``torch.no_grad()`` for a frozen reference policy.
    """

    all_logprobs = model.sequence_logprobs(batch.sequences)
    t = batch.completions.size(1)
    start = batch.prompt_len - 1
    logprobs = all_logprobs[:, start:start + t]
    return logprobs * batch.completion_mask


def cut_cross_entropy_available() -> bool:
    """Return whether the optional fused linear CE package is importable."""

    return _linear_cross_entropy is not None


def pico_hidden_states(model: PiCOFormer, input_ids: torch.Tensor) -> torch.Tensor:
    """Return final normalized PiCOFormer hidden states without LM logits."""

    raw_model = _unwrap_compiled(model)
    x0 = raw_model.embedder(input_ids)
    if getattr(raw_model, "smear_gate", None) is not None:
        x0 = raw_model.smear_gate(x0)

    ves = raw_model._compute_ves(input_ids)
    hidden = x0
    for i, block in enumerate(raw_model.blocks):
        hidden = block(hidden, x0, ve=ves[i])
    return raw_model.final_norm(hidden)


def pico_classifier_weight(model: PiCOFormer) -> torch.Tensor:
    """Return the tied or untied LM classifier weight for hidden-state CE."""

    raw_model = _unwrap_compiled(model)
    return raw_model.unembedder.linear.weight


def linear_lm_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    use_cut_cross_entropy: bool = True,
    softcap: float | None = None,
) -> torch.Tensor:
    """Compute LM CE from hidden states and classifier weight.

    This is the stable library wrapper for monolithic trainers: it uses
    ``cut_cross_entropy`` when that package is installed for the active Torch
    runtime, and otherwise falls back to equivalent PyTorch operations.
    """

    hidden = hidden.reshape(-1, hidden.size(-1))
    targets = targets.reshape(-1)
    if use_cut_cross_entropy and _linear_cross_entropy is not None:
        return _linear_cross_entropy(
            hidden,
            weight,
            targets,
            reduction="mean",
            softcap=softcap if softcap and softcap > 0 else None,
        )

    logits = F.linear(hidden.float(), weight.float())
    if softcap and softcap > 0:
        logits = softcap * torch.tanh(logits / softcap)
    return F.cross_entropy(logits, targets)


def pico_cross_entropy_loss(
    model: PiCOFormer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    use_cut_cross_entropy: bool = True,
    softcap: float | None = None,
) -> torch.Tensor:
    """Compute LM CE from hidden states without requiring ``model`` to emit logits.

    When ``cut_cross_entropy`` is installed, this avoids materializing the full
    ``[batch, seq, vocab]`` logits tensor. Otherwise it falls back to equivalent
    torch CE so callers can keep one training path across environments.
    """

    raw_model = _unwrap_compiled(model)
    hidden = pico_hidden_states(raw_model, input_ids)
    weight = pico_classifier_weight(raw_model)
    cap = raw_model.softcap if softcap is None else softcap
    return linear_lm_cross_entropy(
        hidden,
        weight,
        targets,
        use_cut_cross_entropy=use_cut_cross_entropy,
        softcap=cap,
    )


def _unwrap_compiled(model: torch.nn.Module) -> torch.nn.Module:
    """Return the original module when given a torch.compile wrapper."""

    return getattr(model, "_orig_mod", model)


def save_picoformer(model: PiCOFormer, path: str | Path,
                    cfg: PiCOFormerConfig | None = None) -> None:
    """Save a deployable PiCOFormer checkpoint directory."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model_to_save = _unwrap_compiled(model)
    if cfg is None:
        cfg = getattr(model_to_save, "pico_config", None)
    if cfg is None:
        raise ValueError("cfg is required when model.pico_config is missing")

    with (path / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, sort_keys=True)
        f.write("\n")
    torch.save(model_to_save.state_dict(), path / "model.pt")


def load_picoformer(path: str | Path, *,
                    map_location: str | torch.device | None = None,
                    device: str | torch.device | None = None,
                    dtype: torch.dtype | None = None) -> PiCOFormer:
    """Load a checkpoint created by :func:`save_picoformer`."""

    path = Path(path)
    with (path / "config.json").open("r", encoding="utf-8") as f:
        cfg = PiCOFormerConfig(**json.load(f))
    should_compile = cfg.torch_compile_forward
    model = build_picoformer(replace(cfg, torch_compile_forward=False))
    state = torch.load(path / "model.pt", map_location=map_location or "cpu")
    model.load_state_dict(state)
    model.pico_config = cfg
    return prepare_for_inference(
        model,
        device=device,
        dtype=dtype,
        compile_forward=should_compile,
    )


@dataclass(slots=True)
class FastPiCORunner:
    """Small deployable wrapper for serving or GRPO collection jobs."""

    model: PiCOFormer
    decode: DecodeConfig
    metadata: dict[str, Any] = field(default_factory=dict)

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor) -> torch.Tensor:
        out = self.model.generate_grpo(
            prompt,
            max_new_tokens=self.decode.max_new_tokens,
            group_size=self.decode.group_size,
            temperature=self.decode.temperature,
            top_k=self.decode.top_k,
            top_p=self.decode.top_p,
            eos_token_id=self.decode.eos_token_id,
            pad_token_id=self.decode.pad_token_id,
            return_logprobs=False,
            compile_decode=self.decode.compile_decode,
        )
        return out["sequences"]

    @torch.no_grad()
    def rollouts(self, prompt: torch.Tensor) -> GRPOBatch:
        return generate_grpo_rollouts(self.model, prompt, self.decode)

    def score(self, batch: GRPOBatch) -> torch.Tensor:
        return completion_logprobs(self.model, batch)


__all__ = [
    "AttentionKind",
    "ChannelKind",
    "DecodeConfig",
    "FastPiCORunner",
    "GRPOBatch",
    "PiCOFormerConfig",
    "build_picoformer",
    "completion_logprobs",
    "configure_picoformer_training_runtime",
    "configure_runtime",
    "cut_cross_entropy_available",
    "generate_grpo_rollouts",
    "linear_lm_cross_entropy",
    "load_picoformer",
    "pico_classifier_weight",
    "pico_cross_entropy_loss",
    "pico_hidden_states",
    "prepare_for_inference",
    "prebuild_rope_caches",
    "save_picoformer",
]
