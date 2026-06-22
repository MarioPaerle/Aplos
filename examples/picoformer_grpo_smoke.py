"""Minimal PiCOFormer GRPO rollout smoke.

Run:
    python3 examples/picoformer_grpo_smoke.py

Use this to verify that the production scaffold, static KV cache, completion
scoring, and optional Triton FFN wrapper all agree on a small model.
"""

import torch

from Vathos.picoformer import (
    DecodeConfig,
    PiCOFormerConfig,
    build_picoformer,
    completion_logprobs,
    configure_runtime,
    generate_grpo_rollouts,
    prepare_for_inference,
)


def main():
    configure_runtime()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    cfg = PiCOFormerConfig(
        vocab_size=256,
        d_model=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=128,
        attention="gqa",
        channel="triton_lrelu2_a100",
        ffn_multiplier=2,
    )
    model = build_picoformer(cfg)
    model = prepare_for_inference(model, device=device, dtype=dtype)

    prompt = torch.randint(0, cfg.vocab_size, (12,), device=device)
    decode = DecodeConfig(
        max_new_tokens=16,
        group_size=4,
        temperature=1.0,
        eos_token_id=None,
        return_logprobs=True,
        compile_decode=False,
    )

    batch = generate_grpo_rollouts(model, prompt, decode)
    logprobs = completion_logprobs(model, batch)

    print(f"device={device} dtype={dtype}")
    print(f"sequences={tuple(batch.sequences.shape)}")
    print(f"completion_logprobs={tuple(logprobs.shape)}")
    print(f"mean_logprob={logprobs.mean().item():.4f}")


if __name__ == "__main__":
    main()
