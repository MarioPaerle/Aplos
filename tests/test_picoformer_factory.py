"""Smoke tests for the production-facing PiCOFormer scaffold."""

import tempfile

import torch

from Vathos.picoformer import (
    DecodeConfig,
    PiCOFormerConfig,
    build_picoformer,
    completion_logprobs,
    generate_grpo_rollouts,
    load_picoformer,
    save_picoformer,
)


def _tiny_config(channel="torch_relu2"):
    return PiCOFormerConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
        channel=channel,
        ffn_multiplier=2,
        production_mode=True,
    )


def test_factory_forward_grpo_and_checkpoint_roundtrip():
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = build_picoformer(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (2, 6))

    logits = model(prompt)
    assert logits.shape == (2, 6, cfg.vocab_size)

    decode = DecodeConfig(max_new_tokens=5, group_size=2, return_logprobs=True)
    batch = generate_grpo_rollouts(model, prompt[:1], decode)
    assert batch.sequences.shape == (2, 11)
    assert batch.completions.shape == (2, 5)
    assert batch.completion_mask.shape == (2, 5)
    assert batch.sampling_logprobs.shape == (2, 5)

    logprobs = completion_logprobs(model, batch)
    assert logprobs.shape == (2, 5)
    assert torch.isfinite(logprobs).all()

    with tempfile.TemporaryDirectory() as tmp:
        save_picoformer(model, tmp, cfg)
        loaded = load_picoformer(tmp)
        loaded_logits = loaded(prompt)
    assert torch.allclose(logits, loaded_logits, atol=1e-6, rtol=1e-6)


def test_triton_channel_falls_back_without_triton_gpu():
    torch.manual_seed(1)
    cfg = _tiny_config(channel="triton_lrelu2")
    model = build_picoformer(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    y = model(x)
    assert y.shape == (1, 8, cfg.vocab_size)
    assert torch.isfinite(y).all()


if __name__ == "__main__":
    test_factory_forward_grpo_and_checkpoint_roundtrip()
    test_triton_channel_falls_back_without_triton_gpu()
    print("picoformer factory smoke tests passed")
