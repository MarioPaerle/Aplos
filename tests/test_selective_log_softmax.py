"""Selective log-softmax correctness and gradient behavior."""

import torch
import torch.nn.functional as F

from Vathos.functions import selective_log_softmax


def test_selective_log_softmax_matches_manual_and_backprops():
    torch.manual_seed(0)
    logits = torch.randn(3, 4, 17, requires_grad=True)
    index = torch.randint(0, 17, (3, 4))

    got = selective_log_softmax(logits, index)
    expected = F.log_softmax(logits.float(), dim=-1).gather(
        -1, index.unsqueeze(-1)
    ).squeeze(-1)
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-6)

    loss = -got.mean()
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_selective_log_softmax_no_grad_cuda_kernel_matches_manual_if_available():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(1)
    logits = torch.randn(8, 257, device="cuda", dtype=torch.float16)
    index = torch.randint(0, 257, (8,), device="cuda")

    with torch.no_grad():
        got = selective_log_softmax(logits, index)
        expected = F.log_softmax(logits.float(), dim=-1).gather(
            -1, index.unsqueeze(-1)
        ).squeeze(-1)
    assert torch.allclose(got.cpu(), expected.cpu(), atol=2e-3, rtol=2e-3)


if __name__ == "__main__":
    test_selective_log_softmax_matches_manual_and_backprops()
    test_selective_log_softmax_no_grad_cuda_kernel_matches_manual_if_available()
    print("selective log-softmax tests passed")
