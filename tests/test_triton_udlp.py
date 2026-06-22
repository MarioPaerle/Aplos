"""Correctness tests for Triton UDLP wrappers and their torch fallbacks."""

import torch
import torch.nn.functional as F

from Vathos._triton import (
    TritonLReLU2UDLP_A100,
    TritonLReLU2UDLP,
    TritonReLU2UDLP,
    TritonSwiGLUUDLP,
)


def test_triton_relu2_udlp_matches_eager_formula():
    torch.manual_seed(0)
    layer = TritonReLU2UDLP(d_model=16, d_output=16, M=32, dropout=0.0)
    x = torch.randn(3, 5, 16, requires_grad=True)

    y = layer(x)
    expected = F.relu(F.linear(x, layer.expand.weight)).square() @ layer.contract.weight.T
    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6)

    y.square().mean().backward()
    assert x.grad is not None
    assert layer.expand.weight.grad is not None
    assert layer.contract.weight.grad is not None


def test_triton_lrelu2_udlp_matches_eager_formula():
    torch.manual_seed(1)
    layer = TritonLReLU2UDLP(d_model=16, d_output=16, M=32, dropout=0.0)
    x = torch.randn(3, 5, 16, requires_grad=True)

    y = layer(x)
    expected = F.leaky_relu(F.linear(x, layer.expand.weight), 0.5).square() @ layer.contract.weight.T
    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6)

    y.square().mean().backward()
    assert x.grad is not None
    assert layer.expand.weight.grad is not None
    assert layer.contract.weight.grad is not None


def test_triton_lrelu2_a100_udlp_matches_eager_formula():
    torch.manual_seed(11)
    layer = TritonLReLU2UDLP_A100(d_model=16, d_output=16, M=32, dropout=0.0)
    x = torch.randn(3, 5, 16, requires_grad=True)

    y = layer(x)
    expected = F.leaky_relu(F.linear(x, layer.expand.weight), 0.5).square() @ layer.contract.weight.T
    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6)

    y.square().mean().backward()
    assert x.grad is not None
    assert layer.expand.weight.grad is not None
    assert layer.contract.weight.grad is not None


def test_triton_swiglu_udlp_matches_eager_formula():
    torch.manual_seed(2)
    layer = TritonSwiGLUUDLP(d_model=16, d_output=16, M=32, dropout=0.0)
    x = torch.randn(3, 5, 16, requires_grad=True)

    h = layer.expand(x)
    a, b = h.chunk(2, dim=-1)
    expected = layer.contract(F.silu(a) * b)
    y = layer(x)
    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6)

    y.square().mean().backward()
    assert x.grad is not None
    assert layer.expand.weight.grad is not None
    assert layer.contract.weight.grad is not None


if __name__ == "__main__":
    test_triton_relu2_udlp_matches_eager_formula()
    test_triton_lrelu2_udlp_matches_eager_formula()
    test_triton_lrelu2_a100_udlp_matches_eager_formula()
    test_triton_swiglu_udlp_matches_eager_formula()
    print("triton UDLP smoke tests passed")
