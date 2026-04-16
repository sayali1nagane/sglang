"""Layer Norm Triton kernel example.

Demonstrates how to implement a fused layer normalization kernel using Triton,
following the add-triton-kernel skill conventions.
"""

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused layer norm: compute mean, variance, normalize, scale, and shift."""
    row = tl.program_id(0)
    x_ptr += row * stride
    out_ptr += row * stride

    # Load row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Compute mean
    mean = tl.sum(x, axis=0) / N

    # Compute variance
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N

    # Normalize
    x_norm = diff / tl.sqrt(var + eps)

    # Scale and shift
    weightptr + cols, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = x_norm * weight + bias

    tl.store(out_ptr + cols, out, mask=mask)


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Apply layer normalization using a fused Triton kernel.

    Args:
        x: Input tensor of shape (*, N).
        weight: Learnable scale parameter of shape (N,).
        bias: Learnable shift parameter of shape (N,).
        eps: Small value for numerical stability.

    Returns:
        Normalized tensor with the same shape as x.
    """
    assert x.is_cuda, "Input must be on CUDA"
    orig_shape = x.shape
    x = x.view(-1, orig_shape[-1])
    M, N = x.shape

    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)

    _layer_norm_kernel[(M,)](
        x,
        weight,
        bias,
        out,
        x.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out.view(orig_shape)


def torch_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Reference implementation using PyTorch."""
    return torch.nn.functional.layer_norm(
        x, (x.shape[-1],), weight=weight, bias=bias, eps=eps
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(4, 64), (16, 128), (2, 3, 256)])
def test_layer_norm_correctness(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=torch.float32)
    N = shape[-1]
    weight = torch.randn(N, device="cuda", dtype=torch.float32)
    bias = torch.randn(N, device="cuda", dtype=torch.float32)

    ref = torch_layer_norm(x, weight, bias)
    out = layer_norm(x, weight, bias)

    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_layer_norm_shape_preserved():
    x = torch.randn(8, 32, device="cuda")
    weight = torch.ones(32, device="cuda")
    bias = torch.zeros(32, device="cuda")
    out = layer_norm(x, weight, bias)
    assert out.shape == x.shape


def test_layer_norm_cpu_error():
    x = torch.randn(4, 32)
    weight = torch.ones(32)
    bias = torch.zeros(32)
    with pytest.raises(AssertionError, match="CUDA"):
        layer_norm(x, weight, bias)
