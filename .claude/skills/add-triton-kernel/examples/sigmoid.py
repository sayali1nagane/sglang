"""Triton kernel example: element-wise sigmoid activation.

Demonstrates a simple element-wise unary kernel with Triton.
"""

import torch
import triton
import triton.language as tl
import pytest


@triton.jit
def _sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute sigmoid(x) = 1 / (1 + exp(-x)) element-wise."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    out = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, out, mask=mask)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Triton-accelerated sigmoid activation.

    Args:
        x: Input tensor of any shape on CUDA.

    Returns:
        Tensor of the same shape with sigmoid applied element-wise.
    """
    if x.device.type != "cuda":
        raise ValueError("Input tensor must be on a CUDA device.")

    x_flat = x.contiguous().view(-1)
    out = torch.empty_like(x_flat)
    n_elements = x_flat.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _sigmoid_kernel[grid](
        x_flat,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out.view(x.shape)


def torch_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Reference implementation using PyTorch."""
    return torch.sigmoid(x)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sigmoid_correctness():
    torch.manual_seed(0)
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    out_triton = sigmoid(x)
    out_torch = torch_sigmoid(x)
    assert torch.allclose(out_triton, out_torch, atol=1e-5), (
        f"Max diff: {(out_triton - out_torch).abs().max().item()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sigmoid_shape_preserved():
    x = torch.randn(4, 128, 64, device="cuda")
    out = sigmoid(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sigmoid_bounds():
    """Sigmoid output must be in (0, 1)."""
    x = torch.randn(2048, device="cuda")
    out = sigmoid(x)
    assert out.min().item() > 0.0
    assert out.max().item() < 1.0


def test_sigmoid_cpu_error():
    x = torch.randn(64)
    with pytest.raises(ValueError, match="CUDA"):
        sigmoid(x)
