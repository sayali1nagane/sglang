"""Triton kernel example: GELU activation function.

Implements the GELU (Gaussian Error Linear Unit) activation:
    GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))

Also supports the fast tanh approximation:
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
"""

import math
import torch
import triton
import triton.language as tl
import pytest


@triton.jit
def _gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    approximate: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    if approximate:
        # Tanh approximation
        sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
        coeff = 0.044715
        inner = sqrt_2_over_pi * (x + coeff * x * x * x)
        result = 0.5 * x * (1.0 + tl.math.tanh(inner))
    else:
        # Exact GELU using erf
        sqrt_half = 0.7071067811865476  # 1 / sqrt(2)
        result = x * 0.5 * (1.0 + tl.math.erf(x * sqrt_half))

    tl.store(out_ptr + offsets, result, mask=mask)


def gelu(x: torch.Tensor, approximate: bool = False) -> torch.Tensor:
    """Apply GELU activation using a Triton kernel.

    Args:
        x: Input tensor (must be on CUDA).
        approximate: If True, use the tanh approximation. Default: False.

    Returns:
        Tensor with GELU applied, same shape and dtype as input.
    """
    assert x.is_cuda, "Input tensor must be on CUDA"
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _gelu_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        approximate=approximate,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def torch_gelu(x: torch.Tensor, approximate: bool = False) -> torch.Tensor:
    """Reference GELU implementation using PyTorch."""
    approx_str = "tanh" if approximate else "none"
    return torch.nn.functional.gelu(x, approximate=approx_str)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1024,), (128, 256), (4, 32, 64)])
@pytest.mark.parametrize("approximate", [False, True])
def test_gelu_correctness(shape, approximate):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    expected = torch_gelu(x, approximate=approximate)
    actual = gelu(x, approximate=approximate)
    atol = 1e-4 if not approximate else 1e-3
    assert torch.allclose(actual, expected, atol=atol), (
        f"Max diff: {(actual - expected).abs().max().item()}"
    )


@pytest.mark.parametrize("shape", [(512,), (64, 128)])
def test_gelu_shape_preserved(shape):
    x = torch.randn(shape, device="cuda")
    out = gelu(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_gelu_cpu_error():
    x = torch.randn(256)
    with pytest.raises(AssertionError, match="CUDA"):
        gelu(x)


if __name__ == "__main__":
    x = torch.randn(4096, device="cuda", dtype=torch.float32)
    for approx in [False, True]:
        label = "tanh-approx" if approx else "exact"
        out = gelu(x, approximate=approx)
        ref = torch_gelu(x, approximate=approx)
        diff = (out - ref).abs().max().item()
        print(f"GELU ({label}) max diff vs torch: {diff:.6e}")
