"""Triton kernel example: cumulative sum (prefix sum) along the last dimension."""

import torch
import triton
import triton.language as tl
import pytest


@triton.jit
def _cumsum_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute inclusive prefix sum along each row."""
    row = tl.program_id(0)
    row_start = row * n_cols

    # Load a block of elements
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)

    # Inclusive scan (prefix sum)
    acc = tl.cumsum(x, axis=0)

    tl.store(out_ptr + row_start + offsets, acc, mask=mask)


def cumsum(x: torch.Tensor) -> torch.Tensor:
    """Compute inclusive cumulative sum along the last dimension using Triton.

    Args:
        x: Input tensor of shape (n_rows, n_cols) on CUDA.

    Returns:
        Tensor of the same shape with prefix sums along each row.
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.ndim == 2, "Input must be 2-dimensional"

    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = max(BLOCK_SIZE, 16)

    _cumsum_kernel[(n_rows,)](
        x,
        out,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def torch_cumsum(x: torch.Tensor) -> torch.Tensor:
    """Reference implementation using PyTorch."""
    return torch.cumsum(x, dim=-1)


# ── Tests ──────────────────────────────────────────────────────────────────


def test_cumsum_correctness():
    torch.manual_seed(0)
    x = torch.randn(128, 256, device="cuda", dtype=torch.float32)
    out_triton = cumsum(x)
    out_torch = torch_cumsum(x)
    assert torch.allclose(out_triton, out_torch, atol=1e-4, rtol=1e-4), (
        f"Max diff: {(out_triton - out_torch).abs().max().item()}"
    )


def test_cumsum_shape_preserved():
    x = torch.ones(64, 128, device="cuda", dtype=torch.float32)
    out = cumsum(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_cumsum_ones():
    """cumsum of a row of ones should be [1, 2, 3, ...]."""
    n_cols = 64
    x = torch.ones(4, n_cols, device="cuda", dtype=torch.float32)
    out = cumsum(x)
    expected = torch.arange(1, n_cols + 1, dtype=torch.float32, device="cuda")
    for row in out:
        assert torch.allclose(row, expected), f"Row mismatch: {row[:8]}"


def test_cumsum_cpu_error():
    x = torch.randn(4, 16)
    with pytest.raises(AssertionError, match="CUDA"):
        cumsum(x)


def test_cumsum_non_power_of_two_cols():
    """Ensure kernel handles column counts that aren't powers of two."""
    x = torch.randn(32, 100, device="cuda", dtype=torch.float32)
    out_triton = cumsum(x)
    out_torch = torch_cumsum(x)
    assert torch.allclose(out_triton, out_torch, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    test_cumsum_correctness()
    test_cumsum_shape_preserved()
    test_cumsum_ones()
    test_cumsum_non_power_of_two_cols()
    print("All cumsum tests passed.")
