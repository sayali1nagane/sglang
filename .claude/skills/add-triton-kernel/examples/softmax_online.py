"""Online softmax kernel using Triton.

Implements the numerically stable online softmax algorithm that computes
max and sum in a single pass, avoiding a separate max-reduction pass.
"""

import torch
import triton
import triton.language as tl
import pytest


@triton.jit
def _online_softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Online softmax: compute max and sum in one pass, then normalize."""
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    out_start_ptr = output_ptr + row_idx * output_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load row with masking
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float("inf"))

    # Numerically stable: subtract max before exp
    row_max = tl.max(row, axis=0)
    row = row - row_max
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    tl.store(out_start_ptr + col_offsets, softmax_output, mask=mask)


def online_softmax(x: torch.Tensor) -> torch.Tensor:
    """Compute softmax over last dimension using Triton online softmax kernel.

    Args:
        x: Input tensor of shape (M, N), must be on CUDA.

    Returns:
        Softmax output tensor of same shape as input.

    Raises:
        ValueError: If input is not a CUDA tensor or not 2D.
    """
    if not x.is_cuda:
        raise ValueError("Input tensor must be on a CUDA device.")
    if x.ndim != 2:
        raise ValueError(f"Expected 2D input tensor, got shape {x.shape}.")

    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    # Cap block size to avoid excessive register pressure
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)

    output = torch.empty_like(x)
    grid = (M,)

    _online_softmax_kernel[grid](
        output,
        x,
        x.stride(0),
        output.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def torch_online_softmax(x: torch.Tensor) -> torch.Tensor:
    """Reference implementation using PyTorch."""
    return torch.softmax(x, dim=-1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_online_softmax_correctness():
    torch.manual_seed(0)
    x = torch.randn(128, 256, device="cuda", dtype=torch.float32)
    triton_out = online_softmax(x)
    torch_out = torch_online_softmax(x)
    assert torch.allclose(triton_out, torch_out, atol=1e-5), (
        f"Max diff: {(triton_out - torch_out).abs().max().item()}"
    )


def test_online_softmax_shape_preserved():
    x = torch.randn(64, 512, device="cuda", dtype=torch.float32)
    out = online_softmax(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"


def test_online_softmax_sums_to_one():
    x = torch.randn(32, 128, device="cuda", dtype=torch.float32)
    out = online_softmax(x)
    row_sums = out.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
        f"Row sums not close to 1: {row_sums}"
    )


def test_online_softmax_cpu_error():
    x = torch.randn(4, 16)
    with pytest.raises(ValueError, match="CUDA"):
        online_softmax(x)


def test_online_softmax_non_2d_error():
    x = torch.randn(4, 16, 8, device="cuda")
    with pytest.raises(ValueError, match="2D"):
        online_softmax(x)
