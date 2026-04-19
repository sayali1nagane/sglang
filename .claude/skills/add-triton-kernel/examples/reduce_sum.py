"""Triton kernel example: parallel reduce sum along last dimension."""

import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_sum_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    BLOCK_N: tl.constexpr,
):
    """Reduce sum along the last dimension (axis=-1).

    Each program handles one row of the input matrix.
    """
    row_idx = tl.program_id(0)

    row_start = x_ptr + row_idx * stride_xm
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N

    x = tl.load(rowsets, mask=mask, other=0.0)
    result = tl.sum(x, axis=0)

    tl.store(out_ptr + row_idx, result)


def reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """Compute sum along the last dimension using a Triton kernel.

    Args:
        x: Input tensor of shape (M, N). Must be on CUDA and float32.

    Returns:
        Output tensor of shape (M,) with row-wise sums.
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.dtype == torch.float32, "Input must be float32"
    assert x.ndim == 2, "Input must be 2-dimensional"

    x = x.contiguous()
    M, N = x.shape
    out = torch.empty(M, device=x.device, dtype=x.dtype)

    BLOCK_N = triton.next_power_of_2(N)
    BLOCK_N = max(BLOCK_N, 16)

    grid = (M,)
    _reduce_sum_kernel[grid](
        x,
        out,
        M,
        N,
        x.stride(0),
        BLOCK_N=BLOCK_N,
    )
    return out


def torch_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """Reference implementation using PyTorch."""
    return x.sum(dim=-1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reduce_sum_correctness():
    torch.manual_seed(0)
    x = torch.randn(128, 256, device="cuda", dtype=torch.float32)
    triton_out = reduce_sum(x)
    torch_out = torch_reduce_sum(x)
    assert torch.allclose(triton_out, torch_out, atol=1e-4, rtol=1e-4), (
        f"Max diff: {(triton_out - torch_out).abs().max()}"
    )
    print("test_reduce_sum_correctness passed")


def test_reduce_sum_output_shape():
    x = torch.randn(64, 512, device="cuda", dtype=torch.float32)
    out = reduce_sum(x)
    assert out.shape == (64,), f"Expected (64,), got {out.shape}"
    print("test_reduce_sum_output_shape passed")


def test_reduce_sum_non_power_of_two():
    """Test with N that is not a power of two."""
    torch.manual_seed(42)
    x = torch.randn(32, 100, device="cuda", dtype=torch.float32)
    triton_out = reduce_sum(x)
    torch_out = torch_reduce_sum(x)
    assert torch.allclose(triton_out, torch_out, atol=1e-4, rtol=1e-4), (
        f"Max diff: {(triton_out - torch_out).abs().max()}"
    )
    print("test_reduce_sum_non_power_of_two passed")


def test_reduce_sum_cpu_error():
    x = torch.randn(16, 32)
    try:
        reduce_sum(x)
        assert False, "Expected AssertionError for CPU input"
    except AssertionError:
        print("test_reduce_sum_cpu_error passed")


if __name__ == "__main__":
    test_reduce_sum_correctness()
    test_reduce_sum_output_shape()
    test_reduce_sum_non_power_of_two()
    test_reduce_sum_cpu_error()
