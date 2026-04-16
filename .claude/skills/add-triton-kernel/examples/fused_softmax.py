"""Example: Fused Softmax kernel using Triton.

Demonstrates how to add a Triton kernel to sglang following the
pattern described in SKILL.md.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute softmax over the last dimension of a 2D tensor."""
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    out_start_ptr = output_ptr + row_idx * output_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load row, masking out-of-bounds with -inf
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float("inf"))

    # Subtract max for numerical stability
    row_max = tl.max(row, axis=0)
    row = row - row_max

    # Exponentiate and normalize
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    tl.store(out_start_ptr + col_offsets, softmax_output, mask=mask)


def fused_softmax(x: torch.Tensor) -> torch.Tensor:
    """Apply softmax over the last dimension using a fused Triton kernel.

    Args:
        x: Input tensor of shape (M, N) on CUDA.

    Returns:
        Softmax output tensor of the same shape.
    """
    if x.device.type != "cuda":
        raise ValueError("fused_softmax requires a CUDA tensor")
    if x.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape {x.shape}")

    x = x.contiguous()
    M, N = x.shape
    output = torch.empty_like(x)

    # Each program handles one row; block size is next power of 2 >= N
    BLOCK_SIZE = triton.next_power_of_2(N)

    _fused_softmax_kernel[(M,)](
        x,
        output,
        x.stride(0),
        output.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def torch_fused_softmax(x: torch.Tensor) -> torch.Tensor:
    """Reference implementation using PyTorch (for correctness testing)."""
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fused_softmax_correctness():
    torch.manual_seed(0)
    x = torch.randn(128, 256, device="cuda", dtype=torch.float32)
    out_triton = fused_softmax(x)
    out_torch = torch_fused_softmax(x)
    assert torch.allclose(out_triton, out_torch, atol=1e-5), "Outputs differ!"
    print("test_fused_softmax_correctness passed")


def test_fused_softmax_shape_preserved():
    x = torch.randn(64, 512, device="cuda", dtype=torch.float32)
    out = fused_softmax(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print("test_fused_softmax_shape_preserved passed")


def test_fused_softmax_cpu_error():
    x = torch.randn(4, 8)
    try:
        fused_softmax(x)
        raise AssertionError("Expected ValueError for CPU tensor")
    except ValueError:
        pass
    print("test_fused_softmax_cpu_error passed")


if __name__ == "__main__":
    test_fused_softmax_correctness()
    test_fused_softmax_shape_preserved()
    test_fused_softmax_cpu_error()
    print("All tests passed.")
