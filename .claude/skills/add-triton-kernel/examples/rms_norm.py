"""Example: Fused RMS Normalization Triton kernel.

Demonstrates how to add a custom Triton kernel following the
add-triton-kernel skill guidelines.
"""

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute RMS normalization for a single row."""
    row = tl.program_id(0)
    x_ptr += row * stride
    out_ptr += row * stride

    # Compute sum of squares
    sq_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sq_sum += x * x

    # RMS = sqrt(mean(x^2) + eps)
    rms = tl.sqrt(tl.sum(sq_sum) / N + eps)

    # Normalize and apply weight
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        out = (x / rms) * w
        tl.store(out_ptr + cols, out.to(tl.float16), mask=mask)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Triton-accelerated RMS normalization.

    Args:
        x: Input tensor of shape (M, N), must be on CUDA.
        weight: Scale parameter of shape (N,), must be on CUDA.
        eps: Epsilon for numerical stability.

    Returns:
        Normalized tensor of shape (M, N).
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA"
    assert x.ndim == 2, "Input must be 2D (M, N)"
    assert weight.shape[0] == x.shape[1], "Weight shape must match last dim of x"

    M, N = x.shape
    out = torch.empty_like(x, dtype=torch.float16)
    BLOCK_SIZE = triton.next_power_of_2(N)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)

    _rms_norm_kernel[(M,)](
        x, weight, out,
        x.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def torch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference PyTorch implementation of RMS normalization."""
    rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() * rms * weight.float()).half()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N", [(128, 256), (64, 512), (1, 1024)])
def test_rms_norm_correctness(M, N):
    """Triton output should match PyTorch reference within tolerance."""
    x = torch.randn(M, N, device="cuda", dtype=torch.float16)
    w = torch.randn(N, device="cuda", dtype=torch.float16)

    triton_out = rms_norm(x, w)
    ref_out = torch_rms_norm(x, w)

    torch.testing.assert_close(triton_out, ref_out, atol=1e-2, rtol=1e-2)


def test_rms_norm_shape_preserved():
    """Output shape must equal input shape."""
    x = torch.randn(32, 128, device="cuda", dtype=torch.float16)
    w = torch.ones(128, device="cuda", dtype=torch.float16)
    assert rms_norm(x, w).shape == x.shape


def test_rms_norm_cpu_error():
    """Should raise AssertionError when inputs are on CPU."""
    x = torch.randn(4, 16)
    w = torch.ones(16)
    with pytest.raises(AssertionError, match="CUDA"):
        rms_norm(x, w)
