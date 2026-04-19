"""Triton kernel example: Batch Normalization.

Batch norm normalizes inputs across the batch dimension:
    y = (x - mean) / sqrt(var + eps) * weight + bias
"""

import torch
import triton
import triton.language as tl
import pytest


@triton.jit
def _batch_norm_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rstd_ptr,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Normalize a single feature (column) across N samples."""
    feat_id = tl.program_id(0)

    # Load mean and rstd for this feature
    mean = tl.load(mean_ptr + feat_id)
    rstd = tl.load(rstd_ptr + feat_id)
    weight = tl.load(weight_ptr + feat_id)
    bias = tl.load(bias_ptr + feat_id)

    # Normalize each element in the feature column
    offsets = tl.arange(0, BLOCK_SIZE)
    for start in range(0, N, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < N
        x = tl.load(x_ptr + idx * tl.num_programs(0) + feat_id, mask=mask, other=0.0)
        x_norm = (x - mean) * rstd
        y = x_norm * weight + bias
        tl.store(y_ptr + idx * tl.num_programs(0) + feat_id, y, mask=mask)


def batch_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Apply batch normalization over a 2D input (N, C).

    Args:
        x: Input tensor of shape (N, C).
        weight: Scale parameter of shape (C,).
        bias: Shift parameter of shape (C,).
        eps: Small value for numerical stability.

    Returns:
        Normalized tensor of shape (N, C).
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.ndim == 2, "Input must be 2D (N, C)"
    N, C = x.shape

    x = x.contiguous()
    y = torch.empty_like(x)

    # Compute mean and variance per feature
    mean = x.mean(dim=0)
    var = x.var(dim=0, unbiased=False)
    rstd = 1.0 / torch.sqrt(var + eps)

    BLOCK_SIZE = triton.next_power_of_2(N)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)

    _batch_norm_kernel[(C,)](
        x, y, weight, bias, mean, rstd,
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


def torch_batch_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Reference implementation using PyTorch."""
    return torch.nn.functional.batch_norm(
        x, running_mean=None, running_var=None,
        weight=weight, bias=bias, training=True, eps=eps,
    )


# --- Tests ---

@pytest.mark.parametrize("N,C", [(128, 64), (256, 32), (512, 16)])
def test_batch_norm_correctness(N, C):
    x = torch.randn(N, C, device="cuda", dtype=torch.float32)
    weight = torch.ones(C, device="cuda")
    bias = torch.zeros(C, device="cuda")

    y_triton = batch_norm(x, weight, bias)
    y_torch = torch_batch_norm(x, weight, bias)

    torch.testing.assert_close(y_triton, y_torch, atol=1e-4, rtol=1e-4)


def test_batch_norm_shape_preserved():
    x = torch.randn(64, 32, device="cuda")
    weight = torch.ones(32, device="cuda")
    bias = torch.zeros(32, device="cuda")
    y = batch_norm(x, weight, bias)
    assert y.shape == x.shape


def test_batch_norm_cpu_error():
    x = torch.randn(64, 32)
    weight = torch.ones(32)
    bias = torch.zeros(32)
    with pytest.raises(AssertionError, match="CUDA"):
        batch_norm(x, weight, bias)
