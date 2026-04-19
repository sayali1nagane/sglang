"""Triton kernel example: matrix multiplication with tiling.

Demonstrates a blocked matrix multiplication kernel using Triton,
compared against torch.matmul for correctness verification.
"""

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Tiled matrix multiplication: C = A @ B."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Triton matrix multiplication.

    Args:
        a: Input tensor of shape (M, K), float16, on CUDA.
        b: Input tensor of shape (K, N), float16, on CUDA.

    Returns:
        Output tensor of shape (M, N), float16.
    """
    assert a.is_cuda and b.is_cuda, "Inputs must be on CUDA"
    assert a.dtype == torch.float16 and b.dtype == torch.float16
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"

    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


def torch_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Reference implementation using torch.matmul."""
    return torch.matmul(a.float(), b.float()).half()


# --- Tests ---

@pytest.mark.parametrize("M,N,K", [(128, 128, 64), (256, 512, 128), (64, 64, 32)])
def test_matmul_correctness(M, N, K):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    out_triton = matmul(a, b)
    out_torch = torch_matmul(a, b)
    assert torch.allclose(out_triton, out_torch, atol=1e-2, rtol=1e-2), \
        f"Max diff: {(out_triton - out_torch).abs().max()}"


def test_matmul_output_shape():
    a = torch.randn((64, 32), device="cuda", dtype=torch.float16)
    b = torch.randn((32, 128), device="cuda", dtype=torch.float16)
    out = matmul(a, b)
    assert out.shape == (64, 128)


def test_matmul_cpu_error():
    a = torch.randn((32, 32), dtype=torch.float16)
    b = torch.randn((32, 32), dtype=torch.float16)
    with pytest.raises(AssertionError, match="CUDA"):
        matmul(a, b)
