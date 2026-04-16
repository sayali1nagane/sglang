"""Example Triton kernel: Flash Attention (forward pass).

Demonstrates implementing a memory-efficient attention kernel using Triton,
following the FlashAttention algorithm to avoid materializing the full NxN
attention matrix.
"""

import torch
import triton
import triton.language as tl
import pytest


@triton.jit
def _flash_attention_fwd_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    scale: tl.constexpr,
):
    """Flash attention forward kernel."""
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // tl.num_programs(1)  # not used directly, bh encodes both

    # Offsets for Q block
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    # Pointers to Q, K, V for this (batch, head)
    Q_ptr = Q + off_bh * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    K_ptr = K + off_bh * stride_kh + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
    V_ptr = V + off_bh * stride_vh + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
    O_ptr = Out + off_bh * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok

    # Load Q block
    q = tl.load(Q_ptr, mask=offs_m[:, None] < N_CTX, other=0.0)

    # Running max and sum for online softmax
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Iterate over K/V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n_curr = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(K_ptr + start_n * stride_kn, mask=offs_n_curr[:, None] < N_CTX, other=0.0)
        v = tl.load(V_ptr + start_n * stride_vn, mask=offs_n_curr[:, None] < N_CTX, other=0.0)

        # Compute attention scores
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(offs_n_curr[None, :] < N_CTX, s, float('-inf'))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = alpha[:, None] * acc + tl.dot(p.to(tl.float16), v)
        m_i = m_new

    # Normalize
    acc = acc / l_i[:, None]
    tl.store(O_ptr, acc.to(tl.float16), mask=offs_m[:, None] < N_CTX)


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Memory-efficient attention using Triton Flash Attention kernel.

    Args:
        q: Query tensor of shape (batch, heads, seq_len, head_dim).
        k: Key tensor of shape (batch, heads, seq_len, head_dim).
        v: Value tensor of shape (batch, heads, seq_len, head_dim).

    Returns:
        Output tensor of shape (batch, heads, seq_len, head_dim).
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be on CUDA"
    B, H, N, D = q.shape
    scale = D ** -0.5
    out = torch.empty_like(q)

    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(N, BLOCK_M), B * H)

    _flash_attention_fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        N_CTX=N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D, scale=scale,
    )
    return out


def torch_flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Reference implementation using PyTorch scaled dot-product attention."""
    return torch.nn.functional.scaled_dot_product_attention(q, v, v)


# --- Tests ---

def test_flash_attention_correctness():
    B, H, N, D = 2, 4, 128, 64
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

    tri_out = flash_attention(q, k, v)
    ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=1e-2)


def test_flash_attention_output_shape():
    B, H, N, D = 1, 2, 64, 32
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    out = flash_attention(q, k, v)
    assert out.shape == (B, H, N, D)


def test_flash_attention_cpu_error():
    q = torch.randn(1, 1, 16, 16)
    k = torch.randn(1, 1, 16, 16)
    v = torch.randn(1, 1, 16, 16)
    with pytest.raises(AssertionError, match="CUDA"):
        flash_attention(q, k, v)


if __name__ == "__main__":
    test_flash_attention_output_shape()
    test_flash_attention_correctness()
    print("All tests passed.")
