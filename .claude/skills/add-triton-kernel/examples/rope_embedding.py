"""Rotary Position Embedding (RoPE) Triton kernel example.

This example demonstrates how to implement RoPE using Triton,
which is commonly used in transformer models like LLaMA.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_embedding_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr,
    q_out_ptr, k_out_ptr,
    seq_len, num_heads, head_dim,
    stride_qs, stride_qh, stride_qd,
    stride_ks, stride_kh, stride_kd,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply rotary embeddings to query and key tensors."""
    pid_s = tl.program_id(0)  # sequence position
    pid_h = tl.program_id(1)  # head index

    half_dim = head_dim // 2
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < half_dim

    # Load cos/sin for this position
    cos = tl.load(cos_ptr + pid_s * half_dim + offsets, mask=mask)
    sin = tl.load(sin_ptr + pid_s * half_dim + offsets, mask=mask)

    # Process query
    q_base = pid_s * stride_qs + pid_h * stride_qh
    q1 = tl.load(q_ptr + q_base + offsets, mask=mask)
    q2 = tl.load(q_ptr + q_base + half_dim + offsets, mask=mask)
    q_rot1 = q1 * cos - q2 * sin
    q_rot2 = q1 * sin + q2 * cos
    tl.store(q_out_ptr + q_base + offsets, q_rot1, mask=mask)
    tl.store(q_out_ptr + q_base + half_dim + offsets, q_rot2, mask=mask)

    # Process key
    k_base = pid_s * stride_ks + pid_h * stride_kh
    k1 = tl.load(k_ptr + k_base + offsets, mask=mask)
    k2 = tl.load(k_ptr + k_base + half_dim + offsets, mask=mask)
    k_rot1 = k1 * cos - k2 * sin
    k_rot2 = k1 * sin + k2 * cos
    tl.store(k_out_ptr + k_base + offsets, k_rot1, mask=mask)
    tl.store(k_out_ptr + k_base + half_dim + offsets, k_rot2, mask=mask)


def rope_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor of shape (seq_len, num_heads, head_dim).
        k: Key tensor of shape (seq_len, num_heads, head_dim).
        cos: Cosine values of shape (seq_len, head_dim // 2).
        sin: Sine values of shape (seq_len, head_dim // 2).

    Returns:
        Tuple of rotated (q, k) tensors with same shapes as input.
    """
    assert q.is_cuda and k.is_cuda, "Inputs must be on CUDA device"
    assert q.shape == k.shape, "q and k must have the same shape"
    seq_len, num_heads, head_dim = q.shape
    assert head_dim % 2 == 0, "head_dim must be even"

    q = q.contiguous()
    k = k.contiguous()
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    BLOCK_SIZE = triton.next_power_of_2(head_dim // 2)
    grid = (seq_len, num_heads)

    _rope_embedding_kernel[grid](
        q, k, cos, sin, q_out, k_out,
        seq_len, num_heads, head_dim,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return q_out, k_out


def torch_rope_embedding(q, k, cos, sin):
    """Reference implementation using PyTorch for correctness validation."""
    half = q.shape[-1] // 2
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    cos = cos.unsqueeze(1)  # (seq, 1, half)
    sin = sin.unsqueeze(1)
    q_out = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_out = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q_out, k_out


# --- Tests ---

def test_rope_embedding_correctness():
    seq_len, num_heads, head_dim = 16, 8, 64
    q = torch.randn(seq_len, num_heads, head_dim, device="cuda", dtype=torch.float32)
    k = torch.randn(seq_len, num_heads, head_dim, device="cuda", dtype=torch.float32)
    cos = torch.randn(seq_len, head_dim // 2, device="cuda", dtype=torch.float32)
    sin = torch.randn(seq_len, head_dim // 2, device="cuda", dtype=torch.float32)

    q_tri, k_tri = rope_embedding(q, k, cos, sin)
    q_ref, k_ref = torch_rope_embedding(q, k, cos, sin)

    assert torch.allclose(q_tri, q_ref, atol=1e-5), "Query mismatch"
    assert torch.allclose(k_tri, k_ref, atol=1e-5), "Key mismatch"
    print("test_rope_embedding_correctness passed")


def test_rope_embedding_output_shape():
    seq_len, num_heads, head_dim = 32, 4, 128
    q = torch.randn(seq_len, num_heads, head_dim, device="cuda")
    k = torch.randn(seq_len, num_heads, head_dim, device="cuda")
    cos = torch.randn(seq_len, head_dim // 2, device="cuda")
    sin = torch.randn(seq_len, head_dim // 2, device="cuda")
    q_out, k_out = rope_embedding(q, k, cos, sin)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    print("test_rope_embedding_output_shape passed")


def test_rope_embedding_cpu_error():
    q = torch.randn(4, 2, 32)
    k = torch.randn(4, 2, 32)
    cos = torch.randn(4, 16)
    sin = torch.randn(4, 16)
    try:
        rope_embedding(q, k, cos, sin)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        print("test_rope_embedding_cpu_error passed")


if __name__ == "__main__":
    test_rope_embedding_correctness()
    test_rope_embedding_output_shape()
    test_rope_embedding_cpu_error()
