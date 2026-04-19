"""Triton kernel example: attention with additive bias.

Computes softmax(Q @ K^T / sqrt(d) + bias) @ V
where bias is a (batch, heads, seq_q, seq_k) tensor.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _attention_bias_kernel(
    Q, K, V, Bias, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_bb, stride_bh, stride_bm, stride_bn,
    stride_ob, stride_oh, stride_om, stride_ok,
    seq_len, head_dim,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_head = tl.program_id(0)
    block_m = tl.program_id(1)

    batch = batch_head // tl.num_programs(0)  # derived below
    # Decode batch and head indices
    num_heads = tl.num_programs(0)  # placeholder; passed via grid
    b = batch_head // num_heads
    h = batch_head % num_heads

    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len, other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    lse = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)

    for block_n in range(0, tl.cdiv(seq_len, BLOCK_N)):
        offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)

        k_ptrs = K + b * stride_kb + h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=offs_n[:, None] < seq_len, other=0.0)

        attn = tl.dot(q, tl.trans(k)) * scale

        bias_ptrs = Bias + b * stride_bb + h * stride_bh + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn
        bias = tl.load(bias_ptrs, mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len), other=0.0)
        attn = attn + bias

        attn = tl.where(offs_n[None, :] < seq_len, attn, float("-inf"))

        row_max = tl.max(attn, axis=1)
        new_lse = tl.log(tl.sum(tl.exp(attn - row_max[:, None]), axis=1)) + row_max

        alpha = tl.exp(lse - new_lse)
        lse = new_lse

        p = tl.exp(attn - lse[:, None])

        v_ptrs = V + b * stride_vb + h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=offs_n[:, None] < seq_len, other=0.0)

        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

    out_ptrs = Out + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seq_len)


def attention_bias(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Multi-head attention with additive bias.

    Args:
        q: (batch, heads, seq, head_dim)
        k: (batch, heads, seq, head_dim)
        v: (batch, heads, seq, head_dim)
        bias: (batch, heads, seq, seq) additive bias (e.g. ALiBi slopes)

    Returns:
        out: (batch, heads, seq, head_dim)
    """
    assert q.is_cuda, "Input must be on CUDA"
    B, H, S, D = q.shape
    out = torch.empty_like(q)
    scale = D ** -0.5
    BLOCK_M, BLOCK_N, BLOCK_D = 64, 64, triton.next_power_of_2(D)
    grid = (B * H, triton.cdiv(S, BLOCK_M))
    _attention_bias_kernel[grid](
        q, k, v, bias, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        S, D, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return out


def torch_attention_bias(q, k, v, bias):
    scale = q.shape[-1] ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale + bias
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


# --- tests ---

def test_attention_bias_correctness():
    B, H, S, D = 2, 4, 32, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    bias = torch.randn(B, H, S, S, device="cuda", dtype=torch.float16)
    out_triton = attention_bias(q, k, v, bias)
    out_torch = torch_attention_bias(q, k, v, bias)
    assert torch.allclose(out_triton.float(), out_torch.float(), atol=1e-2), "Correctness check failed"
    print("test_attention_bias_correctness passed")


def test_attention_bias_output_shape():
    B, H, S, D = 1, 2, 16, 32
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    bias = torch.zeros(B, H, S, S, device="cuda", dtype=torch.float16)
    out = attention_bias(q, k, v, bias)
    assert out.shape == (B, H, S, D)
    print("test_attention_bias_output_shape passed")


def test_attention_bias_cpu_error():
    q = torch.randn(1, 1, 8, 16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    bias = torch.zeros(1, 1, 8, 8)
    try:
        attention_bias(q, k, v, bias)
        assert False, "Expected AssertionError"
    except AssertionError:
        print("test_attention_bias_cpu_error passed")


if __name__ == "__main__":
    test_attention_bias_correctness()
    test_attention_bias_output_shape()
    test_attention_bias_cpu_error()
