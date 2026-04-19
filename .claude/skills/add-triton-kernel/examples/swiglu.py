"""SwiGLU activation kernel implemented in Triton.

SwiGLU(x, y) = x * sigmoid(beta * x) * y  (where beta=1 gives standard SiLU gating)
Commonly used in LLaMA/Mistral FFN layers as an alternative to plain SiLU.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused SwiGLU: out = silu(x) * gate"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    gate = tl.load(gate_ptr + offsets, mask=mask)

    # SiLU(x) = x * sigmoid(x)
    silu_x = x * tl.sigmoid(x)
    out = silu_x * gate

    tl.store(out_ptr + offsets, out, mask=mask)


def swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Compute SwiGLU activation: silu(x) * gate.

    Args:
        x: Input tensor of any shape, must be on CUDA.
        gate: Gate tensor, same shape as x, must be on CUDA.

    Returns:
        Output tensor of same shape as inputs.
    """
    assert x.is_cuda and gate.is_cuda, "Inputs must be on CUDA"
    assert x.shape == gate.shape, f"Shape mismatch: {x.shape} vs {gate.shape}"

    x = x.contiguous()
    gate = gate.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _swiglu_kernel[grid](x, gate, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def torch_swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Reference PyTorch implementation of SwiGLU."""
    return torch.nn.functional.silu(x) * gate


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_swiglu_correctness():
    torch.manual_seed(0)
    x = torch.randn(1024, 512, device="cuda", dtype=torch.float32)
    gate = torch.randn_like(x)

    out_triton = swiglu(x, gate)
    out_torch = torch_swiglu(x, gate)

    assert torch.allclose(out_triton, out_torch, atol=1e-5), (
        f"Max diff: {(out_triton - out_torch).abs().max()}"
    )
    print("test_swiglu_correctness passed")


def test_swiglu_shape_preserved():
    for shape in [(128,), (64, 64), (4, 32, 128)]:
        x = torch.randn(*shape, device="cuda")
        gate = torch.randn(*shape, device="cuda")
        out = swiglu(x, gate)
        assert out.shape == x.shape, f"Shape mismatch for input shape {shape}"
    print("test_swiglu_shape_preserved passed")


def test_swiglu_cpu_error():
    x = torch.randn(128)
    gate = torch.randn(128)
    try:
        swiglu(x, gate)
        assert False, "Expected AssertionError for CPU input"
    except AssertionError:
        pass
    print("test_swiglu_cpu_error passed")


def test_swiglu_shape_mismatch_error():
    x = torch.randn(128, device="cuda")
    gate = torch.randn(256, device="cuda")
    try:
        swiglu(x, gate)
        assert False, "Expected AssertionError for shape mismatch"
    except AssertionError:
        pass
    print("test_swiglu_shape_mismatch_error passed")


if __name__ == "__main__":
    test_swiglu_correctness()
    test_swiglu_shape_preserved()
    test_swiglu_cpu_error()
    test_swiglu_shape_mismatch_error()
    print("All tests passed.")
