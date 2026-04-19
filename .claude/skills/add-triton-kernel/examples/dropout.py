"""Triton kernel example: dropout with optional mask output.

Demonstrates element-wise stochastic dropout using Triton,
with a philox-based RNG for reproducible results.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _dropout_kernel(
    x_ptr,
    out_ptr,
    mask_ptr,
    n_elements,
    p,  # drop probability
    seed,
    save_mask: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # Generate uniform random numbers via philox
    rand = tl.rand(seed, offsets)
    keep = rand > p
    scale = 1.0 / (1.0 - p)  # inverted dropout scaling

    out = tl.where(keep, x * scale, tl.zeros_like(x))
    tl.store(out_ptr + offsets, out, mask=mask)

    if save_mask:
        tl.store(mask_ptr + offsets, keep.to(tl.int8), mask=mask)


def dropout(
    x: torch.Tensor,
    p: float = 0.5,
    seed: int = 42,
    return_mask: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply dropout using a Triton kernel.

    Args:
        x: Input tensor (must be on CUDA).
        p: Probability of dropping an element (0 <= p < 1).
        seed: RNG seed for reproducibility.
        return_mask: If True, also return the boolean keep-mask.

    Returns:
        Dropped-out tensor, or (tensor, mask) if return_mask=True.
    """
    if x.device.type != "cuda":
        raise ValueError("dropout requires a CUDA tensor")
    if not (0.0 <= p < 1.0):
        raise ValueError(f"p must be in [0, 1), got {p}")

    x_flat = x.contiguous().view(-1)
    n = x_flat.numel()
    out = torch.empty_like(x_flat)
    mask_buf = torch.empty(n, dtype=torch.int8, device=x.device) if return_mask else out  # dummy

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    _dropout_kernel[grid](
        x_flat, out, mask_buf,
        n, p, seed,
        save_mask=return_mask,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    out = out.view(x.shape)
    if return_mask:
        return out, mask_buf.view(x.shape).bool()
    return out


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def torch_dropout(x: torch.Tensor, p: float = 0.5, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.nn.functional.dropout(x, p=p, training=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dropout_output_shape():
    x = torch.randn(128, 64, device="cuda")
    out = dropout(x, p=0.3)
    assert out.shape == x.shape, f"shape mismatch: {out.shape} vs {x.shape}"


def test_dropout_zero_probability():
    """With p=0, output should equal input (scaled by 1)."""
    x = torch.randn(256, device="cuda")
    out = dropout(x, p=0.0, seed=0)
    torch.testing.assert_close(out, x)


def test_dropout_return_mask():
    x = torch.ones(512, device="cuda")
    out, mask = dropout(x, p=0.5, seed=7, return_mask=True)
    assert mask.dtype == torch.bool
    assert mask.shape == x.shape
    # Kept elements should be scaled to 2.0
    assert torch.all((out[mask] - 2.0).abs() < 1e-5)
    assert torch.all(out[~mask] == 0.0)


def test_dropout_cpu_error():
    x = torch.randn(64)
    try:
        dropout(x)
        assert False, "Expected ValueError for CPU input"
    except ValueError:
        pass


if __name__ == "__main__":
    test_dropout_output_shape()
    test_dropout_zero_probability()
    test_dropout_return_mask()
    test_dropout_cpu_error()
    print("All dropout tests passed.")
