"""Cross entropy loss Triton kernel example.

Demonstrates fused cross entropy with log-softmax for language model training.
"""

import torch
import triton
import triton.language as tl
import pytest


@triton.jit
def _cross_entropy_kernel(
    logits_ptr,
    labels_ptr,
    losses_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused log-softmax + NLL loss kernel."""
    row_idx = tl.program_id(0)
    logits_ptr += row_idx * n_cols
    labels_ptr += row_idx
    losses_ptr += row_idx

    # Load label for this row
    label = tl.load(labels_ptr)

    # Compute max for numerical stability
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
    row_max = tl.max(logits, axis=0)

    # Compute log-sum-exp
    shifted = logits - row_max
    exp_shifted = tl.exp(shifted)
    sum_exp = tl.sum(exp_shifted, axis=0)
    log_sum_exp = tl.log(sum_exp)

    # Load the logit for the correct label
    label_logit = tl.load(logits_ptr + label)
    shifted_label_logit = label_logit - row_max

    # NLL loss = log_sum_exp - shifted_label_logit
    loss = log_sum_exp - shifted_label_logit
    tl.store(losses_ptr, loss)


def cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute cross entropy loss using a fused Triton kernel.

    Args:
        logits: Float tensor of shape (batch_size, vocab_size).
        labels: Long tensor of shape (batch_size,) with class indices.

    Returns:
        losses: Float tensor of shape (batch_size,) with per-sample losses.
    """
    assert logits.is_cuda and labels.is_cuda, "Inputs must be on CUDA"
    assert logits.ndim == 2, "logits must be 2D"
    assert labels.ndim == 1 and labels.shape[0] == logits.shape[0]

    batch_size, n_cols = logits.shape
    losses = torch.empty(batch_size, dtype=logits.dtype, device=logits.device)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    _cross_entropy_kernel[(batch_size,)](
        logits,
        labels,
        losses,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return losses


def torch_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Reference implementation using PyTorch."""
    return torch.nn.functional.cross_entropy(logits, labels, reduction="none")


# --- Tests ---


def test_cross_entropy_correctness():
    torch.manual_seed(42)
    batch_size, vocab_size = 16, 512
    logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=torch.float32)
    labels = torch.randint(0, vocab_size, (batch_size,), device="cuda")

    triton_out = cross_entropy(logits, labels)
    torch_out = torch_cross_entropy(logits, labels)

    torch.testing.assert_close(triton_out, torch_out, atol=1e-4, rtol=1e-4)


def test_cross_entropy_output_shape():
    batch_size, vocab_size = 8, 256
    logits = torch.randn(batch_size, vocab_size, device="cuda")
    labels = torch.randint(0, vocab_size, (batch_size,), device="cuda")
    out = cross_entropy(logits, labels)
    assert out.shape == (batch_size,)


def test_cross_entropy_cpu_error():
    logits = torch.randn(4, 128)
    labels = torch.randint(0, 128, (4,))
    with pytest.raises(AssertionError, match="CUDA"):
        cross_entropy(logits, labels)


if __name__ == "__main__":
    test_cross_entropy_correctness()
    print("All tests passed.")
