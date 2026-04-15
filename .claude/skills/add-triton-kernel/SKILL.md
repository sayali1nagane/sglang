# Skill: Add Triton Kernel

This skill guides you through adding a new Triton kernel to the sglang project, including kernel implementation, Python bindings, tests, and benchmarks.

## Overview

Triton kernels in sglang live under `sgl-kernel/src/sgl-kernel/csrc/` (for any CUDA fallbacks) and the Python Triton code under `sgl-kernel/src/sgl-kernel/ops/`. They are exposed via the `sgl_kernel` Python package.

## File Structure

```
sgl-kernel/
  src/sgl-kernel/
    ops/
      your_op.py          # Triton kernel + Python wrapper
    __init__.py           # Re-export your op here
tests/
  kernels/
    test_your_op.py       # Correctness tests
benchmarks/
  kernels/
    bench_your_op.py      # Performance benchmarks
```

## Step-by-Step Guide

### 1. Implement the Triton Kernel

Create `sgl-kernel/src/sgl-kernel/ops/your_op.py`:

```python
import torch
import triton
import triton.language as tl


@triton.jit
def _your_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Core Triton kernel implementation."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    output = x  # your computation here
    tl.store(out_ptr + offsets, output, mask=mask)


def your_op(x: torch.Tensor) -> torch.Tensor:
    """Python wrapper for the Triton kernel.

    Args:
        x: Input tensor, must be on CUDA.

    Returns:
        Output tensor with same shape and dtype as input.
    """
    assert x.is_cuda, "Input must be on CUDA device"
    assert x.is_contiguous(), "Input must be contiguous"

    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _your_kernel[grid](
        x,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
```

### 2. Export from `__init__.py`

Add to `sgl-kernel/src/sgl-kernel/__init__.py`:

```python
from sgl_kernel.ops.your_op import your_op
```

### 3. Write Correctness Tests

Create `tests/kernels/test_your_op.py`:

```python
import pytest
import torch
from sgl_kernel import your_op


def torch_your_op(x: torch.Tensor) -> torch.Tensor:
    """Reference PyTorch implementation for correctness comparison."""
    return x  # mirror your kernel logic in pure PyTorch


@pytest.mark.parametrize("shape", [(128,), (1024,), (4096,), (128, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_your_op_correctness(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device="cuda")
    ref = torch_your_op(x)
    out = your_op(x)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


def test_your_op_cpu_error():
    x = torch.randn(128, device="cpu")
    with pytest.raises(AssertionError, match="CUDA"):
        your_op(x)


def test_your_op_non_contiguous_error():
    x = torch.randn(256, device="cuda")[::2]  # non-contiguous stride
    with pytest.raises(AssertionError, match="contiguous"):
        your_op(x)


def test_your_op_out_of_place():
    """Verify the kernel does not modify the input tensor."""
    x = torch.randn(512, device="cuda")
    x_clone = x.clone()
    _ = your_op(x)
    torch.testing.assert_close(x, x_clone)
```

### 4. Write Benchmarks

Create `benchmarks/kernels/bench_your_op.py`:

```python
import torch
import triton
from sgl_kernel import your_op


def bench_your_op(shape, dtype, provider):
    x = torch.randn(shape, dtype=dtype, device="cuda")

    if provider == "triton":
        fn = lambda: your_op(x)
    elif provider == "torch":
        fn = lambda: x.clone()  # replace with torch reference
    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms = triton.testing.do_bench(fn)
    gbps = x.numel() * x.element_size() * 2 / ms * 1e-6  # read + write
    return gbps


configs = [
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(10, 24)],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "--")],
        ylabel="GB/s",
        plot_name="your_op-fp16",
        args={"dtype": torch.float16},
    )
]


@triton.testing.perf_report(configs)
def benchmark(N, dtype, provider):
    return bench_your_op((N,), dtype, provider)


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=False)
```

## Common Pitfalls

### Autotuning

For performance-critical kernels, add `@triton.autotune`:

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
    ],
    key=["n_elements"],
)
@triton.jit
def _your_kernel(...):
    ...
```

### Mixed Precision

When inputs may be fp16/bf16 but accumulation needs fp32:

```python
x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
# ... compute in fp32 ...
tl.store(out_ptr + offsets, result.to(tl.float16), mask=mask)
```

### Grid Calculation for 2D Tensors

```python
def your_op_2d(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    BLOCK_M, BLOCK_N = 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _your_kernel_2d[grid](x, out, M, N, x.stride(0), x.stride(1), ...)
```

## Running Tests Locally

```bash
# Install the kernel package in editable mode
pip install -e sgl-kernel/

# Run tests
pytest tests/kernels/test_your_op.py -v

# Run with CUDA_LAUNCH_BLOCKING for easier debugging
CUDA_LAUNCH_BLOCKING=1 pytest tests/kernels/test_your_op.py -v -s

# Run benchmarks
python benchmarks/kernels/bench_your_op.py
```

## CI Integration

Add a test entry in `.github/workflows/kernel-test.yml` under the appropriate test matrix. Triton kernel tests require a GPU runner — use the `gpu` label:

```yaml
- name: Test your_op kernel
  run: pytest tests/kernels/test_your_op.py -v
```

## Examples in This Repo

- `sgl-kernel/src/sgl-kernel/ops/scale.py` — simple element-wise scaling kernel
- See `.claude/skills/add-sgl-kernel/SKILL.md` for the sgl-kernel (pybind/CUDA) variant
- See `.claude/skills/add-jit-kernel/SKILL.md` for the torch.jit variant
