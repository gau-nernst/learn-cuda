import os
from pathlib import Path

os.environ["PYTORCH_ROCM_ARCH"] = "gfx942"

import torch
from torch.utils.cpp_extension import load
from triton.testing import do_bench

CURRENT_DIR = Path(__file__).parent
load(
    "hip_matmul",
    [*list(CURRENT_DIR.glob("*.cu"))],
    is_python_module=False,
    verbose=True,
)
ops = torch.ops.hip_matmul

M, N, K = 4096, 4096, 4096
dtype = torch.bfloat16
torch.set_default_device("cuda")

A = torch.randn(M, K, dtype=dtype)
B = torch.randn(N, K, dtype=dtype).T

ref = torch.mm(A, B)


def bench(name, f):
    out = f(A, B)
    # torch.testing.assert_close(out, ref)
    diff = (out - ref).abs().mean().item()

    latency_ms = do_bench(lambda: f(A, B), warmup=100, rep=500)
    tflops = 2 * M * N * K / (latency_ms * 1e-3) * 1e-12
    print(f"{name}: {tflops:.2f} TFLOPS, {diff}")


bench("PyTorch", torch.mm)
bench("v1", ops.matmul_v1)
