import os
from pathlib import Path

os.environ["PYTORCH_ROCM_ARCH"] = "gfx942"

import torch
from torch.utils.cpp_extension import load

CURRENT_DIR = Path(__file__).parent
load(
    "hip_matmul",
    [*list(CURRENT_DIR.glob("*.cu"))],
    is_python_module=False,
    verbose=True,
)
ops = torch.ops.hip_matmul

M, N, K = 128, 128, 64
dtype = torch.bfloat16
torch.set_default_device("cuda")

A = torch.randn(M, K, dtype=dtype)
B = torch.randn(N, K, dtype=dtype).T

ref = torch.mm(A, B)
print(ref)

C = ops.matmul_v1(A, B)
print(C)
# breakpoint()
torch.testing.assert_close(C, ref)
