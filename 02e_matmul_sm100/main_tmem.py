from pathlib import Path

import torch
import torch.utils.cpp_extension

CURRENT_DIR = Path(__file__).parent

torch.utils.cpp_extension.load(
    "my_module",
    sources=[str(CURRENT_DIR / "_matmul_tmem.cu")],
    extra_cuda_cflags=[
        "-O3",
        "-lineinfo",
        "-Xptxas=-v",
        "-gencode=arch=compute_100a,code=sm_100a",
    ],
    extra_ldflags=["-lcuda"],  # for cuTensorMapEncodeTiled() used by TMA
    verbose=True,
    is_python_module=False,
)

M, N, K = 4096, 4096, 4096
A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).T

out = torch.ops.my_matmul.matmul_tmem(A, B)
out_ref = A @ B
torch.testing.assert_close(out, out_ref)
