from pathlib import Path

import torch
import torch.utils.cpp_extension

CURRENT_DIR = Path(__file__).parent

torch.utils.cpp_extension.load(
    "my_module",
    sources=[str(CURRENT_DIR / "_matmul_v4_trans.cu")],
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
A = torch.randn(K, M, device="cuda", dtype=torch.bfloat16).T
B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

out = torch.ops.my_matmul.matmul_v4_trans(A, B)
out_ref = A @ B
torch.testing.assert_close(out, out_ref)
