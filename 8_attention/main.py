import torch
from torch.utils.cpp_extension import load
from pathlib import Path

CURRENT_DIR = Path(__file__).parent

module = load(
    "my_ext",
    sources=[
        str(CURRENT_DIR / "attention.cu"),
        str(CURRENT_DIR / "attention.cpp"),
    ],
    extra_cuda_cflags=["-lineinfo", "--ptxas-options=-v"],
    verbose=True,
)

Q = torch.randn(1, 128, 128, dtype=torch.bfloat16, device="cuda")
K = torch.randn(1, 64, 128, dtype=torch.bfloat16, device="cuda")
V = torch.randn(1, 64, 128, dtype=torch.bfloat16, device="cuda")

out = module.sdpa(Q, K, V)
print(out.round(decimals=4))

out_ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
print(out_ref.round(decimals=4))
