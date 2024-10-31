from functools import partial

import torch
import torch.utils.cpp_extension
from triton.testing import do_bench


def bench_f(f, *args, **kwargs):
    return do_bench(partial(f, *args, **kwargs), fast_flush=False, return_mode="median")


module = torch.utils.cpp_extension.load(
    "module",
    sources=["matmul.cu", "matmul.cpp"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "--ptxas-options=-v"],
    verbose=True,
)

# for large n, there will be a larger deviation, since sum of many small elements are not accurate
input1 = torch.randn(4096, 4096).bfloat16().cuda()
input2 = torch.randn(4096, 4096).bfloat16().cuda().T

output_ref = torch.matmul(input1, input2)
output_v1 = module.matmul_v1(input1, input2)
output_v2 = module.matmul_v2(input1, input2)
output_v3 = module.matmul_v3(input1, input2)

torch.testing.assert_close(output_v1, output_ref)
torch.testing.assert_close(output_v2, output_ref)
torch.testing.assert_close(output_v3, output_ref)

print("CuBLAS:", bench_f(torch.matmul, input1, input2))
print("v1:", bench_f(module.matmul_v1, input1, input2))
print("v2:", bench_f(module.matmul_v2, input1, input2))
print("v3:", bench_f(module.matmul_v3, input1, input2))
