import time
from functools import partial

import matmul_triton
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
input1 = torch.randn(4096, 4096).cuda()
input2 = torch.randn(4096, 4096).cuda()

output_v1 = module.matmul_v1(input1, input2)
output_v2 = module.matmul_v2(input1, input2)
output_v3 = module.matmul_v3(input1, input2)
output_v4 = module.matmul_v4(input1, input2)
output_v5 = module.matmul_v5(input1, input2)
output_v6a = module.matmul_v6a(input1, input2)
output_v6b = module.matmul_v6b(input1, input2)
output_triton = matmul_triton.matmul(input1, input2)
output_triton_ref = matmul_triton.matmul_ref(input1, input2)

output_ref = torch.matmul(input1, input2)
torch.testing.assert_close(output_v1, output_ref)
torch.testing.assert_close(output_v2, output_ref)
torch.testing.assert_close(output_v3, output_ref)
torch.testing.assert_close(output_v4, output_ref)
torch.testing.assert_close(output_v5, output_ref)
torch.testing.assert_close(output_v6a, output_ref)
torch.testing.assert_close(output_v6b, output_ref)
torch.testing.assert_close(output_triton, output_ref)
torch.testing.assert_close(output_triton_ref, output_ref)

print("CuBLAS:", bench_f(torch.matmul, input1, input2))
print("triton ref:", bench_f(matmul_triton.matmul_ref, input1, input2))
print("v1:", bench_f(module.matmul_v1, input1, input2))
print("v2:", bench_f(module.matmul_v2, input1, input2))
print("v3:", bench_f(module.matmul_v3, input1, input2))
print("v4:", bench_f(module.matmul_v4, input1, input2))
print("v5:", bench_f(module.matmul_v5, input1, input2))
print("v6a:", bench_f(module.matmul_v6a, input1, input2))
print("v6b:", bench_f(module.matmul_v6b, input1, input2))
print("triton:", bench_f(matmul_triton.matmul, input1, input2))
