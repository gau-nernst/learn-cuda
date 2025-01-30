import torch
import torch.utils.cpp_extension
from triton.testing import do_bench


def benchmark(f, *args, **kwargs):
    return do_bench(lambda: f(*args, **kwargs), return_mode="median")


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
output_v1a = module.matmul_v1a(input1, input2)
output_v1b = module.matmul_v1b(input1, input2)
output_v2 = module.matmul_v2(input1, input2)

torch.testing.assert_close(output_v1a, output_ref)
torch.testing.assert_close(output_v1b, output_ref)
torch.testing.assert_close(output_v2, output_ref)

print("CuBLAS:", benchmark(torch.matmul, input1, input2))
print("v1a:", benchmark(module.matmul_v1a, input1, input2))
print("v1b:", benchmark(module.matmul_v1b, input1, input2))
print("v2:", benchmark(module.matmul_v2, input1, input2))
