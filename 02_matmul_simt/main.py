import matmul_triton
import torch
import torch.utils.cpp_extension
from triton.testing import do_bench


def benchmark(f, *args, **kwargs):
    return do_bench(lambda: f(*args, **kwargs), return_mode="median")


module = torch.utils.cpp_extension.load(
    "module",
    sources=["matmul.cu", "matmul.cpp"],
    extra_cuda_cflags=[
        "-O3",
        "-lineinfo",
        "-Xptxas=-v",
    ],
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

print("CuBLAS:", benchmark(torch.matmul, input1, input2))
print("triton ref:", benchmark(matmul_triton.matmul_ref, input1, input2))
print("v1:", benchmark(module.matmul_v1, input1, input2))
print("v2:", benchmark(module.matmul_v2, input1, input2))
print("v3:", benchmark(module.matmul_v3, input1, input2))
print("v4:", benchmark(module.matmul_v4, input1, input2))
print("v5:", benchmark(module.matmul_v5, input1, input2))
print("v6a:", benchmark(module.matmul_v6a, input1, input2))
print("v6b:", benchmark(module.matmul_v6b, input1, input2))
print("triton:", benchmark(matmul_triton.matmul, input1, input2))
