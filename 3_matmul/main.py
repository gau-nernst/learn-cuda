import time

import matmul_triton
import torch
import torch.utils.cpp_extension


module = torch.utils.cpp_extension.load(
    "module",
    sources=["matmul.cu", "matmul.cpp"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "--ptxas-options=-v"],
    verbose=True,
)

# for large n, there will be a larger deviation, since sum of many small elements are not accurate
input1 = torch.randn(4096, 4096, device="cuda")
input2 = torch.randn(4096, 4096, device="cuda")

output_v1 = module.matmul_v1(input1, input2)
output_v2 = module.matmul_v2(input1, input2)
output_v3 = module.matmul_v3(input1, input2)
output_triton = matmul_triton.matmul(input1, input2)
output_triton_ref = matmul_triton.matmul_ref(input1, input2)

output_ref = torch.matmul(input1, input2)
torch.testing.assert_close(output_v1, output_ref)
torch.testing.assert_close(output_v2, output_ref)
torch.testing.assert_close(output_v3, output_ref)
torch.testing.assert_close(output_triton, output_ref)
torch.testing.assert_close(output_triton_ref, output_ref)


def benchmark(fn, *args):
    N = 100

    torch.cuda.synchronize()
    time0 = time.time()
    for _ in range(N):
        fn(*args)
        torch.cuda.synchronize()

    print(N / (time.time() - time0))


benchmark(torch.matmul, input1, input2)
benchmark(module.matmul_v1, input1, input2)
benchmark(module.matmul_v2, input1, input2)
benchmark(module.matmul_v3, input1, input2)
benchmark(matmul_triton.matmul, input1, input2)
benchmark(matmul_triton.matmul_ref, input1, input2)
