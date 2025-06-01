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

# for large K, there will be a larger deviation, since sum of many small elements are not accurate
M, N, K = 4096, 4096, 4096
input1 = torch.randn(M, K).bfloat16().cuda()
input2 = torch.randn(N, K).bfloat16().cuda().T

output_ref = torch.matmul(input1, input2)
output_v1 = module.matmul_v1(input1, input2)
output_v2 = module.matmul_v2(input1, input2)

torch.testing.assert_close(output_v1, output_ref)
torch.testing.assert_close(output_v2, output_ref)


def bench_and_print(f, name):
    latency_ms = benchmark(f, input1, input2)
    tflops = 2 * M * N * K / latency_ms / 1e9
    print(f"{name}:\t{latency_ms:.4f} ms\t{tflops:.2f} TFLOPS")


bench_and_print(torch.matmul, "CuBLAS")
bench_and_print(module.matmul_v1, "v1")
bench_and_print(module.matmul_v2, "v2")
