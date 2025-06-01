import argparse

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    args = parser.parse_args()

    # for large K, there will be a larger deviation, since sum of many small elements are not accurate
    M, N, K = 4096, 4096, 4096
    A = torch.randn(M, K).bfloat16().cuda()
    B = torch.randn(N, K).bfloat16().cuda().T

    if args.profile is not None:
        fn = getattr(module, f"matmul_v{args.profile}")
        fn(A, B)
        torch.cuda.synchronize()
        return

    output_ref = torch.matmul(A, B)
    output_v1 = module.matmul_v1(A, B)
    output_v2 = module.matmul_v2(A, B)
    output_v3 = module.matmul_v3(A, B)
    output_v4 = module.matmul_v4(A, B)

    torch.testing.assert_close(output_v1, output_ref)
    torch.testing.assert_close(output_v2, output_ref)
    torch.testing.assert_close(output_v3, output_ref)
    torch.testing.assert_close(output_v4, output_ref)

    def bench_and_print(f, name):
        latency_ms = benchmark(f, A, B)
        tflops = 2 * M * N * K / latency_ms / 1e9
        print(f"{name}:\t{latency_ms:.4f} ms\t{tflops:.2f} TFLOPS")

    bench_and_print(torch.matmul, "CuBLAS")
    bench_and_print(module.matmul_v1, "v1")
    bench_and_print(module.matmul_v2, "v2")
    bench_and_print(module.matmul_v3, "v3")
    bench_and_print(module.matmul_v4, "v4")


if __name__ == "__main__":
    main()
