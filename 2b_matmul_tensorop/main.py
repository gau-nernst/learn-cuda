import argparse
import time

import torch
import torch.utils.cpp_extension
from triton.testing import do_bench


module = torch.utils.cpp_extension.load(
    "module",
    sources=["matmul.cu", "matmul.cpp"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "--ptxas-options=-v",
        "--generate-line-info",
    ],
    verbose=True,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    args = parser.parse_args()

    torch._inductor.config.max_autotune_gemm_backends = "TRITON"
    # torch._inductor.utils.is_big_gpu = lambda _: True

    # for large K, there will be a larger deviation, since sum of many small elements are not accurate
    M, N, K = 4096, 4096, 4096
    A = torch.randn(M, K).bfloat16().cuda()
    B = torch.randn(N, K).bfloat16().cuda().T
    inductor_mm = torch.compile(torch.matmul, mode="max-autotune-no-cudagraphs", dynamic=False)

    if args.profile is not None:
        if args.profile == "inductor":
            fn = inductor_mm
        else:
            fn = getattr(module, f"matmul_v{args.profile}")
        fn(A, B)
        torch.cuda.synchronize()
        return

    def bench_and_print(f, name):
        latency_ms = do_bench(lambda: f(A, B), return_mode="median")
        tflops = 2 * M * N * K / latency_ms / 1e9
        print(f"{name}:\t{latency_ms:.4f} ms\t{tflops:.2f} TFLOPS")

    output_ref = torch.matmul(A, B)
    bench_and_print(torch.matmul, "CuBLAS")
    bench_and_print(inductor_mm, "Inductor Triton")

    for i in range(5):
        fn = getattr(module, f"matmul_v{i + 1}")
        output = fn(A, B)
        torch.testing.assert_close(output, output_ref)
        bench_and_print(fn, f"v{i + 1}")

        # sleep to stabilize thermal
        time.sleep(1)


if __name__ == "__main__":
    main()
