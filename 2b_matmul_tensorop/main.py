import argparse
import time
from pathlib import Path

import torch
import torch.utils.cpp_extension
from triton.testing import do_bench

CURRENT_DIR = Path(__file__).parent

module = torch.utils.cpp_extension.load(
    "module",
    sources=list(CURRENT_DIR.glob("matmul*")),
    extra_cuda_cflags=[
        "--ptxas-options=-v",
        "--generate-line-info",
    ],
    verbose=True,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument("--shape", type=int, nargs="+", default=[4096, 4096, 4096])
    args = parser.parse_args()

    torch._inductor.config.max_autotune_gemm_backends = "TRITON"
    # torch._inductor.utils.is_big_gpu = lambda _: True

    # for large K, there will be a larger deviation, since sum of many small elements are not accurate
    M, N, K = args.shape
    print(f"{M=}, {N=}, {K=}")
    A = torch.randn(M, K).bfloat16().cuda()
    B = torch.randn(N, K).bfloat16().cuda().T
    inductor_mm = torch.compile(
        torch.mm, mode="max-autotune-no-cudagraphs", dynamic=False
    )

    if args.profile is not None:
        if args.profile == "cublas":
            fn = torch.mm
        elif args.profile == "inductor":
            fn = inductor_mm
        else:
            fn = getattr(module, f"matmul_v{args.profile}")
        fn(A, B)
        torch.cuda.synchronize()
        return

    SOL_LOOKUP = {
        "NVIDIA GeForce RTX 5090": 209.5,
    }
    sol = SOL_LOOKUP.get(torch.cuda.get_device_name(), 0)

    def bench_and_print(f, name):
        # sleep to stabilize thermal
        time.sleep(1)

        latency_ms = do_bench(lambda: f(A, B), return_mode="median")
        tflops = 2 * M * N * K / latency_ms / 1e9
        pct_sol = tflops / sol * 100
        print(f"{name}:\t{latency_ms:.4f} ms\t{tflops:.2f} TFLOPS\t{pct_sol:.2f}% SOL")

    output_ref = torch.matmul(A, B)
    bench_and_print(torch.matmul, "CuBLAS")
    bench_and_print(inductor_mm, "Inductor Triton")

    for i in range(6):
        fn = getattr(module, f"matmul_v{i + 1}")
        output = fn(A, B)
        torch.testing.assert_close(output, output_ref)
        bench_and_print(fn, f"v{i + 1}")


if __name__ == "__main__":
    main()
