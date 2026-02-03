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
        "-O3",
        "-lineinfo",
        "-Xptxas=-v",
        "-gencode=arch=compute_120a,code=sm_120a",
    ],
    extra_ldflags=["-lcuda"],  # for cuTensorMapEncodeTiled() used by TMA
    verbose=True,
)

SOL_LOOKUP = {
    "NVIDIA GeForce RTX 5090": 209.5,
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": 503.8,
}
SOL = SOL_LOOKUP.get(torch.cuda.get_device_name(), 1e9)


def bench_and_print(f, name: str, M: int, N: int, K: int):
    # sleep to stabilize thermal
    time.sleep(1)

    latency_ms = do_bench(f, return_mode="median")
    tflops = 2 * M * N * K / latency_ms / 1e9
    pct_sol = tflops / SOL * 100
    print(f"{name}:\t{latency_ms:.4f} ms\t{tflops:.2f} TFLOPS\t{pct_sol:.2f}% SOL")


def main(args: argparse.Namespace):
    torch._inductor.config.max_autotune_gemm_backends = "TRITON"
    # torch._inductor.utils.is_big_gpu = lambda _: True
    inductor_mm = torch.compile(torch.mm, mode="max-autotune-no-cudagraphs", dynamic=False)

    if args.profile is not None:
        M, N, K = args.shape
        print(f"{M=}, {N=}, {K=}")
        A = torch.randn(M, K, device="cuda").div(1024).bfloat16().cuda()
        B = torch.randn(N, K, device="cuda").div(1024).bfloat16().cuda().T

        if args.profile == "cublas":
            fn = torch.mm
        elif args.profile == "inductor":
            fn = inductor_mm
        else:
            fn = getattr(module, f"matmul_v{args.profile}")
        fn(A, B)
        torch.cuda.synchronize()
        return

    if args.sweep is not None:
        shapes = [(s, s, s) for s in args.sweep]
    else:
        assert len(args.shape) == 3
        shapes = [args.shape]

    for M, N, K in shapes:
        print(f"{M=}, {N=}, {K=}")
        A = torch.randn(M, K, device="cuda").div(1024).bfloat16().cuda()
        B = torch.randn(N, K, device="cuda").div(1024).bfloat16().cuda().T

        # reference in FP32 to avoid things like split-K
        output_ref = torch.matmul(A.float(), B.float()).bfloat16()

        bench_and_print(lambda: torch.matmul(A, B), "CuBLAS", M, N, K)
        bench_and_print(lambda: inductor_mm(A, B), "Inductor Triton", M, N, K)

        for i in range(3):
            fn = getattr(module, f"matmul_v{i}")
            output = fn(A, B)
            torch.testing.assert_close(output, output_ref)
            bench_and_print(lambda: fn(A, B), f"v{i}", M, N, K)

        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument("--shape", type=int, nargs="+", default=[4096, 4096, 4096])
    parser.add_argument("--sweep", type=int, nargs="+")
    args = parser.parse_args()

    main(args)
