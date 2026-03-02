import argparse
import time
from pathlib import Path

import torch
import torch.utils.cpp_extension
from triton.testing import do_bench

CURRENT_DIR = Path(__file__).parent

torch.utils.cpp_extension.load(
    "my_module",
    sources=list(CURRENT_DIR.glob("matmul*")),
    extra_cuda_cflags=[
        "-O3",
        "-lineinfo",
        "-Xptxas=-v",
        "-gencode=arch=compute_120a,code=sm_120a",
    ],
    extra_ldflags=["-lcuda"],  # for cuTensorMapEncodeTiled() used by TMA
    is_python_module=False,
    verbose=True,
)
module = torch.ops.my_module

SOL_LOOKUP = {
    "NVIDIA GeForce RTX 5090": dict(bf16=209.5, int8=838),
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": dict(bf16=503.8, int8=1007.6),
}


def make_inputs(M: int, N: int, K: int, dtype: str):
    if dtype == "bf16":
        A = torch.randn(M, K, device="cuda").mul(K**-0.5).bfloat16()
        B = torch.randn(N, K, device="cuda").mul(K**-0.5).bfloat16().T
    elif dtype == "int8":
        A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="cuda")
        B = torch.randint(-128, 127, (N, K), dtype=torch.int8, device="cuda").T
    return A, B


def bench_and_print(f, name: str, M: int, N: int, K: int, sol: float):
    # sleep to stabilize thermal
    time.sleep(1)

    latency_ms = do_bench(f, return_mode="median")
    tflops = 2 * M * N * K / latency_ms / 1e9
    pct_sol = tflops / sol * 100
    print(f"{name}:\t{latency_ms:.4f} ms\t{tflops:.2f} TFLOPS\t{pct_sol:.2f}% SOL")


def main(args: argparse.Namespace):
    gpu_name = torch.cuda.get_device_name()
    if gpu_name in SOL_LOOKUP:
        sol = SOL_LOOKUP[gpu_name][args.dtype]
    else:
        sol = 1e9

    cublas_mm = dict(bf16=torch.mm, int8=torch._int_mm)[args.dtype]

    torch._inductor.config.max_autotune_gemm_backends = "TRITON"
    # torch._inductor.utils.is_big_gpu = lambda _: True
    inductor_mm = torch.compile(cublas_mm, mode="max-autotune-no-cudagraphs", dynamic=False)

    if args.profile is not None:
        M, N, K = args.shape
        print(f"{M=}, {N=}, {K=}")
        A, B = make_inputs(M, N, K, args.dtype)

        if args.profile == "cublas":
            fn = cublas_mm
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
        A, B = make_inputs(M, N, K, args.dtype)

        if args.dtype == "bf16":
            # reference in FP32 to avoid things like split-K
            output_ref = torch.matmul(A.float(), B.float()).bfloat16()
        else:
            output_ref = cublas_mm(A, B)

        bench_and_print(lambda: cublas_mm(A, B), "CuBLAS", M, N, K, sol)
        bench_and_print(lambda: inductor_mm(A, B), "Inductor Triton", M, N, K, sol)

        for i in range(3):
            fn = getattr(module, f"matmul_v{i}_{args.dtype}")
            output = fn(A, B)
            torch.testing.assert_close(output, output_ref)
            bench_and_print(lambda: fn(A, B), f"v{i}", M, N, K, sol)

        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument("--shape", type=int, nargs="+", default=[4096, 4096, 4096])
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--sweep", type=int, nargs="+")
    args = parser.parse_args()

    main(args)
