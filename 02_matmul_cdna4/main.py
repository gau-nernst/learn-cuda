import os
from pathlib import Path

CURRENT_DIR = Path(__file__).parent

os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(CURRENT_DIR / ".flydsl/cache")
os.environ["FLYDSL_DUMP_DIR"] = str(CURRENT_DIR / ".flydsl/debug")
# os.environ["FLYDSL_DUMP_IR"] = "1"

import argparse

import torch
from matmul_v0 import matmul_v0
from matmul_v1 import matmul_v1


def main(args: argparse.Namespace):
    if args.profile is not None:
        size = 8192

        scale = size**-0.5
        A = torch.randn(size, size, device="cuda").mul(scale).bfloat16()
        B = torch.randn(size, size, device="cuda").mul(scale).bfloat16().T

        f = {
            "1": matmul_v1,
        }[args.profile]

        f(A, B)
        return

    # rocprof will crash if we import this under profiling context
    from triton.testing import do_bench

    for size in (4096, 8192, 16384):
        print(f"M=N=K={size}")

        scale = size**-0.5
        A = torch.randn(size, size, device="cuda").mul(scale).bfloat16()
        B = torch.randn(size, size, device="cuda").mul(scale).bfloat16().T

        out_ref = A @ B

        def benchmark(f, name):
            torch.zeros(size, size, device="cuda", dtype=torch.bfloat16)
            out = f(A, B)
            torch.accelerator.synchronize()
            torch.testing.assert_close(out, out_ref)

            latency_us = do_bench(lambda: f(A, B)) * 1e3
            tflops = 2 * size * size * size / (latency_us * 1e-6) * 1e-12
            print(f"  {name}: {latency_us:.2f} us, {tflops:.2f} TFLOPS")

        benchmark(torch.mm, "PyTorch")
        benchmark(matmul_v0, "v0")
        benchmark(matmul_v1, "v1")

        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    args = parser.parse_args()

    main(args)
