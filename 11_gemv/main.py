import argparse
import time

import torch
from triton.testing import do_bench
from triton_v1 import triton_v1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument("--shape", type=int, nargs="+", default=[4096, 4096])
    args = parser.parse_args()

    M = 1
    N, K = args.shape
    A = torch.randn(M, K).bfloat16().cuda()
    B = torch.randn(N, K).bfloat16().cuda().T
    inductor_mm = torch.compile(torch.mm, mode="max-autotune-no-cudagraphs", dynamic=False)

    SOL_LOOKUP = {
        "NVIDIA GeForce RTX 5090": 1792,
    }
    sol = SOL_LOOKUP.get(torch.cuda.get_device_name(), float("inf"))

    def bench_and_print(f, name):
        time.sleep(1)  # sleep to stabilize thermal
        latency_ms = do_bench(lambda: f(A, B), return_mode="median")
        achieved_bw = 2 * (M * K + N * K + M * N) / 1e9 / (latency_ms / 1e3)
        pct_sol = achieved_bw / sol * 100
        print(f"{name}:\t{latency_ms:.4f} ms\t{achieved_bw:.2f} GB/s\t{pct_sol:.2f}% SOL")

    out_ref = torch.mm(A.float(), B.float()).bfloat16()
    bench_and_print(torch.mm, "CuBLAS")
    bench_and_print(inductor_mm, "Inductor")

    torch.testing.assert_close(triton_v1(A, B), out_ref)
    bench_and_print(triton_v1, "triton_v1")


if __name__ == "__main__":
    main()
