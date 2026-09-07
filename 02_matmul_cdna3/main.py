import argparse
import os
import time
from pathlib import Path

os.environ["PYTORCH_ROCM_ARCH"] = "gfx942"

import torch
from torch.utils.cpp_extension import load
from triton.testing import do_bench

CURRENT_DIR = Path(__file__).parent
load(
    "hip_matmul",
    [str(x) for x in CURRENT_DIR.glob("*.cu")],
    extra_include_paths=[str(CURRENT_DIR)],
    extra_cuda_cflags=[
        "-O3",
    ],
    is_python_module=False,
    verbose=True,
)
ops = torch.ops.hip_matmul


def main(args):
    M, N, K = 4096, 4096, 4096
    dtype = torch.bfloat16

    A = torch.randn(M, K, dtype=dtype).cuda()
    B = torch.randn(N, K, dtype=dtype).cuda().T

    if args.profile is not None:
        if args.profile == "pt":
            f = torch.mm
        else:
            f = getattr(ops, f"matmul_{args.profile}")

        f(A, B)
        return

    ref = torch.mm(A, B)

    def bench(name, f):
        out = f(A, B)
        # torch.testing.assert_close(out, ref)
        diff = (out - ref).abs().mean().item()

        time.sleep(0.5)
        latency_ms = do_bench(lambda: f(A, B), warmup=100, rep=500)
        tflops = 2 * M * N * K / (latency_ms * 1e-3) * 1e-12
        print(f"{name}: {tflops:.2f} TFLOPS, {diff}")

    bench("PyTorch", torch.mm)
    bench("v1a", ops.matmul_v1a)
    bench("v1b", ops.matmul_v1b)
    bench("v2", ops.matmul_v2)
    bench("v3a", ops.matmul_v3a)
    bench("v3b", ops.matmul_v3b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    args = parser.parse_args()

    main(args)
