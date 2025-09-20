import argparse
import os
import time
from pathlib import Path

os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"

import torch
import torch.utils.cpp_extension
from torch import Tensor
from triton.testing import do_bench

CURRENT_DIR = Path(__file__).parent

module = torch.utils.cpp_extension.load(
    "module",
    sources=[
        "ext.cpp",
        *list(CURRENT_DIR.glob("row_scaled_mm*")),
    ],
    extra_cuda_cflags=[
        "--ptxas-options=-v",
        "--generate-line-info",
    ],
    # extra_ldflags=["-lcuda"],  # for cuTensorMapEncodeTiled() used by TMA
    verbose=True,
)


@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
def pytorch_scaled_mm(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor):
    if A.dtype == torch.int8:
        # this is not being fused for some reason
        return (torch._int_mm(A, B).float() * scale_A * scale_B).bfloat16()
    else:
        return torch._scaled_mm(A, B, scale_A, scale_B, out_dtype=torch.bfloat16)


def ref_scaled_mm(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor):
    return (A.float() @ B.float() * scale_A * scale_B).bfloat16()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument("--dtype", choices=["int8", "fp8"], default="int8")
    parser.add_argument("--shape", type=int, nargs="+", default=[4096, 4096, 8192])
    args = parser.parse_args()

    M, N, K = args.shape
    print(f"{M=}, {N=}, {K=}")

    def generate_data(M: int, N: int):
        if args.dtype == "int8":
            dtype = torch.int8
            min_val = torch.iinfo(dtype).min
            max_val = torch.iinfo(dtype).max
        elif args.dtype == "fp8":
            dtype = torch.float8_e4m3fn
            min_val = torch.finfo(dtype).min
            max_val = torch.finfo(dtype).max

        x = torch.randn(M, N) * (K**-0.5)
        scale = x.amax(dim=1) / ((max_val - min_val) * 0.5)
        xq = (x / scale.unsqueeze(1).clip(1e-6)).clip(min_val, max_val).to(dtype)
        return xq.cuda(), scale.cuda()

    A, scale_A = generate_data(M, K)
    B, scale_B = generate_data(N, K)
    scale_A = scale_A.reshape(M, 1)
    scale_B = scale_B.reshape(1, N)

    if args.profile is not None:
        if args.profile == "pt":
            fn = pytorch_scaled_mm
        else:
            fn = getattr(module, f"row_scaled_mm_v{args.profile}")
        fn(A, B.T, scale_A, scale_B)
        torch.cuda.synchronize()
        return

    SOL_LOOKUP = {
        "NVIDIA GeForce RTX 5090": 419,
    }
    sol = SOL_LOOKUP.get(torch.cuda.get_device_name(), 1)
    if args.dtype == "int8":
        sol *= 2

    out_ref = ref_scaled_mm(A, B.T, scale_A, scale_B)

    def bench_and_print(f, name):
        out = f(A, B.T, scale_A, scale_B)
        torch.testing.assert_close(out, out_ref)

        time.sleep(1)  # stabilize thermal
        latency_ms = do_bench(lambda: f(A, B.T, scale_A, scale_B), return_mode="median")
        tflops = 2 * M * N * K / latency_ms / 1e9
        pct_sol = tflops / sol * 100
        print(f"{name}:\t{latency_ms:.4f} ms\t{tflops:.2f} TFLOPS\t{pct_sol:.2f}% SOL")

    bench_and_print(pytorch_scaled_mm, "PyTorch")

    for i in range(1):
        fn = getattr(module, f"row_scaled_mm_v{i + 1}")
        bench_and_print(fn, f"v{i + 1}     ")


if __name__ == "__main__":
    main()
