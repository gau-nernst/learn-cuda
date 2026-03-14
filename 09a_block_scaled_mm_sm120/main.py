import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.cpp_extension
from torch import Tensor
from torch._inductor.utils import do_bench_using_profiling
from triton.testing import do_bench

torch.utils.cpp_extension.include_paths("cuda")
CURRENT_DIR = Path(__file__).parent

torch.utils.cpp_extension.load(
    "module",
    sources=list(CURRENT_DIR.glob("mxfp8*")),
    extra_cuda_cflags=[
        "-O3",
        "-Xptxas=-v",
        "-lineinfo",
        "-gencode=arch=compute_120a,code=sm_120a",
    ],
    extra_ldflags=["-lcuda"],  # for cuTensorMapEncodeTiled() used by TMA
    is_python_module=False,
    verbose=True,
)
module = torch.ops.my_module


def cublas_mxfp8_mm(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor):
    return F.scaled_mm(
        A,
        B,
        scale_a=scale_A,
        scale_b=scale_B,
        scale_recipe_a=F.ScalingType.BlockWise1x32,
        scale_recipe_b=F.ScalingType.BlockWise1x32,
        swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
        output_dtype=torch.bfloat16,
    )


def permute_sf_cublas(scale: Tensor):
    M, N = scale.shape
    scale = scale.view(M // 128, 4, 32, N // 4, 4).permute(0, 3, 2, 1, 4)  # [M/128,N/4,32,4,4]
    return scale.contiguous()


def permute_sf_v2(scale: Tensor):
    M, N = scale.shape
    scale = scale.view(M // 32, 4, 8, N // 4, 4).permute(0, 2, 3, 1, 4)  # [M/32,8,N/4,4,4]
    return scale.contiguous()


def ref_scaled_mm(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor):
    A_f32 = (A.float().unflatten(1, (-1, 32)) * scale_A.float().unsqueeze(2)).flatten(1)
    B_f32 = (B.T.float().unflatten(1, (-1, 32)) * scale_B.float().unsqueeze(2)).flatten(1)
    return (A_f32 @ B_f32.T).bfloat16()


def main(args: argparse.Namespace):
    M, N, K = args.shape
    print(f"{M=}, {N=}, {K=}")

    def generate_tensor(M: int, N: int):
        if args.format == "mxfp8":
            data_lp = torch.randint(-128, 127, size=(M, N), dtype=torch.int8)
            data_lp[(data_lp == -1) | (data_lp == 127)] = 0  # skip NaN
            data_lp = data_lp.view(torch.float8_e4m3fn)
            scale = torch.randn(M, N // 32).div(448).to(torch.float8_e8m0fnu)

        elif args.format == "nvfp4":
            data_lp = torch.randint(-128, 127, size=(M, N // 2), dtype=torch.int8).view(torch.float4_e2m1fn_x2)
            scale = torch.randn(M, N // 16).div(6).clip(-448, 448).to(torch.float8_e4m3fn)

        else:
            raise ValueError(f"Unsupported {args.format=}")

        return data_lp.cuda(), scale.cuda()

    A, SFA = generate_tensor(M, K)
    B, SFB = generate_tensor(N, K)
    print(A.shape, SFA.shape, B.shape, SFB.shape)

    if args.profile is not None:
        if args.profile == "cublas":
            fn = cublas_mxfp8_mm
        else:
            fn = getattr(module, f"mxfp8_mm_v{args.profile}")
        fn(A, B.T, SFA, SFB)
        torch.cuda.synchronize()
        return

    SOL_LOOKUP = {
        "NVIDIA GeForce RTX 5090": 838,
    }
    sol = SOL_LOOKUP.get(torch.cuda.get_device_name(), 1)
    if args.format == "nvfp4":
        sol *= 2

    def bench_and_print(f, name):
        time.sleep(1)  # stabilize thermal
        # latency_ms = do_bench(lambda: f(A, B.T, scale_A, scale_B), return_mode="median")
        latency_ms = do_bench_using_profiling(lambda: f(A, B.T, SFA, SFB))
        tflops = 2 * M * N * K / latency_ms / 1e9
        pct_sol = tflops / sol * 100
        print(f"{name}:\t{latency_ms:.4f} ms\t{tflops:.2f} TFLOPS\t{pct_sol:.2f}% SOL")

    output_ref = ref_scaled_mm(A, B.T, SFA, SFB)

    SFA_cublas = permute_sf_cublas(SFA)
    SFB_cublas = permute_sf_cublas(SFB)
    output = cublas_mxfp8_mm(A, B.T, SFA_cublas, SFB_cublas)
    torch.testing.assert_close(output, output_ref, rtol=1e-2, atol=1e-4)
    bench_and_print(cublas_mxfp8_mm, "CuBLAS")

    permute_fn_map = {
        "1": lambda x: x,
        "2": permute_sf_v2,
        "2b": permute_sf_v2,
        # "3": permute_sf_cublas,
    }

    for name, permute_fn in permute_fn_map.items():
        fn = getattr(module, f"mxfp8_mm_v{name}")
        output = fn(A, B.T, permute_fn(SFA), permute_fn(SFB))
        torch.testing.assert_close(output, output_ref, rtol=1e-2, atol=1e-4)  # is the tolerance too loose?
        bench_and_print(fn, f"v{name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument("--format", choices=["mxfp8", "nvfp4"], default="mxfp8")
    parser.add_argument("--shape", type=int, nargs="+", default=[4096, 4096, 4096])
    args = parser.parse_args()

    main(args)
