import argparse
import time
from pathlib import Path

import torch
from triton.testing import do_bench

CURRENT_DIR = Path(__file__).parent


def get_module(path: Path):
    import torch.utils.cpp_extension

    return torch.utils.cpp_extension.load(
        "module",
        sources=list(path.glob("matmul*")),
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "-Xptxas=-v",
        ],
        verbose=True,
    )


def bench_and_print(f, name: str, M: int, N: int, K: int, sol: float):
    # sleep to stabilize thermal
    time.sleep(1)

    latency_ms = do_bench(f, return_mode="median")
    tflops = 2 * M * N * K / latency_ms / 1e9
    pct_sol = tflops / sol * 100
    print(f"{name}:\t{latency_ms:.4f} ms\t{tflops:.2f} TFLOPS\t{pct_sol:.2f}% SOL")


def main(args: argparse.Namespace, src_path: Path = CURRENT_DIR):
    module = get_module(src_path)
    gpu_name = torch.cuda.get_device_name()

    print(f"{torch.__version__=}")
    print(f"{torch.version.cuda=}")
    print(gpu_name)
    print()

    torch._inductor.config.max_autotune_gemm_backends = "TRITON"
    # torch._inductor.utils.is_big_gpu = lambda _: True
    inductor_mm = torch.compile(torch.mm, mode="max-autotune-no-cudagraphs", dynamic=False)

    if args.profile is not None:
        M, N, K = args.shape
        print(f"{M=}, {N=}, {K=}")
        A = torch.randn(M, K, device="cuda").div(1024).bfloat16()
        B = torch.randn(N, K, device="cuda").div(1024).bfloat16().T

        if args.profile == "cublas":
            fn = torch.mm
        elif args.profile == "inductor":
            fn = inductor_mm
        else:
            fn = getattr(module, f"matmul_v{args.profile}")
        fn(A, B)
        torch.cuda.synchronize()
        return

    if "5090" in gpu_name:
        sol = 209.5
    elif "A100" in gpu_name:
        sol = 312
    else:
        sol = 1e9

    if args.sweep is not None:
        shapes = [(s, s, s) for s in args.sweep]
    else:
        assert len(args.shape) == 3
        shapes = [args.shape]

    for M, N, K in shapes:
        print(f"{M=}, {N=}, {K=}")
        A = torch.randn(M, K, device="cuda").div(1024).bfloat16()
        B = torch.randn(N, K, device="cuda").div(1024).bfloat16().T

        # reference in FP32 to avoid things like split-K
        output_ref = torch.matmul(A.float(), B.float()).bfloat16()

        bench_and_print(lambda: torch.matmul(A, B), "CuBLAS", M, N, K, sol)
        bench_and_print(lambda: inductor_mm(A, B), "Inductor Triton", M, N, K, sol)

        for i in range(7):
            fn = getattr(module, f"matmul_v{i + 1}")
            output = fn(A, B)
            torch.testing.assert_close(output, output_ref)
            bench_and_print(lambda: fn(A, B), f"v{i + 1}", M, N, K, sol)

        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument("--shape", type=int, nargs="+", default=[4096, 4096, 4096])
    parser.add_argument("--sweep", type=int, nargs="+")
    parser.add_argument("--modal")
    args = parser.parse_args()

    # run locally
    if args.modal is None:
        main(args)

    # run on modal
    else:
        import modal

        REMOTE_DIR = Path("/my_extension")

        image = (
            modal.Image.from_registry("nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04", add_python="3.12")
            .entrypoint([])  # remove verbose logging by base image on entry
            .uv_pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu130")
            .uv_pip_install("ninja")
            .add_local_dir(CURRENT_DIR, remote_path=REMOTE_DIR)
        )
        app = modal.App("sm80-matmul", image=image)
        modal_main = app.function(image=image, gpu=args.modal, serialized=True)(main)

        with modal.enable_output(), app.run():
            modal_main.remote(args, REMOTE_DIR)
