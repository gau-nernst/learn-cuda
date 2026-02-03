import argparse
import multiprocessing as mp
from pathlib import Path

import cuda.bench
import pandas as pd
import torch
import torch._inductor.config
import torch.utils.cpp_extension

CURRENT_DIR = Path(__file__).parent
REMOTE_DIR = Path("/my_extension")  # for Modal only


def get_module(path):
    return torch.utils.cpp_extension.load(
        "module",
        sources=list(Path(path).glob("matmul*")),
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "-Xptxas=-v",
        ],
        verbose=True,
    )


def get_sol():
    gpu_name = torch.cuda.get_device_name()
    if "5090" in gpu_name:
        sol = 209.5
    elif "A100" in gpu_name:
        sol = 312
    else:
        sol = 1e9
    return sol


def get_kernel(name: str, source_dir: str):
    if name == "cublas":
        f = torch.mm
    elif name == "inductor":
        torch._inductor.config.max_autotune_gemm_backends = "TRITON"
        f = torch.compile(torch.mm, mode="max-autotune-no-cudagraphs", dynamic=False)
    else:
        module = get_module(source_dir)
        f = getattr(module, name)
    return f


def to_torch_stream(s: cuda.bench.CudaStream, device: int | None):
    return torch.cuda.ExternalStream(stream_ptr=s.addressof(), device=device)


def torch_bench(state: cuda.bench.State) -> None:
    # state.set_throttle_threshold(0.25)
    device = state.get_device()

    # select kernel
    f = get_kernel(state.get_string("kernel"), state.get_string("source_dir"))

    # problem shape
    M = state.get_int64("M")
    N = state.get_int64("N")
    K = state.get_int64("K")

    stream = to_torch_stream(state.get_stream(), device)
    with torch.cuda.stream(stream):
        A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        B = torch.randn(N, K, dtype=torch.bfloat16, device=device).T
        f(A, B)  # trigger torch.compile

    def launcher(launch: cuda.bench.Launch) -> None:
        stream = to_torch_stream(launch.get_stream(), device)
        with torch.cuda.stream(stream):
            f(A, B)

    state.exec(launcher, sync=True)


def run_nvbench(M: int, N: int, K: int, source_dir: str):
    kernels_list = []
    kernels_list += ["cublas", "inductor"]
    kernels_list += [f"matmul_v{i}" for i in range(1, 9)]

    bench = cuda.bench.register(torch_bench)
    bench.add_string_axis("source_dir", [source_dir])
    bench.add_string_axis("kernel", kernels_list)
    bench.add_int64_axis("M", [M])
    bench.add_int64_axis("N", [N])
    bench.add_int64_axis("K", [K])

    result_path = "/tmp/result.csv"
    cuda.bench.run_all_benchmarks(["--csv", result_path])

    df = pd.read_csv(result_path)
    df["latency (us)"] = df["GPU Time (sec)"] * 1e6
    df["TFLOPS"] = 2 * M * N * K / df["GPU Time (sec)"] * 1e-12
    df["% SOL"] = df["TFLOPS"] / get_sol()

    # apply formatting
    df["latency (us)"] = df["latency (us)"].map("{:.2f}".format)
    df["TFLOPS"] = df["TFLOPS"].map("{:.2f}".format)
    df["% SOL"] = df["% SOL"].map("{:.2%}".format)
    df["Noise"] = df["Noise"].map("{:.2%}".format)

    print()
    print(df[["kernel", "latency (us)", "Noise", "TFLOPS", "% SOL"]].to_markdown(index=False))


def main(args: argparse.Namespace):
    source_dir = str(REMOTE_DIR if args.modal else CURRENT_DIR)

    if args.profile is not None:
        M, N, K = args.shape
        scale = K**-0.5  # make sure output doesn't explode
        A = torch.randn(M, K, device="cuda").mul(scale).bfloat16()
        B = torch.randn(N, K, device="cuda").mul(scale).bfloat16().T

        f = get_kernel(args.profile, source_dir)
        f(A, B)
        return

    if args.sweep is not None:
        shapes = [(s, s, s) for s in args.sweep]
    else:
        assert len(args.shape) == 3
        shapes = [args.shape]

    mp_context = mp.get_context("spawn")

    for M, N, K in shapes:
        # correctness check
        scale = K**-0.5  # make sure output doesn't explode
        A = torch.randn(M, K, device="cuda").mul(scale).bfloat16()
        B = torch.randn(N, K, device="cuda").mul(scale).bfloat16().T

        # compute in FP32 to avoid split-K
        out_ref = torch.mm(A.float(), B.float()).bfloat16()

        module = get_module(source_dir)
        for i in range(1, 9):
            print(f"correctness check for matmul_v{i}")
            out = getattr(module, f"matmul_v{i}")(A, B)
            torch.testing.assert_close(out, out_ref)

        # nvbench can only be run once per process. hence, we spawn subprocess for each shape.
        proc = mp_context.Process(target=run_nvbench, args=(M, N, K, source_dir))
        proc.start()
        proc.join()
        if proc.exitcode != 0:
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument("--shape", type=int, nargs="+", default=[4096, 4096, 4096])
    parser.add_argument("--sweep", type=int, nargs="+")
    parser.add_argument("--modal")
    args = parser.parse_args()

    # local
    if args.modal is None:
        main(args)

    # modal
    else:
        import modal

        image = (
            modal.Image.from_registry("nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04", add_python="3.12")
            .entrypoint([])  # remove verbose logging by base image on entry
            .uv_pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu130")
            .uv_pip_install("ninja", "pandas", "tabulate", "cuda-bench[cu13]")
            .add_local_dir(CURRENT_DIR, remote_path=REMOTE_DIR)
        )
        app = modal.App("sm80-matmul", image=image)
        modal_main = app.function(image=image, gpu=args.modal)(main)

        with modal.enable_output(), app.run():
            modal_main.remote(args)
