import argparse
import importlib
import math
from typing import TYPE_CHECKING

import pandas as pd
import torch
import torch._inductor.config
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    import cuda.bench


def mlp_ref(x: Tensor, w1: Tensor, w3: Tensor, w2: Tensor):
    return (F.silu(x @ w1.T) * (x @ w3.T)) @ w2.T


# TFLOPS is from specsheet, membw is measured memcpy bw.
def get_sol():
    gpu_name = torch.cuda.get_device_name()
    if "5090" in gpu_name:
        sol = 209.5, 1500
    elif "A100" in gpu_name:
        sol = 312, 1700
    elif "H200" in gpu_name:
        sol = 1979, 4000
    else:
        sol = 1e9, 1e9
    return sol


def get_kernel(name: str):
    if name == "eager":
        f = mlp_ref
    elif name == "inductor":
        # torch._inductor.config.max_autotune_gemm_backends = "TRITON"
        f = torch.compile(mlp_ref, mode="max-autotune-no-cudagraphs", dynamic=False)
    else:
        fullname = f"mlp_{name}"
        f = getattr(importlib.import_module(fullname), fullname)
    return f


def to_torch_stream(s: "cuda.bench.CudaStream", device: int | None):
    return torch.cuda.ExternalStream(stream_ptr=s.addressof(), device=device)


def torch_bench(state: "cuda.bench.State") -> None:
    # state.set_throttle_threshold(0.25)
    device = state.get_device()

    # select kernel
    f = get_kernel(state.get_string("kernel"))

    # problem shape
    M = state.get_int64("M")
    N = state.get_int64("N")
    K = state.get_int64("K")

    stream = to_torch_stream(state.get_stream(), device)
    with torch.cuda.stream(stream):
        scale = K**-0.5  # make sure output doesn't explode
        X = torch.randn(M, K, device=device).mul(scale).bfloat16()
        W1 = torch.randn(N, K, device=device).mul(scale).bfloat16()
        W3 = torch.randn(N, K, device=device).mul(scale).bfloat16()
        W2 = torch.randn(K, N, device=device).mul(scale).bfloat16()

        # correctness check
        out_ref = mlp_ref(X, W1, W3, W2)
        out = f(X, W1, W3, W2)
        torch.testing.assert_close(out, out_ref)

        inputs_list = []
        for _ in range(state.get_int64("num_inputs")):
            X = torch.randn(M, K, device=device).mul(scale).bfloat16()
            W1 = torch.randn(N, K, device=device).mul(scale).bfloat16()
            W3 = torch.randn(N, K, device=device).mul(scale).bfloat16()
            W2 = torch.randn(K, N, device=device).mul(scale).bfloat16()
            inputs_list.append((X, W1, W3, W2))

    def launcher(launch: "cuda.bench.Launch") -> None:
        stream = to_torch_stream(launch.get_stream(), device)
        with torch.cuda.stream(stream):
            for X, W1, W3, W2 in inputs_list:
                f(X, W1, W3, W2)

    state.exec(launcher, sync=True)


def benchmark(shape: list[int]):
    import cuda.bench

    print(f"{torch.__version__=}")
    print(f"{torch.version.cuda=}")

    M, N, K = shape

    num_flops = 3 * 2 * M * N * K
    num_gb = 2 * (2 * M * K + 3 * N * K) * 1e-9

    # duplicate inputs to make sure each measurement is at least 10ms
    SOL_COMPUTE, SOL_MEMORY = get_sol()
    min_compute_latency_ms = num_flops / (SOL_COMPUTE * 1e12) * 1e3
    min_memory_latency_ms = num_gb / SOL_MEMORY * 1e3
    min_latency_ms = max(min_compute_latency_ms, min_memory_latency_ms)
    num_inputs = max(math.ceil(10 / min_latency_ms), 1000)

    kernels_list = []
    kernels_list += ["eager", "inductor"]
    kernels_list += ["triton_v1"]

    bench = cuda.bench.register(torch_bench)
    bench.add_string_axis("kernel", kernels_list)
    bench.add_int64_axis("M", [M])
    bench.add_int64_axis("N", [N])
    bench.add_int64_axis("K", [K])
    bench.add_int64_axis("num_inputs", [num_inputs])

    result_path = "/tmp/result.csv"
    cuda.bench.run_all_benchmarks(["--csv", result_path])

    df = pd.read_csv(result_path)
    df["GPU Time (sec)"] /= num_inputs  # rescale
    df["latency (us)"] = df["GPU Time (sec)"] * 1e6
    df["TFLOPS"] = num_flops / df["GPU Time (sec)"] * 1e-12
    df["membw (GB/s)"] = num_gb / df["GPU Time (sec)"]

    # apply formatting
    df["latency (us)"] = df["latency (us)"].map("{:.2f}".format)
    df["TFLOPS"] = df["TFLOPS"].map("{:.2f}".format)
    df["membw (GB/s)"] = df["membw (GB/s)"].map("{:.2f}".format)
    df["Noise"] = df["Noise"].map("{:.2%}".format)

    print()
    print(df[["kernel", "latency (us)", "Noise", "TFLOPS", "membw (GB/s)"]].to_markdown(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=int, nargs="+", default=[128, 1024, 3072])
    parser.add_argument("--modal")
    args = parser.parse_args()

    # local
    if args.modal is None:
        benchmark(args.shape)

    # modal
    else:
        import modal

        image = (
            modal.Image.from_registry("nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04", add_python="3.12")
            .entrypoint([])  # remove verbose logging by base image on entry
            .uv_pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu130")
            .uv_pip_install("ninja", "pandas", "tabulate", "cuda-bench[cu13]")
            .add_local_python_source("mlp_triton_v1")
        )
        app = modal.App("megakernel-mlp", image=image)
        modal_main = app.function(image=image, gpu=args.modal)(benchmark)

        with modal.enable_output(), app.run():
            modal_main.remote(args.shape)
