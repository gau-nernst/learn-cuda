import argparse
import importlib
import math
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch
from reference import get_sol, mlp_ref
from torch.utils.cpp_extension import load

if TYPE_CHECKING:
    import cuda.bench


CURRENT_DIR = Path(__file__).parent


def get_kernel(name: str):
    if name == "eager":
        f = mlp_ref
    elif name == "inductor":
        # torch._inductor.config.max_autotune_gemm_backends = "TRITON"
        f = torch.compile(mlp_ref, mode="max-autotune-no-cudagraphs", dynamic=False, fullgraph=True)
    elif "cuda" in name:
        sources = [str(CURRENT_DIR / "mlp_cuda.cpp")]
        sources.extend(str(x) for x in CURRENT_DIR.glob("mlp_*.cu"))
        load(
            "my_mlp",
            sources,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            is_python_module=False,
        )
        f = getattr(torch.ops.my_mlp, name)
    else:
        m_name, f_name = name.split(".")
        f = getattr(importlib.import_module(m_name), f_name)
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
        # apply scaling to make sure the output doesn't explode
        X = torch.randn(M, K, device=device).mul(K**-0.5).bfloat16()
        norm = torch.randn(K, device=device).mul(K**-0.5).bfloat16()
        W13 = torch.randn(N * 2, K, device=device).mul(K**-0.5).bfloat16()
        W2 = torch.randn(K, N, device=device).mul(N**-0.5).bfloat16()

        # correctness check
        out_ref = mlp_ref(X, norm, W13, W2)
        out = f(X, norm, W13, W2)
        torch.testing.assert_close(out, out_ref)

        inputs_list = []
        for _ in range(state.get_int64("num_inputs")):
            X = torch.randn(M, K, device=device).mul(K**-0.5).bfloat16()
            norm = torch.randn(K, device=device).mul(K**-0.5).bfloat16()
            W13 = torch.randn(N * 2, K, device=device).mul(K**-0.5).bfloat16()
            W2 = torch.randn(K, N, device=device).mul(N**-0.5).bfloat16()
            inputs_list.append((X, norm, W13, W2))

    def launcher(launch: "cuda.bench.Launch") -> None:
        stream = to_torch_stream(launch.get_stream(), device)
        with torch.cuda.stream(stream):
            for X, norm, W13, W2 in inputs_list:
                f(X, norm, W13, W2)

    state.exec(launcher, sync=True)


def benchmark(shape: list[int]):
    import cuda.bench

    print(f"{torch.__version__=}")
    print(f"{torch.version.cuda=}")

    M, N, K = shape

    # only tensor core flops
    num_flops = 3 * 2 * M * N * K

    # we also count writing and reading tmp buffers in total memory traffic
    num_elems = 2 * M * K + K  # RMS norm
    num_elems += M * K + 2 * N * K + M * N  # gate + up proj
    num_elems += M * N + N * K + M * K  # down proj
    num_gb = num_elems * 2 * 1e-9

    # duplicate inputs to make sure each measurement is at least 10ms
    SOL_COMPUTE, SOL_MEMORY = get_sol()
    min_compute_latency_ms = num_flops / (SOL_COMPUTE * 1e12) * 1e3
    min_memory_latency_ms = num_gb / SOL_MEMORY * 1e3
    min_latency_ms = max(min_compute_latency_ms, min_memory_latency_ms)
    num_inputs = min(math.ceil(10 / min_latency_ms), 1000)

    kernels_list = []
    kernels_list += ["eager", "inductor"]
    kernels_list += ["mlp_triton_v1.mlp_triton_v1", "mlp_triton_v1.mlp_triton_v1_2stage"]
    if M == 1:
        kernels_list += ["mlp_gemv_triton_v1.mlp_gemv_triton_v1"]
        kernels_list += ["mlp_gemv_cuda_v1"]

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
    parser.add_argument("--shape", type=int, nargs="+", default=[128, 3072, 1024])
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
            .uv_pip_install("torch==2.11.0")
            .uv_pip_install("transformers", "ninja", "pandas", "tabulate", "cuda-bench[cu13]")
            .workdir("/workspace")
            .add_local_dir(CURRENT_DIR, remote_path="/workspace", ignore=["*.venv"])
        )
        app = modal.App("megakernel-mlp", image=image)
        modal_main = app.function(image=image, gpu=args.modal)(benchmark)

        with modal.enable_output(), app.run():
            modal_main.remote(args.shape)
