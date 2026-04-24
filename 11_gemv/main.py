import argparse
import importlib
import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.cpp_extension import load

if TYPE_CHECKING:
    import cuda.bench


CURRENT_DIR = Path(__file__).parent


def get_membw():
    prop = torch.cuda.get_device_properties()
    print(f"Memory bus width: {prop.memory_bus_width}-bit")
    print(f"Memory clock: {prop.memory_clock_rate / 1e3} kHz")
    return prop.memory_bus_width * prop.memory_clock_rate / 8.0 * 2


def get_kernel(name: str):
    if name == "eager":
        f = torch.mm
    elif name == "inductor":
        # torch._inductor.config.max_autotune_gemm_backends = "TRITON"
        # torch._inductor.utils.is_big_gpu = lambda _: True
        f = torch.compile(torch.mm, mode="max-autotune-no-cudagraphs", dynamic=False)
    elif name.startswith("cuda_"):
        sources = [str(CURRENT_DIR / "gemv.cpp")]
        sources.extend(str(x) for x in CURRENT_DIR.glob("cuda_*.cu"))
        load(
            "my_gemv",
            sources,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            is_python_module=False,
        )
        f = getattr(torch.ops.my_gemv, name)
    else:
        f = getattr(importlib.import_module(name), name)
    return f


def to_torch_stream(s: "cuda.bench.CudaStream", device: int | None):
    return torch.cuda.ExternalStream(stream_ptr=s.addressof(), device=device)


def torch_bench(state: "cuda.bench.State") -> None:
    # state.set_throttle_threshold(0.25)
    device = state.get_device()

    # select kernel
    f = get_kernel(state.get_string("kernel"))

    M = 1
    N, K = [int(x) for x in state.get_string("shape").split("_")]

    stream = to_torch_stream(state.get_stream(), device)
    with torch.cuda.stream(stream):
        scale = K**-0.5  # make sure output doesn't explode
        A = torch.randn(M, K, device=device).mul(scale).bfloat16()
        B = torch.randn(N, K, device=device).mul(scale).bfloat16().T

        # correctness check
        # compute in FP32 to avoid split-K
        out_ref = torch.mm(A.float(), B.float()).bfloat16()
        out = f(A, B)
        torch.testing.assert_close(out, out_ref)

        inputs_list = []
        for _ in range(state.get_int64("num_inputs")):
            A = torch.randn(M, K, device=device).mul(scale).bfloat16()
            B = torch.randn(N, K, device=device).mul(scale).bfloat16().T
            inputs_list.append((A, B))

    def launcher(launch: "cuda.bench.Launch") -> None:
        stream = to_torch_stream(launch.get_stream(), device)
        with torch.cuda.stream(stream):
            for A, B in inputs_list:
                f(A, B)

    state.exec(launcher, sync=True)


def benchmark(shape: str):
    import cuda.bench
    import pandas as pd

    print(f"{torch.__version__=}")
    print(f"{torch.version.cuda=}")

    M = 1
    N, K = map(int, shape.split("_"))

    # duplicate inputs to make sure each measurement is at least 10ms
    membw = get_membw()
    min_latency_ms = (M * K + N * K + M * N) * 2 / membw * 1e3
    num_inputs = math.ceil(10 / min_latency_ms)

    kernels_list = []
    kernels_list += ["eager", "inductor"]
    kernels_list += ["cuda_v1"]
    kernels_list += ["cuda_persistent_v1"]

    bench = cuda.bench.register(torch_bench)
    bench.add_string_axis("kernel", kernels_list)
    bench.add_string_axis("shape", [shape])
    bench.add_int64_axis("num_inputs", [num_inputs])

    result_path = "/tmp/result.csv"
    cuda.bench.run_all_benchmarks(["--csv", result_path])

    df = pd.read_csv(result_path)
    df["GPU Time (sec)"] /= num_inputs  # rescale
    df["latency (us)"] = df["GPU Time (sec)"] * 1e6
    df["GB/s"] = (M * K + N * K + M * N) * 2 / df["GPU Time (sec)"] * 1e-9

    # apply formatting
    df["latency (us)"] = df["latency (us)"].map("{:.2f}".format)
    df["GB/s"] = df["GB/s"].map("{:.2f}".format)
    df["Noise"] = df["Noise"].map("{:.2%}".format)

    print()
    print(df[["kernel", "latency (us)", "Noise", "GB/s"]].to_markdown(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument("--shape", default="4096_4096")
    parser.add_argument("--modal")
    args = parser.parse_args()

    if args.modal is not None:
        import modal

        image = (
            modal.Image.from_registry("nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04", add_python="3.12")
            .entrypoint([])  # remove verbose logging by base image on entry
            .uv_pip_install("torch==2.11.0")
            .uv_pip_install("ninja", "pandas", "tabulate", "cuda-bench[cu13]")
            .workdir("/workspace")
            .add_local_python_source("triton_v1")
            .add_local_dir(CURRENT_DIR, remote_path="/workspace")
        )
        app = modal.App("gemv", image=image)
        modal_benchmark = app.function(gpu=args.modal)(benchmark)

        with modal.enable_output(), app.run():
            modal_benchmark.remote(args.shape)

    else:
        benchmark(args.shape)
