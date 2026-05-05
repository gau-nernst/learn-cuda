# run this script with
#   python main.py --action benchmark
#   python main.py --action benchmark --modal  # run on Modal

import argparse
import importlib
import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.utils.cpp_extension

if TYPE_CHECKING:
    import cuda.bench


CURRENT_DIR = Path(__file__).parent


MY_KERNELS = [
    # "matmul_v0",
    # "matmul_v1a",
    # "matmul_v1b",
    # "matmul_v2a",
    # "matmul_v2b",
    # "matmul_v3",
    # "matmul_v4",
    # "matmul_v5",
    # "matmul_v6",
    "matmul_v7a",
    "matmul_v7b",
    "matmul_v7c",
    # "cutedsl_v1",
]


def get_module():
    torch.utils.cpp_extension.load(
        "module",
        sources=list(CURRENT_DIR.glob("matmul*")),
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "-Xptxas=-v",
            "-gencode=arch=compute_100a,code=sm_100a",
        ],
        extra_ldflags=["-lcuda"],  # for cuTensorMapEncodeTiled() used by TMA
        verbose=True,
        is_python_module=False,
    )
    return torch.ops.my_matmul


def get_kernel(name: str):
    if name == "cublas":
        f = torch.mm
    elif name.startswith("cutedsl_"):
        f = getattr(importlib.import_module(name), name)
    else:
        f = getattr(get_module(), name)
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

    M, N, K = map(int, shape.split(","))

    # duplicate inputs to make sure each measurement is at least 10ms
    SOL = 2000  # just an estimate, we can't reach this number anyway
    min_latency_ms = 2 * M * N * K / (SOL * 1e12) * 1e3
    num_inputs = math.ceil(10 / min_latency_ms)

    kernels_list = []
    kernels_list += ["cublas"]
    kernels_list += MY_KERNELS

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
    df["TFLOPS"] = 2 * M * N * K / df["GPU Time (sec)"] * 1e-12

    # apply formatting
    df["latency (us)"] = df["latency (us)"].map("{:.2f}".format)
    df["TFLOPS"] = df["TFLOPS"].map("{:.2f}".format)
    df["Noise"] = df["Noise"].map("{:.2%}".format)

    print()
    print(df[["kernel", "latency (us)", "Noise", "TFLOPS"]].to_markdown(index=False))


def profile(shape: str):
    # this must match what's defined in profiler.h
    TAGS = [
        "SETUP",
        "ISSUE_TMA",
        "ISSUE_MMA",
        "WAIT_TMA",
        "WAIT_MMA",
        "WAIT_MAINLOOP",
        "WAIT_EPILOGUE",
        "EPILOGUE",
    ]

    my_matmul = get_module()

    M, N, K = map(int, shape.split(","))
    print(f"{M=}, {N=}, {K=}")
    A = torch.randn(M, K).bfloat16().cuda()
    B = torch.randn(N, K).bfloat16().cuda().T

    # f = my_matmul.profile_matmul_v5
    # NUM_BLOCKS = 10_000
    # NUM_ENTRIES = 1000

    # for persistent kernel, there are only 148 threadblocks, but each
    # threadblock produces a lot of entries
    f = my_matmul.profile_matmul_v6
    NUM_BLOCKS = 200 * 6  # 6 warps per threadblock
    NUM_ENTRIES = 100_000

    profiler = torch.zeros(NUM_BLOCKS, 1 + NUM_ENTRIES * 4, dtype=torch.int64, device="cuda")
    # warm up
    for _ in range(5):
        f(A, B, profiler, NUM_ENTRIES)
    # actual run
    torch.cuda.synchronize()
    profiler.zero_()
    f(A, B, profiler, NUM_ENTRIES)

    profile_data = profiler.tolist()
    events = []

    for bid, data in enumerate(profile_data):
        for i in range(data[0]):
            sm_id, tag, start, duration = data[1 + i * 4 : 1 + (i + 1) * 4]
            events.append(dict(name=TAGS[tag], ph="X", ts=start, dur=duration, pid=sm_id, tid=sm_id + bid))

    offset = min([evt["ts"] for evt in events])
    for evt in events:
        evt["ts"] -= offset
    return events


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["benchmark", "profile"], default="benchmark")
    parser.add_argument("--shape", default="4096,4096,4096")
    parser.add_argument("--modal", action="store_true")
    args = parser.parse_args()

    # build Modal stuff
    if args.modal:
        import modal

        image = (
            modal.Image.from_registry("nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04", add_python="3.12")
            .entrypoint([])  # remove verbose logging by base image on entry
            .uv_pip_install("torch==2.11.0")
            .uv_pip_install("ninja", "pandas", "tabulate", "nvidia-cutlass-dsl", "apache-tvm-ffi", "cuda-bench[cu13]")
            .workdir("/workspace")
            .add_local_dir(CURRENT_DIR, remote_path="/workspace", ignore=["*.venv"])
        )
        app = modal.App("sm100-matmul", image=image)
        modal_benchmark = app.function(gpu="B200")(benchmark)
        modal_profile = app.function(gpu="B200")(profile)

    # dispatch action
    if args.action == "benchmark":
        if args.modal:
            with modal.enable_output(), app.run():
                modal_benchmark.remote(args.shape)
        else:
            benchmark(args.shape)

    elif args.action == "profile":
        import gzip
        import json

        if args.modal:
            with modal.enable_output(), app.run():
                events = modal_profile.remote(args.shape)
        else:
            events = profile(args.shape)

        trace = dict(traceEvents=events)
        gzip.open("trace.json.gz", "w").write(json.dumps(trace).encode("utf-8"))
