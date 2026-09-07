import argparse
import math
from pathlib import Path

import cuda.bench
import pandas as pd
import torch
import torch.utils.cpp_extension

CURRENT_DIR = Path(__file__).parent

torch.utils.cpp_extension.load(
    "my_module",
    sources=list(CURRENT_DIR.glob("matmul*")),
    extra_cuda_cflags=[
        "-O3",
        "-lineinfo",
        "-Xptxas=-v",
        "-gencode=arch=compute_120a,code=sm_120a",
    ],
    extra_ldflags=["-lcuda"],  # for cuTensorMapEncodeTiled() used by TMA
    is_python_module=False,
    verbose=True,
)
module = torch.ops.my_module

SOL_LOOKUP = {
    "NVIDIA GeForce RTX 5090": dict(bf16=209.5, int8=838),
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": dict(bf16=503.8, int8=1007.6),
}


def make_inputs(M: int, N: int, K: int, dtype: str):
    if dtype == "bf16":
        A = torch.randn(M, K, device="cuda").mul(K**-0.5).bfloat16()
        B = torch.randn(N, K, device="cuda").mul(K**-0.5).bfloat16().T
    elif dtype == "int8":
        A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="cuda")
        B = torch.randint(-128, 127, (N, K), dtype=torch.int8, device="cuda").T
    return A, B


def get_kernel(name: str, dtype: str):
    cublas_mm = dict(bf16=torch.mm, int8=torch._int_mm)[dtype]

    if name == "cublas":
        fn = cublas_mm
    elif name == "inductor":
        torch._inductor.config.max_autotune_gemm_backends = "TRITON"
        # torch._inductor.utils.is_big_gpu = lambda _: True
        fn = torch.compile(cublas_mm, mode="max-autotune-no-cudagraphs", dynamic=False)
    else:
        fn = getattr(module, name)
    return fn


def to_torch_stream(s: cuda.bench.CudaStream, device: int | None):
    return torch.cuda.ExternalStream(stream_ptr=s.addressof(), device=device)


def torch_bench(state: cuda.bench.State) -> None:
    # state.set_throttle_threshold(0.25)
    device = state.get_device()

    # select kernel
    dtype = state.get_string("dtype")
    f = get_kernel(state.get_string("kernel"), dtype)

    # problem shape
    M, N, K = [int(x) for x in state.get_string("shape").split("_")]

    # duplicate inputs to make sure each measurement is at least 10ms
    sol = SOL_LOOKUP.get(torch.cuda.get_device_name(), dict(bf16=1e9, int8=1e9))[args.dtype]
    min_latency_ms = 2 * M * N * K / (sol * 1e12) * 1e3
    num_inputs = math.ceil(10 / min_latency_ms)
    state.add_element_count(num_inputs, "num_inputs")

    # create inputs and warmup
    stream = to_torch_stream(state.get_stream(), device)
    with torch.cuda.stream(stream):
        A, B = make_inputs(M, N, K, dtype)
        for _ in range(5):
            f(A, B)

        inputs_list = [make_inputs(M, N, K, dtype) for _ in range(num_inputs)]

    def launcher(launch: cuda.bench.Launch) -> None:
        stream = to_torch_stream(launch.get_stream(), device)
        with torch.cuda.stream(stream):
            for A, B in inputs_list:
                f(A, B)

    state.exec(launcher, sync=True)


def main(args: argparse.Namespace):
    for shape in args.shape:
        M, N, K = [int(x) for x in shape.split("_")]
        print(f"{M=}, {N=}, {K=}")
        A, B = make_inputs(M, N, K, args.dtype)

        if args.profile is not None:
            fn = get_kernel(args.profile, args.dtype)
            with torch.cuda.nvtx.range("kernel"):
                fn(A, B)
            torch.cuda.synchronize()
            return

        # correctness check
        if args.dtype == "bf16":
            # reference in FP32 to avoid things like split-K
            output_ref = torch.matmul(A.float(), B.float()).bfloat16()
        else:
            output_ref = torch._int_mm(A, B)

        for i in range(4):
            fn = getattr(module, f"matmul_v{i}")
            output = fn(A, B)
            torch.testing.assert_close(output, output_ref)

    # benchmark with nvbench
    kernels_list = []
    kernels_list += ["cublas", "inductor"]
    kernels_list += [f"matmul_v{i}" for i in range(4)]

    bench = cuda.bench.register(torch_bench)
    bench.add_string_axis("kernel", kernels_list)
    bench.add_string_axis("dtype", [args.dtype])
    bench.add_string_axis("shape", args.shape)

    result_path = "/tmp/result.csv"
    cuda.bench.run_all_benchmarks(["--csv", result_path])

    df = pd.read_csv(result_path)
    df["GPU Time (sec)"] /= df["num_inputs"]
    df["latency (us)"] = df["GPU Time (sec)"] * 1e6

    def parse_tflops(shape: str, time: float):
        M, N, K = [int(x) for x in shape.split("_")]
        return 2 * M * N * K / time * 1e-12

    sol = SOL_LOOKUP.get(torch.cuda.get_device_name(), dict(bf16=1e9, int8=1e9))[args.dtype]
    df["TFLOPS"] = [parse_tflops(shape, time) for shape, time in zip(df["shape"], df["GPU Time (sec)"])]
    df["% SOL"] = df["TFLOPS"] / sol

    # apply formatting
    df["latency (us)"] = df["latency (us)"].map("{:.2f}".format)
    df["TFLOPS"] = df["TFLOPS"].map("{:.2f}".format)
    df["% SOL"] = df["% SOL"].map("{:.2%}".format)
    df["Noise"] = df["Noise"].map("{:.2%}".format)

    for shape, sub_df in df.groupby("shape"):
        print()
        print(shape)
        print(sub_df[["kernel", "latency (us)", "Noise", "TFLOPS", "% SOL"]].to_markdown(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument("--shape", nargs="+", default=["4096_4096_4096"])
    parser.add_argument("--dtype", default="bf16")
    args = parser.parse_args()

    main(args)
