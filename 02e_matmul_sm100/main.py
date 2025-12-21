# run this script with
#   modal run main.py --action benchmark

from pathlib import Path

import modal

CURRENT_DIR = Path(__file__).parent
REMOTE_DIR = Path("/my_extension")

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])  # remove verbose logging by base image on entry
    .uv_pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu130")
    .uv_pip_install("ninja")
    .add_local_dir(CURRENT_DIR, remote_path=REMOTE_DIR)
)
app = modal.App("sm100-matmul", image=image)


def get_module():
    import torch
    import torch.utils.cpp_extension

    print(f"{torch.__version__=}")
    print(f"{torch.version.cuda=}")

    torch.utils.cpp_extension.load(
        "module",
        sources=list(REMOTE_DIR.glob("matmul*")),
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


@app.function(gpu="B200")
def profile(shape: str):
    import torch

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


@app.function(gpu="B200")
def benchmark(shape: str):
    import time

    import torch
    from triton.testing import do_bench

    my_matmul = get_module()

    M, N, K = map(int, shape.split(","))
    print(f"{M=}, {N=}, {K=}")
    A = torch.randn(M, K).bfloat16().cuda()
    B = torch.randn(N, K).bfloat16().cuda().T

    def bench_and_print(f, name):
        # sleep to stabilize thermal
        torch.cuda.synchronize()
        time.sleep(1)

        # NOTE: B200 is too fast. the current way of benchmarking might be flawed.
        latency_ms = do_bench(lambda: f(A, B), warmup=10, rep=100, return_mode="median")
        tflops = 2 * M * N * K / latency_ms / 1e9
        print(f"{name}:\t{latency_ms:.4f} ms\t{tflops:.2f} TFLOPS")

    output_ref = torch.matmul(A, B)
    bench_and_print(torch.matmul, "CuBLAS")

    for version in [
        "v0",
        "v1a",
        "v1b",
        "v2a",
        "v2b",
        "v3",
        "v4",
        "v5",
        "v6",
    ]:
        f = getattr(my_matmul, f"matmul_{version}")
        out = f(A, B)
        torch.cuda.synchronize()
        try:
            torch.testing.assert_close(out, output_ref)
        except:
            print(output_ref)
            print(out)
            raise
        bench_and_print(f, version)


@app.local_entrypoint()
def main(action: str, shape: str = "4096,4096,4096"):
    if action == "benchmark":
        benchmark.remote(shape)

    elif action == "profile":
        import gzip
        import json

        events = profile.remote(shape)
        trace = dict(traceEvents=events)
        gzip.open("trace.json.gz", "w").write(json.dumps(trace).encode("utf-8"))
