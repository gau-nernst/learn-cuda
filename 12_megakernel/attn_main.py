# NOTE: only measure decode speed

import argparse
import importlib
import math
from typing import TYPE_CHECKING

import pandas as pd
import torch
from reference import _precompute_rope, attn_ref, get_sol

if TYPE_CHECKING:
    import cuda.bench


def get_kernel(name: str):
    if name == "eager":
        f = attn_ref
    elif name == "inductor":
        # torch._inductor.config.max_autotune_gemm_backends = "TRITON"
        f = torch.compile(attn_ref, mode="max-autotune-no-cudagraphs", dynamic=False, fullgraph=True)
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
    dim = state.get_int64("dim")
    num_heads = state.get_int64("num_heads")
    num_kv_heads = state.get_int64("num_kv_heads")
    kv_size = state.get_int64("kv_size")
    head_dim = 128

    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    qkv_dim = q_dim + kv_dim * 2

    all_rope = _precompute_rope(kv_dim + 1, head_dim, theta=1e6)
    rope = all_rope[kv_size : kv_size + 1].to(device)

    stream = to_torch_stream(state.get_stream(), device)
    with torch.cuda.stream(stream):
        # apply scaling to make sure the output doesn't explode
        X = torch.randn(1, dim, device=device).mul(dim**-0.5).bfloat16()
        norm = torch.randn(dim, device=device).mul(dim**-0.5).bfloat16()
        kv_cache = torch.randn(2, kv_size * 2, num_kv_heads, head_dim, device=device).bfloat16()
        Wqkv = torch.randn(qkv_dim, dim, device=device).mul(dim**-0.5).bfloat16()
        q_norm = torch.randn(head_dim, device=device).mul(head_dim**-0.5).bfloat16()
        k_norm = torch.randn(head_dim, device=device).mul(head_dim**-0.5).bfloat16()
        Wo = torch.randn(dim, q_dim, device=device).mul(q_dim**-0.5).bfloat16()

        # correctness check
        kv_cache_ref = kv_cache.clone()
        out_ref = attn_ref(X, norm, kv_cache_ref, Wqkv, q_norm, k_norm, rope, Wo, kv_size)
        out = f(X, norm, kv_cache, Wqkv, q_norm, k_norm, rope, Wo, kv_size)

        # only check atol
        atol, rtol = 1e-4, float("inf")
        torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(kv_cache[:, kv_size], kv_cache_ref[:, kv_size], atol=atol, rtol=rtol)

        inputs_list = []
        for _ in range(state.get_int64("num_inputs")):
            X = torch.randn(1, dim, device=device).mul(dim**-0.5).bfloat16()
            norm = torch.randn(dim, device=device).mul(dim**-0.5).bfloat16()
            kv_cache = torch.randn(2, kv_size * 2, num_kv_heads, head_dim, device=device).bfloat16()
            Wqkv = torch.randn(qkv_dim, dim, device=device).mul(dim**-0.5).bfloat16()
            q_norm = torch.randn(head_dim, device=device).mul(head_dim**-0.5).bfloat16()
            k_norm = torch.randn(head_dim, device=device).mul(head_dim**-0.5).bfloat16()
            Wo = torch.randn(dim, q_dim, device=device).mul(q_dim**-0.5).bfloat16()
            inputs_list.append((X, norm, kv_cache, Wqkv, q_norm, k_norm, Wo))

    def launcher(launch: "cuda.bench.Launch") -> None:
        stream = to_torch_stream(launch.get_stream(), device)
        with torch.cuda.stream(stream):
            for X, norm, kv_cache, Wqkv, q_norm, k_norm, Wo in inputs_list:
                f(X, norm, kv_cache, Wqkv, q_norm, k_norm, rope, Wo, kv_size)

    state.exec(launcher, sync=True)


def benchmark(args: argparse.Namespace):
    import cuda.bench

    print(f"{torch.__version__=}")
    print(f"{torch.version.cuda=}")

    dim = args.dim
    num_heads = args.num_heads
    num_kv_heads = args.num_kv_heads
    kv_size = args.kv_size
    head_dim = 128

    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    qkv_dim = q_dim + kv_dim * 2

    num_elems = dim + dim  # RMS norm. don't count output
    num_elems += dim * qkv_dim  # qkv proj. don't count input/output
    num_elems += head_dim * 2  # QK norm. don't count input/output
    num_elems += head_dim * 2  # RoPE. don't count input/output
    num_elems += kv_size * kv_dim * 2  # KV cache
    num_elems += kv_dim * 2  # update KV cache (write to gmem)
    num_elems += q_dim * dim + dim  # output projection. don't include input
    num_gb = num_elems * 2 * 1e-9

    # duplicate inputs to make sure each measurement is at least 10ms
    SOL_COMPUTE, SOL_MEMORY = get_sol()
    min_memory_latency_ms = num_gb / SOL_MEMORY * 1e3
    min_latency_ms = min_memory_latency_ms
    num_inputs = min(math.ceil(10 / min_latency_ms), 1000)

    kernels_list = []
    kernels_list += ["eager", "inductor"]
    kernels_list += ["attn_triton_v1.attn_triton_v1"]

    bench = cuda.bench.register(torch_bench)
    bench.add_string_axis("kernel", kernels_list)
    bench.add_int64_axis("dim", [dim])
    bench.add_int64_axis("num_heads", [num_heads])
    bench.add_int64_axis("num_kv_heads", [num_kv_heads])
    bench.add_int64_axis("kv_size", [kv_size])
    bench.add_int64_axis("num_inputs", [num_inputs])

    result_path = "/tmp/result.csv"
    cuda.bench.run_all_benchmarks(["--csv", result_path])

    df = pd.read_csv(result_path)
    df["GPU Time (sec)"] /= num_inputs  # rescale
    df["latency (us)"] = df["GPU Time (sec)"] * 1e6
    df["membw (GB/s)"] = num_gb / df["GPU Time (sec)"]

    # apply formatting
    df["latency (us)"] = df["latency (us)"].map("{:.2f}".format)
    df["membw (GB/s)"] = df["membw (GB/s)"].map("{:.2f}".format)
    df["Noise"] = df["Noise"].map("{:.2%}".format)

    print()
    print(df[["kernel", "latency (us)", "Noise", "membw (GB/s)"]].to_markdown(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--num_kv_heads", type=int, default=8)
    parser.add_argument("--kv_size", type=int, default=1024)
    parser.add_argument("--modal")
    args = parser.parse_args()

    # local
    if args.modal is None:
        benchmark(args)

    # modal
    else:
        import modal

        image = (
            modal.Image.from_registry("nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04", add_python="3.12")
            .entrypoint([])  # remove verbose logging by base image on entry
            .uv_pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu130")
            .uv_pip_install("transformers", "ninja", "pandas", "tabulate", "cuda-bench[cu13]")
            .add_local_python_source("reference", "attn_triton_v1")
        )
        app = modal.App("megakernel-mlp", image=image)
        modal_main = app.function(image=image, gpu=args.modal)(benchmark)

        with modal.enable_output(), app.run():
            modal_main.remote(args)
