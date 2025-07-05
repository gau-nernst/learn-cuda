import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.cpp_extension import load
from triton.testing import do_bench

CURRENT_DIR = Path(__file__).parent

module = load(
    "my_ext",
    sources=list(CURRENT_DIR.glob("attention*")),
    extra_cuda_cflags=["-lineinfo", "--ptxas-options=-v"],
    verbose=True,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument("--bs", default=4)
    parser.add_argument("--nh", default=8)
    parser.add_argument("--lq", default=2048)
    parser.add_argument("--lkv", default=4096)
    args = parser.parse_args()

    bs = args.bs
    nh = args.nh
    lq = args.lq
    lkv = args.lkv
    head_dim = 128

    # add a small offset so that output does not have a mean of zero,
    # which will result in large relative error
    # F.sdpa doesn't use FA/CuDNN for 3D inputs
    def generate_input(*shape):
        return torch.randn(shape).add(0.5).bfloat16().cuda()

    Q = generate_input(bs, nh, lq, head_dim)
    K = generate_input(bs, nh, lkv, head_dim)
    V = generate_input(bs, nh, lkv, head_dim)

    if args.profile is not None:
        if args.profile == "fa":
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                F.scaled_dot_product_attention(Q, K, V)

        elif args.profile == "cudnn":
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                F.scaled_dot_product_attention(Q, K, V)

        else:
            f = getattr(module, f"sdpa_v{args.profile}")
            f(Q, K, V)

        torch.cuda.synchronize()
        return

    SOL_LOOKUP = {
        "NVIDIA GeForce RTX 5090": 209.5,
    }
    sol = SOL_LOOKUP.get(torch.cuda.get_device_name(), 0)

    def bench_and_print(f, name):
        # sleep to stabilize thermal
        time.sleep(1)

        latency_ms = do_bench(lambda: f(Q, K, V), return_mode="median")
        tflops = 4 * bs * nh * lq * lkv * head_dim / latency_ms / 1e9
        pct_sol = tflops / sol * 100
        print(f"{name}:\t{latency_ms:.4f} ms\t{tflops:.2f} TFLOPS\t{pct_sol:.2f}% SOL")

    out_ref = F.scaled_dot_product_attention(Q, K, V)
    bench_and_print(F.scaled_dot_product_attention, "F.sdpa")

    for i in range(1):
        f = getattr(module, f"sdpa_v{i + 1}")
        out = f(Q, K, V)
        torch.testing.assert_close(out, out_ref)
        bench_and_print(f, f"v{i + 1}")


if __name__ == "__main__":
    main()
