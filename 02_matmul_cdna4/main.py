import torch
from triton.testing import do_bench


def main():
    for size in (4096, 8192, 16384):
        print(f"M=N=K={size}")

        scale = size**-0.5
        A = torch.randn(size, size, device="cuda").mul(scale).bfloat16()
        B = torch.randn(size, size, device="cuda").mul(scale).bfloat16().T

        out_ref = A @ B

        def benchmark(f, name):
            torch.testing.assert_close(f(A, B), out_ref)

            latency_us = do_bench(lambda: f(A, B)) * 1e3
            tflops = 2 * size * size * size / (latency_us * 1e-6) * 1e-12
            print(f"  {name}: {latency_us:.2f} us, {tflops:.2f} TFLOPS")

        benchmark(torch.mm, "PyTorch")

        print()


if __name__ == "__main__":
    main()
