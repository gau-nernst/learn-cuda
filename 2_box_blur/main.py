import time

import torch
import torch.nn.functional as F
import torch.utils.cpp_extension

module = torch.utils.cpp_extension.load(
    "module",
    sources=["box_blur.cu"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)

# Example usage
input = torch.randn(1, 1000, 1000, device="cuda")
kernel_size = 5
output = module.box_blur(input, kernel_size)

cached_kernels = dict()


def box_blur_ref(input: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size not in cached_kernels:
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=input.device)
        kernel /= kernel.sum()
        cached_kernels[kernel_size] = kernel

    kernel = cached_kernels[kernel_size]
    padding = (kernel_size - 1) // 2
    return F.conv2d(input.unsqueeze(0), kernel, padding=padding).squeeze(0)


torch.testing.assert_close(output, box_blur_ref(input, kernel_size))


def benchmark(fn, *args):
    N = 100

    torch.cuda.synchronize()
    time0 = time.time()
    for _ in range(N):
        fn(*args)
        torch.cuda.synchronize()

    print(N / (time.time() - time0))


benchmark(box_blur_ref, input, kernel_size)
benchmark(module.box_blur, input, kernel_size)
