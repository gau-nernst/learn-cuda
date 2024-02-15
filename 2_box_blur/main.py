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
input = torch.randn(4, 1000, 800, device="cuda")
kernel_size = 3
output_v1 = module.box_blur_v1(input, kernel_size)
output_v2 = module.box_blur_v2(input, kernel_size)

cached_kernels = dict()


def box_blur_ref(input: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size not in cached_kernels:
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=input.device)
        kernel /= kernel.sum()
        cached_kernels[kernel_size] = kernel

    kernel = cached_kernels[kernel_size]
    padding = (kernel_size - 1) // 2
    return F.conv2d(input.unsqueeze(1), kernel, padding=padding).squeeze(1)


output_ref = box_blur_ref(input, kernel_size)
torch.testing.assert_close(output_v1, output_ref)
torch.testing.assert_close(output_v2, output_ref)


def benchmark(fn, *args):
    N = 100

    torch.cuda.synchronize()
    time0 = time.time()
    for _ in range(N):
        fn(*args)
        torch.cuda.synchronize()

    print(N / (time.time() - time0))


benchmark(box_blur_ref, input, kernel_size)
benchmark(module.box_blur_v1, input, kernel_size)
benchmark(module.box_blur_v2, input, kernel_size)
