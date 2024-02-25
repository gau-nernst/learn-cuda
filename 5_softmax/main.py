import time

import torch
import torch.nn.functional as F
import torch.utils.cpp_extension

module = torch.utils.cpp_extension.load(
    "module",
    sources=["softmax.cu"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "--ptxas-options=-v"],
    verbose=True,
)

# for large n, there will be a larger deviation, since sum of many small elements are not accurate
input = torch.randn(1000, 1024, device="cuda")

output_v1 = module.softmax_v1(input)

output_ref = torch.softmax(input, 1)
torch.testing.assert_close(output_v1, output_ref)


def benchmark(fn, *args):
    N = 100

    torch.cuda.synchronize()
    time0 = time.time()
    for _ in range(N):
        fn(*args)
        torch.cuda.synchronize()

    print(N / (time.time() - time0))


benchmark(torch.softmax, input, 1)
benchmark(module.softmax_v1, input)
