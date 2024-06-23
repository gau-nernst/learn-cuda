import time

import torch
import torch.utils.cpp_extension

module = torch.utils.cpp_extension.load(
    "module",
    sources=["sum.cu", "sum.cpp"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "--ptxas-options=-v"],
    verbose=True,
)

# for large n, there will be a larger deviation, since sum of many small elements are not accurate
input = torch.randn(64, 32000).cuda()

output_v1 = module.sum_v1(input)
output_v2 = module.sum_v2(input)
output_v3 = module.sum_v3(input)
output_v4 = module.sum_v4(input)
output_v5 = module.sum_v5(input)

output_ref = torch.sum(input, dim=-1)
torch.testing.assert_close(output_v1, output_ref, atol=1e-5, rtol=1e-5)
torch.testing.assert_close(output_v2, output_ref, atol=1e-4, rtol=1e-4)
torch.testing.assert_close(output_v3, output_ref, atol=1e-4, rtol=1e-4)
torch.testing.assert_close(output_v4, output_ref, atol=1e-4, rtol=1e-4)
torch.testing.assert_close(output_v5, output_ref, atol=1e-4, rtol=1e-4)


def benchmark(fn, *args):
    N = 100

    torch.cuda.synchronize()
    time0 = time.time()
    for _ in range(N):
        fn(*args)
        torch.cuda.synchronize()

    print(N / (time.time() - time0))


benchmark(torch.sum, input, -1)
benchmark(module.sum_v1, input)
benchmark(module.sum_v2, input)
benchmark(module.sum_v3, input)
benchmark(module.sum_v4, input)
benchmark(module.sum_v5, input)
