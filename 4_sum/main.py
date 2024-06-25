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

output_ref = torch.sum(input, dim=-1)
output_v1 = module.sum_v1(input)
output_v2 = module.sum_v2(input)
output_v3 = module.sum_v3(input)
output_v4a = module.sum_v4a(input)
output_v4b = module.sum_v4b(input)
output_v4c = module.sum_v4c(input)
output_v5 = module.sum_v5(input)

torch.testing.assert_close(output_v1, output_ref, atol=1e-5, rtol=1e-5)
torch.testing.assert_close(output_v2, output_ref, atol=1e-4, rtol=1e-4)
torch.testing.assert_close(output_v3, output_ref, atol=1e-4, rtol=1e-4)
torch.testing.assert_close(output_v4a, output_ref, atol=1e-4, rtol=1e-4)
torch.testing.assert_close(output_v4b, output_ref, atol=1e-4, rtol=1e-4)
torch.testing.assert_close(output_v4c, output_ref, atol=1e-4, rtol=1e-4)
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
benchmark(module.sum_v4a, input)
benchmark(module.sum_v4b, input)
benchmark(module.sum_v4c, input)
benchmark(module.sum_v5, input)
