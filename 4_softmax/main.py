import torch
import torch.utils.cpp_extension
from triton.testing import do_bench


def benchmark(f, *args):
    return do_bench(lambda: f(*args), return_mode="median") * 1e3  # return in us


module = torch.utils.cpp_extension.load(
    "module",
    sources=["softmax.cu", "softmax.cpp"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "--ptxas-options=-v"],
    verbose=True,
)
compiled_softmax = torch.compile(torch.softmax, mode="max-autotune", dynamic=False)

for M, N in [
    (1023, 518),
    (8192, 8192),
    (1, 128256),
]:
    print(f"{M=}, {N=}")

    input = torch.randn(M, N).cuda()
    output_ref = torch.softmax(input, dim=1)
    output_naive = module.softmax_naive(input)
    output_naive_split = module.softmax_naive_split(input)
    output_online = module.softmax_online(input)
    output_online_split = module.softmax_online_split(input)

    torch.testing.assert_close(output_naive, output_ref)
    torch.testing.assert_close(output_naive_split, output_ref)
    torch.testing.assert_close(output_online, output_ref)
    torch.testing.assert_close(output_online_split, output_ref)

    print(f"PyTorch: {benchmark(torch.softmax, input, 1):.2f}us")
    print(f"torch.compile: {benchmark(compiled_softmax, input, 1):.2f}us")
    print(f"Softmax naive: {benchmark(module.softmax_naive, input):.2f}us")
    print(f"Softmax naive split: {benchmark(module.softmax_naive_split, input):.2f}us")
    print(f"Softmax online: {benchmark(module.softmax_online, input):.2f}us")
    print(f"Softmax online split: {benchmark(module.softmax_online_split, input):.2f}us")
    print()
