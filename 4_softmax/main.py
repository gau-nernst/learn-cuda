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

for M, N in [(8096, 8096), (1, 128256)]:
    print(f"{M=}, {N=}")

    input = torch.randn(M, N).cuda()
    output_ref = torch.softmax(input, dim=1)
    output_v1 = module.softmax_v1(input)
    output_v2 = module.softmax_v2(input)

    # torch.testing.assert_close(output_v1, output_ref)
    # torch.testing.assert_close(output_v2, output_ref)
    print(((output_v1 - output_ref).abs() / output_ref.abs()).mean())
    print(((output_v2 - output_ref).abs() / output_ref.abs()).mean())  # pretty high. seems like sth is wrong

    print(f"PyTorch: {benchmark(torch.softmax, input, 1):.2f}us")
    print(f"torch.compile: {benchmark(compiled_softmax, input, 1):.2f}us")
    print(f"Softmax v1: {benchmark(module.softmax_v1, input):.2f}us")
    print(f"Softmax v2: {benchmark(module.softmax_v2, input):.2f}us")
