import torch
import torch.utils.cpp_extension

module = torch.utils.cpp_extension.load(
    "module",
    sources=["add.cu"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)

# Example usage
input1 = torch.randn(1000, device="cuda")
input2 = torch.randn(1000, device="cuda")
output = module.add(input1, input2)

torch.testing.assert_close(output, input1 + input2)
