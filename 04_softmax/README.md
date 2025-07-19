# Softmax

Resources:
- Online softmax: https://arxiv.org/abs/1805.02867 (https://github.com/NVIDIA/online-softmax)
- https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/cuda/SoftMax.cu

Given a matrix of size (M, N), we want to calculate softmax along the last dimension.

**`M=8192, N=8192`**: this is a possible attention logits for seq_len=8192.

Kernel name                       | Latency (us) | % of PyTorch
----------------------------------|--------------|--------------
PyTorch `cunn_SoftMaxForwardSmem` |       877.57 |      100.00%
`torch.compile()`                 |     1,780.74 |       49.28%
Naive softmax                     |       874.69 |      100.33%
Naive softmax (split)             |     1,726.50 |       50.83%
Online softmax                    |       874.85 |      100.31%
Online softmax (split)            |     1,303.55 |       67.32%

**`M=1, N=128256`**: this is the logit outputs of Llama3 with batch_size=1.

Kernel name                   | Latency (us) | % of PyTorch
------------------------------|--------------|--------------
PyTorch `cunn_SoftMaxForward` |        31.74 |      100.00%
`torch.compile()`             |        24.58 |      129.13%
Naive softmax                 |        25.60 |      123.98%
Naive softmax (split)         |        11.26 |      281.88%
Online softmax                |        19.46 |      163.10%
Online softmax (split)        |        28.67 |      110.71%

Lessons learned:
- Set initial value for max to `-FLT_MAX` (requires `float.h`) instead of `-INFINITY` since doing floating point math with infinity may result in NaNs.
- FP32 atomic max: implement via uint/int atomic max, since we can compare FP32 numbers with their bit representations directly. Some special care is required for sign-ness.
- Online softmax doesn't seem to be faster than naive softmax. It's also problematic for split-N implementation, since we have to use the slow `atomicCAS()`.
- Use `atomicCAS()` for custom atomic op.
- For small batch size, split-N implementation outperforms single-block implementation. It makes sense, since we parallelize across softmax dim.
- PyTorch has 3 implementations of softmax:
  1. `cunn_SoftMaxForward`: similar to our naive softmax.
  2. `cunn_SoftMaxForwardSmem`: same as above, but cache inputs to shared memory (each row must fit in shared memory).
  3. `cunn_SpatialSoftMaxForward`: similar to our naive softmax, but supports non-contiguous input (stride > 1).
