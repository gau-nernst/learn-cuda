# Softmax

Resources:
- Online softmax: https://arxiv.org/abs/1805.02867
- https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/cuda/SoftMax.cu

Given a matrix of size (M, N), we want to calculate softmax along the last dimension.

**`M=8192, N=8192`**: this is a possible attention logits for seq_len=8192.

Kernel name               | Latency (us) | % of PyTorch
--------------------------|--------------|--------------
Max theoretical bandwidth |           -- |           --
PyTorch                   |       875.52 |      100.00%
`torch.compile()`         |     1,776.64 |       49.28%


**`M=1, N=128256`**: this is the logit outputs of Llama3 with batch_size=1.

Lessons learned:
- Set initial value for max to `-FLT_MAX` (requires `float.h`) instead of `-INFINITY` since doing floating point math with infinity may result in NaNs.
