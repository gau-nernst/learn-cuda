# Softmax

Resources:
- Online softmax: https://arxiv.org/abs/1805.02867
- https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/cuda/SoftMax.cu

Given a matrix of size (M, N), we want to calculate softmax along the last dimension. We benchmarks the following scenarios
1. M=8192, N=8192: this is a possible attention logits for seq_len=8192.
2. M=1, N=128256: this is the logit outputs of Llama3 with batch_size=1.

Kernel name               | Latency (us) | % of PyTorch | Bandwidth (GB/s)
--------------------------|--------------|--------------|-----------------
Max theoretical bandwidth |           -- |           -- |           672.00
PyTorch

Lessons learned:
- Set initial value for max to `-FLT_MAX` (requires `float.h`) instead of `-INFINITY` since doing floating point math with infinity may result in NaNs.
