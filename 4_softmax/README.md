# Softmax

Resources:
- Online softmax: https://arxiv.org/abs/1805.02867

Given a matrix of size (M, N), we want to calculate softmax along the last dimension. We benchmarks the following scenarios
1. M=1, N=128256: this is the logit outputs of Llama3 with batch_size=1. 
2. M=8096, N=8096: this is a possible attention logits for seq_len=8096.

Kernel name               | Latency (ms) | % of PyTorch | Bandwidth (GB/s)
--------------------------|--------------|--------------|-----------------
Max theoretical bandwidth |           -- |           -- |           672.00
PyTorch
