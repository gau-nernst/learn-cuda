import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def kernel_v1(
    A_ptr,  # [1, K]
    B_ptr,  # [N, K]
    C_ptr,  # [1, N]
    N,
    K,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_DOT: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + offs_k[None, :]
    B_ptrs = B_ptr + (offs_n[:, None] * K + offs_k[None, :])

    if USE_DOT:
        acc = tl.zeros((1, BLOCK_N), dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    for _ in range(tl.cdiv(K, BLOCK_K)):
        a = tl.load(A_ptrs, eviction_policy="evict_first")
        b = tl.load(B_ptrs, eviction_policy="evict_last")
        A_ptrs += BLOCK_K
        B_ptrs += BLOCK_K

        if USE_DOT:
            acc = tl.dot(a, b.T, acc)
        else:
            acc += a.to(tl.float32) * b.to(tl.float32)

    if not USE_DOT:
        acc = tl.sum(acc, axis=1)[None, :]

    # rematerialize offs_n
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    C_ptrs = C_ptr + offs_n[None, :]
    tl.store(C_ptrs, acc)


def triton_v1(A: Tensor, B: Tensor, *, return_kernel: bool = False):
    K, N = B.shape
    C = A.new_empty(1, N)

    BLOCK_N = 32
    BLOCK_K = 512
    USE_DOT = False
    num_blocks = triton.cdiv(N, BLOCK_N)

    kernel = kernel_v1[(num_blocks,)](A, B, C, N, K, BLOCK_N, BLOCK_K, USE_DOT)
    return kernel if return_kernel else C


if __name__ == "__main__":
    M, N, K = 1, 7168, 8192
    A = torch.randn(M, K).bfloat16().cuda()
    B = torch.randn(N, K).bfloat16().cuda().T

    kernel = triton_v1(A, B, return_kernel=True)
    print(kernel.asm["ptx"])
