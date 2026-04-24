import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def kernel_v1(
    A_ptr,  # [K]
    B_ptr,  # [N, K]
    C_ptr,  # [N]
    N,
    K,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + offs_k
    B_ptrs = B_ptr + (offs_n[:, None] * K + offs_k)

    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    for _ in range(tl.cdiv(K, BLOCK_K)):
        a = tl.load(A_ptrs, eviction_policy="evict_last")
        b = tl.load(B_ptrs, eviction_policy="evict_first")
        acc += a.to(tl.float32) * b.to(tl.float32)
        A_ptrs += BLOCK_K
        B_ptrs += BLOCK_K

    acc = tl.sum(acc, axis=1)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(C_ptr + offs_n, acc)


def triton_v1(A: Tensor, B: Tensor):
    K, N = B.shape
    C = A.new_empty(1, N)

    BLOCK_N, BLOCK_K = 32, 512
    num_blocks = triton.cdiv(N, BLOCK_N)

    kernel_v1[(num_blocks,)](A, B, C, N, K, BLOCK_N, BLOCK_K)
    return C
