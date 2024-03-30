import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": size}) for size in (16, 32, 64)],
    key=["m", "n", "k"],
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, m, n, k, BLOCK_SIZE: tl.constexpr):
    # A: (m, k), B: (k, n), C: (m, n)
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    offsets_m = pid0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets_n = pid1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # each program calculate a block in C
    c = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # iterate over inner dim k
    for block_id_k in range(0, tl.cdiv(k, BLOCK_SIZE)):
        offsets_k = block_id_k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        a_ptrs = a_ptr + offsets_m[:, None] * k + offsets_k[None, :]
        b_ptrs = b_ptr + offsets_k[:, None] * n + offsets_n[None, :]

        a = tl.load(a_ptrs, mask=(offsets_m[:, None] < m) & (offsets_k[None, :] < k), other=0.0)
        b = tl.load(b_ptrs, mask=(offsets_k[:, None] < k) & (offsets_n[None, :] < n), other=0.0)

        c += tl.dot(a, b, allow_tf32=False)

    c_ptrs = c_ptr + offsets_m[:, None] * n + offsets_n[None, :]
    tl.store(c_ptrs, c, mask=(offsets_m[:, None] < m) & (offsets_n[None, :] < n))


def matmul(a: Tensor, b: Tensor):
    assert a.is_cuda and b.is_cuda
    assert a.shape[1] == b.shape[0]
    assert a.is_contiguous() and b.is_contiguous()

    out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)

    def grid(meta):
        return (
            triton.cdiv(a.shape[0], meta["BLOCK_SIZE"]),
            triton.cdiv(b.shape[1], meta["BLOCK_SIZE"]),
        )

    matmul_kernel[grid](a, b, out, a.shape[0], b.shape[1], a.shape[1])
    return out


if __name__ == "__main__":
    n = 1000

    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        print(dtype)
        a = torch.randn((n, n), device="cuda", dtype=dtype)
        b = torch.randn((n, n), device="cuda", dtype=dtype)
        out_torch = a @ b
        out_triton = matmul(a, b)

        try:
            torch.testing.assert_close(out_triton, out_torch)
        except Exception as e:
            print(e)
