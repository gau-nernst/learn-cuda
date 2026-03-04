# decode only

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def mlp_triton_v2_kernel(
    x_ptr,  # (hidden_dim)
    norm_ptr,  # (hidden_dim)
    w13_ptr,  # (mlp_dim * 2, hidden_dim)
    w2_ptr,  # (hidden_dim, mlp_dim)
    tmp_ptr,  # (batch_size, mlp_dim) - for w13
    flag_ptr,
    batch_size,
    hidden_dim: tl.constexpr,
    mlp_dim: tl.constexpr,
    # currently using the same hparams for the 1st and 2nd matmul,
    # which is probably not optimal
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    raw_pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    # do input RMS norm on all threadblocks.
    # duplicate work, but avoid grid-wide sync.
    # assume hidden_dim is small
    offs = tl.arange(0, hidden_dim)
    x = tl.load(x_ptr + offs).to(tl.float32)
    norm = tl.load(norm_ptr + offs).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) * (1 / hidden_dim) + 1e-6
    x_normed = x * norm * tl.rsqrt(mean_sq)

    # persistent kernel
    w1_ptr = w13_ptr
    w3_ptr = w13_ptr + mlp_dim * hidden_dim

    for pid_n in range(raw_pid, mlp_dim // BLOCK_N, num_pids):
        # one-shot
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        w1_ptrs = w1_ptr + (offs_n[:, None] * hidden_dim + offs[None, :])
        w3_ptrs = w3_ptr + (offs_n[:, None] * hidden_dim + offs[None, :])

        w1 = tl.load(w1_ptrs)  # [BLOCK_N, hidden_dim]
        w3 = tl.load(w3_ptrs)

        acc1 = tl.sum(x_normed * w1, axis=1)  # [BLOCK_N]
        acc3 = tl.sum(x_normed * w3, axis=1)
        acc = acc1 * acc3 * tl.sigmoid(acc1)

        tmp_ptrs = tmp_ptr + offs_n
        tl.store(tmp_ptrs, acc)

    # grid-wide sync
    tl.atomic_add(flag_ptr, 1, sem="release", scope="gpu")
    while tl.atomic_add(flag_ptr, 0, sem="acquire", scope="gpu") != num_pids:
        pass

    # persistent kernel
    for pid_n in range(raw_pid, hidden_dim // BLOCK_N, num_pids):
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        tmp_ptrs = tmp_ptr + offs_k
        w2_ptrs = w2_ptr + (offs_n[:, None] * mlp_dim + offs_k[None, :])

        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

        for _ in range(mlp_dim // BLOCK_K):
            TMP = tl.load(tmp_ptrs)  # [BLOCK_K]
            W2 = tl.load(w2_ptrs)  # [BLOCK_N, BLOCK_K]
            acc += TMP * W2

            tmp_ptrs += BLOCK_K
            w2_ptrs += BLOCK_K

        acc = tl.sum(acc, axis=1)  # [BLOCK_N]

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc += tl.load(x_ptr + offs_n)
        tl.store(x_ptr + offs_n, acc)

    # signal done. last pid issues reset.
    before = tl.atomic_add(flag_ptr + 1, 1, sem="release", scope="gpu")
    if before == num_pids - 1:
        for i in range(2):
            tl.store(flag_ptr + i, 0)


_FLAG: Tensor | None = None


def mlp_triton_v2(x: Tensor, norm: Tensor, w13: Tensor, w2: Tensor):
    # lazily init _FLAG
    # NOTE: since this flag is shared module-wide, this function is not thread-safe
    global _FLAG
    if _FLAG is None:
        _FLAG = torch.zeros(2, dtype=torch.int32, device=x.device)

    batch_size, hidden_dim = x.shape
    _, mlp_dim = w2.shape
    assert batch_size == 1, "Only supports decode"

    tmp = x.new_empty(batch_size, mlp_dim)

    # NOTE: may want to limit num SMs used
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    BLOCK_N = 4
    BLOCK_K = 512

    mlp_triton_v2_kernel[(num_sms,)](
        x,
        norm,
        w13,
        w2,
        tmp,
        _FLAG,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return x
