# decode only

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton_utils import _grid_sync, _rms_norm, _spin_wait


@triton.jit
def mlp_triton_v2_kernel(
    x_ptr,  # (hidden_dim)
    norm_ptr,  # (hidden_dim)
    w13_ptr,  # (mlp_dim * 2, hidden_dim)
    w2_ptr,  # (hidden_dim, mlp_dim)
    tmp_ptr,  # (hidden_dim + mlp_dim) - for w13
    flag_ptr,
    hidden_dim: tl.constexpr,
    mlp_dim: tl.constexpr,
    # currently using the same hparams for the 1st and 2nd matmul,
    # which is probably not optimal
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    raw_pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    # first pid performs RMS norm
    # NOTE: we can avoid grid-sync here by performing rms norm by all threadblocks,
    # which writes data to L2. and issue __syncthreads() / bar.sync so that subsequent reads
    # can see the rms norm result in L2 cache.
    if raw_pid == 0:
        _rms_norm(x_ptr, norm_ptr, tmp_ptr, hidden_dim)
        tl.atomic_add(flag_ptr, 1, sem="release", scope="gpu")
    else:
        _spin_wait(flag_ptr, 1)

    # persistent kernel
    w1_ptr = w13_ptr
    w3_ptr = w13_ptr + mlp_dim * hidden_dim

    for pid_n in range(raw_pid, mlp_dim // BLOCK_N, num_pids):
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        tmp_ptrs = tmp_ptr + offs_k
        w1_ptrs = w1_ptr + (offs_n[:, None] * hidden_dim + offs_k[None, :])
        w3_ptrs = w3_ptr + (offs_n[:, None] * hidden_dim + offs_k[None, :])

        acc1 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

        for _ in range(hidden_dim // BLOCK_K):
            x_normed = tl.load(tmp_ptrs)
            w1 = tl.load(w1_ptrs)  # [BLOCK_N, BLOCK_K]
            w3 = tl.load(w3_ptrs)

            acc1 += x_normed * w1
            acc3 += x_normed * w3

            tmp_ptrs += BLOCK_K
            w1_ptrs += BLOCK_K
            w3_ptrs += BLOCK_K

        acc1 = tl.sum(acc1, axis=1)  # [BLOCK_N]
        acc3 = tl.sum(acc3, axis=1)
        acc = acc1 * acc3 * tl.sigmoid(acc1)
        tl.store(tmp_ptr + (hidden_dim + offs_n), acc)

    _grid_sync(flag_ptr + 1, num_pids)

    # persistent kernel
    for pid_n in range(raw_pid, hidden_dim // BLOCK_N, num_pids):
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        tmp_ptrs = tmp_ptr + (hidden_dim + offs_k)
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
    before = tl.atomic_add(flag_ptr + 2, 1, sem="release", scope="gpu")
    if before == num_pids - 1:
        for i in range(3):
            tl.store(flag_ptr + i, 0)


_FLAG: Tensor | None = None


def mlp_triton_v2(x: Tensor, norm: Tensor, w13: Tensor, w2: Tensor):
    # lazily init _FLAG
    # NOTE: since this flag is shared module-wide, this function is not thread-safe
    global _FLAG
    if _FLAG is None:
        _FLAG = torch.zeros(100, dtype=torch.int32, device=x.device)

    batch_size, hidden_dim = x.shape
    _, mlp_dim = w2.shape
    assert batch_size == 1, "Only supports decode"

    tmp = x.new_empty(batch_size, hidden_dim + mlp_dim)

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
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return x
