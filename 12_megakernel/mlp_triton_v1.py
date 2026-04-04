import torch
import triton
import triton.language as tl
from torch import Tensor
from triton_utils import _grid_sync, _rms_norm


@triton.jit
def _stage1_mainloop(
    tmp1_ptr,  # (batch_size, hidden_dim)
    w1_ptr,  # (mlp_dim, hidden_dim)
    w3_ptr,  # (mlp_dim, hidden_dim)
    tmp2_ptr,  # (batch_size, mlp_dim)
    pid,
    batch_size,
    hidden_dim: tl.constexpr,
    mlp_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    grid_n0: tl.constexpr = mlp_dim // BLOCK_N
    pid_m = pid // grid_n0
    pid_n = pid % grid_n0

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % batch_size
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    tmp1_ptrs = tmp1_ptr + (offs_m[:, None] * hidden_dim + offs_k[None, :])
    w1_ptrs = w1_ptr + (offs_n[None, :] * hidden_dim + offs_k[:, None])
    w3_ptrs = w3_ptr + (offs_n[None, :] * hidden_dim + offs_k[:, None])

    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(hidden_dim // BLOCK_K):
        TMP1 = tl.load(tmp1_ptrs)  # [BLOCK_M, BLOCK_K]
        W1 = tl.load(w1_ptrs)  # [BLOCK_K, BLOCK_N2]
        acc1 = tl.dot(TMP1, W1, acc=acc1)

        W3 = tl.load(w3_ptrs)  # [BLOCK_K, BLOCK_N2]
        acc3 = tl.dot(TMP1, W3, acc=acc3)

        tmp1_ptrs += BLOCK_K
        w1_ptrs += BLOCK_K
        w3_ptrs += BLOCK_K

    acc = acc1 * acc3 * tl.sigmoid(acc1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tmp2_ptrs = tmp2_ptr + (offs_m[:, None] * mlp_dim + offs_n[None, :])
    tl.store(tmp2_ptrs, acc, mask=offs_m[:, None] < batch_size)  # [BLOCK_M, BLOCK_N]


@triton.jit
def _stage2_mainloop(
    tmp2_ptr,  # (batch_size, mlp_dim)
    w2_ptr,  # (hidden_dim, mlp_dim)
    x_ptr,  # (batch_size, hidden_dim)
    pid,
    batch_size,
    hidden_dim: tl.constexpr,
    mlp_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    grid_n1: tl.constexpr = hidden_dim // BLOCK_N
    pid_m = pid // grid_n1
    pid_n = pid % grid_n1

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % batch_size
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    tmp2_ptrs = tmp2_ptr + (offs_m[:, None] * mlp_dim + offs_k[None, :])
    w2_ptrs = w2_ptr + (offs_n[None, :] * mlp_dim + offs_k[:, None])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(mlp_dim // BLOCK_K):
        TMP = tl.load(tmp2_ptrs)  # [BLOCK_M, BLOCK_K]
        W2 = tl.load(w2_ptrs)  # [BLOCK_K, BLOCK_N]
        acc = tl.dot(TMP, W2, acc=acc)

        tmp2_ptrs += BLOCK_K
        w2_ptrs += BLOCK_K

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs = offs_m[:, None] * hidden_dim + offs_n[None, :]
    mask = offs_m[:, None] < batch_size

    acc += tl.load(x_ptr + offs, mask=mask)
    tl.store(x_ptr + offs, acc, mask=mask)  # [BLOCK_M, BLOCK_N]


# TODO:
# - autotune?
# - add cache hints
# - swap A/B when M is small?
@triton.jit
def mlp_triton_v1_kernel(
    x_ptr,  # (batch_size, hidden_dim)
    norm_ptr,  # (hidden_dim)
    w13_ptr,  # (mlp_dim * 2, hidden_dim)
    w2_ptr,  # (hidden_dim, mlp_dim)
    tmp1_ptr,  # (batch_size, hidden_dim) - for rms norm
    tmp2_ptr,  # (batch_size, mlp_dim) - for w13
    flag_ptr,
    batch_size,
    hidden_dim: tl.constexpr,
    mlp_dim: tl.constexpr,
    # currently using the same hparams for the 1st and 2nd matmul,
    # which is probably not optimal
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    raw_pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    for batch_id in range(raw_pid, batch_size, num_pids):
        _rms_norm(x_ptr + batch_id * hidden_dim, norm_ptr, tmp1_ptr + batch_id * hidden_dim, hidden_dim)
    _grid_sync(flag_ptr, num_pids)

    # NOTE: typically mlp_dim is 2-5x hidden_dim
    # hence, the 1st MMA has many output tiles, with short mainloop
    #   -> larger BM/BN, shallower pipeline
    # while the 2nd MMA has few output tiles, but long mainloop
    #   -> smaller BM/BN, deeper pipeline

    # persistent kernel
    grid_m = tl.cdiv(batch_size, BLOCK_M)
    grid_n0: tl.constexpr = mlp_dim // BLOCK_N
    num_tiles0 = grid_m * grid_n0

    w1_ptr = w13_ptr
    w3_ptr = w13_ptr + mlp_dim * hidden_dim

    for pid in range(raw_pid, num_tiles0, num_pids):
        _stage1_mainloop(
            tmp1_ptr, w1_ptr, w3_ptr, tmp2_ptr, pid, batch_size, hidden_dim, mlp_dim, BLOCK_M, BLOCK_N, BLOCK_K
        )
    _grid_sync(flag_ptr + 1, num_pids)

    # persistent kernel
    grid_n1: tl.constexpr = hidden_dim // BLOCK_N
    num_tiles1 = grid_m * grid_n1

    for pid in range(raw_pid, num_tiles1, num_pids):
        _stage2_mainloop(tmp2_ptr, w2_ptr, x_ptr, pid, batch_size, hidden_dim, mlp_dim, BLOCK_M, BLOCK_N, BLOCK_K)

    # signal done. last pid issues reset.
    before = tl.atomic_add(flag_ptr + 2, 1, sem="release", scope="gpu")
    if before == num_pids - 1:
        for i in range(3):
            tl.store(flag_ptr + i, 0)


@triton.jit
def mlp_triton_v1_rms_norm_kernel(
    x_ptr,  # (batch_size, hidden_dim)
    norm_ptr,  # (hidden_dim)
    tmp1_ptr,  # (batch_size, hidden_dim)
    hidden_dim: tl.constexpr,
):
    pid = tl.program_id(0)
    _rms_norm(x_ptr + pid * hidden_dim, norm_ptr, tmp1_ptr + pid * hidden_dim, hidden_dim)


@triton.jit
def mlp_triton_v1_stage1_kernel(
    tmp1_ptr,  # (batch_size, hidden_dim)
    w13_ptr,  # (mlp_dim * 2, hidden_dim)
    tmp2_ptr,  # (batch_size, mlp_dim)
    batch_size,
    hidden_dim: tl.constexpr,
    mlp_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    w1_ptr = w13_ptr
    w3_ptr = w13_ptr + mlp_dim * hidden_dim
    _stage1_mainloop(
        tmp1_ptr, w1_ptr, w3_ptr, tmp2_ptr, pid, batch_size, hidden_dim, mlp_dim, BLOCK_M, BLOCK_N, BLOCK_K
    )


@triton.jit
def mlp_triton_v1_stage2_kernel(
    tmp2_ptr,  # (batch_size, mlp_dim)
    w2_ptr,  # (hidden_dim, mlp_dim)
    x_ptr,  # (batch_size, hidden_dim)
    batch_size,
    hidden_dim: tl.constexpr,
    mlp_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    _stage2_mainloop(tmp2_ptr, w2_ptr, x_ptr, pid, batch_size, hidden_dim, mlp_dim, BLOCK_M, BLOCK_N, BLOCK_K)


_FLAG: Tensor | None = None


def _heuristics(batch_size: int, hidden_dim: int, mlp_dim: int):
    if batch_size > 256:
        BLOCK_M = 128
    else:
        # BLOCK_M needs to be at least 16 to activate MMA pipeline(?)
        BLOCK_M = min(max(triton.next_power_of_2(batch_size), 16), 64)

    gpu_name = torch.cuda.get_device_name()

    if "5090" in gpu_name:
        # tuned a bit for 5090
        # small BLOCK_N since 5090 has a lot of SMs
        # may want to increase this if hidden_dim is larger
        BLOCK_N = 32

        if BLOCK_M == 16:
            BLOCK_K, num_stages = 32, 12
        elif BLOCK_M == 32:
            BLOCK_K, num_stages = 32, 10
        else:
            BLOCK_K, num_stages = 64, 4

    else:
        # tuned a bit for H200
        BLOCK_N = 64
        BLOCK_K = 64
        num_stages = 4

    return BLOCK_M, BLOCK_N, BLOCK_K, num_stages


def mlp_triton_v1(x: Tensor, norm: Tensor, w13: Tensor, w2: Tensor):
    # lazily init _FLAG
    # NOTE: since this flag is shared module-wide, this function is not thread-safe
    global _FLAG
    if _FLAG is None:
        _FLAG = torch.zeros(3, dtype=torch.int32, device=x.device)

    batch_size, hidden_dim = x.shape
    _, mlp_dim = w2.shape

    tmp1 = x.new_empty(batch_size, hidden_dim)
    tmp2 = x.new_empty(batch_size, mlp_dim)

    # NOTE: may want to limit num SMs used
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    BLOCK_M, BLOCK_N, BLOCK_K, num_stages = _heuristics(batch_size, hidden_dim, mlp_dim)

    mlp_triton_v1_kernel[(num_sms,)](
        x,
        norm,
        w13,
        w2,
        tmp1,
        tmp2,
        _FLAG,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_stages=num_stages,
    )
    return x


def mlp_triton_v1_2stage(x: Tensor, norm: Tensor | None, w13: Tensor, w2: Tensor):
    batch_size, hidden_dim = x.shape
    _, mlp_dim = w2.shape

    tmp1 = x.new_empty(batch_size, hidden_dim)
    tmp2 = x.new_empty(batch_size, mlp_dim)

    BLOCK_M, BLOCK_N, BLOCK_K, num_stages = _heuristics(batch_size, hidden_dim, mlp_dim)
    grid_m = triton.cdiv(batch_size, BLOCK_M)

    mlp_triton_v1_rms_norm_kernel[(batch_size,)](x, norm, tmp1, hidden_dim)
    mlp_triton_v1_stage1_kernel[(grid_m * (mlp_dim // BLOCK_N),)](
        tmp1,
        w13,
        tmp2,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_stages=num_stages,
    )
    mlp_triton_v1_stage2_kernel[(grid_m * (hidden_dim // BLOCK_N),)](
        tmp2,
        w2,
        x,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_stages=num_stages,
    )
    return x
