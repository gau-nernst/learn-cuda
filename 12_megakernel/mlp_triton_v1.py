import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def load_flag(ptr):
    return tl.inline_asm_elementwise(
        "ld.acquire.gpu.global.b32 $0, [$1];",
        constraints="=r,l",
        args=[ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


# TODO:
# - support bs not a multiple of BLOCK_M, especially bs=1
# - autotune?
# - add cache hints
@triton.jit
def mlp_triton_v1_kernel(
    x_ptr,  # (batch_size, hidden_dim)
    w1_ptr,  # (mlp_dim, hidden_dim)
    w3_ptr,  # (mlp_dim, hidden_dim)
    tmp_ptr,  # (batch_size, mlp_dim)
    w2_ptr,  # (hidden_dim, mlp_dim)
    o_ptr,  # (batch_size, hidden_dim)
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

    # persistent kernel
    grid_m = batch_size // BLOCK_M
    grid_n0: tl.constexpr = mlp_dim // BLOCK_N
    num_tiles0 = grid_m * grid_n0

    for pid in range(raw_pid, num_tiles0, num_pids):
        pid_m = pid // grid_n0
        pid_n = pid % grid_n0

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        x_ptrs = x_ptr + (offs_m[:, None] * hidden_dim + offs_k[None, :])
        w1_ptrs = w1_ptr + (offs_n[None, :] * hidden_dim + offs_k[:, None])
        w3_ptrs = w3_ptr + (offs_n[None, :] * hidden_dim + offs_k[:, None])

        acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for _ in range(hidden_dim // BLOCK_K):
            X = tl.load(x_ptrs)  # [BLOCK_M, BLOCK_K]
            W1 = tl.load(w1_ptrs)  # [BLOCK_K, BLOCK_N2]
            W3 = tl.load(w3_ptrs)  # [BLOCK_K, BLOCK_N2]

            acc1 = tl.dot(X, W1, acc=acc1)
            acc3 = tl.dot(X, W3, acc=acc3)

            x_ptrs += BLOCK_K
            w1_ptrs += BLOCK_K
            w3_ptrs += BLOCK_K

        acc = acc1 * acc3 * tl.sigmoid(acc1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        tmp_ptrs = tmp_ptr + (offs_m[:, None] * mlp_dim + offs_n[None, :])
        tl.store(tmp_ptrs, acc)  # [BLOCK_M, BLOCK_N]

    # signal done
    tl.atomic_add(flag_ptr, 1, sem="release", scope="gpu")

    # spin
    # NOTE: we can prefetch W2 during this time
    while load_flag(flag_ptr) != num_pids:
        pass

    # persistent kernel
    grid_n1: tl.constexpr = hidden_dim // BLOCK_N
    num_tiles1 = grid_m * grid_n1

    for pid in range(raw_pid, num_tiles1, num_pids):
        pid_m = pid // grid_n1
        pid_n = pid % grid_n1

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        tmp_ptrs = tmp_ptr + (offs_m[:, None] * mlp_dim + offs_k[None, :])
        w2_ptrs = w2_ptr + (offs_n[None, :] * mlp_dim + offs_k[:, None])

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for _ in range(mlp_dim // BLOCK_K):
            TMP = tl.load(tmp_ptrs)  # [BLOCK_M, BLOCK_K]
            W2 = tl.load(w2_ptrs)  # [BLOCK_K, BLOCK_N]
            acc = tl.dot(TMP, W2, acc=acc)

            tmp_ptrs += BLOCK_K
            w2_ptrs += BLOCK_K

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        o_ptrs = o_ptr + (offs_m[:, None] * hidden_dim + offs_n[None, :])
        tl.store(o_ptrs, acc)  # [BLOCK_M, BLOCK_N]

    # signal done
    before = tl.atomic_add(flag_ptr + 1, 1, sem="release", scope="gpu")

    # last pid issues reset
    if before == num_pids - 1:
        tl.store(flag_ptr, 0)
        tl.store(flag_ptr + 1, 0)


_FLAG: Tensor | None = None


def mlp_triton_v1(x: Tensor, w1: Tensor, w3: Tensor, w2: Tensor):
    # lazily init _FLAG
    # NOTE: since this flag is shared module-wide, this function is not thread-safe
    global _FLAG
    if _FLAG is None:
        _FLAG = torch.zeros(2, dtype=torch.int32, device=x.device)

    batch_size, hidden_dim = x.shape
    mlp_dim, _ = w1.shape

    out = torch.empty_like(x)
    tmp = x.new_empty(batch_size, mlp_dim)

    # NOTE: may want to limit num SMs used
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    mlp_triton_v1_kernel[(num_sms,)](
        x,
        w1,
        w3,
        tmp,
        w2,
        out,
        _FLAG,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=64,
        num_stages=4,
    )
    return out
