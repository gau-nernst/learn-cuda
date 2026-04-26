# decode only

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton_utils import _grid_sync, _rms_norm, _spin_wait


@triton.jit
def attn_triton_v1_kernel(
    x_ptr,  # (dim)
    norm_ptr,  # (dim)
    kv_cache_ptr,  # (2, max_context, num_kv_heads, head_dim)
    wqkv_ptr,  # (qkv_dim, dim)
    q_norm_ptr,  # (head_dim)
    k_norm_ptr,  # (head_dim)
    rope_ptr,  # (head_dim * 2)
    wo_ptr,  # (dim, q_dim)
    tmp_ptr,  # (dim + qkv_dim)
    flag_ptr,
    position,
    max_context,
    dim: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_ATTN: tl.constexpr,
    head_dim: tl.constexpr = 128,
):
    raw_pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    qkv_dim: tl.constexpr = (num_heads + num_kv_heads * 2) * head_dim
    q_dim: tl.constexpr = num_heads * head_dim
    kv_dim: tl.constexpr = num_kv_heads * head_dim

    if raw_pid == 0:
        _rms_norm(x_ptr, norm_ptr, tmp_ptr, dim)
        tl.debug_barrier()
        tl.atomic_add(flag_ptr, 1, sem="release", scope="gpu")
    else:
        _spin_wait(flag_ptr, 1)

    # QKV projection
    for pid_n in range(raw_pid, qkv_dim // BLOCK_N, num_pids):
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        tmp_ptrs = tmp_ptr + offs_k
        wqkv_ptrs = wqkv_ptr + (offs_n[:, None] * dim + offs_k[None, :])

        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

        for _ in range(dim // BLOCK_K):
            x_normed = tl.load(tmp_ptrs).to(tl.float32)
            wqkv = tl.load(wqkv_ptrs).to(tl.float32)  # [BLOCK_N, BLOCK_K]
            acc += x_normed * wqkv
            tmp_ptrs += BLOCK_K
            wqkv_ptrs += BLOCK_K

        # NOTE: since v doesn't have QK-norm or RoPE, we can write v directly to KV cache
        tl.store(tmp_ptr + (dim + offs_n), tl.sum(acc, axis=1))  # [BLOCK_N]

    _grid_sync(flag_ptr + 1, num_pids)

    q_ptr = tmp_ptr + dim
    k_ptr = q_ptr + q_dim
    v_ptr = k_ptr + kv_dim

    # QK norm
    if raw_pid < num_heads:
        q_head_ptr = q_ptr + raw_pid * head_dim
        _rms_norm(q_head_ptr, q_norm_ptr, q_head_ptr, head_dim, BLOCK_SIZE=head_dim)
        tl.atomic_add(flag_ptr + 2, 1, sem="release", scope="gpu")

    elif raw_pid < num_heads + num_kv_heads:
        k_head_ptr = k_ptr + (raw_pid - num_heads) * head_dim
        _rms_norm(k_head_ptr, k_norm_ptr, k_head_ptr, head_dim, BLOCK_SIZE=head_dim)
        tl.atomic_add(flag_ptr + 2, 1, sem="release", scope="gpu")

    _spin_wait(flag_ptr + 2, num_heads + num_kv_heads)

    # each threadblock = 1 attention head
    # this won't perform well for long context
    if raw_pid < num_heads:
        head_id = raw_pid
        kv_head_id = head_id // (num_heads // num_kv_heads)

        cos_ptr = rope_ptr
        sin_ptr = cos_ptr + head_dim

        # QK norm and RoPE
        offs_hdim = tl.arange(0, head_dim)
        v = tl.load(v_ptr + kv_head_id * head_dim + offs_hdim)

        offs_half = tl.arange(0, head_dim // 2)
        cos = tl.load(cos_ptr + offs_half).to(tl.float32)
        sin = tl.load(sin_ptr + offs_half).to(tl.float32)

        q_normed = tl.load(q_ptr + (head_id * head_dim + offs_hdim)).to(tl.float32)
        k_normed = tl.load(k_ptr + (kv_head_id * head_dim + offs_hdim)).to(tl.float32)

        q1, q2 = tl.split(tl.reshape(q_normed, (2, head_dim // 2)).T)  # [head_dim/2] each
        k1, k2 = tl.split(tl.reshape(k_normed, (2, head_dim // 2)).T)

        q_new = tl.reshape(tl.join(q1 * cos - q2 * sin, q1 * sin + q2 * cos).T, (head_dim,))
        k_new = tl.reshape(tl.join(k1 * cos - k2 * sin, k1 * sin + k2 * cos).T, (head_dim,))

        # pre-apply exp2 scaling and softmax scale
        q_new *= 1.4426950408889634 * (head_dim**-0.5)

        # update KV cache
        k_cache_ptr = kv_cache_ptr + kv_head_id * head_dim
        v_cache_ptr = kv_cache_ptr + max_context * kv_dim + kv_head_id * head_dim
        tl.store(k_cache_ptr + (position * kv_dim + offs_hdim), k_new)
        tl.store(v_cache_ptr + (position * kv_dim + offs_hdim), v)

        # iterate over KV cache
        offs_hdim = tl.arange(0, head_dim)
        offs_len = tl.arange(0, BLOCK_ATTN)
        k_cache_ptrs = k_cache_ptr + (offs_len[:, None] * kv_dim + offs_hdim)
        v_cache_ptrs = v_cache_ptr + (offs_len[:, None] * kv_dim + offs_hdim)

        # attention w/ online softmax
        max_s = tl.full((1,), float("-inf"), dtype=tl.float32)
        sum_exp = tl.zeros((BLOCK_ATTN,), dtype=tl.float32)
        o = tl.zeros((BLOCK_ATTN, head_dim), dtype=tl.float32)

        # unroll the last tile so that we don't do masking here
        num_iters = tl.cdiv(position + 1, BLOCK_ATTN) - 1
        for _ in range(num_iters):
            k_block = tl.load(k_cache_ptrs).to(tl.float32)  # [BLOCK_K, head_dim]
            v_block = tl.load(v_cache_ptrs).to(tl.float32)  # [BLOCK_K, head_dim]

            s = tl.sum(q_new * k_block, axis=1)  # [BLOCK_K]
            new_max_s = tl.maximum(max_s, tl.max(s, axis=0))
            rescale = tl.exp2(max_s - new_max_s)

            p = tl.exp2(s - new_max_s)  # [BLOCK_K]
            sum_exp = sum_exp * rescale + p
            o = o * rescale + p[:, None] * v_block
            max_s = new_max_s

            k_cache_ptrs += BLOCK_ATTN * kv_dim
            v_cache_ptrs += BLOCK_ATTN * kv_dim

        # last tile w/ masking
        mask = offs_len < position + 1 - num_iters * BLOCK_ATTN
        k_block = tl.load(k_cache_ptrs, mask[:, None], other=0.0).to(tl.float32)  # [BLOCK_K, head_dim]
        v_block = tl.load(v_cache_ptrs, mask[:, None], other=0.0).to(tl.float32)  # [BLOCK_K, head_dim]

        s = tl.sum(q_new * k_block, axis=1)  # [BLOCK_K]
        s = tl.where(mask, s, float("-inf"))
        new_max_s = tl.maximum(max_s, tl.max(s, axis=0))
        rescale = tl.exp2(max_s - new_max_s)

        p = tl.exp2(s - new_max_s)  # [BLOCK_K]
        sum_exp = sum_exp * rescale + p
        o = o * rescale + p[:, None] * v_block

        # store to the same tmp memory for query
        o = tl.sum(o, axis=0) / tl.sum(sum_exp)  # [head_dim]
        offs_hdim = tl.arange(0, head_dim)
        tl.store(q_ptr + head_id * head_dim + offs_hdim, o)

        tl.atomic_add(flag_ptr + 3, 1, sem="release", scope="gpu")

    _spin_wait(flag_ptr + 3, num_heads)

    # output projection
    for pid_n in range(raw_pid, dim // BLOCK_N, num_pids):
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        o_ptrs = q_ptr + offs_k
        wo_ptrs = wo_ptr + (offs_n[:, None] * q_dim + offs_k[None, :])

        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

        for _ in range(q_dim // BLOCK_K):
            o = tl.load(o_ptrs).to(tl.float32)  # [BLOCK_K]
            wo = tl.load(wo_ptrs).to(tl.float32)  # [BLOCK_N, BLOCK_K]
            acc += o * wo
            o_ptrs += BLOCK_K
            wo_ptrs += BLOCK_K

        acc = tl.sum(acc, axis=1)  # [BLOCK_N]

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc += tl.load(x_ptr + offs_n).to(tl.float32)
        tl.store(x_ptr + offs_n, acc)

    # signal done. last pid issues reset.
    before = tl.atomic_add(flag_ptr + 4, 1, sem="release", scope="gpu")
    if before == num_pids - 1:
        for i in range(5):
            tl.store(flag_ptr + i, 0)


_FLAG: Tensor | None = None


def attn_triton_v1(
    x: Tensor,  # (num_tokens, dim)
    norm: Tensor,  # (dim)
    kv_cache: Tensor,  # (2, max_context, num_kv_heads, head_dim)
    wqkv: Tensor,  # (qkv_dim, dim)
    q_norm: Tensor,
    k_norm: Tensor,
    rope: Tensor,  # (num_tokens, head_dim * 2)
    wo: Tensor,  # (dim, q_dim)
    position: int,
):
    # lazily init _FLAG
    # NOTE: since this flag is shared module-wide, this function is not thread-safe
    global _FLAG
    if _FLAG is None:
        _FLAG = torch.zeros(100, dtype=torch.int32, device=x.device)

    batch_size, dim = x.shape
    qkv_dim, _ = wqkv.shape
    _, max_context, num_kv_heads, head_dim = kv_cache.shape

    num_heads = qkv_dim // head_dim - num_kv_heads * 2
    assert batch_size == 1, "Only supports decode"

    tmp = x.new_empty(batch_size, dim + qkv_dim)

    # NOTE: may want to limit num SMs used
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    BLOCK_N = 16
    BLOCK_K = 512
    BLOCK_ATTN = 128

    attn_triton_v1_kernel[(num_sms,)](
        x,
        norm,
        kv_cache,
        wqkv,
        q_norm,
        k_norm,
        rope,
        wo,
        tmp,
        _FLAG,
        position=position,
        max_context=max_context,
        dim=dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        BLOCK_ATTN=BLOCK_ATTN,
        head_dim=head_dim,
        num_warps=8,
    )
    return x
