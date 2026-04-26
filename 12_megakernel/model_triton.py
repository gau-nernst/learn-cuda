# decode only
# attn_triton_v1 + mlp_triton_v2

import time

import reference
import torch
import triton
import triton.language as tl
from attn_triton_v2 import qk_norm_rope
from torch import Tensor
from triton_utils import _grid_sync, _rms_norm, _spin_wait


@triton.jit
def model_triton_kernel(
    input_ids_ptr,  # ()
    x_ptr,  # (dim)
    input_embeds_ptr,  # (vocab_size, dim)
    num_layers,
    # attention
    attn_norm_ptr,  # (num_layers, dim)
    kv_cache_ptr,  # (num_layers, 2, max_context, num_kv_heads, head_dim)
    wqkv_ptr,  # (num_layers, qkv_dim, dim)
    q_norm_ptr,  # (num_layers, head_dim)
    k_norm_ptr,  # (num_layers, head_dim)
    rope_ptr,  # (head_dim * 2)
    wo_ptr,  # (num_layers, dim, q_dim)
    attn_tmp_ptr,  # (dim + q_dim) - BF16
    attn_tmp2_ptr,  # (num_pids_per_head, num_heads, head_dim + 2) - FP32
    position,
    max_context,
    # mlp
    mlp_norm_ptr,  # (num_layers, hidden_dim)
    w13_ptr,  # (num_layers, mlp_dim * 2, hidden_dim)
    w2_ptr,  # (num_layers, hidden_dim, mlp_dim)
    mlp_tmp_ptr,  # (mlp_dim) - for w13
    # the rest
    norm_ptr,  # (dim)
    lm_head_ptr,  # (vocab_size, dim)
    temperature,
    tmp_ptr,  # (num_pids * 2)
    token_ptr,
    flag_ptr,
    rng,
    vocab_size,
    dim: tl.constexpr,
    mlp_dim: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_pids_per_head: tl.constexpr,
    #
    BLOCK_K: tl.constexpr,
    BLOCK_N_ATTN: tl.constexpr,
    BLOCK_ATTN: tl.constexpr,
    #
    BLOCK_N_W13: tl.constexpr,
    BLOCK_N_W2: tl.constexpr,
    #
    BLOCK_PIDS: tl.constexpr,
    head_dim: tl.constexpr = 128,
):
    raw_pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    qkv_dim: tl.constexpr = (num_heads + num_kv_heads * 2) * head_dim
    q_dim: tl.constexpr = num_heads * head_dim
    kv_dim: tl.constexpr = num_kv_heads * head_dim

    # load input embedding on all threadblocks to avoid grid-wide sync
    input_id = tl.load(input_ids_ptr)
    for k in range(tl.cdiv(dim, 1024)):
        offs = k * 1024 + tl.arange(0, 1024)
        x = tl.load(input_embeds_ptr + (input_id * dim + offs))
        tl.store(x_ptr + offs, x, mask=offs < dim)
    tl.debug_barrier()

    q_tmp_ptr = attn_tmp_ptr + dim

    for layer_id in range(num_layers):
        if raw_pid == 0:
            _rms_norm(x_ptr, attn_norm_ptr, attn_tmp_ptr, dim)
            tl.debug_barrier()
            tl.atomic_add(flag_ptr + layer_id * 9, 1, sem="release", scope="gpu")
        else:
            _spin_wait(flag_ptr + layer_id * 9, 1)

        # QKV projection
        for pid_n in range(raw_pid, qkv_dim // BLOCK_N_ATTN, num_pids):
            offs_n = pid_n * BLOCK_N_ATTN + tl.arange(0, BLOCK_N_ATTN)
            offs_k = tl.arange(0, BLOCK_K)

            tmp_ptrs = attn_tmp_ptr + offs_k
            wqkv_ptrs = wqkv_ptr + (offs_n[:, None] * dim + offs_k)

            acc = tl.zeros((BLOCK_N_ATTN, BLOCK_K), dtype=tl.float32)

            for _ in range(dim // BLOCK_K):
                x_normed = tl.load(tmp_ptrs).to(tl.float32)
                wqkv = tl.load(wqkv_ptrs).to(tl.float32)  # [BLOCK_N, BLOCK_K]
                acc += x_normed * wqkv
                tmp_ptrs += BLOCK_K
                wqkv_ptrs += BLOCK_K

            acc = tl.sum(acc, axis=1)

            # write q to tmp buffer, kv to KV cache
            if pid_n * BLOCK_N_ATTN < q_dim:
                tl.store(q_tmp_ptr + offs_n, acc)
            elif pid_n * BLOCK_N_ATTN < q_dim + kv_dim:
                tl.store(kv_cache_ptr + (position * kv_dim + (offs_n - q_dim)), acc)
            else:
                tl.store(kv_cache_ptr + ((max_context + position) * kv_dim + (offs_n - (q_dim + kv_dim))), acc)

        _grid_sync(flag_ptr + (layer_id * 9 + 1), num_pids)

        # QK norm
        if raw_pid < num_heads:
            q_head_ptr = q_tmp_ptr + raw_pid * head_dim
            qk_norm_rope(q_head_ptr, q_norm_ptr, rope_ptr, head_dim)
            tl.atomic_add(flag_ptr + (layer_id * 9 + 2), 1, sem="release", scope="gpu")

        elif raw_pid < num_heads + num_kv_heads:
            k_head_id = raw_pid - num_heads
            k_head_ptr = kv_cache_ptr + (position * num_kv_heads + k_head_id) * head_dim
            qk_norm_rope(k_head_ptr, k_norm_ptr, rope_ptr, head_dim)
            tl.atomic_add(flag_ptr + (layer_id * 9 + 2), 1, sem="release", scope="gpu")

        _spin_wait(flag_ptr + (layer_id * 9 + 2), num_heads + num_kv_heads)

        # each threadblock = 1 attention head
        if raw_pid < num_pids_per_head * num_heads:
            head_id = raw_pid % num_heads
            kv_head_id = head_id // (num_heads // num_kv_heads)

            offs_hdim = tl.arange(0, head_dim)
            q = tl.load(q_tmp_ptr + (head_id * head_dim + offs_hdim)).to(tl.float32)
            q *= 1.4426950408889634 * (head_dim**-0.5)  # apply exp2 scaling and softmax scale

            k_cache_ptr = kv_cache_ptr + kv_head_id * head_dim
            v_cache_ptr = kv_cache_ptr + (max_context * num_kv_heads + kv_head_id) * head_dim

            # attention w/ online softmax
            max_s = tl.full((), float("-inf"), dtype=tl.float32)
            sum_exp = tl.zeros((BLOCK_ATTN,), dtype=tl.float32)
            o = tl.zeros((BLOCK_ATTN, head_dim), dtype=tl.float32)

            # iterate over KV cache
            num_kv_blocks = tl.cdiv(position + 1, BLOCK_ATTN)

            # handle last tile first, with masking
            if raw_pid // num_heads == num_pids_per_head - 1:
                kv_block = num_kv_blocks - 1
                offs_len = kv_block * BLOCK_ATTN + tl.arange(0, BLOCK_ATTN)
                mask = offs_len < position + 1
                k_cache_ptrs = k_cache_ptr + (offs_len[:, None] * kv_dim + offs_hdim)
                v_cache_ptrs = v_cache_ptr + (offs_len[:, None] * kv_dim + offs_hdim)
                k_block = tl.load(k_cache_ptrs, mask[:, None], other=0.0).to(tl.float32)  # [BLOCK_ATTN, head_dim]
                v_block = tl.load(v_cache_ptrs, mask[:, None], other=0.0).to(tl.float32)  # [BLOCK_ATTN, head_dim]

                s = tl.sum(q * k_block, axis=1)  # [BLOCK_ATTN]
                s = tl.where(mask, s, float("-inf"))
                new_max_s = tl.maximum(max_s, tl.max(s))
                rescale = tl.exp2(max_s - new_max_s)

                p = tl.exp2(s - new_max_s)  # [BLOCK_ATTN]
                sum_exp = sum_exp * rescale + p
                o = o * rescale + p[:, None] * v_block
                max_s = new_max_s

            # handle the remaining, no masking
            for kv_block in range(raw_pid // num_heads, num_kv_blocks - 1, num_pids_per_head):
                offs_len = kv_block * BLOCK_ATTN + tl.arange(0, BLOCK_ATTN)
                k_cache_ptrs = k_cache_ptr + (offs_len[:, None] * kv_dim + offs_hdim)
                v_cache_ptrs = v_cache_ptr + (offs_len[:, None] * kv_dim + offs_hdim)

                k_block = tl.load(k_cache_ptrs).to(tl.float32)  # [BLOCK_ATTN, head_dim]
                v_block = tl.load(v_cache_ptrs).to(tl.float32)  # [head_dim, BLOCK_ATTN]

                s = tl.sum(q * k_block, axis=1)  # [BLOCK_ATTN]
                new_max_s = tl.maximum(max_s, tl.max(s))
                rescale = tl.exp2(max_s - new_max_s)

                p = tl.exp2(s - new_max_s)  # [BLOCK_ATTN]
                sum_exp = sum_exp * rescale + p
                o = o * rescale + p[:, None] * v_block
                max_s = new_max_s

            # final reduction. store to tmp buffer
            sum_exp = tl.sum(sum_exp)  # [1]
            o = tl.sum(o, axis=0)  # [head_dim]

            o_tmp_ptr = attn_tmp2_ptr  # size: [num_pids_per_head, num_heads, head_dim]
            other_ptr = o_tmp_ptr + num_pids_per_head * num_heads * head_dim  # size: [num_pids_per_head, num_heads, 2]
            tl.store(o_tmp_ptr + (raw_pid * head_dim + offs_hdim), o)
            tl.store(other_ptr + (raw_pid * 2 + 0), sum_exp)
            tl.store(other_ptr + (raw_pid * 2 + 1), max_s)
            tl.atomic_add(flag_ptr + (layer_id * 9 + 3), 1, sem="release", scope="gpu")

            # combine attention states from all threadblocks
            if raw_pid < num_heads:
                _spin_wait(flag_ptr + (layer_id * 9 + 3), num_pids_per_head * num_heads)  # all blocks finish

                BLOCK_SIZE: tl.constexpr = triton.next_power_of_2(num_pids_per_head)
                head_id = raw_pid
                offs_yo = tl.arange(0, BLOCK_SIZE)
                mask = offs_yo < num_pids_per_head

                o_tmp_ptrs = o_tmp_ptr + (offs_yo[:, None] * num_heads * head_dim + head_id * head_dim + offs_hdim)
                o = tl.load(o_tmp_ptrs, mask[:, None], other=0.0)  # [BLOCK_SIZE, head_dim]

                sum_exp = tl.load(other_ptr + (offs_yo * num_heads + head_id) * 2, mask, other=0.0)  # [BLOCK_SIZE]
                max_s_ = tl.load(
                    other_ptr + (offs_yo * num_heads + head_id) * 2 + 1, mask, other=float("-inf")
                )  # [BLOCK_SIZE]
                rescale = tl.exp2(max_s_ - tl.max(max_s_))  # [BLOCK_SIZE]

                sum_exp = tl.sum(sum_exp * rescale)
                o = tl.sum(o * rescale[:, None], axis=0)  # [head_dim]

                # store to the same tmp memory for query
                o /= sum_exp
                tl.store(q_tmp_ptr + head_id * head_dim + offs_hdim, o)
                tl.atomic_add(flag_ptr + (layer_id * 9 + 4), 1, sem="release", scope="gpu")

        _spin_wait(flag_ptr + (layer_id * 9 + 4), num_heads)

        # output projection
        for pid_n in range(raw_pid, dim // BLOCK_N_ATTN, num_pids):
            offs_n = pid_n * BLOCK_N_ATTN + tl.arange(0, BLOCK_N_ATTN)
            offs_k = tl.arange(0, BLOCK_K)

            o_ptrs = q_tmp_ptr + offs_k
            wo_ptrs = wo_ptr + (offs_n[:, None] * q_dim + offs_k)

            acc = tl.zeros((BLOCK_N_ATTN, BLOCK_K), dtype=tl.float32)

            for _ in range(q_dim // BLOCK_K):
                o = tl.load(o_ptrs).to(tl.float32)  # [BLOCK_K]
                wo = tl.load(wo_ptrs).to(tl.float32)  # [BLOCK_N, BLOCK_K]
                acc += o * wo
                o_ptrs += BLOCK_K
                wo_ptrs += BLOCK_K

            acc = tl.sum(acc, axis=1)  # [BLOCK_N]

            offs_n = pid_n * BLOCK_N_ATTN + tl.arange(0, BLOCK_N_ATTN)
            acc += tl.load(x_ptr + offs_n).to(tl.float32)
            tl.store(x_ptr + offs_n, acc)

        _grid_sync(flag_ptr + (layer_id * 9 + 5), num_pids)

        # start of MLP
        if raw_pid == 0:
            _rms_norm(x_ptr, mlp_norm_ptr, mlp_tmp_ptr, dim)
            tl.debug_barrier()
            tl.atomic_add(flag_ptr + (layer_id * 9 + 6), 1, sem="release", scope="gpu")
        else:
            _spin_wait(flag_ptr + (layer_id * 9 + 6), 1)

        # w13 projection
        w1_ptr = w13_ptr
        w3_ptr = w1_ptr + mlp_dim * dim

        tl.static_assert(mlp_dim % BLOCK_N_W13 == 0)
        for pid_n in range(raw_pid, mlp_dim // BLOCK_N_W13, num_pids):
            offs_n = pid_n * BLOCK_N_W13 + tl.arange(0, BLOCK_N_W13)
            offs_k = tl.arange(0, BLOCK_K)

            tmp_ptrs = mlp_tmp_ptr + offs_k
            w1_ptrs = w1_ptr + (offs_n[:, None] * dim + offs_k)
            w3_ptrs = w3_ptr + (offs_n[:, None] * dim + offs_k)

            acc1 = tl.zeros((BLOCK_N_W13, BLOCK_K), dtype=tl.float32)
            acc3 = tl.zeros((BLOCK_N_W13, BLOCK_K), dtype=tl.float32)

            tl.static_assert(dim % BLOCK_K == 0)
            for _ in range(dim // BLOCK_K):
                x_normed = tl.load(tmp_ptrs).to(tl.float32)
                w1 = tl.load(w1_ptrs, eviction_policy="evict_first").to(tl.float32)  # [BLOCK_N, BLOCK_K]
                w3 = tl.load(w3_ptrs, eviction_policy="evict_first").to(tl.float32)
                acc1 += x_normed * w1
                acc3 += x_normed * w3
                tmp_ptrs += BLOCK_K
                w1_ptrs += BLOCK_K
                w3_ptrs += BLOCK_K

            acc1 = tl.sum(acc1, axis=1)  # [BLOCK_N]
            acc3 = tl.sum(acc3, axis=1)
            acc = acc1 * acc3 * tl.sigmoid(acc1)
            tl.store(mlp_tmp_ptr + (dim + offs_n), acc)

        _grid_sync(flag_ptr + (layer_id * 9 + 7), num_pids)

        # persistent kernel
        tl.static_assert(dim % BLOCK_N_W2 == 0)
        for pid_n in range(raw_pid, dim // BLOCK_N_W2, num_pids):
            offs_n = pid_n * BLOCK_N_W2 + tl.arange(0, BLOCK_N_W2)
            offs_k = tl.arange(0, BLOCK_K)

            mlp_tmp_ptrs = mlp_tmp_ptr + (dim + offs_k)
            w2_ptrs = w2_ptr + (offs_n[:, None] * mlp_dim + offs_k)

            acc = tl.zeros((BLOCK_N_W2, BLOCK_K), dtype=tl.float32)

            tl.static_assert(mlp_dim % BLOCK_K == 0)
            for _ in range(mlp_dim // BLOCK_K):
                TMP = tl.load(mlp_tmp_ptrs).to(tl.float32)  # [BLOCK_K]
                W2 = tl.load(w2_ptrs, eviction_policy="evict_first").to(tl.float32)  # [BLOCK_N, BLOCK_K]
                acc += TMP * W2
                mlp_tmp_ptrs += BLOCK_K
                w2_ptrs += BLOCK_K

            acc = tl.sum(acc, axis=1)  # [BLOCK_N]

            offs_n = pid_n * BLOCK_N_W2 + tl.arange(0, BLOCK_N_W2)
            acc += tl.load(x_ptr + offs_n).to(tl.float32)
            tl.store(x_ptr + offs_n, acc)

        # increment layer
        attn_norm_ptr += dim
        kv_cache_ptr += 2 * max_context * kv_dim
        wqkv_ptr += qkv_dim * dim
        q_norm_ptr += head_dim
        k_norm_ptr += head_dim
        wo_ptr += dim * q_dim

        mlp_norm_ptr += dim
        w13_ptr += (mlp_dim * 2) * dim
        w2_ptr += dim * mlp_dim

        _grid_sync(flag_ptr + (layer_id * 9 + 8), num_pids)

    # LM head and sampling
    _rms_norm(x_ptr, norm_ptr, x_ptr, dim)
    tl.debug_barrier()

    curr_max = tl.full((), float("-inf"), dtype=tl.float32)
    curr_argmax = tl.zeros((), dtype=tl.int32)

    BLOCK_N: tl.constexpr = 16
    for pid_n in range(raw_pid, vocab_size // BLOCK_N, num_pids):
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        x_ptrs = x_ptr + offs_k
        lm_head_ptrs = lm_head_ptr + (offs_n[:, None] * dim + offs_k)

        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

        for _ in range(dim // BLOCK_K):
            x = tl.load(x_ptrs).to(tl.float32)
            lm_head = tl.load(lm_head_ptrs).to(tl.float32)
            acc += x * lm_head
            x_ptrs += BLOCK_K
            lm_head_ptrs += BLOCK_K

        acc = tl.sum(acc, axis=1) / temperature

        # gumbel-max
        acc -= tl.log(-tl.log(tl.rand(rng, 47 + offs_n * 1007)))
        new_max = tl.max(acc)
        if new_max > curr_max:
            curr_max = new_max
            curr_argmax = pid_n * BLOCK_N + tl.argmax(acc, axis=0)

    tl.store(tmp_ptr + raw_pid, curr_max)
    tl.store(tmp_ptr + (num_pids + raw_pid), curr_argmax)
    _grid_sync(flag_ptr + num_layers * 9, num_pids)

    if raw_pid == 0:
        offs = tl.arange(0, BLOCK_PIDS)
        vals = tl.load(tmp_ptr + offs, mask=offs < num_pids, other=float("-inf"))
        token = tl.load(tmp_ptr + (num_pids + tl.argmax(vals, axis=0)))
        tl.store(token_ptr, token)

    elif raw_pid == 1:
        # reset flag
        for i in range(num_layers * 9 + 1):
            tl.store(flag_ptr + i, 0)


_FLAG: Tensor | None = None


def model_triton(
    input_ids: Tensor, params: reference.ModelParams, buffers: reference.ModelBuffers, temperature: float = 1.0
):
    # lazily init _FLAG
    # NOTE: since this flag is shared module-wide, this function is not thread-safe
    global _FLAG
    if _FLAG is None:
        _FLAG = torch.zeros(1000, dtype=torch.int32, device=input_ids.device)

    batch_size = input_ids.shape[0]
    assert batch_size == 1, "Only supports decode"

    _, qkv_dim, dim = params.l_wqkv.shape
    _, _, mlp_dim = params.l_w2.shape
    _, _, max_context, num_kv_heads, head_dim = buffers.kv_cache.shape
    num_heads = qkv_dim // head_dim - num_kv_heads * 2

    # NOTE: may want to limit num SMs used
    num_sms = torch.cuda.get_device_properties(input_ids.device).multi_processor_count
    num_pids_per_head = num_sms // num_heads

    input_embeds = params.input_embeds
    x = input_embeds.new_empty(batch_size, dim)
    attn_tmp = x.new_empty(batch_size, dim + num_heads * head_dim)
    attn_tmp2 = x.new_empty(batch_size, num_pids_per_head, num_heads, head_dim + 2, dtype=torch.float)
    mlp_tmp = x.new_empty(batch_size, dim + mlp_dim)

    lm_head = params.lm_head if params.lm_head is not None else params.input_embeds
    tmp = x.new_empty(batch_size, num_sms * 2, dtype=torch.float32)
    token = x.new_empty(batch_size, dtype=torch.int32)
    vocab_size = lm_head.shape[0]

    BLOCK_K = 512
    BLOCK_N_ATTN = 16
    BLOCK_ATTN = 128
    BLOCK_N_W13 = 4
    BLOCK_N_W2 = 8

    model_triton_kernel[(num_sms,)](
        input_ids,
        x,
        params.input_embeds,
        params.num_layers,
        # attention
        params.l_attn_norm,
        buffers.kv_cache,
        params.l_wqkv,
        params.l_q_norm,
        params.l_k_norm,
        buffers.rope[buffers.position],
        params.l_wo,
        attn_tmp,
        attn_tmp2,
        buffers.position,
        max_context,
        # mlp
        params.l_mlp_norm,
        params.l_w13,
        params.l_w2,
        mlp_tmp,
        #
        params.norm,
        lm_head,
        temperature,
        tmp,
        token,
        flag_ptr=_FLAG,
        rng=time.process_time_ns(),
        vocab_size=vocab_size,
        dim=dim,
        mlp_dim=mlp_dim,
        num_heads=params.num_heads,
        num_kv_heads=params.num_kv_heads,
        num_pids_per_head=num_pids_per_head,
        BLOCK_K=BLOCK_K,
        BLOCK_N_ATTN=BLOCK_N_ATTN,
        BLOCK_ATTN=BLOCK_ATTN,
        BLOCK_N_W13=BLOCK_N_W13,
        BLOCK_N_W2=BLOCK_N_W2,
        BLOCK_PIDS=triton.next_power_of_2(num_sms),
        head_dim=head_dim,
        num_warps=8,
    )
    # assert not x.isnan().any()

    # increment position
    buffers.position += 1

    return token
