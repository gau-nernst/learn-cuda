# decode only
# attn_triton_v1 + mlp_triton_v2

import reference
import torch
import triton
import triton.language as tl
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
    attn_tmp_ptr,  # (qkv_dim)
    position,
    max_context,
    # mlp
    mlp_norm_ptr,  # (num_layers, hidden_dim)
    w13_ptr,  # (num_layers, mlp_dim * 2, hidden_dim)
    w2_ptr,  # (num_layers, hidden_dim, mlp_dim)
    mlp_tmp_ptr,  # (mlp_dim) - for w13
    #
    norm_ptr,  # (dim)
    flag_ptr,
    dim: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    mlp_dim: tl.constexpr,
    ATTN_BLOCK_N: tl.constexpr,
    ATTN_BLOCK_K: tl.constexpr,
    MLP_BLOCK_N: tl.constexpr,
    MLP_BLOCK_K: tl.constexpr,
    head_dim: tl.constexpr = 128,
):
    raw_pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    # load input embedding on all threadblocks to avoid grid-wide sync
    input_id = tl.load(input_ids_ptr)
    for k in range(tl.cdiv(dim, 1024)):
        offs = k * 1024 + tl.arange(0, 1024)
        x = tl.load(input_embeds_ptr + (input_id * dim + offs))
        tl.store(x_ptr + offs, x, mask=offs < dim)
    tl.debug_barrier()

    for layer_id in range(num_layers):
        if raw_pid == 0:
            _rms_norm(x_ptr, attn_norm_ptr + layer_id * dim, attn_tmp_ptr, dim)
            tl.atomic_add(flag_ptr + layer_id * 8, 1, sem="release", scope="gpu")
        else:
            _spin_wait(flag_ptr + layer_id * 8, 1)

        # QKV projection
        qkv_dim: tl.constexpr = (num_heads + num_kv_heads * 2) * head_dim
        for pid_n in range(raw_pid, qkv_dim // ATTN_BLOCK_N, num_pids):
            offs_n = pid_n * ATTN_BLOCK_N + tl.arange(0, ATTN_BLOCK_N)
            offs_k = tl.arange(0, ATTN_BLOCK_K)

            tmp_ptrs = attn_tmp_ptr + offs_k
            wqkv_ptrs = wqkv_ptr + (layer_id * qkv_dim * dim + offs_n[:, None] * dim + offs_k[None, :])

            acc = tl.zeros((ATTN_BLOCK_N, ATTN_BLOCK_K), dtype=tl.float32)

            for _ in range(dim // ATTN_BLOCK_K):
                x_normed = tl.load(tmp_ptrs)
                wqkv = tl.load(wqkv_ptrs)  # [BLOCK_N, BLOCK_K]
                acc += x_normed * wqkv

                tmp_ptrs += ATTN_BLOCK_K
                wqkv_ptrs += ATTN_BLOCK_K

            # NOTE: since v doesn't have QK-norm or RoPE, we can write v directly to KV cache
            tl.store(attn_tmp_ptr + (dim + offs_n), tl.sum(acc, axis=1))  # [BLOCK_N]

        _grid_sync(flag_ptr + (layer_id * 8 + 1), num_pids)

        q_ptr = attn_tmp_ptr + dim
        k_ptr = q_ptr + num_heads * head_dim
        v_ptr = k_ptr + num_kv_heads * head_dim

        # QK norm
        if raw_pid < num_heads:
            q_head_ptr = q_ptr + raw_pid * head_dim
            _rms_norm(q_head_ptr, q_norm_ptr + layer_id * head_dim, q_head_ptr, head_dim, BLOCK_SIZE=head_dim)
            tl.atomic_add(flag_ptr + (layer_id * 8 + 2), 1, sem="release", scope="gpu")

        elif raw_pid < num_heads + num_kv_heads:
            k_head_ptr = k_ptr + (raw_pid - num_heads) * head_dim
            _rms_norm(k_head_ptr, k_norm_ptr + layer_id * head_dim, k_head_ptr, head_dim, BLOCK_SIZE=head_dim)
            tl.atomic_add(flag_ptr + (layer_id * 8 + 2), 1, sem="release", scope="gpu")

        _spin_wait(flag_ptr + (layer_id * 8 + 2), num_heads + num_kv_heads)

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

            q_normed = tl.load(q_ptr + (head_id * head_dim + offs_hdim))
            k_normed = tl.load(k_ptr + (kv_head_id * head_dim + offs_hdim))

            q1, q2 = tl.split(tl.reshape(q_normed, (2, head_dim // 2)).T)  # [head_dim/2] each
            k1, k2 = tl.split(tl.reshape(k_normed, (2, head_dim // 2)).T)

            q_new = tl.reshape(tl.join(q1 * cos - q2 * sin, q1 * sin + q2 * cos).T, (head_dim,))
            k_new = tl.reshape(tl.join(k1 * cos - k2 * sin, k1 * sin + k2 * cos).T, (head_dim,))

            # pre-apply exp2 scaling and softmax scale
            q_new *= 1.4426950408889634 * (head_dim**-0.5)

            # update KV cache
            k_cache_ptr = kv_cache_ptr
            v_cache_ptr = kv_cache_ptr + max_context * num_kv_heads * head_dim

            # there is a bug in Triton? increment kv_cache_ptr directly works.
            # but if we do k_cache_ptr = kv_cache_ptr + layer_id * 2 * max_context * num_kv_heads * head_dim
            # there will be out-of-bounds access for KV cache.
            kv_cache_ptr += 2 * max_context * num_kv_heads * head_dim

            kv_offsets = (position * num_kv_heads + kv_head_id) * head_dim + offs_hdim
            tl.store(k_cache_ptr + kv_offsets, k_new)
            tl.store(v_cache_ptr + kv_offsets, v)

            # iterate over KV cache
            offs_hdim = tl.arange(0, head_dim)
            offs_len = tl.arange(0, ATTN_BLOCK_K)

            k_cache_ptrs = k_cache_ptr + (
                offs_len[:, None] * num_kv_heads * head_dim + kv_head_id * head_dim + offs_hdim
            )
            v_cache_ptrs = v_cache_ptr + (
                offs_len * num_kv_heads * head_dim + kv_head_id * head_dim + offs_hdim[:, None]
            )

            # attention w/ online softmax
            max_s = tl.full((1,), float("-inf"), dtype=tl.float32)
            sum_exp = tl.zeros((ATTN_BLOCK_K,), dtype=tl.float32)
            o = tl.zeros((head_dim, ATTN_BLOCK_K), dtype=tl.float32)

            # unroll the last tile so that we don't do masking here
            num_iters = tl.cdiv(position + 1, ATTN_BLOCK_K) - 1
            for _ in range(num_iters):
                k_block = tl.load(k_cache_ptrs)  # [BLOCK_K, head_dim]
                v_block = tl.load(v_cache_ptrs)  # [head_dim, BLOCK_K]

                s = tl.sum(q_new * k_block, axis=1)  # [BLOCK_K]
                new_max_s = tl.maximum(max_s, tl.max(s, axis=0))
                rescale = tl.exp2(max_s - new_max_s)

                p = tl.exp2(s - new_max_s)  # [BLOCK_K]
                sum_exp = sum_exp * rescale + p
                o = o * rescale + p * v_block
                max_s = new_max_s

                k_cache_ptrs += ATTN_BLOCK_K * num_kv_heads * head_dim
                v_cache_ptrs += ATTN_BLOCK_K * num_kv_heads * head_dim

            # last tile w/ masking
            mask = offs_len < position + 1 - num_iters * ATTN_BLOCK_K
            k_block = tl.load(k_cache_ptrs)  # [BLOCK_K, head_dim]
            v_block = tl.load(v_cache_ptrs, mask=mask, other=0.0)  # [head_dim, BLOCK_K]

            s = tl.sum(q_new * k_block, axis=1)  # [BLOCK_K]
            s = tl.where(mask, s, float("-inf"))
            new_max_s = tl.maximum(max_s, tl.max(s, axis=0))
            rescale = tl.exp2(max_s - new_max_s)

            p = tl.exp2(s - new_max_s)  # [BLOCK_K]
            sum_exp = sum_exp * rescale + p
            o = o * rescale + p * v_block

            # store to the same tmp memory for query
            o = tl.sum(o, axis=1) / tl.sum(sum_exp)  # [head_dim]
            offs_hdim = tl.arange(0, head_dim)
            tl.store(q_ptr + head_id * head_dim + offs_hdim, o)

            tl.atomic_add(flag_ptr + (layer_id * 8 + 3), 1, sem="release", scope="gpu")

        _spin_wait(flag_ptr + (layer_id * 8 + 3), num_heads)

        # output projection
        q_dim: tl.constexpr = num_heads * head_dim
        for pid_n in range(raw_pid, dim // ATTN_BLOCK_N, num_pids):
            offs_n = pid_n * ATTN_BLOCK_N + tl.arange(0, ATTN_BLOCK_N)
            offs_k = tl.arange(0, ATTN_BLOCK_K)

            o_ptrs = q_ptr + offs_k
            wo_ptrs = wo_ptr + (layer_id * dim * q_dim + offs_n[:, None] * q_dim + offs_k[None, :])

            acc = tl.zeros((ATTN_BLOCK_N, ATTN_BLOCK_K), dtype=tl.float32)

            for _ in range(q_dim // ATTN_BLOCK_K):
                o = tl.load(o_ptrs)  # [BLOCK_K]
                wo = tl.load(wo_ptrs)  # [BLOCK_N, BLOCK_K]
                acc += o * wo

                o_ptrs += ATTN_BLOCK_K
                wo_ptrs += ATTN_BLOCK_K

            acc = tl.sum(acc, axis=1)  # [BLOCK_N]

            offs_n = pid_n * ATTN_BLOCK_N + tl.arange(0, ATTN_BLOCK_N)
            acc += tl.load(x_ptr + offs_n)
            tl.store(x_ptr + offs_n, acc)

        _grid_sync(flag_ptr + (layer_id * 8 + 4), num_pids)

        # start of MLP
        if raw_pid == 0:
            _rms_norm(x_ptr, mlp_norm_ptr + layer_id * dim, mlp_tmp_ptr, dim)
            tl.atomic_add(flag_ptr + (layer_id * 8 + 5), 1, sem="release", scope="gpu")
        else:
            _spin_wait(flag_ptr + (layer_id * 8 + 5), 1)

        # w13 projection
        w1_ptr = w13_ptr + layer_id * 2 * mlp_dim * dim
        w3_ptr = w1_ptr + mlp_dim * dim

        for pid_n in range(raw_pid, mlp_dim // MLP_BLOCK_N, num_pids):
            offs_n = pid_n * MLP_BLOCK_N + tl.arange(0, MLP_BLOCK_N)
            offs_k = tl.arange(0, MLP_BLOCK_K)

            tmp_ptrs = mlp_tmp_ptr + offs_k
            w1_ptrs = w1_ptr + (offs_n[:, None] * dim + offs_k[None, :])
            w3_ptrs = w3_ptr + (offs_n[:, None] * dim + offs_k[None, :])

            acc1 = tl.zeros((MLP_BLOCK_N, MLP_BLOCK_K), dtype=tl.float32)
            acc3 = tl.zeros((MLP_BLOCK_N, MLP_BLOCK_K), dtype=tl.float32)

            for _ in range(dim // MLP_BLOCK_K):
                x_normed = tl.load(tmp_ptrs)
                w1 = tl.load(w1_ptrs)  # [BLOCK_N, BLOCK_K]
                w3 = tl.load(w3_ptrs)

                acc1 += x_normed * w1
                acc3 += x_normed * w3

                tmp_ptrs += MLP_BLOCK_K
                w1_ptrs += MLP_BLOCK_K
                w3_ptrs += MLP_BLOCK_K

            acc1 = tl.sum(acc1, axis=1)  # [BLOCK_N]
            acc3 = tl.sum(acc3, axis=1)
            acc = acc1 * acc3 * tl.sigmoid(acc1)
            tl.store(mlp_tmp_ptr + (dim + offs_n), acc)

        _grid_sync(flag_ptr + (layer_id * 8 + 6), num_pids)

        # persistent kernel
        for pid_n in range(raw_pid, dim // MLP_BLOCK_N, num_pids):
            offs_n = pid_n * MLP_BLOCK_N + tl.arange(0, MLP_BLOCK_N)
            offs_k = tl.arange(0, MLP_BLOCK_K)

            mlp_tmp_ptrs = mlp_tmp_ptr + (dim + offs_k)
            w2_ptrs = w2_ptr + (layer_id * dim * mlp_dim + offs_n[:, None] * mlp_dim + offs_k[None, :])

            acc = tl.zeros((MLP_BLOCK_N, MLP_BLOCK_K), dtype=tl.float32)

            for _ in range(mlp_dim // MLP_BLOCK_K):
                TMP = tl.load(mlp_tmp_ptrs)  # [BLOCK_K]
                W2 = tl.load(w2_ptrs)  # [BLOCK_N, BLOCK_K]
                acc += TMP * W2

                mlp_tmp_ptrs += MLP_BLOCK_K
                w2_ptrs += MLP_BLOCK_K

            acc = tl.sum(acc, axis=1)  # [BLOCK_N]

            offs_n = pid_n * MLP_BLOCK_N + tl.arange(0, MLP_BLOCK_N)
            acc += tl.load(x_ptr + offs_n)
            tl.store(x_ptr + offs_n, acc)

        _grid_sync(flag_ptr + (layer_id * 8 + 7), num_pids)

    # output norm
    if raw_pid == 0:
        _rms_norm(x_ptr, norm_ptr, x_ptr, dim)

    elif raw_pid == 1:
        # reset flag
        for i in range(num_layers * 8):
            tl.store(flag_ptr + i, 0)


_FLAG: Tensor | None = None


def model_triton(input_ids: Tensor, params: reference.ModelParams, buffers: reference.ModelBuffers):
    # lazily init _FLAG
    # NOTE: since this flag is shared module-wide, this function is not thread-safe
    global _FLAG
    if _FLAG is None:
        _FLAG = torch.zeros(1000, dtype=torch.int32, device=input_ids.device)

    batch_size = input_ids.shape[0]
    assert batch_size == 1, "Only supports decode"

    _, qkv_dim, dim = params.l_wqkv.shape
    _, _, mlp_dim = params.l_w2.shape
    _, _, max_context, _, head_dim = buffers.kv_cache.shape

    input_embeds = params.input_embeds
    x = input_embeds.new_empty(batch_size, dim)
    attn_tmp = input_embeds.new_empty(batch_size, dim + qkv_dim)
    mlp_tmp = input_embeds.new_empty(batch_size, dim + mlp_dim)

    # NOTE: may want to limit num SMs used
    num_sms = torch.cuda.get_device_properties(input_ids.device).multi_processor_count
    ATTN_BLOCK_N = 16
    ATTN_BLOCK_K = 128
    MLP_BLOCK_N = 4
    MLP_BLOCK_K = 512

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
        buffers.position,
        max_context,
        # mlp
        params.l_mlp_norm,
        params.l_w13,
        params.l_w2,
        mlp_tmp,
        #
        params.norm,
        flag_ptr=_FLAG,
        dim=dim,
        num_heads=params.num_heads,
        num_kv_heads=params.num_kv_heads,
        mlp_dim=mlp_dim,
        ATTN_BLOCK_N=ATTN_BLOCK_N,
        ATTN_BLOCK_K=ATTN_BLOCK_K,
        MLP_BLOCK_N=MLP_BLOCK_N,
        MLP_BLOCK_K=MLP_BLOCK_K,
        head_dim=head_dim,
    )

    # increment position
    buffers.position += 1

    # NOTE: right now it's much faster to do LM head separately
    # (or perhaps i need a better way to write this in triton)
    lm_head = params.lm_head if params.lm_head is not None else params.input_embeds
    logits = x @ lm_head.T
    return logits.argmax(-1)
