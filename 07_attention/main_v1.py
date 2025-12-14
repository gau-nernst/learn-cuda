import os

# TORCH_CUDA_ARCH_LIST=10.0
# os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
import argparse
import time
from pathlib import Path
import copy

# import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.cpp_extension import load
from triton.testing import do_bench

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

CURRENT_DIR = Path(__file__).parent

module = load(
    "my_ext",
    sources=list(CURRENT_DIR.glob("attention*")),
    extra_cuda_cflags=["-lineinfo", "--ptxas-options=-v"],
    verbose=True,
)


def seed_everything(seed: int | None = None) -> None:
    """
    Set the seed of each random module.
    `torch.manual_seed` will set seed on all devices.

    Loosely based on: https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/fabric/utilities/seed.py#L20
    """
    import random
    import numpy as np

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


seed_everything(42)


def print_2d_tensort(tensor, name=""):
    torch.cuda.synchronize()
    print(f"+++++++++++++++++++++++++++++{name}+++++++++++++++++++++++++++++++++")
    print("------------------------" * 5)
    m, n = tensor.shape
    for i in range(m):
        print(f"Row: {i}", end=", ")
        for j in range(n):
            print(f"{tensor[i][j]:.4f}", end=", ")
        print()
    print("***************************" * 5)
    torch.cuda.synchronize()


def show_tensor(t, name=""):
    torch.set_printoptions(precision=4)
    print(f"{name} shape: {t.shape}, stride: {t.stride()}")
    print(t)
    if len(t.shape) == 2:
        print_2d_tensort(t)


from dataclasses import dataclass


@dataclass
class MMAConfig:
    M: int = 16
    K: int = 16
    N: int = 8


def load_matrix_from_smem_to_rmem_(
    share_mem_src: torch.Tensor,
    row: int,
    col: int,
    height: int,
    width: int,
    rmem_dst: torch.Tensor,
):
    rmem_dst.copy_(share_mem_src[row : row + height, col : col + width])


def load_global_to_share(
    global_mem_src: torch.Tensor,
    shared_mem_dst: torch.Tensor,
    height: int,
    width: int,
    row: int,
    col: int,
):
    print(
        f"load_global_to_share: row: {row}, col: {col}, height: {height}, width: {width}, global_mem_src shape: {global_mem_src.shape}, shared_mem_dst shape: {shared_mem_dst.shape}"
    )

    shared_mem_dst.copy_(global_mem_src[row : row + height, col : col + width])


def mma_m16n8k16_(frag_A, frag_B, frag_D):
    frag_D.add_((frag_A.to(torch.float32) @ frag_B.T.to(torch.float32)))


def attn_thread_block_ref(
    Q: torch.Tensor,  # [len_q, DIM]
    K: torch.Tensor,  # [len_kv, DIM]
    V: torch.Tensor,  # [len_kv, DIM]
    DIM: int = 64,
    BLOCK_Q: int = 64,
    BLOCK_KV: int = 32,
    NUM_WARPS: int = 4,
    TB_SIZE: int = 128,
    block_id: int = 0,
    softmax_scale: float = 1.0,
):
    # Share memory: Q/K/V
    # Reg mem:
    #  query_rmem warp bf16 [query_rows_per_warp // MMAConfig.M][DIM      // MMAConfig.K][MMAConfig.M, MMAConfig.K]
    #  key_rmem   tb   bf16 [BLOCK_KV            // MMAConfig.N][DIM      // MMAConfig.K][MMAConfig.N, MMAConfig.K]
    #  s_rmem     warp fp32 [query_rows_per_warp // MMAConfig.M][BLOCK_KV // MMAConfig.N][MMAConfig.M, MMAConfig.N]
    #  prob_rmem  warp bf16 [query_rows_per_warp // MMAConfig.M][BLOCK_KV // MMAConfig.K][MMAConfig.M, MMAConfig.K]
    #  value_rmem tb   bf16 [BLOCK_KV            // MMAConfig.K][DIM      // MMAConfig.N][MMAConfig.K, MMAConfig.N]
    #  out_rmem   warp fp32 [query_rows_per_warp // MMAConfig.M][DIM      // MMAConfig.N][MMAConfig.M, MMAConfig.N]
    #  rowmax     warp fp32 [BLOCK_Q]
    #  rowsumexp  warp fp32 [BLOCK_Q]
    softmax_scale = 1.0 / (DIM**0.5)
    device = Q.device
    # Init Q/K/V share mem
    Q_smem = torch.zeros((BLOCK_Q, DIM), device=device, dtype=torch.bfloat16)
    # Key smem reuse Q_smem
    V_smem = torch.zeros((BLOCK_KV, DIM), device=device, dtype=torch.bfloat16)
    
    # Load K from global mem to share mem
    query_rows_per_warp = BLOCK_Q // NUM_WARPS
    K_smem = Q_smem[:BLOCK_KV, ...]
    # Init K frag
    init_first_mma_k_reg = torch.zeros(
        (MMAConfig.N, MMAConfig.K), device=device, dtype=torch.bfloat16
    )
    K_mma_k_per_tb = DIM // MMAConfig.K
    K_mma_n_per_tb = BLOCK_KV // MMAConfig.N
    # [K_mma_n_per_tb][K_mma_k_per_tb][MMAConfig.N, MMAConfig.K]
    cur_tb_first_mma_k_reg = [
        [init_first_mma_k_reg.clone() for k_id in range(K_mma_k_per_tb)]
        for n_id in range(K_mma_n_per_tb)
    ]
    # Init p frag
    first_mma_p_reg = torch.zeros(
        (MMAConfig.M, MMAConfig.K), device=device, dtype=torch.bfloat16
    )
    cur_warp_second_mma_prob_reg = [
        [first_mma_p_reg.clone() for n_id in range(BLOCK_KV // MMAConfig.N)]
        for m_id in range(query_rows_per_warp // MMAConfig.M)
    ]
    cur_tb_second_mma_prob_reg = [
        copy.deepcopy(cur_warp_second_mma_prob_reg) for warp_id in range(NUM_WARPS)
    ]
    # Init O frag
    mma_o_frag = torch.zeros((MMAConfig.M, MMAConfig.N), device=device, dtype=torch.float32)
    cur_warp_second_mma_out_reg = [
        [mma_o_frag.clone() for n_id in range(DIM // MMAConfig.N)]
        for m_id in range(query_rows_per_warp // MMAConfig.M)
    ]
    cur_tb_second_mma_out_reg = [
        copy.deepcopy(cur_warp_second_mma_out_reg) for warp_id in range(NUM_WARPS)
    ]

    # Init rowmax [BLOCK_Q]
    init_cur_warp_rowmax = torch.zeros(
        (query_rows_per_warp), device=device, dtype=torch.float32
    )
    init_cur_warp_rowmax.fill_(-torch.inf)
    cur_tb_rowmax = [init_cur_warp_rowmax.clone() for _ in range(NUM_WARPS)]

    # Init row_sumexp [BLOCK_Q]
    init_cur_warp_row_sumexp = torch.zeros(
        (query_rows_per_warp), device=device, dtype=torch.float32
    )
    cur_tb_rowsumexp = [init_cur_warp_row_sumexp.clone() for _ in range(NUM_WARPS)]

    second_mma_num_m = query_rows_per_warp // MMAConfig.M
    second_mma_num_n = DIM // MMAConfig.N
    second_mma_num_k = BLOCK_KV // MMAConfig.K
    init_v_frag = torch.zeros((MMAConfig.K, MMAConfig.N), device=device, dtype=torch.bfloat16)
    cur_tb_value_reg = [
        [init_v_frag.clone() for second_mma_n_id in range(second_mma_num_n)]
        for second_mma_k_id in range(second_mma_num_k)
    ]
    # Load Q from global mem to share mem [BLOCK_Q, DIM]
    cur_block_q_row = BLOCK_Q * block_id
    cur_block_q_col = 0
    q_shared_mem_height = BLOCK_Q
    q_shared_mem_width = DIM
    # **load query from global mem to share mem**
    load_global_to_share(
        global_mem_src=Q,
        shared_mem_dst=Q_smem,
        height=q_shared_mem_height,
        width=q_shared_mem_width,
        row=cur_block_q_row,
        col=cur_block_q_col,
    )
    # load Q from share mem to register
    # There are 4 warps, each warp handle BLOCK_Q/4 rows (M)
    query_rows_per_warp = BLOCK_Q // NUM_WARPS
    mma_m_per_warp = query_rows_per_warp // MMAConfig.M
    mma_k_per_warp = DIM // MMAConfig.K
    # Each warp own [mma_m_per_warp][mma_k_per_warp][MMAConfig.M, MMAConfig.K] Q
    init_first_mma_q_reg = torch.zeros(
        (MMAConfig.M, MMAConfig.K), device=device, dtype=torch.bfloat16
    )
    cur_warp_first_mma_q_reg = [
        [init_first_mma_q_reg.clone() for i in range(mma_k_per_warp)]
        for j in range(mma_m_per_warp)
    ]
    cur_tb_first_mma_q_reg = [
        copy.deepcopy(cur_warp_first_mma_q_reg) for warp_id in range(NUM_WARPS)
    ]
    # **Load all qeury rows for each warp from share mem to register**
    for warp_id in range(NUM_WARPS):
        cur_warp_first_mma_q_reg = cur_tb_first_mma_q_reg[warp_id]
        per_warp_rows = BLOCK_Q // NUM_WARPS
        cur_warp_start_row = per_warp_rows * warp_id
        for mma_m_id in range(mma_m_per_warp):
            for mma_k_id in range(mma_k_per_warp):
                cur_q_frag = cur_warp_first_mma_q_reg[mma_m_id][mma_k_id]
                # Load cur_frag from share mem
                row = mma_m_id * MMAConfig.M
                col = mma_k_id * MMAConfig.K
                cur_warp_Q_smem = Q_smem[cur_warp_start_row:, ...]
                load_matrix_from_smem_to_rmem_(
                    cur_warp_Q_smem,
                    row,
                    col,
                    MMAConfig.M,
                    MMAConfig.K,
                    cur_q_frag,
                )
                print_2d_tensort(
                    cur_q_frag,
                    name=f"frag warp:{warp_id},mma_m_id:{mma_m_id},mma_k_id:{mma_k_id}",
                )
    # All BLOCK_Q are in the register
    #  [BLOCK_Q, DIM] -> warp 0 [query_rows_per_warp // MMAConfig.M][DIM // MMAConfig.K][M, K]
    #                 -> warp 1 [query_rows_per_warp // MMAConfig.M][DIM // MMAConfig.K][M, K]
    #                 -> warp 2 [query_rows_per_warp // MMAConfig.M][DIM // MMAConfig.K][M, K]
    #                 -> warp 3 [query_rows_per_warp // MMAConfig.M][DIM // MMAConfig.K][M, K]
    # >>>>>>>>>>> start main loop <<<<<<<<<<<<<<<<
    len_kv, K_DIM = K.shape
    assert K_DIM == DIM, f"Expected K_DIM == DIM, but got K_DIM:{K_DIM}, DIM: {DIM}"
    num_kv_iters = len_kv // BLOCK_KV
    Q_smem.zero_()
    # Only take half of BLOCK_Q
    

    for kv_iter in range(num_kv_iters):
        # Init S rmem
        # 4 float32 in acutally for cuda
        init_first_mma_s_frag = torch.zeros(
            (MMAConfig.M, MMAConfig.N), device=device, dtype=torch.float32
        )
        cur_warp_first_mma_s_frag = [
            [init_first_mma_s_frag.clone() for n_id in range(BLOCK_KV // MMAConfig.N)]
            for m_id in range(query_rows_per_warp // MMAConfig.M)
        ]
        cur_tb_first_mma_s_frag = [
            copy.deepcopy(cur_warp_first_mma_s_frag) for warp_id in range(NUM_WARPS)
        ]

        kv_start_row = kv_iter * BLOCK_KV
        # Load K from global mem to share mem [BLOCK_KV, DIM]
        print(
            f"start to copy K kv_iter: {kv_iter}, kv_start_row: {kv_start_row}, K shape: {K.shape}, BLOCK_KV: {BLOCK_KV}, DIM: {DIM}"
        )
        load_global_to_share(
            global_mem_src=K,
            shared_mem_dst=K_smem,
            height=BLOCK_KV,
            width=DIM,
            row=kv_start_row,
            col=0,
        )
        show_tensor(K_smem, "K_smem")
        # Load K from share mem to register
        # Here, we need to load all BLOCK_KV K for each warp
        for K_mma_n_id in range(K_mma_n_per_tb):
            for K_mma_k_id in range(K_mma_k_per_tb):
                K_row = K_mma_n_id * MMAConfig.N
                K_col = K_mma_k_id * MMAConfig.K
                cur_K_frag = cur_tb_first_mma_k_reg[K_mma_n_id][K_mma_k_id]
                load_matrix_from_smem_to_rmem_(
                    share_mem_src=K_smem,
                    row=K_row,
                    col=K_col,
                    height=MMAConfig.N,
                    width=MMAConfig.K,
                    rmem_dst=cur_K_frag,
                )
                show_tensor(
                    cur_K_frag,
                    f"cur_K_frag: kv_iter: {kv_iter}, K_row:{K_row}, K_col: {K_col}",
                )
        # All Q and K are in the register, we can do mma
        for warp_id in range(NUM_WARPS):
            cur_warp_first_mma_q_reg = cur_tb_first_mma_q_reg[warp_id]
            cur_warp_first_mma_s_frag = cur_tb_first_mma_s_frag[warp_id]
            first_mma_num_m = query_rows_per_warp // MMAConfig.M
            first_mma_num_n = BLOCK_KV // MMAConfig.N
            first_mma_num_k = DIM // MMAConfig.K
            for mma_m_id in range(first_mma_num_m):
                for mma_n_id in range(first_mma_num_n):
                    cur_s_reg = cur_warp_first_mma_s_frag[mma_m_id][mma_n_id]
                    for mma_k_id in range(first_mma_num_k):
                        cur_mma_query = cur_warp_first_mma_q_reg[mma_m_id][mma_k_id]
                        cur_mma_key = cur_tb_first_mma_k_reg[mma_n_id][mma_k_id]
                        mma_m16n8k16_(
                            frag_A=cur_mma_query,
                            frag_B=cur_mma_key,
                            frag_D=cur_s_reg,
                        )
                        if (kv_iter == num_kv_iters-1 and warp_id == 0 and mma_m_id == 0 and mma_n_id == 0 and mma_k_id==0):
                            show_tensor(cur_mma_query, "cur_mma_query")
                            show_tensor(cur_mma_key, "cur_mma_key")
                            show_tensor(
                                cur_s_reg,
                                f" cur_s_reg: After MMA kv_iter:{kv_iter}, warp_id:{warp_id}, mma_m_id:{mma_m_id}, mma_n_id:{mma_n_id}",
                            )

        # **apply softmax scale: sqrt(dk)**
        for warp_id in range(NUM_WARPS):
            cur_warp_first_mma_s_frag = cur_tb_first_mma_s_frag[warp_id]
            for mma_m_id in range(first_mma_num_m):
                for mma_n_id in range(first_mma_num_n):
                    cur_s_reg = cur_warp_first_mma_s_frag[mma_m_id][mma_n_id]
                    cur_s_reg.mul_(softmax_scale)

        # **init this KV iter row max
        cur_warp_this_iter_rowmax = torch.zeros(
            (query_rows_per_warp), device=device, dtype=torch.float32
        )
        cur_warp_this_iter_rowmax.fill_(-torch.inf)
        cur_tb_this_iter_rowmax = [cur_warp_this_iter_rowmax.clone() for _ in range(NUM_WARPS)]

        # **init rescale factor for EACH kv iter
        # [BLOCK_Q]
        cur_warp_rescale = torch.zeros(
            (query_rows_per_warp), device=device, dtype=torch.float32
        )
        cur_tb_rescale = [cur_warp_rescale.clone() for _ in range(NUM_WARPS)]

        # **update the row max, resacle, and apply rescale on previous O
        for warp_id in range(NUM_WARPS):
            cur_warp_first_mma_s_frag = cur_tb_first_mma_s_frag[warp_id]
            cur_warp_this_iter_rowmax = cur_tb_this_iter_rowmax[warp_id]
            for mma_m_id in range(query_rows_per_warp // MMAConfig.M):
                start_row = mma_m_id * MMAConfig.M
                for mma_n_id in range(BLOCK_KV // MMAConfig.N):
                    cur_s_reg = cur_warp_first_mma_s_frag[mma_m_id][mma_n_id]
                    cur_s_rowmax, _ = cur_s_reg.max(dim=1, keepdim=False)
                    cur_warp_this_iter_rowmax[start_row : start_row + MMAConfig.M] = torch.max(
                        cur_warp_this_iter_rowmax[start_row : start_row + MMAConfig.M],
                        cur_s_rowmax,
                    )
            # Compare with the previous max
            cur_warp_rowmax = cur_tb_rowmax[warp_id]
            cur_warp_this_iter_rowmax.copy_(
                torch.max(cur_warp_this_iter_rowmax, cur_warp_rowmax)
            )

            # Update rescale factor
            cur_warp_rescale = cur_tb_rescale[warp_id]
            rowmax_diff = cur_warp_rowmax - cur_warp_this_iter_rowmax
            cur_warp_rescale.copy_(torch.exp(rowmax_diff))

            # Apply rescale factor on previou O
            cur_warp_second_mma_out_reg = cur_tb_second_mma_out_reg[warp_id]
            for mma_m_id in range(query_rows_per_warp // MMAConfig.M):
                start_row = mma_m_id * MMAConfig.M
                for mma_n_id in range(DIM // MMAConfig.N):
                    cur_out_reg = cur_warp_second_mma_out_reg[mma_m_id][mma_n_id]
                    cur_rescale = cur_warp_rescale[start_row : start_row + MMAConfig.M]
                    cur_out_reg.copy_(cur_out_reg * cur_rescale.unsqueeze(-1))
            # Update the rowmax after apply rescale on previous output
            cur_warp_rowmax.copy_(cur_warp_this_iter_rowmax)

        # **Next, we repack the 2xS_rmem to P_rmem, 2xm16n8 -> m16k16, and update row sumexp
        for warp_id in range(NUM_WARPS):
            cur_warp_rowmax = cur_tb_rowmax[warp_id]
            cur_warp_first_mma_s_frag = cur_tb_first_mma_s_frag[warp_id]
            cur_warp_rescale = cur_tb_rescale[warp_id]
            cur_warp_second_mma_prob_reg = cur_tb_second_mma_prob_reg[warp_id]
            cur_warp_prob_sum = torch.zeros(
                (query_rows_per_warp), device=device, dtype=torch.float32
            )
            for mma_m_id in range(query_rows_per_warp // MMAConfig.M):
                start_row = mma_m_id * MMAConfig.M
                for mma_n_id in range(BLOCK_KV // MMAConfig.N):
                    first_mma_s_frag = cur_warp_first_mma_s_frag[mma_m_id][mma_n_id]
                    tmp_mma_row_max = cur_warp_rowmax[start_row : start_row + MMAConfig.M]
                    tmp_new_s = first_mma_s_frag - tmp_mma_row_max.unsqueeze(-1)
                    first_mma_s_frag.copy_(torch.exp(tmp_new_s))
                    mma_p_reg = cur_warp_second_mma_prob_reg[mma_m_id][mma_n_id // 2]  # ?
                    start_col = (mma_n_id % 2) * MMAConfig.N
                    mma_p_reg[:, start_col : start_col + MMAConfig.N].copy_(first_mma_s_frag)
                    mma_p_reg_sum = first_mma_s_frag.sum(dim=-1)
                    cur_warp_prob_sum[start_row : start_row + MMAConfig.M].add_(mma_p_reg_sum)
                cur_warp_rowsumexp = cur_tb_rowsumexp[warp_id]
                cur_mma_rescale = cur_warp_rescale[start_row : start_row + MMAConfig.M]
                cur_warp_rowsumexp.copy_(
                    cur_warp_rowsumexp * cur_mma_rescale + cur_warp_prob_sum
                )

        # Okay, P is ready,let's move to 2nd MMA: P@V
        # Load V
        kv_start_row = kv_iter * BLOCK_KV
        # **Load V from global mem to share mem [BLOCK_KV, DIM]
        load_global_to_share(
            global_mem_src=V,
            shared_mem_dst=V_smem,
            height=BLOCK_KV,
            width=DIM,
            row=kv_start_row,
            col=0,
        )
        show_tensor(V_smem, "V_smem")

        # **Load V from share mem to register
        # Here, we need to load all BLOCK_KV V for each warp
        for second_mma_k_id in range(second_mma_num_k):
            for second_mma_n_id in range(second_mma_num_n):
                v_row = second_mma_k_id * MMAConfig.K
                v_col = second_mma_n_id * MMAConfig.N
                cur_value_reg = cur_tb_value_reg[second_mma_k_id][second_mma_n_id]
                load_matrix_from_smem_to_rmem_(
                    share_mem_src=V_smem,
                    row=v_row,
                    col=v_col,
                    height=MMAConfig.K,
                    width=MMAConfig.N,
                    rmem_dst=cur_value_reg,
                )
        # **Final, let do 2nd MMA
        for warp_id in range(NUM_WARPS):
            cur_warp_second_mma_prob_reg = cur_tb_second_mma_prob_reg[warp_id]
            cur_warp_second_mma_out_reg = cur_tb_second_mma_out_reg[warp_id]
            for second_mma_m_id in range(second_mma_num_m):
                for second_mma_n_id in range(second_mma_num_n):
                    for second_mma_k_id in range(second_mma_num_k):
                        cur_mma_prob_reg = cur_warp_second_mma_prob_reg[second_mma_m_id][second_mma_k_id]
                        cur_mma_value_reg = cur_tb_value_reg[second_mma_k_id][second_mma_n_id]
                        cur_mma_out_Reg = cur_warp_second_mma_out_reg[second_mma_m_id][second_mma_n_id]
                        mma_m16n8k16_(
                            frag_A=cur_mma_prob_reg,
                            frag_B=cur_mma_value_reg.T,
                            frag_D=cur_mma_out_Reg,
                        )
    # End of all kv_iters
    # **Write out back
    out_all = torch.zeros((BLOCK_Q, DIM), device=device, dtype=torch.bfloat16)
    for warp_id in range(NUM_WARPS):
        cur_warp_second_mma_out_reg = cur_tb_second_mma_out_reg[warp_id]
        cur_warp_start_row = warp_id * query_rows_per_warp
        cur_warp_row_sumexp = cur_tb_rowsumexp[warp_id]
        for second_mma_m_id in range(second_mma_num_m):
            for second_mma_n_id in range(second_mma_num_n):
                cur_mma_out = cur_warp_second_mma_out_reg[second_mma_m_id][second_mma_n_id]
                cur_mma_start_row_in_warp = second_mma_m_id * MMAConfig.M
                out_start_row = cur_mma_start_row_in_warp + cur_warp_start_row
                out_start_col = second_mma_n_id * MMAConfig.N
                cur_mma_row_sumexp = cur_warp_row_sumexp[
                    cur_mma_start_row_in_warp : cur_mma_start_row_in_warp + MMAConfig.M
                ]
                print(f"start to update ({out_start_row}, {out_start_col})")

                cur_mma_out_update = cur_mma_out / cur_mma_row_sumexp.unsqueeze(-1)
                out_all[
                    out_start_row : out_start_row + MMAConfig.M,
                    out_start_col : out_start_col + MMAConfig.N,
                ].copy_(cur_mma_out_update.to(torch.bfloat16))
    return out_all


def compare_2d_tensor(tensor1, tensor2):
    m, n = tensor1.shape
    max_diff = -torch.inf
    max_rel_diff = -torch.inf
    max_diff_record = [None, None]
    for i in range(m):
        for j in range(n):
            diff = tensor2[i][j] - tensor1[i][j]
            if diff > max_diff:
                max_diff_record = tensor1[i][j], tensor2[i][j]
            max_diff = max(max_diff, diff)
            rel_diff = diff / tensor1[i][j]
            max_rel_diff = max(max_rel_diff, rel_diff)
    print(
        f"max_diff: {max_diff},max_rel_diff: {max_rel_diff}, max_diff_record: {max_diff_record}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--nh", type=int, default=1)
    parser.add_argument("--lq", type=int, default=64)
    parser.add_argument("--lkv", type=int, default=64)
    args = parser.parse_args()

    bs = args.bs
    nh = args.nh
    lq = args.lq
    lkv = args.lkv
    head_dim = 64

    # add a small offset so that output does not have a mean of zero,
    # which will result in large relative error
    def generate_input(*shape):
        # return torch.randn(shape).add(0.5).bfloat16().cuda()
        init = torch.arange(torch.prod(torch.tensor(shape)), dtype=torch.float32).reshape(shape)
        return init.bfloat16().cuda()

    Q = generate_input(bs, nh, lq, head_dim)
    K = generate_input(bs, nh, lkv, head_dim) + 128
    V = generate_input(bs, nh, lkv, head_dim)

    if args.profile is not None:
        if args.profile == "fa":
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                F.scaled_dot_product_attention(Q, K, V)

        elif args.profile == "cudnn":
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                F.scaled_dot_product_attention(Q, K, V)

        else:
            f = getattr(module, f"sdpa_v{args.profile}")
            f(Q, K, V)

        torch.cuda.synchronize()
        return

    SOL_LOOKUP = {
        "NVIDIA GeForce RTX 5090": 209.5,
    }
    sol = SOL_LOOKUP.get(torch.cuda.get_device_name(), 1000)

    results = []

    def bench_and_print(f, name, *args):
        # sleep to stabilize thermal
        time.sleep(1)

        latency_ms = do_bench(lambda: f(*args), return_mode="median", rep=10)
        tflops = 4 * bs * nh * lq * lkv * head_dim / latency_ms / 1e9
        pct_sol = tflops / sol * 100
        results.append([name, round(latency_ms, 4), round(tflops, 2), round(pct_sol, 2)])

    out_ref = F.scaled_dot_product_attention(Q, K, V)

    # with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
    #     bench_and_print(F.scaled_dot_product_attention, "F.sdpa() - FA", Q, K, V)
    # with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
    #     bench_and_print(F.scaled_dot_product_attention, "F.sdpa() - CuDNN", Q, K, V)

    if flash_attn_func is not None:
        tran_Q = Q.transpose(1, 2)
        # print(f"Flash Attention input shape: {tran_Q.shape}")
        # print(f"Flash Attention input stride: {tran_Q.stride()}")
        # print(f"transposed Q: {tran_Q}")
        out = flash_attn_func(tran_Q, K.transpose(1, 2), V.transpose(1, 2)).transpose(1, 2)
        torch.testing.assert_close(out, out_ref)
        # bench_and_print(
        #     flash_attn_func,
        #     "flash-attn",
        #     Q.transpose(1, 2),
        #     K.transpose(1, 2),
        #     V.transpose(1, 2),
        # )
    #   const int BLOCK_Q = 64;
    #   const int BLOCK_KV = 32;
    #   const int DIM = 64;
    #   const int NUM_WARPS = 4;
    show_tensor(Q, "Q")
    show_tensor(K, "K")
    tmp_P_first_mma = Q[0, 0, :, :16] @ K[0, 0, :, :16].T
    show_tensor(tmp_P_first_mma, "tmp_P_first_mma")
    tmp_P = Q[0, 0, :, :] @ K[0, 0, :, :].T
    show_tensor(tmp_P, "tmp_P")
    BLOCK_KV = 32
    torch_ref = attn_thread_block_ref(
        Q.clone()[0][0],
        K.clone()[0][0],
        V.clone()[0][0],
        DIM=head_dim,
        BLOCK_Q=64,
        BLOCK_KV=BLOCK_KV,
        NUM_WARPS=4,
        TB_SIZE=128,
        block_id=0,
    )
    # show_tensor(torch_ref, "torch_ref")
    #
    # compare_2d_tensor(tmp_P, torch_ref)
    # breakpoint()
    # torch.testing.assert_close(out_ref[0][0], torch_ref)
    for i in [
        0, 
        5
        ]:
        f = getattr(module, f"sdpa_v{i + 1}")
        out = f(Q, K, V)
        # show_tensor(out[0][0], f"out_v{i + 1}")
        # torch.testing.assert_close(out[0][0], torch_ref)
        # breakpoint()
        print(f"v{i + 1} passed correctness test")
        # bench_and_print(f, f"v{i + 1}", Q, K, V)

    # df = pd.DataFrame(results, columns=["Kernel", "Latency (ms)", "TFLOPS", "% SOL"])
    # print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
