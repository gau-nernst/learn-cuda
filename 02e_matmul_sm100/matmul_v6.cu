#include "common.h"
#include "profiler.h"

#include <cuda_bf16.h>

constexpr int NUM_WARPS = 6;  // 1 warp for TMA, 1 warp for MMA, and 4 warps for epilogue
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

constexpr int BLOCK_M = 128;
constexpr int MMA_K = 16;

template <int BLOCK_N, int BLOCK_K, int CTA_GROUP, int NUM_STAGES, bool DO_PROFILE>
__global__
__cluster_dims__(CTA_GROUP, 1, 1)
__launch_bounds__(TB_SIZE)
void matmul_v6_kernel(
  const __grid_constant__ CUtensorMap A_tmap,
  const __grid_constant__ CUtensorMap B_tmap,
  nv_bfloat16 *C_ptr,
  int M, int N, int K,
  int64_t *profiler_ptr,
  int num_entries
) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const int grid_m = M / BLOCK_M;
  const int grid_n = N / BLOCK_N;

  Profiler profiler;
  if constexpr (DO_PROFILE) if (elect_sync()) {
    profiler.init(num_entries, profiler_ptr, bid * NUM_WARPS + warp_id);
    profiler.start(ProfilerTag::Setup);
  }

  // CTA rank in a cluster
  int cta_rank;
  asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

  // set up smem
  // each CTA only loads half of B
  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_size = BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);
  constexpr int B_size = (BLOCK_N / CTA_GROUP) * BLOCK_K * sizeof(nv_bfloat16);

  // set up mbarrier and tmem
  // we have NUM_STAGES mbars for TMA
  //         NUM_STAGES mbars for MMA
  //                  2 mbars for mainloop
  //                  2 mbars for epilogue
  // TMA warp -> TMA mbar -> MMA warp -> mainloop mbar -> epilogue warps
  //          <- MMA mbar <-          <- epilogue mbar
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ uint64_t mbars[NUM_STAGES * 2 + 4];
  __shared__ int tmem_addr[1];  // tmem address is 32-bit
  const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
  const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;
  const int epilogue_mbar_addr = mainloop_mbar_addr + 2 * 8;

  if (warp_id == 0 && elect_sync()) {
    for (int i = 0; i < NUM_STAGES; i++) {
      mbarrier_init(tma_mbar_addr + i * 8, CTA_GROUP);  // both CTAs report TMA to CTA0 only
      mbarrier_init(mma_mbar_addr + i * 8, 1);          // CTA0 reports MMA to BOTH CTAs (multicast)
    }
    for (int i = 0; i < 2; i++) {
      mbarrier_init(mainloop_mbar_addr + i * 8, 1);              // CTA0 reports mainloop to BOTH CTAs (multicast)
      mbarrier_init(epilogue_mbar_addr + i * 8, 4 * CTA_GROUP);  // 4 epilogue warps x both CTAs report to CTA0 only
    }
    asm volatile("fence.mbarrier_init.release.cluster;");  // visible to async proxy
  }
  else if (warp_id == 1) {
    // allocate tmem for output (issued by both CTAs)
    // we allocate double BLOCK_N to double-buffer accumulator
    // it's unlikely that we use BLOCK_N > 256 (tmem limit is 512 columns)
    static_assert(BLOCK_N * 2 <= 512);
    const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
    asm volatile("tcgen05.alloc.cta_group::%2.sync.aligned.shared::cta.b32 [%0], %1;"
                :: "r"(addr), "r"(BLOCK_N * 2), "n"(CTA_GROUP));
  }

  if constexpr (CTA_GROUP > 1) {
    // visible to all threads in a cluster
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
  }
  else {
    // visible to all threads in a threadblock
    __syncthreads();
  }
  const int taddr = tmem_addr[0];  // this will be 0
  if constexpr (DO_PROFILE) if (elect_sync()) profiler.stop();

  // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor
  constexpr int MMA_M = BLOCK_M * CTA_GROUP;  // 128 for 1SM, 256 for 2SM
  constexpr uint32_t i_desc = (1U << 4U)   // dtype=FP32
                            | (1U << 7U)   // atype=BF16
                            | (1U << 10U)  // btype=BF16
                            | ((uint32_t)BLOCK_N >> 3U << 17U)  // MMA_N
                            | ((uint32_t)MMA_M >> 4U << 24U)  // MMA_M
                            ;

  auto load = [&](int tma_stage, int mma_phase, int iter_k, int bid_m, int bid_n) {
    // wait for MMA on the local MMA mbar
    if constexpr (DO_PROFILE) profiler.start(ProfilerTag::WaitMMA);
    mbarrier_wait(mma_mbar_addr + tma_stage * 8, mma_phase);
    if constexpr (DO_PROFILE) profiler.stop();

    if constexpr (DO_PROFILE) profiler.start(ProfilerTag::IssueTMA);
    // both CTA ranks update tx-count of CTA0's mbar
    // https://github.com/NVIDIA/cutlass/blob/v4.3.1/include/cute/arch/copy_sm100_tma.hpp#L113-L115
    const int mbar_addr = (tma_mbar_addr + tma_stage * 8) & 0xFEFFFFFF;  // this is on CTA0
    const int A_smem = smem + tma_stage * (A_size + B_size);
    const int B_smem = A_smem + A_size;

    const int off_m = bid_m * BLOCK_M;
    const int off_n = bid_n * BLOCK_N + cta_rank * (BLOCK_N / CTA_GROUP);
    const int off_k = iter_k * BLOCK_K;
    tma_3d_gmem2smem<CTA_GROUP>(A_smem, &A_tmap, 0, off_m, off_k / 64, mbar_addr);
    tma_3d_gmem2smem<CTA_GROUP>(B_smem, &B_tmap, 0, off_n, off_k / 64, mbar_addr);

    // NOTE: we are using .shared::cluster here
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                :: "r"(mbar_addr), "r"(A_size + B_size) : "memory");
    if constexpr (DO_PROFILE) profiler.stop();
  };

  auto compute = [&](int tma_stage, int tma_phase, int mainloop_stage, int enable_input_d) {
    // wait for TMA on the local TMA mbar
    if constexpr (DO_PROFILE) profiler.start(ProfilerTag::WaitTMA);
    mbarrier_wait(tma_mbar_addr + tma_stage * 8, tma_phase);
    asm volatile("tcgen05.fence::after_thread_sync;");  // (why) do we need this? from DeepGEMM
    if constexpr (DO_PROFILE) profiler.stop();

    if constexpr (DO_PROFILE) profiler.start(ProfilerTag::IssueMMA);
    // select TMA buffer
    const int A_smem = smem + tma_stage * (A_size + B_size);
    const int B_smem = A_smem + A_size;

    // set up shared memory descriptors for A and B
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor
    // 128-byte swizzling. LBO is implied to be 1.
    auto make_desc = [](int addr) -> uint64_t {
      const int SBO = 8 * 128;
      return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
    };

    // select tmem buffer
    const int tmem = taddr + mainloop_stage * BLOCK_N;

    // we specify the LOCAL A smem and B smem here. the tensor cores hardware will know
    // to fetch data from BOTH CTAs, assuming we use the same offset across CTAs.
    // manually unroll 1st iteration to disable accumulation
    {
      tcgen05_mma_f16<CTA_GROUP>(tmem, make_desc(A_smem), make_desc(B_smem), i_desc, enable_input_d);
      for (int k2 = 1; k2 < 64 / MMA_K; k2++) {
        uint64_t a_desc = make_desc(A_smem + k2 * 32);
        uint64_t b_desc = make_desc(B_smem + k2 * 32);
        tcgen05_mma_f16<CTA_GROUP>(tmem, a_desc, b_desc, i_desc, 1);
      }
    }
    // k1 selects the (BLOCK_M, 64) tile.
    // k2 selects the (BLOCK_M, 16) tile, whose rows are swizzled.
    for (int k1 = 1; k1 < BLOCK_K / 64; k1++)
      for (int k2 = 0; k2 < 64 / MMA_K; k2++) {
        uint64_t a_desc = make_desc(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
        uint64_t b_desc = make_desc(B_smem + k1 * (BLOCK_N / CTA_GROUP) * 128 + k2 * 32);
        tcgen05_mma_f16<CTA_GROUP>(tmem, a_desc, b_desc, i_desc, 1);
      }
    // this signals to mbar on BOTH CTAs (thanks to .multicast::cluster)
    constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;
    asm volatile("tcgen05.commit.cta_group::%2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                :: "r"(mma_mbar_addr + tma_stage * 8), "h"(cta_mask), "n"(CTA_GROUP) : "memory");
    if constexpr (DO_PROFILE) profiler.stop();
  };

  auto epilogue = [&](int mainloop_stage, int bid_m, int bid_n) {
    if constexpr (DO_PROFILE) if (elect_sync()) profiler.start(ProfilerTag::Epilogue);
    // we are using warp2-warp5 for epilogue. hence, we need to remap the warp_id
    // for accessing tmem.
    const int epilogue_warp_id = warp_id % 4;
    const int epilogue_tid = epilogue_warp_id * WARP_SIZE + lane_id;

    // load 8 columns from tmem at a time -> store 16 bytes per thread to smem
    // (still strided though)
    for (int n = 0; n < BLOCK_N / 8; n++) {
      // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-a
      // Layout A
      float tmp[8];
      // select tmem buffer
      const int row = cta_rank * 128 + epilogue_warp_id * 32;
      const int col = mainloop_stage * BLOCK_N + n * 8;
      const int addr = taddr + (row << 16) + col;
      asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                  : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
                    "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
                  : "r"(addr));
      asm volatile("tcgen05.wait::ld.sync.aligned;");

      nv_bfloat162 out[4];
      for (int i = 0; i < 4; i++)
        out[i] = __float22bfloat162_rn({tmp[i * 2], tmp[i * 2 + 1]});

      // uncoalesced writes weeee
      nv_bfloat16 *out_ptr = C_ptr + (bid_m * BLOCK_M + epilogue_tid) * N + (bid_n * BLOCK_N + n * 8);
      reinterpret_cast<int4 *>(out_ptr)[0] = reinterpret_cast<int4 *>(out)[0];
    }
    if constexpr (DO_PROFILE) if (elect_sync()) profiler.stop();
  };

  const int num_tiles = grid_m * grid_n;
  const int num_iters = K / BLOCK_K;

  auto compute_bid = [&](int bid) -> std::tuple<int, int> {
    // bid must run along M-mode first so that .cta_group::2 works correctly
    constexpr int GROUP_M = 2;
    int bid_m = bid / (grid_n * GROUP_M) * GROUP_M + (bid % GROUP_M);
    int bid_n = (bid / GROUP_M) % grid_n;
    return {bid_m, bid_n};
  };

  if (warp_id == 0 && elect_sync()) {
    // TMA warp
    int tma_stage = 0;
    int mma_phase = 1;  // the initial MMA phase is 0, and it is available. so we initialize it with 1.

    for (int this_bid = bid; this_bid < num_tiles; this_bid += num_bids) {
      auto [bid_m, bid_n] = compute_bid(this_bid);
      for (int iter_k = 0; iter_k < num_iters; iter_k++) {
        load(tma_stage, mma_phase, iter_k, bid_m, bid_n);

        // flip phase when we have cycled through all TMA buffers
        tma_stage = (tma_stage + 1) % NUM_STAGES;
        if (tma_stage == 0)
          mma_phase ^= 1;
      }
    }
  }
  else if (cta_rank == 0 && warp_id == 1 && elect_sync()) {
    // MMA warp
    int tma_stage = 0;
    int tma_phase = 0;
    int mainloop_stage = 0;
    int epilogue_phase = 1;  // the initial epilogue phase is 0, and it is available. hence we initialize it with 1.

    for (int this_bid = bid; this_bid < num_tiles; this_bid += num_bids) {
      // wait for epilogue to finish
      if constexpr (DO_PROFILE) profiler.start(ProfilerTag::WaitEpilogue);
      mbarrier_wait(epilogue_mbar_addr + mainloop_stage * 8, epilogue_phase);
      if constexpr (DO_PROFILE) profiler.stop();

      for (int iter_k = 0; iter_k < num_iters; iter_k++) {
        compute(tma_stage, tma_phase, mainloop_stage, iter_k);

        // flip phase when we have cycled through all TMA buffers
        tma_stage = (tma_stage + 1) % NUM_STAGES;
        if (tma_stage == 0)
          tma_phase ^= 1;
      }

      // signal when tcgen05 finishes with the main loop to BOTH CTAs
      // (notice .multicast::cluster)
      constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;
      asm volatile("tcgen05.commit.cta_group::%2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                  :: "r"(mainloop_mbar_addr + mainloop_stage * 8), "h"(cta_mask), "n"(CTA_GROUP) : "memory");

      // flip phase when we have cycled through all tmem buffers
      mainloop_stage = (mainloop_stage + 1) % 2;
      if (mainloop_stage == 0)
        epilogue_phase ^= 1;
    }
  }
  else if (warp_id >= 2) {
    int mainloop_stage = 0;
    int mainloop_phase = 0;

    // epilogue warps
    for (int this_bid = bid; this_bid < num_tiles; this_bid += num_bids) {
      // wait for mainloop to finish
      if constexpr (DO_PROFILE) if (elect_sync()) profiler.start(ProfilerTag::WaitMainloop);
      mbarrier_wait(mainloop_mbar_addr + mainloop_stage * 8, mainloop_phase);
      // PTX doc says we need to add this before tcgen05.ld, after tcgen05.mma
      asm volatile("tcgen05.fence::after_thread_sync;");
      if constexpr (DO_PROFILE) if (elect_sync()) profiler.stop();

      auto [bid_m, bid_n] = compute_bid(this_bid);
      epilogue(mainloop_stage, bid_m, bid_n);

      // all epilogue warps report to CTA0 mbar
      if (elect_sync()) {
        const int mbar_addr = (epilogue_mbar_addr + mainloop_stage * 8) & 0xFEFFFFFF;
        asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" :: "r"(mbar_addr) : "memory");
      }

      // flip phase when we have cycled through all tmem buffers
      mainloop_stage = (mainloop_stage + 1) % 2;
      if (mainloop_stage == 0)
        mainloop_phase ^= 1;
    }
  }

  if constexpr (CTA_GROUP > 1) {
    // this is important. otherwise the kernel may fail.
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
  } else {
    __syncthreads();  // all threads finish reading data from tmem
  }

  if (warp_id == 0)  // deallocate tmem (issued by both CTAs)
    asm volatile("tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;"
                :: "r"(taddr), "r"(BLOCK_N * 2), "n"(CTA_GROUP));
  if constexpr (DO_PROFILE) if (elect_sync()) profiler.flush();
}

template <int BLOCK_N, int BLOCK_K, int CTA_GROUP, int NUM_STAGES, bool DO_PROFILE>
void matmul_v6_launch(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K,
  int64_t *profiler_ptr,
  int num_entries
) {
  CUtensorMap A_tmap, B_tmap;

  // input layout for tcgen05: contiguous blocks of (BLOCK_M, 64)
  // then we perform swizzling within this block
  // 3D tensormap (WIDTH / 64, HEIGHT, 64) : (64, WIDTH, 1)
  auto init_tmap_AB = [&](CUtensorMap *tmap, const nv_bfloat16 *ptr, uint64_t global_height, uint32_t shared_height) {
    constexpr uint32_t rank = 3;
    uint64_t globalDim[rank]       = {64, global_height, (uint64_t)K / 64};
    uint64_t globalStrides[rank-1] = {(uint64_t)K * sizeof(nv_bfloat16), 128};  // in bytes
    uint32_t boxDim[rank]          = {64, shared_height, (uint32_t)BLOCK_K / 64};
    uint32_t elementStrides[rank]  = {1, 1, 1};

    auto err = cuTensorMapEncodeTiled(
      tmap,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      rank,
      (void *)ptr,
      globalDim,
      globalStrides,
      boxDim,
      elementStrides,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    check_cu(err);
  };
  // when using threadblock cluster, each threadblock:
  // - still loads (BLOCK_M, BLOCK_K) of A
  // - only loads (BLOCK_N / 2, BLOCK_K) of B.
  // - and still stores (BLOCK_M, BLOCK_N) of C.
  init_tmap_AB(&A_tmap, A_ptr, M, BLOCK_M);
  init_tmap_AB(&B_tmap, B_ptr, N, BLOCK_N / CTA_GROUP);

  // int grid = (M / BLOCK_M) * (N / BLOCK_N);
  int grid = 148;
  int size_AB = (BLOCK_M + BLOCK_N / CTA_GROUP) * BLOCK_K * NUM_STAGES;
  int smem_size = size_AB * sizeof(nv_bfloat16);

  auto this_kernel = matmul_v6_kernel<BLOCK_N, BLOCK_K, CTA_GROUP, NUM_STAGES, DO_PROFILE>;
  if (smem_size > 48'000)
    cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  this_kernel<<<grid, TB_SIZE, smem_size>>>(A_tmap, B_tmap, C_ptr, M, N, K, profiler_ptr, num_entries);
  check_cuda(cudaGetLastError());
}

void matmul_v6(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  matmul_v6_launch<256, 64, 2, 7, false>(A_ptr, B_ptr, C_ptr, M, N, K, nullptr, 0);
}

void profile_matmul_v6(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K,
  int64_t *profiler_ptr,
  int num_entries
) {
  matmul_v6_launch<256, 64, 2, 7, true>(A_ptr, B_ptr, C_ptr, M, N, K, profiler_ptr, num_entries);
}
