#include "common.h"
#include <assert.h>
#include <cstdint>
#include <cuda_bf16.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// trick to get 128-byte aligned shared memory
typedef struct __align__(128) {} Aligned128B;

template <int SWIZZLE_WIDTH>
__device__
uint32_t tma_swizzle(uint32_t addr) {
  const uint32_t row = (addr / 128) % (SWIZZLE_WIDTH / 16);
  return addr ^ (row << 4);
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE) // maxThreadsPerBlock
__global__
void matmul_v7_kernel(const __grid_constant__ CUtensorMap A_tensor_map,
                      const __grid_constant__ CUtensorMap B_tensor_map,
                      nv_bfloat16 *C,
                      int M, int N, int K) {
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;
  static_assert(BLOCK_M % NUM_WARP_M == 0);
  static_assert(BLOCK_N % NUM_WARP_N == 0);
  static_assert(BLOCK_K % MMA_K == 0);
  constexpr int WARP_M = BLOCK_M / NUM_WARP_M;
  constexpr int WARP_N = BLOCK_N / NUM_WARP_N;
  static_assert(WARP_M % MMA_M == 0);
  static_assert(WARP_N % MMA_N == 0);
  constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
  constexpr int NUM_MMA_M = WARP_M / MMA_M;
  constexpr int NUM_MMA_N = WARP_N / MMA_N;
  constexpr int NUM_MMA_K = BLOCK_K / MMA_K;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  // TODO: threadblock swizzling to improve L2 cache hit rate
  const int num_blocks_n = cdiv(N, BLOCK_N);
  const int bid_m = bid / num_blocks_n;
  const int bid_n = bid % num_blocks_n;
  const int offset_m = bid_m * BLOCK_M;
  const int offset_n = bid_n * BLOCK_N;

  const int warp_id_m = warp_id / NUM_WARP_N;
  const int warp_id_n = warp_id % NUM_WARP_N;

  // convert shared memory address to 32-bit from the start
  extern __shared__ Aligned128B smem[];
  nv_bfloat16 *A_smem = reinterpret_cast<nv_bfloat16 *>(smem);
  nv_bfloat16 *B_smem = A_smem + BLOCK_M * BLOCK_K;

  constexpr int num_acc_regs = MMA_M * MMA_N / WARP_SIZE;
  constexpr int num_A_regs = MMA_M * MMA_K * sizeof(nv_bfloat16) / 4 / WARP_SIZE; // 4
  constexpr int num_B_regs = MMA_N * MMA_K * sizeof(nv_bfloat16) / 4 / WARP_SIZE; // 4
  float acc[NUM_MMA_M][NUM_MMA_N][num_acc_regs] = {};
  uint32_t A_regs[NUM_MMA_M][NUM_MMA_K][num_A_regs];
  uint32_t B_regs[NUM_MMA_N][NUM_MMA_K][num_B_regs];

  // pre-compute address used for ldmatrix
  // also pre-compute swizzling
  const int A_offm = (warp_id_m * WARP_M) + (lane_id % 16);
  const int A_offk = (lane_id / 16) * 8;
  uint32_t A_shm_thread = cvta_shared(A_smem) + (A_offm * BLOCK_K + A_offk) * sizeof(nv_bfloat16);
  A_shm_thread = tma_swizzle<BLOCK_K * sizeof(nv_bfloat16)>(A_shm_thread);

  const int B_offn = (warp_id_n * WARP_N) + (lane_id % 8);
  const int B_offk = (lane_id / 8) * 8;
  uint32_t B_shm_thread = cvta_shared(B_smem) + (B_offn * BLOCK_K + B_offk) * sizeof(nv_bfloat16);
  B_shm_thread = tma_swizzle<BLOCK_K * sizeof(nv_bfloat16)>(B_shm_thread);

  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-data-copies-using-the-tensor-memory-accelerator-tma
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;
  if (tid == 0) {
    init(&bar, TB_SIZE);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  const int num_k_iters = cdiv(K, BLOCK_K);
  for (int k_iter = 0; k_iter < num_k_iters; k_iter++) {
    // 1st thread initiate TMA copy / cp.async.bulk
    barrier::arrival_token token;
    if (tid == 0) {
      cde::cp_async_bulk_tensor_2d_global_to_shared(A_smem, &A_tensor_map, k_iter * BLOCK_K, offset_m, bar);
      cde::cp_async_bulk_tensor_2d_global_to_shared(B_smem, &B_tensor_map, k_iter * BLOCK_K, offset_n, bar);
      token = cuda::device::barrier_arrive_tx(bar, 1, (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16));
    } else {
      token = bar.arrive();
    }

    // wait for global->shared to finish
    bar.wait(std::move(token));

    // A shared->regs
    for (int mma_id_m = 0; mma_id_m < NUM_MMA_M; mma_id_m++)
      for (int mma_id_k = 0; mma_id_k < NUM_MMA_K; mma_id_k++) {
        uint32_t A_addr = A_shm_thread;
        A_addr += mma_id_m * MMA_M * BLOCK_K * sizeof(nv_bfloat16);
        A_addr ^= mma_id_k * MMA_K * sizeof(nv_bfloat16);
        ldmatrix_x4(A_regs[mma_id_m][mma_id_k], A_addr);
      }

    // B shared->regs
    for (int mma_id_n = 0; mma_id_n < NUM_MMA_N; mma_id_n++)
      for (int mma_id_k = 0; mma_id_k < NUM_MMA_K; mma_id_k += 2) {
        uint32_t B_addr = B_shm_thread;
        B_addr += mma_id_n * MMA_N * BLOCK_K * sizeof(nv_bfloat16);
        B_addr ^= mma_id_k * MMA_K * sizeof(nv_bfloat16);
        ldmatrix_x4(B_regs[mma_id_n][mma_id_k], B_addr);
      }

    // do MMA. NUM_STAGES-1 prefetch stages are still on-going
    for (int mma_id_m = 0; mma_id_m < NUM_MMA_M; mma_id_m++)
      for (int mma_id_n = 0; mma_id_n < NUM_MMA_N; mma_id_n++)
        for (int mma_id_k = 0; mma_id_k < NUM_MMA_K; mma_id_k++)
          mma_m16n8k16(A_regs[mma_id_m][mma_id_k],
                       B_regs[mma_id_n][mma_id_k],
                       acc[mma_id_m][mma_id_n]);
    __syncthreads();
  }

  C += (offset_m + warp_id_m * WARP_M) * N + (offset_n + warp_id_n * WARP_N);
  for (int mma_id_m = 0; mma_id_m < NUM_MMA_M; mma_id_m++)
    for (int mma_id_n = 0; mma_id_n < NUM_MMA_N; mma_id_n++) {
      const int row = mma_id_m * MMA_M + (lane_id / 4);
      const int col = mma_id_n * MMA_N + (lane_id % 4) * 2;
      nv_bfloat16 *C_local = C + row * N + col;

      float *regs = acc[mma_id_m][mma_id_n];
      reinterpret_cast<nv_bfloat162 *>(C_local)[0]         = __float22bfloat162_rn({regs[0], regs[1]});
      reinterpret_cast<nv_bfloat162 *>(C_local + 8 * N)[0] = __float22bfloat162_rn({regs[2], regs[3]});
    }
}

void matmul_v7(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  assert(is_power_of_two(M) && "M must be a power of 2");
  assert(is_power_of_two(N) && "N must be a power of 2");
  assert(is_power_of_two(K) && "K must be a power of 2");

  // 4 warps
  const int BLOCK_M = 128, BLOCK_N = 64, BLOCK_K = 64;
  const int NUM_WARP_M = 2, NUM_WARP_N = 2;

  auto kernel = matmul_v7_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N>;

  const int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
  const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
  const int shm_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16);

  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#using-tma-to-transfer-multi-dimensional-arrays
  auto init_tensor_map = [&](CUtensorMap *tensor_map, const nv_bfloat16 *gmem_ptr,
                             uint64_t GMEM_WIDTH, uint64_t GMEM_HEIGHT,
                             uint32_t SMEM_WIDTH, uint32_t SMEM_HEIGHT) {
    constexpr uint32_t rank = 2;
    uint64_t size[rank] = {GMEM_WIDTH, GMEM_HEIGHT};
    uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(nv_bfloat16)};
    uint32_t box_size[rank] = {SMEM_WIDTH, SMEM_HEIGHT};
    uint32_t elem_stride[rank] = {1, 1};

    // BLOCK_K = 128 will violate the constraint that shared box's inner dim <= 128 bytes for CU_TENSOR_MAP_SWIZZLE_128B.
    // TODO: support BLOCK_K = 128
    CUtensorMapSwizzle_enum swizzle_mode;
    if constexpr (BLOCK_K <= 8)
      swizzle_mode = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE;
    else if constexpr (BLOCK_K == 16)
      swizzle_mode = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B;
    else if constexpr (BLOCK_K == 32)
      swizzle_mode = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B;
    else if constexpr (BLOCK_K == 64)
      swizzle_mode = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B;
    else {
      std::cerr << "Unsupported BLOCK_K=" << BLOCK_K << std::endl;
      exit(1);
    }

    auto res = cuTensorMapEncodeTiled(
      tensor_map,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      rank,
      (void *)gmem_ptr,
      size,
      stride,
      box_size,
      elem_stride,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle_mode,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) {
      const char *error_msg_ptr;
      if (cuGetErrorString(res, &error_msg_ptr) != CUDA_SUCCESS)
        error_msg_ptr = "unable to get error string";
      std::cerr << "cuTensorMapEncodeTiled error: " << error_msg_ptr << std::endl;
      exit(1);
    }
  };
  CUtensorMap A_tensor_map, B_tensor_map;
  init_tensor_map(&A_tensor_map, A, K, M, BLOCK_K, BLOCK_M);
  init_tensor_map(&B_tensor_map, B, K, N, BLOCK_K, BLOCK_N);

  launch_kernel(kernel, grid_size, TB_SIZE, shm_size, A_tensor_map, B_tensor_map, C, M, N, K);
}
