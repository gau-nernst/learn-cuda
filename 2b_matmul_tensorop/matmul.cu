#include <cmath>
#include <stdio.h>
#include <assert.h>
#include <cstdint>
#include <cuda_bf16.h>

#define PRINT_IF(cond, ...) if (cond) printf(__VA_ARGS__);

__host__ __device__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }
constexpr bool is_power_of_two(int x) { return x > 0 && (x & (x - 1)) == 0; }  // https://stackoverflow.com/a/1804686
constexpr int WARP_SIZE = 32;


template <int BLOCK_SIZE, int HEIGHT, int WIDTH, typename T>
__device__ void load_shared_128(const T *in, int in_row_stride, T *out, int tid) {
  // number of elements to do 128-bit load
  // e.g. FP32 -> 4 elements, BF16 -> 8 elements.
  constexpr int num_elems = 128 / sizeof(T);

  for (int idx = tid * num_elems; idx < HEIGHT * WIDTH; idx += BLOCK_SIZE * num_elems) {
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;
    uint4 tmp = reinterpret_cast<const uint4 *>(&in[row * in_row_stride + col])[0];
    reinterpret_cast<uint4 *>(&out[row * WIDTH + col])[0] = tmp;
  }
}

__device__ uint32_t cvta_shared(void const *ptr) { return static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); }

template <
  int BLOCK_M, int BLOCK_N, int BLOCK_K,
  int WARP_M, int WARP_N, int WARP_K,
  typename T>
__global__ void matmul_v1_kernel(const T *A, const T *B, T *C, int M, int N, int K) {
  constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 8;
  static_assert(BLOCK_M % WARP_M == 0);
  static_assert(BLOCK_N % WARP_N == 0);
  static_assert(BLOCK_K % WARP_K == 0);
  static_assert(WARP_M % MMA_M == 0);
  static_assert(WARP_N % MMA_N == 0);
  static_assert(WARP_K % MMA_K == 0);
  constexpr int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
  constexpr int NUM_MMA_M = WARP_M / MMA_M;
  constexpr int NUM_MMA_N = WARP_N / MMA_N;
  constexpr int NUM_MMA_K = WARP_K / MMA_K;

  const int tid = threadIdx.x;
  const int block_id = blockIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const int num_blocks_per_row = cdiv(N, BLOCK_N);
  const int block_id_m = block_id / num_blocks_per_row;
  const int block_id_n = block_id % num_blocks_per_row;
  const int offset_m = block_id_m * BLOCK_M;
  const int offset_n = block_id_n * BLOCK_N;

  constexpr int num_warps_per_row = BLOCK_N / WARP_N;
  const int warp_id_m = warp_id / num_warps_per_row;
  const int warp_id_n = warp_id % num_warps_per_row;
  const int warp_tile_offset_m = warp_id_m * WARP_M;
  const int warp_tile_offset_n = warp_id_n * WARP_N;

  // A is row-major, B is column-major
  A += offset_m * K;
  B += offset_n * K;

  __shared__ T A_shared[BLOCK_M * BLOCK_K];
  __shared__ T B_shared[BLOCK_N * BLOCK_K];

  float acc[NUM_MMA_M][NUM_MMA_N][4] = {0};
  uint32_t A_reg[NUM_MMA_M][NUM_MMA_K][2];      // 2x (8,8) matrix
  uint32_t B_reg[NUM_MMA_N][NUM_MMA_K];         // 1x (8,8) matrix

  // first A and B warp-tile along the BLOCK_K (we will iterate along BLOCK_K with step_size=WARP_K)
  const T *A_warp_tile = reinterpret_cast<const T *>(A_shared) + warp_tile_offset_m * BLOCK_K;
  const T *B_warp_tile = reinterpret_cast<const T *>(B_shared) + warp_tile_offset_n * BLOCK_K;

  for (int block_k = 0; block_k < K; block_k += BLOCK_K) {
    load_shared_128<BLOCK_SIZE, BLOCK_M, BLOCK_K>(A, K, A_shared, tid);
    load_shared_128<BLOCK_SIZE, BLOCK_N, BLOCK_K>(B, K, B_shared, tid);
    __syncthreads();

    for (int warp_k = 0; warp_k < BLOCK_K; warp_k += WARP_K) {
      // load data from shared memory to registers using ldmatrix
      // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix

      // 1. convert generic address to .shared state space address expected by inline PTX
      uint32_t A_tile_addr = cvta_shared(A_warp_tile);
      uint32_t B_tile_addr = cvta_shared(B_warp_tile);

      // 2. thread 0 holds address of row 0
      // thread 1 holds address of row 1, and so on
      A_tile_addr += lane_id * BLOCK_K * sizeof(T);
      B_tile_addr += lane_id * BLOCK_K * sizeof(T);

      // load A to registers
      // ldmatrix can only load 8x8 matrix. for 16x8 tile, we need to use x2
      for (int mma_tile_id_m = 0; mma_tile_id_m < NUM_MMA_M; mma_tile_id_m++) {
        for (int mma_tile_id_k = 0; mma_tile_id_k < NUM_MMA_K; mma_tile_id_k++) {
          uint32_t *A_reg_frag = A_reg[mma_tile_id_m][mma_tile_id_k];
          asm volatile (
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
            : "=r"(A_reg_frag[0]), "=r"(A_reg_frag[1])  // output
            : "r"(A_tile_addr)  // input
          );
          A_tile_addr += MMA_K * sizeof(T);
        }
        A_tile_addr += MMA_M * BLOCK_K * sizeof(T);
      }

      // load B to registers
      for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++) {
        for (int mma_tile_id_k = 0; mma_tile_id_k < NUM_MMA_K; mma_tile_id_k++) {
          asm volatile (
            "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
            : "=r"(B_reg[mma_tile_id_n][mma_tile_id_k]) // output
            : "r"(B_tile_addr)  // input
          );
          B_tile_addr += MMA_K * sizeof(T);
        }
        B_tile_addr += MMA_N * BLOCK_K * sizeof(T);
      }

      // call mma
      // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-1688
      for (int mma_tile_id_m = 0; mma_tile_id_m < NUM_MMA_M; mma_tile_id_m++)
        for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++)
          for (int mma_tile_id_k = 0; mma_tile_id_k < NUM_MMA_K; mma_tile_id_k++) {
            float *acc_frag = acc[mma_tile_id_m][mma_tile_id_n];
            uint32_t *A_reg_frag = A_reg[mma_tile_id_m][mma_tile_id_k];
            asm volatile (
              "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
              "{%0, %1, %2, %3}, "  // D
              "{%4, %5}, "          // A
              "{%6}, "              // B
              "{%7, %8, %9, %10};"  // C
              : "=f"(acc_frag[0]), "=f"(acc_frag[1]), "=f"(acc_frag[2]), "=f"(acc_frag[3])
              : "r"(A_reg_frag[0]), "r"(A_reg_frag[1]),
                "r"(B_reg[mma_tile_id_n][mma_tile_id_k]),
                "f"(acc_frag[0]), "f"(acc_frag[1]), "f"(acc_frag[2]), "f"(acc_frag[3])
            );
          }

      A_warp_tile += WARP_K;
      B_warp_tile += WARP_K;
    }
    A_warp_tile -= BLOCK_K;  // compensate
    B_warp_tile -= BLOCK_K;
    __syncthreads();

    A += BLOCK_K;
    B += BLOCK_K;
  }

  const int C_offset_m = offset_m + warp_tile_offset_m;
  const int C_offset_n = offset_n + warp_tile_offset_n;
  C += C_offset_m * N + C_offset_n;

  // check output layout here
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-1688
  const int a0_row = lane_id >> 2; 
  const int a0_col = (lane_id % 4) * 2;
  C += a0_row * N + a0_col;

  for (int mma_tile_id_m = 0; mma_tile_id_m < NUM_MMA_M; mma_tile_id_m++) {
    for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++) {
      if constexpr (std::is_same<T, __half>::value) {
        __half2 tmp;
        float *acc_frag = acc[mma_tile_id_m][mma_tile_id_n];

        // write a0 and a1
        tmp.x = __float2half(acc_frag[0]);
        tmp.y = __float2half(acc_frag[1]);
        reinterpret_cast<__half2 *>(C)[0] = tmp;

        // write a2 and a3
        tmp.x = __float2half(acc_frag[2]);
        tmp.y = __float2half(acc_frag[3]);
        reinterpret_cast<__half2 *>(C + 8 * N)[0] = tmp;
      }
      else if constexpr (std::is_same<T, nv_bfloat16>::value) {
        nv_bfloat162 tmp;
        float *acc_frag = acc[mma_tile_id_m][mma_tile_id_n];

        // write a0 and a1
        tmp.x = __float2bfloat16(acc_frag[0]);
        tmp.y = __float2bfloat16(acc_frag[1]);
        reinterpret_cast<nv_bfloat162 *>(C)[0] = tmp;

        // write a2 and a3
        tmp.x = __float2bfloat16(acc_frag[2]);
        tmp.y = __float2bfloat16(acc_frag[3]);
        reinterpret_cast<nv_bfloat162 *>(C + 8 * N)[0] = tmp;
      }

      C += MMA_N;
    }
    C += MMA_M * N;
  }
}

void matmul_v1(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  assert(is_power_of_two(M) && "M must be a power of 2");
  assert(is_power_of_two(N) && "N must be a power of 2");
  assert(is_power_of_two(K) && "K must be a power of 2");

  const int BLOCK_M = 256, BLOCK_N = 128, BLOCK_K = 32;
  const int WARP_M = 32, WARP_N = 32, WARP_K = 32;

  const int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
  const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
  matmul_v1_kernel<
    BLOCK_M, BLOCK_N, BLOCK_K,
    WARP_M, WARP_N, WARP_K><<<grid_size, BLOCK_SIZE>>>(A, B, C, M, N, K);
}
