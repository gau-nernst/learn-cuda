#include <cmath>
#include <stdio.h>

#define PRINT_IF(cond, ...) if (cond) printf(__VA_ARGS__);

__host__ __device__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

// naive kernel. 1 row dot 1 column
__global__ void matmul_v1_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row >= M || col >= N)
    return;

  float total = 0.0f;

  // broadcast read from A since each warp reads the same A value
  // coalesce read from B since each warp reads consecutive B values
  for (int k = 0; k < K; k++)
    total += A[row * K + k] * B[k * N + col];

  // coalesce write to C since each warp writes consecutive C values
  C[row * N + col] = total;
}

void matmul_v1(const float *A, const float *B, float *C, int M, int N, int K) {
  // determine optimal block size at runtime
  int block_size_total;
  int min_grid_size; // we don't need this
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size_total, matmul_v1_kernel, 0, 0);

  // NOTE: blockIdx.x is the fastest changing dimension. thus, we assign column index to it
  // intuitively, block dimensions will be PyTorch's dimensions in reverse.
  // NOTE: blockDim.x must be multiple of 32 (warpSize) to ensure coalesce memory access
  dim3 block_size(32, block_size_total / 32);
  dim3 grid_size(cdiv(N, 32), cdiv(M, block_size.y));
  matmul_v1_kernel<<<grid_size, block_size>>>(A, B, C, M, N, K);
}

// read 2D block into shared memory for caching
template <int BLOCK_SIZE>
__global__ void matmul_v2_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
  const int tid_x = threadIdx.x;
  const int tid_y = threadIdx.y;

  const int offset_m = blockIdx.y * BLOCK_SIZE;
  const int offset_n = blockIdx.x * BLOCK_SIZE;

  A += offset_m * K;             // skip x rows
  B += offset_n;                 // skip y columns
  C += offset_m * N + offset_n;  // skip x rows and y columns

  // we cannot return early since all threads need to synchronize
  __shared__ float A_shmem[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float B_shmem[BLOCK_SIZE][BLOCK_SIZE];
  float acc = 0.0f;

  // we move block by block along K dim
  for (int offset_k = 0; offset_k < K; offset_k += BLOCK_SIZE) {
    // load data from global memory (DDR/HBM) to shared memory (SRAM)
    // notice now each thread only loads 2 x n_blocks elements
    // coalesced memory read for both A and B
    A_shmem[tid_y][tid_x] = tid_y < (M - offset_m) && tid_x < (K - offset_k) ? A[tid_y * K + tid_x] : 0.0f;
    B_shmem[tid_y][tid_x] = tid_y < (K - offset_k) && tid_x < (N - offset_n) ? B[tid_y * N + tid_x] : 0.0f;

    // wait for all threads in a block to load data
    __syncthreads();

    // compute from shared memory
    for (int k = 0; k < BLOCK_SIZE; k++)
      acc += A_shmem[tid_y][k] * B_shmem[k][tid_x];

    // wait to finish before moving to the next tile
    __syncthreads();

    A += BLOCK_SIZE;      // stride 1 in K dim
    B += BLOCK_SIZE * N;  // stride N in K dim
  }

  if (tid_y < (M - offset_m) && tid_x < (N - offset_n))
    C[tid_y * N + tid_x] = acc;
}

void matmul_v2(const float *A, const float *B, float *C, int M, int N, int K) {
  // we can't use a larger block size since we are limited by 1024 threads per block
  constexpr int BLOCK_SIZE = 32;
  dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_size(cdiv(N, BLOCK_SIZE), cdiv(M, BLOCK_SIZE));
  matmul_v2_kernel<BLOCK_SIZE><<<grid_size, block_size>>>(A, B, C, M, N, K);
}

// we want to load a (HEIGHT, WIDTH) tile from global to shared memory.
// just load a BLOCK_SIZE of data until the whole tile is loaded.
template <int BLOCK_SIZE, int HEIGHT, int WIDTH>
__device__ void load_shmem(const float *in, int in_row_stride, int in_max_row, int in_max_col,
                           float out[HEIGHT][WIDTH], int tid) {
  for (int idx = tid; idx < HEIGHT * WIDTH; idx += BLOCK_SIZE) {
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;
    out[row][col] = row < in_max_row && col < in_max_col ? in[row * in_row_stride + col] : 0.0f;
  }
}

// thread coarsening
template <int BLOCK_SIZE, int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void matmul_v3_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
  const int tid = threadIdx.x;
  const int block_id = blockIdx.x;

  // assign block linearly
  const int grid_width = cdiv(N, BLOCK_N);
  const int block_id_m = block_id / grid_width;
  const int block_id_n = block_id % grid_width;

  const int offset_m = block_id_m * BLOCK_M;
  const int offset_n = block_id_n * BLOCK_N;

  A += offset_m * K;
  B += offset_n;
  C += offset_m * N + offset_n;

  __shared__ float A_shmem[BLOCK_M][BLOCK_K];
  __shared__ float B_shmem[BLOCK_K][BLOCK_N];

  // each thread is responsible for (BLOCK_M * BLOCK_N / BLOCK_SIZE) output elements
  float acc[BLOCK_M * BLOCK_N / BLOCK_SIZE] = {0.0f};

  // we move block by block along K dim
  for (int offset_k = 0; offset_k < K; offset_k += BLOCK_K) {
    // decouple global memory read, so we don't need to care about assigning which thread
    // to read which element.
    // load (BLOCK_M, BLOCK_K) from A and (BLOCK_K, BLOCK_N) from B
    load_shmem<BLOCK_SIZE, BLOCK_M, BLOCK_K>(A, K, M - offset_m, K - offset_k, A_shmem, tid);
    load_shmem<BLOCK_SIZE, BLOCK_K, BLOCK_N>(B, N, K - offset_k, N - offset_n, B_shmem, tid);
    __syncthreads();

    // do a mini matmul of (BLOCK_M, BLOCK_K) x (BLOCK_K, BLOCK_N) = (BLOCK_M, BLOCK_N)
    // simply assign a BLOCK_SIZE of threads to a BLOCK_SIZE of elements in output tile
    for (int idx = tid; idx < BLOCK_M * BLOCK_N; idx += BLOCK_SIZE) {
      const int local_idx = idx / BLOCK_SIZE;
      const int col = idx % BLOCK_N;
      const int row = idx / BLOCK_N;

      for (int k = 0; k < BLOCK_K; k++)
        acc[local_idx] += A_shmem[row][k] * B_shmem[k][col];
    }
    __syncthreads();

    A += BLOCK_K;
    B += BLOCK_K * N;
  }

  // write (BLOCK_M, BLOCK_N) to C
  for (int idx = tid; idx < BLOCK_M * BLOCK_N; idx += BLOCK_SIZE) {
    const int local_idx = idx / BLOCK_SIZE;
    const int row = idx / BLOCK_N;
    const int col = idx % BLOCK_N;

    if (row < (M - offset_m) && col < (N - offset_n))
      C[row * N + col] = acc[local_idx];
  }
}

void matmul_v3(const float *A, const float *B, float *C, int M, int N, int K) {
  // we are limited by the amount of shared memory
  // 128 * 32 * 2 * 4 = 32kB
  const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32;
  const int BLOCK_SIZE = 256;
  const int grid_size = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N);
  matmul_v3_kernel<BLOCK_SIZE, BLOCK_M, BLOCK_N, BLOCK_K><<<grid_size, BLOCK_SIZE>>>(A, B, C, M, N, K);
}

// register cache with 2D thread tiling
// only mini matmul is different from v3
template <int BLOCK_SIZE, int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_N>
__global__ void matmul_v4_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
  const int tid = threadIdx.x;
  const int block_id = blockIdx.x;

  const int grid_width = cdiv(N, BLOCK_N);
  const int block_id_m = block_id / grid_width;
  const int block_id_n = block_id % grid_width;

  const int offset_m = block_id_m * BLOCK_M;
  const int offset_n = block_id_n * BLOCK_N;

  A += offset_m * K;
  B += offset_n;

  __shared__ float A_shmem[BLOCK_M][BLOCK_K];
  __shared__ float B_shmem[BLOCK_K][BLOCK_N];

  // each thread will calculate (THREAD_M, THREAD_N) thread-tile of output (BLOCK_M, BLOCK_N) block-tile
  constexpr int THREAD_M = BLOCK_M * BLOCK_N / BLOCK_SIZE / THREAD_N;
  float acc[THREAD_M][THREAD_N] = {0.0f};

  const int mini_grid_width = BLOCK_N / THREAD_N;
  const int thread_tile_id_m = tid / mini_grid_width;
  const int thread_tile_id_n = tid % mini_grid_width;

  const int thread_tile_offset_m = thread_tile_id_m * THREAD_M;
  const int thread_tile_offset_n = thread_tile_id_n * THREAD_N;

  const float *A_thread_tile = reinterpret_cast<const float *>(A_shmem) + thread_tile_offset_m * BLOCK_K;
  const float *B_thread_tile = reinterpret_cast<const float *>(B_shmem) + thread_tile_offset_n;

  for (int offset_k = 0; offset_k < K; offset_k += BLOCK_K) {
    load_shmem<BLOCK_SIZE, BLOCK_M, BLOCK_K>(A, K, M - offset_m, K - offset_k, A_shmem, tid);
    load_shmem<BLOCK_SIZE, BLOCK_K, BLOCK_N>(B, N, K - offset_k, N - offset_n, B_shmem, tid);
    __syncthreads();

    // mini-matmul with thread-tile
    // notice that we put k as the outermost loop.
    // column of A_thread_tile and row of B_thread_tile is cached to A_reg[] and B_reg[].
    // there is shared memory bank conflict
    for (int k = 0; k < BLOCK_K; k++) {
      float A_reg[THREAD_M];
      float B_reg[THREAD_N];

      for (int m = 0; m < THREAD_M; m++)
        A_reg[m] = A_thread_tile[m * BLOCK_K + k];
      
      for (int n = 0; n < THREAD_N; n++)
        B_reg[n] = B_thread_tile[k * BLOCK_N + n];

      for (int m = 0; m < THREAD_M; m++)
        for (int n = 0; n < THREAD_N; n++)
          acc[m][n] += A_reg[m] * B[n];
    }
    __syncthreads();

    A += BLOCK_K;
    B += BLOCK_K * N;
  }

  C += (offset_m + thread_tile_offset_m) * N + (offset_n + thread_tile_offset_n);
  float4 *C_float4 = reinterpret_cast<float4 *>(C);

  // uncoalesced memory write
  // fixing it doesn't seem to make the kernel faster.
  // using vectorized write will help with uncoalesced memory write (issue fewer txn).
  for (int m = 0; m < THREAD_M; m++) {
    for (int n = 0; n < THREAD_N; n += 4) {
      float4 tmp = {acc[m][n], acc[m][n+1], acc[m][n+2], acc[m][n+3]};

      // TODO: handle n % 4 != 0
      if (m < (M - (offset_m + thread_tile_offset_m)) && n < (N - (offset_n + thread_tile_offset_n)))
        C_float4[(m * N + n) / 4] = tmp;
    }
  }
}

void matmul_v4(const float *A, const float *B, float *C, int M, int N, int K) {
  const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32;
  const int THREAD_N = 32;  // THREAD_M will be 2
  const int BLOCK_SIZE = 256;
  const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
  matmul_v4_kernel<BLOCK_SIZE, BLOCK_M, BLOCK_N, BLOCK_K, THREAD_N><<<grid_size, BLOCK_SIZE>>>(A, B, C, M, N, K);
}
