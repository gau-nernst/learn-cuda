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

// we can't use a larger block size since we are limited by 1024 threads per block
constexpr int V2_BLOCK_SIZE = 32;

// read 2D block into shared memory for caching
__global__ void matmul_v2_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
  const int tid_x = threadIdx.x;
  const int tid_y = threadIdx.y;

  const int m_offset = blockIdx.y * V2_BLOCK_SIZE;
  const int n_offset = blockIdx.x * V2_BLOCK_SIZE;

  A += m_offset * K;             // skip x rows
  B += n_offset;                 // skip y columns
  C += m_offset * N + n_offset;  // skip x rows and y columns

  // we cannot return early since all threads need to synchronize
  __shared__ float A_shmem[V2_BLOCK_SIZE][V2_BLOCK_SIZE];
  __shared__ float B_shmem[V2_BLOCK_SIZE][V2_BLOCK_SIZE];
  float acc = 0.0f;

  // we move block by block along K dim
  for (int k_offset = 0; k_offset < K; k_offset += V2_BLOCK_SIZE) {
    // load data from global memory (DDR/HBM) to shared memory (SRAM)
    // notice now each thread only loads 2 x n_blocks elements
    // coalesced memory read for both A and B
    A_shmem[tid_y][tid_x] = tid_y < (M - m_offset) && tid_x < (K - k_offset) ? A[tid_y * K + tid_x] : 0.0f;
    B_shmem[tid_y][tid_x] = tid_y < (K - k_offset) && tid_x < (N - n_offset) ? B[tid_y * N + tid_x] : 0.0f;

    // wait for all threads in a block to load data
    __syncthreads();

    // compute from shared memory
    for (int k = 0; k < V2_BLOCK_SIZE; k++)
      acc += A_shmem[tid_y][k] * B_shmem[k][tid_x];

    // wait to finish before moving to the next tile
    __syncthreads();

    A += V2_BLOCK_SIZE;      // stride 1 in K dim
    B += V2_BLOCK_SIZE * N;  // stride N in K dim
  }

  if (tid_y < (M - m_offset) && tid_x < (N - n_offset))
    C[tid_y * N + tid_x] = acc;
}

void matmul_v2(const float *A, const float *B, float *C, int M, int N, int K) {
  dim3 block_size(V2_BLOCK_SIZE, V2_BLOCK_SIZE);
  dim3 grid_size(cdiv(N, V2_BLOCK_SIZE), cdiv(M, V2_BLOCK_SIZE));
  matmul_v2_kernel<<<grid_size, block_size>>>(A, B, C, M, N, K);
}

// NOTE: to make it clear, BLOCK now only refers to block of threads. TILE refers to tile of data.
// we want to load a (HEIGHT, WIDTH) tile from global to shared memory.
// just load a BLOCK_SIZE of data until the whole tile is loaded.
template <int HEIGHT, int WIDTH, int BLOCK_SIZE>
__device__ void load_shmem(const float *in, int in_row_stride, int in_max_row, int in_max_col,
                           float out[HEIGHT][WIDTH], int tid) {
  for (int idx = tid; idx < HEIGHT * WIDTH; idx += BLOCK_SIZE) {
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;
    out[row][col] = row < in_max_row && col < in_max_col ? in[row * in_row_stride + col] : 0.0f;
  }
}

// thread coarsening
template <int TILE_M, int TILE_N, int TILE_K, int BLOCK_SIZE>
__global__ void matmul_v3_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
  const int tid = threadIdx.x;
  const int block_id = blockIdx.x;

  // assign block linearly
  const int grid_width = cdiv(N, TILE_N);
  const int block_id_m = block_id / grid_width;
  const int block_id_n = block_id % grid_width;

  const int m_offset = block_id_m * TILE_M;
  const int n_offset = block_id_n * TILE_N;

  A += m_offset * K;
  B += n_offset;
  C += m_offset * N + n_offset;

  __shared__ float A_shmem[TILE_M][TILE_K];
  __shared__ float B_shmem[TILE_K][TILE_N];

  // each thread is responsible for (TILE_M * TILE_N / BLOCK_SIZE) output elements
  float acc[TILE_M * TILE_N / BLOCK_SIZE] = {0.0f};

  // we move block by block along K dim
  for (int k_offset = 0; k_offset < K; k_offset += TILE_K) {
    // decouple global memory read, so we don't need to care about assigning which thread
    // to read which element.
    // load (TILE_M, TILE_K) from A and (TILE_K, TILE_N) from B
    load_shmem<TILE_M, TILE_K, BLOCK_SIZE>(A, K, M - m_offset, K - k_offset, A_shmem, tid);
    load_shmem<TILE_K, TILE_N, BLOCK_SIZE>(B, N, K - k_offset, N - n_offset, B_shmem, tid);
    __syncthreads();

    // do a mini matmul of (TILE_M, TILE_K) x (TILE_K, TILE_N) = (TILE_M, TILE_N)
    // simply assign a BLOCK_SIZE of threads to a BLOCK_SIZE of elements in output tile
    for (int idx = tid; idx < TILE_M * TILE_N; idx += BLOCK_SIZE) {
      const int local_idx = idx / BLOCK_SIZE;
      const int col = idx % TILE_N;
      const int row = idx / TILE_N;

      for (int k = 0; k < TILE_K; k++)
        acc[local_idx] += A_shmem[row][k] * B_shmem[k][col];
    }
    __syncthreads();

    A += TILE_K;
    B += TILE_K * N;
  }

  // write (TILE_M, TILE_N) to C
  for (int idx = tid; idx < TILE_M * TILE_N; idx += BLOCK_SIZE) {
    const int local_idx = idx / BLOCK_SIZE;
    const int col = idx % TILE_N;
    const int row = idx / TILE_N;

    if (row < (M - m_offset) && col < (N - n_offset))
      C[row * N + col] = acc[local_idx];
  }
}

void matmul_v3(const float *A, const float *B, float *C, int M, int N, int K) {
  const int TILE_M = 128, TILE_N = 128, TILE_K = 32;
  const int block_size = 256;
  const int grid_size = cdiv(M * N, TILE_M * TILE_N);
  matmul_v3_kernel<TILE_M, TILE_N, TILE_K, block_size><<<grid_size, block_size>>>(A, B, C, M, N, K);
}
