#include <torch/library.h>
#include <ATen/ATen.h>

#define STRINGIFY(x) #x
#define CUDA_CHECK(call)                                                             \
  do {                                                                               \
    cudaError_t err = call;                                                          \
    TORCH_CHECK(err == cudaSuccess, STRINGIFY(call), ": ", cudaGetErrorString(err)); \
  } while (0)

void enable_p2p(int64_t rank, int64_t world_size) {
  CUDA_CHECK(cudaSetDevice(rank));
  for (int i = 0; i < world_size; i++) {
    if (i == rank)
      continue;
    auto res = cudaDeviceEnablePeerAccess(i, 0);
    if (res != cudaSuccess && res != cudaErrorPeerAccessAlreadyEnabled)
      CUDA_CHECK(res);
  }
}

// input is CUDA, but output is CPU
at::Tensor get_ipc_handle(const at::Tensor& x) {
  // IPC handle as a tensor
  at::Tensor h = at::empty({sizeof(cudaIpcMemHandle_t)}, at::TensorOptions().dtype(at::kChar).device(at::kCPU));
  auto h_ptr = reinterpret_cast<cudaIpcMemHandle_t *>(h.data_ptr());
  CUDA_CHECK(cudaIpcGetMemHandle(h_ptr, x.data_ptr()));
  return h;
}

int64_t open_ipc_handle(const at::Tensor& handle) {
  void *ptr;
  auto h = reinterpret_cast<cudaIpcMemHandle_t *>(handle.data_ptr())[0];
  CUDA_CHECK(cudaIpcOpenMemHandle(&ptr, h, cudaIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<int64_t>(ptr);
}

template <typename T>
__device__
T *translate(T *ptr, const int64_t *heap_bases, int local_rank, int remote_rank) {
  static_assert(sizeof(ptr) == sizeof(int64_t));
  const int64_t ptr_i64 = reinterpret_cast<int64_t>(ptr);
  const int64_t new_ptr_i64 = heap_bases[remote_rank] + (ptr_i64 - heap_bases[local_rank]);
  return reinterpret_cast<T *>(new_ptr_i64);
}

__global__
void check_p2p_write_kernel(
        int32_t *symmetric_ptr,
  const int64_t *heap_bases,
  int local_rank
) {
  const int remote_rank = threadIdx.x;
  translate(symmetric_ptr, heap_bases, local_rank, remote_rank)[local_rank] = local_rank;
}

void check_p2p_write(
        at::Tensor& symmetric_data,
  const at::Tensor& heap_bases,
  int64_t local_rank,
  int64_t world_size
) {
  check_p2p_write_kernel<<<1, world_size>>>(symmetric_data.data_ptr<int32_t>(),
                                            heap_bases.data_ptr<int64_t>(),
                                            local_rank);
}

__global__
void check_p2p_read_kernel(
        int32_t *local_ptr,
  const int32_t *symmetric_ptr,
  const int64_t *heap_bases,
  int local_rank
) {
  const int remote_rank = threadIdx.x;
  // this is basically a2a - read remote, write local
  const int32_t data = translate(symmetric_ptr, heap_bases, local_rank, remote_rank)[local_rank];
  local_ptr[remote_rank] = data;
}

void check_p2p_read(
        at::Tensor& local_data,
  const at::Tensor& symmetric_data,
  const at::Tensor& heap_bases,
  int64_t local_rank,
  int64_t world_size
) {
  check_p2p_read_kernel<<<1, world_size>>>(local_data.data_ptr<int32_t>(),
                                           symmetric_data.data_ptr<int32_t>(),
                                           heap_bases.data_ptr<int64_t>(),
                                           local_rank);
}

TORCH_LIBRARY(p2p_module, m) {
  m.def("enable_p2p(int rank, int world_size) -> ()");
  m.impl("enable_p2p", &enable_p2p);

  m.def("get_ipc_handle(Tensor x) -> Tensor");
  m.impl("get_ipc_handle", &get_ipc_handle);

  m.def("open_ipc_handle(Tensor handle) -> int");
  m.impl("open_ipc_handle", &open_ipc_handle);

  m.def("check_p2p_write(Tensor(a!) symmetric_data, Tensor heap_bases, int local_rank, int world_size) -> ()");
  m.impl("check_p2p_write", at::kCUDA, &check_p2p_write);

  m.def("check_p2p_read(Tensor(a!) local_data, Tensor symmetric, Tensor heap_bases, int local_rank, int world_size) -> ()");
  m.impl("check_p2p_read", at::kCUDA, &check_p2p_read);
}
