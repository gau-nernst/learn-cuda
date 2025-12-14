#include <torch/extension.h>
#include <cuda_bf16.h>

using AttentionFn = void(
  const nv_bfloat16 *Q,  // [bs, len_q, DIM]
  const nv_bfloat16 *K,  // [bs, len_kv, DIM]
  const nv_bfloat16 *V,  // [bs, len_kv, DIM]
  nv_bfloat16 *O,        // [bs, len_q, DIM]
  int bs,
  int len_q,
  int len_kv,
  int dim);

AttentionFn attention_v1;
// AttentionFn attention_v2;
// AttentionFn attention_v3;
// AttentionFn attention_v4;
// AttentionFn attention_v5;

template<AttentionFn attention>
at::Tensor sdpa(
  const at::Tensor& Q,
  const at::Tensor& K,
  const at::Tensor& V) {

  const int bs = Q.size(0) * Q.size(1);
  const int len_q = Q.size(2);
  const int len_kv = K.size(2);
  const int dim = Q.size(3);

  at::Tensor O = at::empty_like(Q);

  auto Q_ptr = reinterpret_cast<const nv_bfloat16 *>(Q.data_ptr());
  auto K_ptr = reinterpret_cast<const nv_bfloat16 *>(K.data_ptr());
  auto V_ptr = reinterpret_cast<const nv_bfloat16 *>(V.data_ptr());
  auto O_ptr = reinterpret_cast<nv_bfloat16 *>(O.data_ptr());

  attention(Q_ptr, K_ptr, V_ptr, O_ptr, bs, len_q, len_kv, dim);

  return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sdpa_v1", &sdpa<attention_v1>);
//   m.def("sdpa_v2", &sdpa<attention_v2>);
//   m.def("sdpa_v3", &sdpa<attention_v3>);
//   m.def("sdpa_v4", &sdpa<attention_v4>);
//   m.def("sdpa_v5", &sdpa<attention_v5>);
}
