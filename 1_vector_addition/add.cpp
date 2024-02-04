#include <torch/extension.h>

torch::Tensor add(torch::Tensor input1, torch::Tensor input2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add", &add, "Add two vectors");
}
