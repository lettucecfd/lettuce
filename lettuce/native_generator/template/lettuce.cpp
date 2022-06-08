#include "lettuce.hpp"

#define CHECK_CUDA(x) TORCH_CHECK((x).device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

void
lettuce_{name}({cpp_wrapper_parameter})
{{
    CHECK_CUDA(f); CHECK_CONTIGUOUS(f);
    lettuce_cuda_{name}({cuda_wrapper_parameter_values});
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{{
    m.def("collide_and_stream_{name}", &lettuce_{name}, "collide_and_stream_{name}");
}}
