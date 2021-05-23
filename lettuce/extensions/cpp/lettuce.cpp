#include <torch/extension.h>
#include <vector>

at::Tensor
stream_standard_collision_bgk(at::Tensor input)
{
    at::Tensor output{};

    stream_standard(output);
    collision_bgk(output);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "collection of native implementations of steam and collision algorithms";
    m.def("stream_standard_collision_bgk", &stream_standard_collision_bgk, "native implementation of StandardStream and BGKCollision combined");
}
