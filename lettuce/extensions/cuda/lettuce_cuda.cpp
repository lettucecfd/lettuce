#if _MSC_VER && !__INTEL_COMPILER
#pragma warning ( push )
#pragma warning ( disable : 4067 )
#pragma warning ( disable : 4624 )
#endif

#include <torch/extension.h>

#if _MSC_VER && !__INTEL_COMPILER
#pragma warning ( pop )
#endif

// Forward declare Cuda Function

void
lettuce_cuda_stream_and_collide(at::Tensor f, at::Tensor f_next, at::Tensor collision);

// C++ Interface

void
lettuce_stream_and_collide(at::Tensor f, at::Tensor f_next, at::Tensor collision)
{
    assert(f.sizes().size() == 3);
    assert(f.sizes()[0] == 9);

    lettuce_cuda_stream_and_collide(f, f_next, collision);
}

#ifdef TORCH_EXTENSION_NAME
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("stream_and_collide", &lettuce_stream_and_collide, "lettuce stream and collide (Cuda)");
}
#endif
