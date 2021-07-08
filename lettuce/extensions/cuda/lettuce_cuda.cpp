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
lettuce_cuda_stream_and_collide(at::Tensor f, at::Tensor f_next, double tau);

// C++ Interface

#define CHECK_CUDA(x) TORCH_CHECK((x).device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_SHAPE0(x) TORCH_CHECK((x).sizes()[0] == 9, #x " must have shape (9,x,y)")
#define CHECK_SHAPE1(x) TORCH_CHECK((x).sizes()[1] % 16 == 0, #x " must have x dimension that is dividable by 16")
#define CHECK_SHAPE2(x) TORCH_CHECK((x).sizes()[2] % 16 == 0, #x " must have y dimension that is dividable by 16")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_SHAPE0(x); CHECK_SHAPE1(x); CHECK_SHAPE2(x)

void
lettuce_stream_and_collide(at::Tensor f, at::Tensor f_next, double tau)
{
    CHECK_INPUT(f);
    CHECK_INPUT(f_next);
    TORCH_CHECK(f.sizes() == f_next.sizes(), "f and f_next need to have the same dimensions!")

    lettuce_cuda_stream_and_collide(f, f_next, tau);
}

#ifdef TORCH_EXTENSION_NAME
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("stream_and_collide", &lettuce_stream_and_collide, "lettuce stream and collide (Cuda)");
}
#endif
