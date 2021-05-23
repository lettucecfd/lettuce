// BEGIN SUPPRESS WARNINGS FOR THIRD PARTY INCLUDES
#if _MSC_VER && !__INTEL_COMPILER
#pragma warning ( push )
#pragma warning ( disable : 4067 )
#pragma warning ( disable : 4624 )
#endif

#include <torch/extension.h>

// END SUPPRESS WARNINGS FOR THIRD PARTY INCLUDES
#if _MSC_VER && !__INTEL_COMPILER
#pragma warning ( pop )
#endif

template<typename scalar_t>
__global__ void
lettuce_cuda_stream_and_collide_kernel(const scalar_t *f, scalar_t *f_next)
{
    // todo stream with 2d9 while allowing *f = *f_next
}

void
lettuce_cuda_stream_and_collide(at::Tensor f, at::Tensor f_next)
{
    // todo stream with 2d9 while allowing *f = *f_next
    // todo collide while allowing *f = *f_next

    const int threads = 512;
    const dim3 blocks(1, 1, 1);

    AT_DISPATCH_FLOATING_TYPES(f.scalar_type(), "lettuce_cuda_stream_and_collide", ([&]
    { lettuce_cuda_stream_and_collide_kernel<scalar_t><<<blocks, threads>>>(f.data<scalar_t>(), f_next.data<scalar_t>()); }));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("lettuce_cuda_stream_and_collide", &lettuce_cuda_stream_and_collide, "lettuce stream and collide");
}
