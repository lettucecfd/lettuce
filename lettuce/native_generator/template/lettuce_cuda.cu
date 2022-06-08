#if _MSC_VER && !__INTEL_COMPILER
#pragma warning ( push )
#pragma warning ( disable : 4067 )
#pragma warning ( disable : 4624 )
#endif

#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>

#if _MSC_VER && !__INTEL_COMPILER
#pragma warning ( pop )
#endif

using index_t = int;
using byte_t = unsigned char;

template<typename scalar_t>
__global__ void
lettuce_cuda_{name}_kernel({kernel_parameter})
{{
  {constexpr_buffer}

  {index_buffer}
  {{
    {node_buffer}

#pragma unroll
    for (index_t i = 0; i < q; ++i)
    {{
      {distribution_buffer}
    }}
  }}
  {write_buffer}
}}

void
lettuce_cuda_{name}({cuda_wrapper_parameter})
{{
  {cuda_wrapper_buffer}

  AT_DISPATCH_FLOATING_TYPES(f.scalar_type(), "lettuce_cuda_{name}", [&]
  {{
    lettuce_cuda_{name}_kernel<scalar_t><<<block_count, thread_count>>>({kernel_parameter_values});
  }});
}}
