#if _MSC_VER && !__INTEL_COMPILER
#pragma warning ( push )
#pragma warning ( disable : 4067 )
#pragma warning ( disable : 4624 )
#endif

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#if _MSC_VER && !__INTEL_COMPILER
#pragma warning ( pop )
#endif

using index_t = unsigned int;
using c_index_t = const unsigned int;

/*
 * sample for a cuda helper function:
 *
 * ```CUDA
 *
 *     template<typename scalar_t>
 *     __device__ __forceinline__ scalar_t
 *     identity(scalar_t x)
 *     {
 *         return x;
 *     }
 *
 * ```
 */

/**
 * collide and stream the given field (f)
 *
 * steps:
 * 1. read all nodes (from f)
 * 2. add collision value (local)
 * 3. stream values (local)
 * 4. write all nodes (to f_next)
 *
 * @tparam scalar_t the scalar type of the tensor passed. typically defined by AT_DISPATCH_FLOATING_TYPES
 * @tparam horizontal_offset number of ghost nodes to the left and right of the processing data
 * @tparam vertical_offset number of ghost nodes to the top and bottom of the processing data
 * @param f the fluid forces at time t (at the moment)
 * @param f_next a memory region as big as f which is used to write the simulation results into
 * @param collision the calculated collision value which will be added to f before streaming. TODO later this value must be calculated locally
 * @param width the width of the field
 * @param length the length of the memory region (f/f_next) (second dimension) which is equal to with*height of the field
 */
template<typename scalar_t>
__global__ void
lettuce_cuda_stream_and_collide_kernel(const scalar_t *f, scalar_t *f_next, scalar_t *collision, c_index_t width, c_index_t height, c_index_t length)
{
    // pre calculate the vertical and horizontal indices before streaming
    const auto horizontal_index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto vertical_index   = blockIdx.y * blockDim.y + threadIdx.y;

    // pre calculate the vertical and horizontal offsets before streaming
    const auto &horizontal_offset = horizontal_index;
    const auto vertical_offset    = vertical_index * width;

    // pre calculate the vertical and horizontal offsets after streaming
    const auto vertical_t_offset   = ((vertical_index == 0)            ? height : (vertical_index - 1)) * width;
    const auto vertical_b_offset   = (((vertical_index + 1) == height) ? 0      : (vertical_index + 1)) * width;
    const auto horizontal_l_offset = (horizontal_index == 0)           ? width  : (horizontal_index - 1);
    const auto horizontal_r_offset = ((horizontal_index + 1) == width) ? 0      : (horizontal_index + 1);

    // pre calculate the current index
    const auto index = vertical_offset + horizontal_offset;

    /*
     * read and collide
     */

    // TODO write some inline documentation
    auto index_it = index; const auto force_next    = f[index_it] + collision[index_it];
    index_it += length;    const auto force_r_next  = f[index_it] + collision[index_it];
    index_it += length;    const auto force_b_next  = f[index_it] + collision[index_it];
    index_it += length;    const auto force_l_next  = f[index_it] + collision[index_it];
    index_it += length;    const auto force_t_next  = f[index_it] + collision[index_it];
    index_it += length;    const auto force_br_next = f[index_it] + collision[index_it];
    index_it += length;    const auto force_tr_next = f[index_it] + collision[index_it];
    index_it += length;    const auto force_tl_next = f[index_it] + collision[index_it];
    index_it += length;    const auto force_bl_next = f[index_it] + collision[index_it];

    /*
     * steam and write
     */

    f_next[index] = force_next;

    // TODO write some inline documentation
    auto dim_offset_it = length; f_next[dim_offset_it + horizontal_r_offset + vertical_offset  ] = force_r_next;
    dim_offset_it += length;     f_next[dim_offset_it + horizontal_offset   + vertical_b_offset] = force_b_next;
    dim_offset_it += length;     f_next[dim_offset_it + horizontal_l_offset + vertical_offset  ] = force_l_next;
    dim_offset_it += length;     f_next[dim_offset_it + horizontal_offset   + vertical_t_offset] = force_t_next;
    dim_offset_it += length;     f_next[dim_offset_it + horizontal_r_offset + vertical_b_offset] = force_br_next;
    dim_offset_it += length;     f_next[dim_offset_it + horizontal_r_offset + vertical_t_offset] = force_tr_next;
    dim_offset_it += length;     f_next[dim_offset_it + horizontal_l_offset + vertical_t_offset] = force_tl_next;
    dim_offset_it += length;     f_next[dim_offset_it + horizontal_l_offset + vertical_b_offset] = force_bl_next;
}

void
lettuce_cuda_stream_and_collide(at::Tensor f, at::Tensor f_next, at::Tensor collision)
{
    /*
     * Use all threads of one block (asserting the block support 1024 threads)
     */

    const auto thread_count = dim3{32u, 32u};

    /*
     * calculate constant values
     */

    const auto width = static_cast<index_t> (f.sizes()[1]);
    const auto height = static_cast<index_t> (f.sizes()[2]);

    const auto block_count = ([&]()
    {
        assert((width % thread_count.x) == 0u);
        assert((height % thread_count.y) == 0u);

        const auto horizontal_block_count = width / thread_count.x;
        const auto vertical_block_count = height / thread_count.y;

        return dim3{horizontal_block_count, vertical_block_count};
    }());

    /*
     * call the cuda kernel in a safe way for all supported float types
     */

    AT_DISPATCH_FLOATING_TYPES(f.scalar_type(), "lettuce_cuda_stream_and_collide", ([&]
    {
        lettuce_cuda_stream_and_collide_kernel<scalar_t><<<block_count, thread_count>>>(
                f.data<scalar_t>(),
                f_next.data<scalar_t>(),
                collision.data<scalar_t>(),
                width,
                height,
                width * height
        );
    }));
}
