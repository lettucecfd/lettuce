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
 * @param length the length of the memory region (f/f_next) which is equal to with*height of the field
 */
template<typename scalar_t, index_t horizontal_offset, index_t vertical_offset>
__global__ void
lettuce_cuda_stream_and_collide_kernel(const scalar_t *f, scalar_t *f_next, scalar_t *collision, c_index_t width, c_index_t length)
{
    const auto horizontal_index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto vertical_index = blockIdx.y * blockDim.y + threadIdx.y;

    /*
     * read and collide
     */

    // calculate the index for each force
    const auto index = ([&]()
    {
        const auto previous_lines_offset = (vertical_index + vertical_offset) * width;
        const auto previous_nodes_offset = horizontal_index + horizontal_offset;
        return previous_lines_offset + previous_nodes_offset;
    }());
    const auto index_r = (length * 1u) + index;
    const auto index_b = (length * 2u) + index;
    const auto index_l = (length * 3u) + index;
    const auto index_t = (length * 4u) + index;
    const auto index_br = (length * 5u) + index;
    const auto index_tr = (length * 6u) + index;
    const auto index_tl = (length * 7u) + index;
    const auto index_bl = (length * 8u) + index;

    // apply collision
    const auto force_tl_next = f[index_tl] + collision[index_tl];
    const auto force_t_next  = f[index_t]  + collision[index_t];
    const auto force_tr_next = f[index_tr] + collision[index_tr];
    const auto force_l_next  = f[index_l]  + collision[index_l];
    const auto force_next    = f[index]    + collision[index];
    const auto force_r_next  = f[index_r]  + collision[index_r];
    const auto force_bl_next = f[index_bl] + collision[index_bl];
    const auto force_b_next  = f[index_b]  + collision[index_b];
    const auto force_br_next = f[index_br] + collision[index_br];

    /*
     * steam and write
     */

    // helper functions (for readability) (will be optimised away)
    const auto stream_l = [](c_index_t i)
    { return i - 1; };
    const auto stream_r = [](c_index_t i)
    { return i + 1; };
    const auto stream_t = [&width](c_index_t i)
    { return i - width; };
    const auto stream_b = [&width](c_index_t i)
    { return i + width; };

    // calculate the new index for each force
    const auto &streamed_index = index;
    const auto streamed_index_r = stream_r(index_r);
    const auto streamed_index_b = stream_b(index_b);
    const auto streamed_index_l = stream_l(index_l);
    const auto streamed_index_t = stream_t(index_t);
    const auto streamed_index_br = stream_b(streamed_index_r);
    const auto streamed_index_tr = stream_t(streamed_index_r);
    const auto streamed_index_tl = stream_t(streamed_index_l);
    const auto streamed_index_bl = stream_b(streamed_index_l);

    // write next forces
    f_next[streamed_index_tl] = force_tl_next;
    f_next[streamed_index_t] = force_t_next;
    f_next[streamed_index_tr] = force_tr_next;
    f_next[streamed_index_l] = force_l_next;
    f_next[streamed_index] = force_next;
    f_next[streamed_index_r] = force_r_next;
    f_next[streamed_index_bl] = force_bl_next;
    f_next[streamed_index_b] = force_b_next;
    f_next[streamed_index_br] = force_br_next;
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

    // TODO these values are used in a template. either dispatch the template or make these a preprocessor constant!
    constexpr auto horizontal_offset = 1u;
    constexpr auto vertical_offset = 1u;

    const auto width = static_cast<index_t> (f.sizes()[1]);
    const auto height = static_cast<index_t> (f.sizes()[2]);

    const auto block_count = ([&]()
    {
        const auto horizontal_ghost_node_count = 2u * horizontal_offset;
        const auto vertical_ghost_node_count = 2u * vertical_offset;

        const auto processing_width = width - horizontal_ghost_node_count;
        const auto processing_height = height - vertical_ghost_node_count;

        // TODO check weather to wrap into a debug only
        assert((processing_width % thread_count.x) == 0u);
        assert((processing_height % thread_count.y) == 0u);

        const auto horizontal_block_count = processing_width / thread_count.x;
        const auto vertical_block_count = processing_height / thread_count.y;

        return dim3{horizontal_block_count, vertical_block_count};
    }());

    /*
     * call the cuda kernel in a safe way for all supported float types
     */

    AT_DISPATCH_FLOATING_TYPES(f.scalar_type(), "lettuce_cuda_stream_and_collide", ([&]
    {
        lettuce_cuda_stream_and_collide_kernel<scalar_t, 1u, 1u><<<block_count, thread_count>>>(
                f.data<scalar_t>(),
                f_next.data<scalar_t>(),
                collision.data<scalar_t>(),
                width,
                width * height
        );
    }));
}
