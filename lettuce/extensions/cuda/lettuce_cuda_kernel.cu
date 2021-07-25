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

#include "lettuce_cuda_stencil.cu"
#include "lettuce_cuda_lattice.cu"

using index_t = unsigned int;

template<typename scalar_t, index_t D, index_t Q, stencil<scalar_t, D, Q> STENCIL>
__device__ __forceinline__ void
quadratic_equilibrium_collision(scalar_t *f, scalar_t tau)
{
    /*
     * avoid recalculating the inverse of tau
     */

    const scalar_t tau_inv = 1.0 / tau;

    /*
     * begin calculating the equilibrium
     */

    constexpr auto cs_pow_two = STENCIL.cs * STENCIL.cs;
    constexpr auto two_cs_pow_two = cs_pow_two + cs_pow_two;

    scalar_t rho;
    lattice::rho<scalar_t, D>(f, &rho);

    scalar_t j[2];
    lattice::j<scalar_t, D, Q, STENCIL>(f, &j);

    const scalar_t u[2] = {j[0] / rho, j[1] / rho};
    const scalar_t uxu = u[0] * u[0] + u[1] * u[1];

#pragma unroll
    for (index_t i = 0; i < 9; ++i)
    {
        const scalar_t exu = STENCIL.e[i][0] * u[0] + STENCIL.e[i][1] * u[1];

        const scalar_t tmp0 = exu / cs_pow_two;
        const scalar_t tmp1 = rho * (((exu + exu - uxu) / two_cs_pow_two) + (0.5 * (tmp0 * tmp0)) + 1.0);
        const scalar_t feq = tmp1 * STENCIL.w[i];

        /*
         * finally apply the collision operator
         */

        f[i] = f[i] - (tau_inv * (f[i] - feq));
    }
}

template<typename scalar_t>
__device__ __forceinline__ void
d2q9_read(const scalar_t *f, scalar_t *f_reg, index_t length, index_t index)
{
    // the reading index is trivial as it is the same relative index in each dimension
    // [by using an iterator bypass some multiplications]
    auto index_it = index; f_reg[0] = f[index_it];
    index_it += length;    f_reg[1] = f[index_it];
    index_it += length;    f_reg[2] = f[index_it];
    index_it += length;    f_reg[3] = f[index_it];
    index_it += length;    f_reg[4] = f[index_it];
    index_it += length;    f_reg[5] = f[index_it];
    index_it += length;    f_reg[6] = f[index_it];
    index_it += length;    f_reg[7] = f[index_it];
    index_it += length;    f_reg[8] = f[index_it];
}

template<typename scalar_t>
__device__ __forceinline__ void
d2q9_standard_stream_read(
        const scalar_t *f, scalar_t *f_reg,
        index_t width, index_t height, index_t length,
        index_t index, index_t horizontal_index, index_t vertical_index,
        index_t vertical_m_offset)
{
    /*
     * define needed variables for the streaming
     */

    // alter name for convenience
    const auto &horizontal_m_offset = horizontal_index;

    // pre calculate the vertical and horizontal offsets
    const auto vertical_t_offset = ((vertical_index == 0) ? height - 1 : (vertical_index - 1)) * width;
    const auto vertical_b_offset = (((vertical_index + 1) == height) ? 0 : (vertical_index + 1)) * width;
    const auto horizontal_l_offset = (horizontal_index == 0) ? width - 1 : (horizontal_index - 1);
    const auto horizontal_r_offset = ((horizontal_index + 1) == width) ? 0 : (horizontal_index + 1);

    /*
     * read the neighbor distributions into the current/register node
     */

    // center force is trivial as it stays in place
    f_reg[0] = f[index];

    // the index from which to stream from is calculated by:
    // - a dimensional offset (which is calculated by iteration)
    //   [by using an iterator bypass some multiplications]
    // - a relative horizontal offset (corresponding to the dimension)
    // - a relative vertical offset (corresponding to the dimension)
    auto dim_offset_it = length;
    f_reg[1] = f[dim_offset_it + horizontal_m_offset + vertical_t_offset];
    dim_offset_it += length;
    f_reg[2] = f[dim_offset_it + horizontal_l_offset + vertical_m_offset];
    dim_offset_it += length;
    f_reg[3] = f[dim_offset_it + horizontal_m_offset + vertical_b_offset];
    dim_offset_it += length;
    f_reg[4] = f[dim_offset_it + horizontal_r_offset + vertical_m_offset];
    dim_offset_it += length;
    f_reg[5] = f[dim_offset_it + horizontal_l_offset + vertical_t_offset];
    dim_offset_it += length;
    f_reg[6] = f[dim_offset_it + horizontal_l_offset + vertical_b_offset];
    dim_offset_it += length;
    f_reg[7] = f[dim_offset_it + horizontal_r_offset + vertical_b_offset];
    dim_offset_it += length;
    f_reg[8] = f[dim_offset_it + horizontal_r_offset + vertical_t_offset];
}

template<typename scalar_t>
__device__ __forceinline__ void
d2q9_write(const scalar_t *f_reg, scalar_t *f_next, index_t length, index_t index)
{
    // the writing index is trivial as it is the same relative index in each dimension
    // [by using an iterator bypass some multiplications]
    auto index_it = index;
    f_next[index_it] = f_reg[0];
    index_it += length;
    f_next[index_it] = f_reg[1];
    index_it += length;
    f_next[index_it] = f_reg[2];
    index_it += length;
    f_next[index_it] = f_reg[3];
    index_it += length;
    f_next[index_it] = f_reg[4];
    index_it += length;
    f_next[index_it] = f_reg[5];
    index_it += length;
    f_next[index_it] = f_reg[6];
    index_it += length;
    f_next[index_it] = f_reg[7];
    index_it += length;
    f_next[index_it] = f_reg[8];
}

/**
 * collide and stream the given field (f)
 *
 * steps: TODO out of date documentation
 * 1. read all nodes (from f)
 * 2. add collision value (local)
 * 3. stream values (local)
 * 4. write all nodes (to f_next)
 *
 * @tparam scalar_t the scalar type of the tensor passed. typically defined by AT_DISPATCH_FLOATING_TYPES
 * @param f the fluid forces at time t (at the moment)
 * @param f_next a memory region as big as f which is used to write the simulation results into
 * @param tau TODO document better what tau is
 * @param width the width of the field
 * @param length the length of the memory region (f/f_next) (second dimension) which is equal to with*height of the field
 */
template<typename scalar_t>
__global__ void
lettuce_cuda_stream_and_collide_kernel(const scalar_t *f, scalar_t *f_next, scalar_t tau, index_t width, index_t height, index_t length)
{
    /*
     * define needed variables for the streaming
     */

    // pre calculate the vertical and horizontal indices before streaming
    const auto horizontal_index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto vertical_index = blockIdx.y * blockDim.y + threadIdx.y;

    // pre calculate the vertical and horizontal offsets before streaming
    const auto &horizontal_m_offset = horizontal_index;
    const auto vertical_m_offset = vertical_index * width;

    // pre calculate the current index
    const auto index = vertical_m_offset + horizontal_m_offset;

    /*
     * do the work
     */

    // standard stream & read
    scalar_t f_reg[9];
    d2q9_standard_stream_read(f, &(f_reg[0]), width, height, length, index, horizontal_index, vertical_index, vertical_m_offset);

    // collide & write
    quadratic_equilibrium_collision<scalar_t, 2, 9, d2q9<scalar_t>>(&(f_reg[0]), tau);
    d2q9_write(f_reg, f_next, length, index);
}

/**
 * TODO document
 */
template<typename scalar_t>
__global__ void
lettuce_cuda_stream_kernel(const scalar_t *f, scalar_t *f_next, index_t width, index_t height, index_t length)
{
    /*
     * define needed variables for the streaming
     */

    // pre calculate the vertical and horizontal indices before streaming
    const auto horizontal_index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto vertical_index = blockIdx.y * blockDim.y + threadIdx.y;

    // pre calculate the vertical and horizontal offsets before streaming
    const auto &horizontal_m_offset = horizontal_index;
    const auto vertical_m_offset = vertical_index * width;

    // pre calculate the current index
    const auto index = vertical_m_offset + horizontal_m_offset;

    /*
     * do the work
     */

    // standard stream & read & write
    scalar_t f_reg[9];
    d2q9_standard_stream_read(f, &(f_reg[0]), width, height, length, index, horizontal_index, vertical_index, vertical_m_offset);
    d2q9_write(f_reg, f_next, length, index);
}

/**
 * TODO document
 */
template<typename scalar_t>
__global__ void
lettuce_cuda_collide_kernel(const scalar_t *f, scalar_t *f_next, scalar_t tau, index_t width, index_t height, index_t length)
{
    /*
     * define needed variables for the streaming
     */

    // pre calculate the vertical and horizontal indices before streaming
    const auto horizontal_index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto vertical_index = blockIdx.y * blockDim.y + threadIdx.y;

    // pre calculate the vertical and horizontal offsets before streaming
    const auto &horizontal_m_offset = horizontal_index;
    const auto vertical_m_offset = vertical_index * width;

    // pre calculate the current index
    const auto index = vertical_m_offset + horizontal_m_offset;

    /*
     * do the work
     */

    // read
    scalar_t f_reg[9];
    d2q9_read(f, &(f_reg[0]), length, index);

    // collide & write
    quadratic_equilibrium_collision<scalar_t, 2, 9, d2q9<scalar_t>>(&(f_reg[0]), tau);
    d2q9_write(f_reg, f_next, length, index);
}

void
lettuce_cuda_stream_and_collide(at::Tensor f, at::Tensor f_next, double tau)
{
    /*
     * Use all threads of one block (asserting the block support 1024 threads)
     */

    const auto thread_count = dim3{16u, 16u};

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
                static_cast<scalar_t>(tau),
                width,
                height,
                width * height
        );
        cudaDeviceSynchronize(); // TODO maybe replace with torch.cuda.sync().
        //      this may be more efficient as is bridges
        //      the time to return from the native code.
        //      but maybe this time is not noticeable ...
    }));
}

void
lettuce_cuda_stream(at::Tensor f, at::Tensor f_next)
{
    /*
     * Use all threads of one block (asserting the block support 1024 threads)
     */

    const auto thread_count = dim3{16u, 16u};

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
        lettuce_cuda_stream_kernel<scalar_t><<<block_count, thread_count>>>(
                f.data<scalar_t>(),
                f_next.data<scalar_t>(),
                width,
                height,
                width * height
        );
        cudaDeviceSynchronize(); // TODO maybe replace with torch.cuda.sync().
        //      this may be more efficient as is bridges
        //      the time to return from the native code.
        //      but maybe this time is not noticeable ...
    }));
}

void
lettuce_cuda_collide(at::Tensor f, at::Tensor f_next, double tau)
{
    /*
     * Use all threads of one block (asserting the block support 1024 threads)
     */

    const auto thread_count = dim3{16u, 16u};

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
        lettuce_cuda_collide_kernel<scalar_t><<<block_count, thread_count>>>(
                f.data<scalar_t>(),
                f_next.data<scalar_t>(),
                static_cast<scalar_t>(tau),
                width,
                height,
                width * height
        );
        cudaDeviceSynchronize(); // TODO maybe replace with torch.cuda.sync().
        //      this may be more efficient as is bridges
        //      the time to return from the native code.
        //      but maybe this time is not noticeable ...
    }));
}
