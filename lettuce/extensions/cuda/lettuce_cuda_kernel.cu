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

template<typename scalar_t>
__device__ __forceinline__ void
d3q9_quadratic_equilibrium_collision(scalar_t *f, scalar_t tau)
{
    /*
     * define some constants for the d2q9 stencil
     */

    constexpr auto sqrt_3 = static_cast<scalar_t> (1.7320508075688772935274463415058723669428052538103806280558069794);
    constexpr scalar_t d2q9_e[9][2] = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}, {-1.0, -1.0}, {1.0, -1.0}};
    constexpr scalar_t d2q9_w[9] = {4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};
    constexpr scalar_t d2q9_cs = 1.0 / sqrt_3;
    constexpr scalar_t d2q9_cs_pow_two = d2q9_cs * d2q9_cs;
    constexpr scalar_t d2q9_two_cs_pow_two = d2q9_cs_pow_two + d2q9_cs_pow_two;

    /*
     * begin calculating the equilibrium
     */

    const scalar_t j[2] = {
              d2q9_e[0][0] * f[0]
            + d2q9_e[1][0] * f[1]
            + d2q9_e[2][0] * f[2]
            + d2q9_e[3][0] * f[3]
            + d2q9_e[4][0] * f[4]
            + d2q9_e[5][0] * f[5]
            + d2q9_e[6][0] * f[6]
            + d2q9_e[7][0] * f[7]
            + d2q9_e[8][0] * f[8],
              d2q9_e[0][1] * f[0]
            + d2q9_e[1][1] * f[1]
            + d2q9_e[2][1] * f[2]
            + d2q9_e[3][1] * f[3]
            + d2q9_e[4][1] * f[4]
            + d2q9_e[5][1] * f[5]
            + d2q9_e[6][1] * f[6]
            + d2q9_e[7][1] * f[7]
            + d2q9_e[8][1] * f[8]
    };

    const scalar_t rho = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8];

    const scalar_t u[2] = {j[0] / rho, j[1] / rho};

    const scalar_t exu[9] = {
            d2q9_e[0][0] * u[0] + d2q9_e[0][1] * u[1],
            d2q9_e[1][0] * u[0] + d2q9_e[1][1] * u[1],
            d2q9_e[2][0] * u[0] + d2q9_e[2][1] * u[1],
            d2q9_e[3][0] * u[0] + d2q9_e[3][1] * u[1],
            d2q9_e[4][0] * u[0] + d2q9_e[4][1] * u[1],
            d2q9_e[5][0] * u[0] + d2q9_e[5][1] * u[1],
            d2q9_e[6][0] * u[0] + d2q9_e[6][1] * u[1],
            d2q9_e[7][0] * u[0] + d2q9_e[7][1] * u[1],
            d2q9_e[8][0] * u[0] + d2q9_e[8][1] * u[1]
    };
    const scalar_t uxu = u[0] * u[0] + u[1] * u[1];

    // TODO is there a better name for this variables?

    const scalar_t tmp0[9] = {
            exu[0] / d2q9_cs_pow_two,
            exu[1] / d2q9_cs_pow_two,
            exu[2] / d2q9_cs_pow_two,
            exu[3] / d2q9_cs_pow_two,
            exu[4] / d2q9_cs_pow_two,
            exu[5] / d2q9_cs_pow_two,
            exu[6] / d2q9_cs_pow_two,
            exu[7] / d2q9_cs_pow_two,
            exu[8] / d2q9_cs_pow_two
    };
    const scalar_t tmp[9] = {
            rho * (((exu[0] + exu[0] - uxu) / d2q9_two_cs_pow_two) + (0.5 * (tmp0[0] * tmp0[0])) + 1.0),
            rho * (((exu[1] + exu[1] - uxu) / d2q9_two_cs_pow_two) + (0.5 * (tmp0[1] * tmp0[1])) + 1.0),
            rho * (((exu[2] + exu[2] - uxu) / d2q9_two_cs_pow_two) + (0.5 * (tmp0[2] * tmp0[2])) + 1.0),
            rho * (((exu[3] + exu[3] - uxu) / d2q9_two_cs_pow_two) + (0.5 * (tmp0[3] * tmp0[3])) + 1.0),
            rho * (((exu[4] + exu[4] - uxu) / d2q9_two_cs_pow_two) + (0.5 * (tmp0[4] * tmp0[4])) + 1.0),
            rho * (((exu[5] + exu[5] - uxu) / d2q9_two_cs_pow_two) + (0.5 * (tmp0[5] * tmp0[5])) + 1.0),
            rho * (((exu[6] + exu[6] - uxu) / d2q9_two_cs_pow_two) + (0.5 * (tmp0[6] * tmp0[6])) + 1.0),
            rho * (((exu[7] + exu[7] - uxu) / d2q9_two_cs_pow_two) + (0.5 * (tmp0[7] * tmp0[7])) + 1.0),
            rho * (((exu[8] + exu[8] - uxu) / d2q9_two_cs_pow_two) + (0.5 * (tmp0[8] * tmp0[8])) + 1.0)
    };

    const scalar_t feq[9] = {
            tmp[0] * d2q9_w[0],
            tmp[1] * d2q9_w[1],
            tmp[2] * d2q9_w[2],
            tmp[3] * d2q9_w[3],
            tmp[4] * d2q9_w[4],
            tmp[5] * d2q9_w[5],
            tmp[6] * d2q9_w[6],
            tmp[7] * d2q9_w[7],
            tmp[8] * d2q9_w[8]
    };

    /*
     * avoid recalculating the inverse of tau
     */

    const scalar_t tau_inv = 1.0 / tau;

    /*
     * finally apply the collision operator
     */

    f[0] = f[0] - (tau_inv * (f[0] - feq[0]));
    f[1] = f[1] - (tau_inv * (f[1] - feq[1]));
    f[2] = f[2] - (tau_inv * (f[2] - feq[2]));
    f[3] = f[3] - (tau_inv * (f[3] - feq[3]));
    f[4] = f[4] - (tau_inv * (f[4] - feq[4]));
    f[5] = f[5] - (tau_inv * (f[5] - feq[5]));
    f[6] = f[6] - (tau_inv * (f[6] - feq[6]));
    f[7] = f[7] - (tau_inv * (f[7] - feq[7]));
    f[8] = f[8] - (tau_inv * (f[8] - feq[8]));
}

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
 * @param f the fluid forces at time t (at the moment)
 * @param f_next a memory region as big as f which is used to write the simulation results into
 * @param tau TODO document better what tau is
 * @param width the width of the field
 * @param length the length of the memory region (f/f_next) (second dimension) which is equal to with*height of the field
 */
template<typename scalar_t>
__global__ void
lettuce_cuda_stream_and_collide_kernel(const scalar_t *f, scalar_t *f_next, scalar_t tau, c_index_t width, c_index_t height, c_index_t length)
{
    // pre calculate the vertical and horizontal indices before streaming
    const auto horizontal_index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto vertical_index   = blockIdx.y * blockDim.y + threadIdx.y;

    // pre calculate the vertical and horizontal offsets before streaming
    const auto &horizontal_m_offset = horizontal_index;
    const auto vertical_m_offset    = vertical_index * width;

    // pre calculate the vertical and horizontal offsets after streaming
    const auto vertical_t_offset   = ((vertical_index == 0)            ? height-1 : (vertical_index - 1)) * width;
    const auto vertical_b_offset   = (((vertical_index + 1) == height) ? 0        : (vertical_index + 1)) * width;
    const auto horizontal_l_offset = (horizontal_index == 0)           ? width-1  : (horizontal_index - 1);
    const auto horizontal_r_offset = ((horizontal_index + 1) == width) ? 0        : (horizontal_index + 1);

    // pre calculate the current index
    const auto index = vertical_m_offset + horizontal_m_offset;

    /*
     * standard stream & read
     */

    scalar_t next[9];
    {
        // constants to visualize the streaming better
        // maybe this is not necessary and we replace the values inline
        constexpr index_t mm = 0;
        constexpr index_t rm = 1;
        constexpr index_t mb = 2;
        constexpr index_t lm = 3;
        constexpr index_t mt = 4;
        constexpr index_t rb = 5;
        constexpr index_t lb = 6;
        constexpr index_t lt = 7;
        constexpr index_t rt = 8;
        constexpr index_t opposite[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

        // center force is trivial as it stays in place
        next[mm] = f[index];

        // the index from which to stream from is calculated by:
        // - a dimensional offset (which is calculated by iteration)
        //   [by using an iterator bypass some multiplications]
        // - a relative horizontal offset (corresponding to the dimension)
        // - a relative vertical offset (corresponding to the dimension)
        auto dim_offset_it = length; next[opposite[rm]] = f[dim_offset_it + horizontal_r_offset + vertical_m_offset];
        dim_offset_it += length;     next[opposite[mb]] = f[dim_offset_it + horizontal_m_offset + vertical_b_offset];
        dim_offset_it += length;     next[opposite[lm]] = f[dim_offset_it + horizontal_l_offset + vertical_m_offset];
        dim_offset_it += length;     next[opposite[mt]] = f[dim_offset_it + horizontal_m_offset + vertical_t_offset];
        dim_offset_it += length;     next[opposite[rb]] = f[dim_offset_it + horizontal_r_offset + vertical_b_offset];
        dim_offset_it += length;     next[opposite[lb]] = f[dim_offset_it + horizontal_r_offset + vertical_t_offset];
        dim_offset_it += length;     next[opposite[lt]] = f[dim_offset_it + horizontal_l_offset + vertical_t_offset];
        dim_offset_it += length;     next[opposite[rt]] = f[dim_offset_it + horizontal_l_offset + vertical_b_offset];
    }

    /*
     * collide & write
     */

    {
        d3q9_quadratic_equilibrium_collision<scalar_t>(&(next[0]), tau);

        // the writing index is trivial as it is the same relative index in each dimension
        // [by using an iterator bypass some multiplications]
        auto index_it = index; f_next[index_it] = next[0];
        index_it += length;    f_next[index_it] = next[1];
        index_it += length;    f_next[index_it] = next[2];
        index_it += length;    f_next[index_it] = next[3];
        index_it += length;    f_next[index_it] = next[4];
        index_it += length;    f_next[index_it] = next[5];
        index_it += length;    f_next[index_it] = next[6];
        index_it += length;    f_next[index_it] = next[7];
        index_it += length;    f_next[index_it] = next[8];
    }
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
