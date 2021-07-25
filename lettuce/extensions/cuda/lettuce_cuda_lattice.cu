#include "lettuce_cuda_stencil.cu"

using index_t = unsigned int;

namespace lattice
{
    template<typename scalar_t, index_t Q>
    __device__ __forceinline__ void
    rho(const scalar_t *f, scalar_t *rho_f)
    {
        static_assert(Q > 0, "rho can not implemented for zero distributions!");

        rho_f = f[0];

#pragma unroll
        for (index_t q = 1; q < Q; ++q)
            rho_f += f[q];
    }

    template<typename scalar_t, index_t D, index_t Q, stencil<scalar_t, D, Q> Stencil>
    __device__ __forceinline__ void
    j(const scalar_t *f, scalar_t *j_f)
    {
        static_assert(Q > 0, "rho can not implemented for zero distributions!");

#pragma unroll
        for (index_t d = 0; d < D; ++d)
            j_f[d] = Stencil.e[0][d] * f[0];

#pragma unroll
        for (index_t d = 0; d < D; ++d)
#pragma unroll
                for (index_t q = 1; q < Q; ++q)
                    j_f[d] += Stencil.e[q][d] * f[q];
    }
}
