using index_t = unsigned int;

template<typename scalar_t, index_t D, index_t Q, typename stencil_t>
struct standard_streaming
{
public:

    const index_t offset[3][D];
    const index_t index[D];

    standard_streaming(size_t *resolution, size_t *block_id, size_t *block_dim, size_t *thread_idx)
    {
        // initialize index
        {
#pragma unroll
            for (index_t d = 0; d < D; ++d)
            {
                index[d] = block_id[d] * block_dim[d] + thread_idx[d];
            }
        }

        // initialize offset
        {
            offset[0] = index[0];

            index_t dimension_offset = 1;
#pragma unroll
            for (index_t d = 1; d < D; ++d)
            {
                dimension_offset *= resolution[d - 1];
                offset[d] = index[d] * dimension_offset;
            }
        }
    }

public:

    constexpr void
    read(const scalar_t *f, scalar_t *f_reg)
    {
    }

    constexpr void
    write(const scalar_t *f_reg, scalar_t *f)
    {
    }
};
