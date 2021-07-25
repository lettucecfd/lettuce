using index_t = unsigned int;

namespace
{
    template<typename scalar_t>
    constexpr scalar_t sqrt3 = 1.7320508075688772935274463415058723669428052538103806280558069794;
}

template<typename scalar_t, index_t D, index_t Q>
struct stencil
{
    // TODO document e
    scalar_t e[Q][D]{};

    /// weights to compensate distance to other nodes
    scalar_t w[Q]{};

    // TODO document cs
    scalar_t cs{};

    /// opposite distribution mapping
    index_t opposite[Q]{};
};

template<typename scalar_t>
constexpr stencil<scalar_t, 2, 9>
        d2q9
        {
                .e={{0.0,  0.0},
                    {1.0,  0.0},
                    {0.0,  1.0},
                    {-1.0, 0.0},
                    {0.0,  -1.0},
                    {1.0,  1.0},
                    {-1.0, 1.0},
                    {-1.0, -1.0},
                    {1.0,  -1.0}},
                .w={4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0},
                .cs=1.0 / sqrt3<scalar_t>,
                .opposite={0, 3, 4, 1, 2, 7, 8, 5, 6}
        };

template<typename scalar_t>
constexpr stencil<scalar_t, 1, 3>
        d1q3
        {
                .e = {{0.0},
                      {1.0},
                      {-1.0}},
                .w = {2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0},
                .cs = 1.0 / sqrt3<scalar_t>,
                .opposite ={0.0, 2.0, 1.0}
        };
