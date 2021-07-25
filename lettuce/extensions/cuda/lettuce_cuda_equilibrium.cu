#include "lettuce_cuda_stencil.cu"

using index_t = unsigned int;

template<typename scalar_t, index_t D, index_t Q, stencil<scalar_t, D, Q> Stencil, typename PerNodeVariables>
struct equilibrium
{
    PerNodeVariables
    (*per_node)(const scalar_t *f){};

    scalar_t
    (*equilibrium)(const scalar_t *f, const PerNodeVariables *per_node_variables){};
};

struct QuadraticEquilibriumPerNodeVariables
{

};

template<typename scalar_t, index_t D, index_t Q, stencil<scalar_t, D, Q> Stencil>
constexpr equilibrium<scalar_t, D, Q, Stencil, QuadraticEquilibriumPerNodeVariables>
        quadratic_equilibrium
        {
                .per_node=[](const scalar_t *f)
                {},
                .equilibrium=[](const scalar_t *f, const QuadraticEquilibriumPerNodeVariables *per_node_variables)
                {}
        };
