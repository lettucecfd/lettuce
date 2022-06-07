"""
Utility functions.
"""

import inspect
import torch

__all__ = [
    "get_subclasses", 'all_stencils',
    "LettuceException", "LettuceWarning", "InefficientCodeWarning", "ExperimentalWarning",
    "torch_gradient", "torch_jacobi", "grid_fine_to_coarse", "pressure_poisson", "append_axes"
]


def get_subclasses(classname, module):
    for name, obj in inspect.getmembers(module):
        if hasattr(obj, "__bases__") and classname in obj.__bases__:
            yield obj


def all_stencils():
    import lettuce.stencils
    return list(get_subclasses(lettuce.stencils.Stencil, module=lettuce.stencils))


class LettuceException(Exception):
    pass


class LettuceWarning(UserWarning):
    pass


class InefficientCodeWarning(LettuceWarning):
    pass


class ExperimentalWarning(LettuceWarning):
    pass


def torch_gradient(f, dx=1, order=2):
    """
    Function to calculate the first derivative of tensors.
    Orders O(h²); O(h⁴); O(h⁶) are implemented.

    Notes
    -----
    See [1]_. The function only works for periodic domains

    References
    ----------
    .. [1]  Fornberg B. (1988) Generation of Finite Difference Formulas on
        Arbitrarily Spaced Grids,
        Mathematics of Computation 51, no. 184 : 699-706.
        `PDF <http://www.ams.org/journals/mcom/1988-51-184/
        S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf>`_.
    """
    dim = f.ndim
    weights = {
        2: [-1 / 2, 1 / 2, 0, 0, 0, 0],
        4: [1 / 12, -2 / 3, 2 / 3, -1 / 12, 0, 0],
        6: [-1 / 60, 3 / 20, -3 / 4, 3 / 4, -3 / 20, 1 / 60],
    }
    weight = weights.get(order)
    if dim == 2:
        dims = (0, 1)
        stencil = {
            2: [[[1, 0], [-1, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 1], [0, -1], [0, 0], [0, 0], [0, 0], [0, 0]]],
            4: [[[2, 0], [1, 0], [-1, 0], [-2, 0], [0, 0], [0, 0]],
                [[0, 2], [0, 1], [0, -1], [0, -2], [0, 0], [0, 0]]],
            6: [[[3, 0], [2, 0], [1, 0], [-1, 0], [-2, 0], [-3, 0]],
                [[0, 3], [0, 2], [0, 1], [0, -1], [0, -2], [0, -3]]],
        }
        shift = stencil.get(order)
    if dim == 3:
        dims = (0, 1, 2)
        stencil = {
            2: [[[1, 0, 0], [-1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 1, 0], [0, -1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 1], [0, 0, -1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]],
            4: [[[2, 0, 0], [1, 0, 0], [-1, 0, 0], [-2, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 2, 0], [0, 1, 0], [0, -1, 0], [0, -2, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 2], [0, 0, 1], [0, 0, -1], [0, 0, -2], [0, 0, 0], [0, 0, 0]]],
            6: [[[3, 0, 0], [2, 0, 0], [1, 0, 0], [-1, 0, 0], [-2, 0, 0], [-3, 0, 0]],
                [[0, 3, 0], [0, 2, 0], [0, 1, 0], [0, -1, 0], [0, -2, 0], [0, -3, 0]],
                [[0, 0, 3], [0, 0, 2], [0, 0, 1], [0, 0, -1], [0, 0, -2], [0, 0, -3]]]
        }
        shift = stencil.get(order)
    with torch.no_grad():
        out = torch.cat(dim * [f[None, ...]])
        for i in range(dim):
            out[i, ...] = (
                                  weight[0] * f.roll(shifts=shift[i][0], dims=dims) +
                                  weight[1] * f.roll(shifts=shift[i][1], dims=dims) +
                                  weight[2] * f.roll(shifts=shift[i][2], dims=dims) +
                                  weight[3] * f.roll(shifts=shift[i][3], dims=dims) +
                                  weight[4] * f.roll(shifts=shift[i][4], dims=dims) +
                                  weight[5] * f.roll(shifts=shift[i][5], dims=dims)
                          ) * torch.tensor(1.0 / dx, dtype=f.dtype, device=f.device)
    return out


def grid_fine_to_coarse(lattice, f_fine, tau_fine, tau_coarse):
    if f_fine.shape.__len__() == 3:
        f_eq = lattice.equilibrium(lattice.rho(f_fine[:, ::2, ::2]), lattice.u(f_fine[:, ::2, ::2]))
        f_neq = f_fine[:, ::2, ::2] - f_eq
    if f_fine.shape.__len__() == 4:
        f_eq = lattice.equilibrium(lattice.rho(f_fine[:, ::2, ::2, ::2]), lattice.u(f_fine[:, ::2, ::2, ::2]))
        f_neq = f_fine[:, ::2, ::2, ::2] - f_eq
    f_coarse = f_eq + 2 * tau_coarse / tau_fine * f_neq
    return f_coarse


def torch_jacobi(f, p, dx, device, dim, tol_abs=1e-10, max_num_steps=100000):
    """Jacobi solver for the Poisson pressure equation"""

    ## transform to torch.tensor
    # p = torch.tensor(p, device=device, dtype=torch.double)
    # dx = torch.tensor(dx, device=device, dtype=torch.double)
    error, it = 1, 0
    while error > tol_abs and it < max_num_steps:
        it += 1
        if dim == 2:
            # Difference quotient for second derivative O(h²) for index i=0,1
            p = (f * (dx ** 2) - (p.roll(shifts=1, dims=0)
                                  + p.roll(shifts=1, dims=1)
                                  + p.roll(shifts=-1, dims=0)
                                  + p.roll(shifts=-1, dims=1))) * -1 / 4
            residuum = f - (p.roll(shifts=1, dims=0)
                            + p.roll(shifts=1, dims=1)
                            + p.roll(shifts=-1, dims=0)
                            + p.roll(shifts=-1, dims=1)
                            - 4 * p) / (dx ** 2)
        if dim == 3:
            # Difference quotient for second derivative O(h²) for index i=0,1,2
            p = (f * (dx ** 2) - (p.roll(shifts=1, dims=0)
                                  + p.roll(shifts=1, dims=1)
                                  + p.roll(shifts=1, dims=2)
                                  + p.roll(shifts=-1, dims=0)
                                  + p.roll(shifts=-1, dims=1)
                                  + p.roll(shifts=-1, dims=2))) * -1 / 6
            residuum = f - (p.roll(shifts=1, dims=0)
                            + p.roll(shifts=1, dims=1)
                            + p.roll(shifts=1, dims=2)
                            + p.roll(shifts=-1, dims=0)
                            + p.roll(shifts=-1, dims=1)
                            + p.roll(shifts=-1, dims=2)
                            - 6 * p) / (dx ** 2)
        # Error is defined as the mean value of the residuum
        error = torch.mean(residuum ** 2)
    return p


def pressure_poisson(units, u, rho0, tol_abs=1e-10, max_num_steps=100000):
    """
    Solve the pressure poisson equation using a jacobi scheme.

    Parameters
    ----------
    units : lettuce.UnitConversion
        The flow instance.
    u : torch.Tensor
        The velocity tensor.
    rho0 : torch.Tensor
        Initial guess for the density (i.e., pressure).
    tol_abs : float
        The tolerance for pressure convergence.


    Returns
    -------
    rho : torch.Tensor
        The converged density (i.e., pressure).
    """
    # convert to physical units
    dx = units.convert_length_to_pu(1.0)
    u = units.convert_velocity_to_pu(u)
    p = units.convert_density_lu_to_pressure_pu(rho0)

    # compute laplacian
    with torch.no_grad():
        u_mod = torch.zeros_like(u[0])
        dim = u.shape[0]
        for i in range(dim):
            for j in range(dim):
                derivative = torch_gradient(torch_gradient(u[i] * u[j], dx)[i], dx)[j]
                u_mod -= derivative
    # TODO(@MCBs): still not working in 3D

    p_mod = torch_jacobi(
        u_mod,
        p[0],
        dx,
        units.lattice.device,
        dim=units.lattice.D,
        tol_abs=tol_abs,
        max_num_steps=max_num_steps
    )[None, ...]

    return units.convert_pressure_pu_to_density_lu(p_mod)


def append_axes(array, n):
    index = (Ellipsis, ) + (None, ) * n
    return array[index]
