"""
Utility functions.
"""

import inspect
import torch


class LettuceException(Exception):
    pass


class LettuceWarning(UserWarning):
    pass


class InefficientCodeWarning(LettuceWarning):
    pass


class ExperimentalWarning(LettuceWarning):
    pass


def get_subclasses(classname, module):
    for name, obj in inspect.getmembers(module):
        if hasattr(obj, "__bases__") and classname in obj.__bases__:
            yield obj


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
        2: [-1/2, 1/2, 0, 0, 0, 0],
        4: [1/12, -2/3, 2/3, -1/12, 0, 0],
        6: [-1/60, 3/20, -3/4, 3/4, -3/20, 1/60],
    }
    weight = weights.get(order)
    if dim == 2:
        dims = (0, 1)
        stencil = {
            2: [[[0, 1], [0, -1], [0, 0], [0, 0], [0, 0], [0, 0]],
                [[1, 0], [-1, 0], [0, 0], [0, 0], [0, 0], [0, 0]]],
            4: [[[0, 2], [0, 1], [0, -1], [0, -2], [0, 0], [0, 0]],
                [[2, 0], [1, 0], [-1, 0], [-2, 0], [0, 0], [0, 0]]],
            6: [[[0, 3], [0, 2], [0, 1], [0, -1], [0, -2], [0, -3]],
                [[3, 0], [2, 0], [1, 0], [-1, 0], [-2, 0], [-3, 0]]],
        }
        shift = stencil.get(order)
    if dim == 3:
        dims = (0, 1, 2)
        stencil = {
            2: [[[0, 1, 0], [0, -1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[1, 0, 0], [-1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 1], [0, 0, -1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]],
            4: [[[0, 2, 0], [0, 1, 0], [0, -1, 0], [0, -2, 0], [0, 0, 0], [0, 0, 0]],
                [[2, 0, 0], [1, 0, 0], [-1, 0, 0], [-2, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 2], [0, 0, 1], [0, 0, -1], [0, 0, -2], [0, 0, 0], [0, 0, 0]]],
            6: [[[0, 3, 0], [0, 2, 0], [0, 1, 0], [0, -1, 0], [0, -2, 0], [0, -3, 0]],
                [[3, 0, 0], [2, 0, 0], [1, 0, 0], [-1, 0, 0], [-2, 0, 0], [-3, 0, 0]],
                [[0, 0, 3], [0, 0, 2], [0, 0, 1], [0, 0, -1], [0, 0, -2], [0, 0, -3]]]
        }
        shift = stencil.get(order)
    with torch.no_grad():
        out = torch.cat(dim*[f[None,...]])
        for i in range(dim):
            out[i, ...] = (
                weight[0] * f.roll(shifts=shift[i][0], dims=dims) +
                weight[1] * f.roll(shifts=shift[i][1], dims=dims) +
                weight[2] * f.roll(shifts=shift[i][2], dims=dims) +
                weight[3] * f.roll(shifts=shift[i][3], dims=dims) +
                weight[4] * f.roll(shifts=shift[i][4], dims=dims) +
                weight[5] * f.roll(shifts=shift[i][5], dims=dims)
            ) * torch.tensor(1.0/dx, dtype=f.dtype, device=f.device)
    return out

def torch_jacobi(f, p, dx, device, dim, tol_abs=1e-5):
    # f = torch.tensor(f, device=device, dtype=torch.double)
    p = torch.tensor(p, device=device, dtype=torch.double)
    #dx = self.units.convert_length_to_pu(1)
    dx = torch.tensor(dx, device=device, dtype=torch.double)
    error, it = 1, 0
    while error > tol_abs and it < 100000:
        it += 1
        if dim== 2:
            p = (f * (dx ** 2) - (-p.roll(shifts=1, dims=0)
                                  - p.roll(shifts=1, dims=1)
                                  - p.roll(shifts=-1, dims=0)
                                  - p.roll(shifts=-1, dims=1))) * 0.25
            residuum = f - 1 / (dx ** 2) * (-p.roll(shifts=1, dims=0)
                                            - p.roll(shifts=1, dims=1)
                                            - p.roll(shifts=-1, dims=0)
                                            - p.roll(shifts=-1, dims=1)
                                            + 4 * p)
            error = torch.mean(residuum)
        if dim == 3:
            p = (f * (dx ** 2) - (-p.roll(shifts=1, dims=0)
                                  - p.roll(shifts=1, dims=1)
                                  - p.roll(shifts=1, dims=2)
                                  - p.roll(shifts=-1, dims=0)
                                  - p.roll(shifts=-1, dims=1)
                                  - p.roll(shifts=-1, dims=2))) * 1 / 6
            residuum = f - 1 / (dx ** 2) * (-p.roll(shifts=1, dims=0)
                                            - p.roll(shifts=1, dims=1)
                                            - p.roll(shifts=1, dims=2)
                                            - p.roll(shifts=-1, dims=0)
                                            - p.roll(shifts=-1, dims=1)
                                            - p.roll(shifts=-1, dims=2)
                                            + 6 * p)
        error = torch.mean(residuum)

    print(f'Error: {error}')
    return p.detach().cpu().numpy()
