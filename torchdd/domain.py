__all__ = ["Domain", "BoxDomain"]

import copy
from typing import Sequence, Union
import torch
import torch.distributed as dist
import os
import numpy as np
#from math import prod

# TODO: (v0.1.5) Add Endpoint (TRUE/FALSE)
# TODO: (v0.1.5) Concept for ghostcells
# TODO: (v0.2.0) Concept for periodicity (see. mpi4py intercomm/create_cart)
# TODO: (v0.2.0) Add META Attribute

class Domain:
    def __init__(self,
                 mpi_rank: int = 0,
                 meta: "DomainMeta" = None):
        self._meta = meta
        self._mpi_rank = mpi_rank

    @property
    def mpi_rank(self) -> int:
        return self._mpi_rank

    @mpi_rank.setter
    def mpi_rank(self, new_rank: int):
        # if new_rank >= self.mpi_size:
        #     raise ValueError(f"mpi_rank ({new_rank}) has to be smaller than mpi_size ({self.mpi_size()})")
        self._mpi_rank = new_rank

    @property
    def rank(self):
        return self.mpi_rank

    @rank.setter
    def rank(self, new_rank: int):
        self.mpi_rank = new_rank

    @property
    def meta(self) -> Union["DomainMeta", None]:
        if hasattr(self, "_meta"):
            return self._meta
        return None

    @meta.setter
    def meta(self, meta: Union["DomainMeta", None]):
        self._meta = meta

    @property
    def dim(self) -> int:
        raise NotImplementedError()

    @property
    def tensor_shape(self) -> torch.Size:
        raise NotImplementedError()

    @property
    def ghosts(self):
        raise NotImplementedError()

    @property
    def grid(self, as_numpy=False, with_ghost=False) -> Union[Sequence[torch.Tensor], Sequence[np.ndarray]]:
        raise NotImplementedError()

    def contains(self, points: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class BoxDomain(Domain):
    """(Hyper-)Rectangular domain of dimension `D`

    Attributes
    ----------
    lower : torch.Tensor
        lower bound of the box
    ... # TODO
    n_ghost : Sequence[Sequence[int]]
        Int tensor of shape `(D, 2)`. Number of ghost nodes below and above domain.
        The first and second column denotes the number of lower und upper ghost cells
        along each axis, respectively.
    """
    def __init__(
            self,
            lower: torch.Tensor,
            upper: torch.Tensor,
            resolution: Sequence[int],
            mpi_rank: int = 0,
            cubic_cells: bool = True,
            n_ghost: Sequence[Sequence[int]] = None,
            device: str or torch.device = None,
            dtype: torch.dtype = None,
            endpoint: bool = True,
    ):
        super().__init__(mpi_rank=mpi_rank)
        self.endpoint = endpoint
        self.device = device if device is not None else lower.device
        self.dtype = dtype if dtype is not None else lower.dtype
        self._cubic_cells = cubic_cells
        self._lower = lower.to(device=device,dtype=dtype)
        self._upper = upper.to(device=device,dtype=dtype)
        self.resolution = resolution
        self.n_ghost = n_ghost
        self._coord = [0] * self.dim

    @property
    def coord(self) ->  Sequence[int]:
        return self._coord

    @coord.setter
    def coord(self, new_coord: Sequence[int]):
        self._coord = new_coord

    @property
    def dim(self) -> int:
        return self._lower.shape[0]

    @property
    def lower(self) -> torch.Tensor:
        return self._lower

    @property
    def upper(self) -> torch.Tensor:
        return self._upper

    @property
    def resolution(self) -> torch.Size:
        return torch.Size(self._resolution)

    @resolution.setter
    def resolution(self, new_resolution: Sequence[int]):
        if self._cubic_cells:
            cell_lengths = torch.tensor([length / res for length, res in zip(self.lengths, new_resolution)])
            if not all(torch.allclose(cell_length, cell_lengths[0]) for cell_length in cell_lengths):
                raise ValueError(
                    f"Quad domain with domain lengths {[x for x in self.lengths]} cannot be "
                    f"represented by {[x for x in new_resolution]} cubic cells. "
                    f"Cell lengths {[x for x in cell_lengths]} would be non-cubic."
                )
        self._resolution = torch.Size(new_resolution)
        # TODO: (v0.2.0) Implement meta data
        # if self._meta is not None:
        #     self._meta.update(self)


    @property
    def shape(self) -> torch.Size:
        return torch.Size([res + 1 for res in self.resolution])

    @property
    def tensor_shape(self) -> torch.Size:
        return torch.Size([length + ghost.sum() for length, ghost in zip(self.shape, self.n_ghost)])

    @shape.setter
    def shape(self, new_shape: Sequence[int]):
        self.resolution = [s - 1 for s in new_shape]

    @property
    def lengths(self) -> torch.Tensor:
        return self.upper - self.lower

    @property
    def cell_lengths(self) -> torch.Tensor:
        modifier = (torch.tensor(self.resolution, device=self.upper.device) if self.endpoint
                    else torch.tensor(self.resolution, device=self.upper.device)+1)
        return (self.upper - self.lower)/modifier

    @property
    def n_cells(self) -> int:
        return torch.prod(torch.tensor(self.resolution, device=self.upper.device), dtype= int)

    @property
    def n_points(self) -> int:
        return torch.prod(torch.tensor(self.shape, device=self.upper.device), dtype= int)

    @property
    def n_ghost(self) -> torch.IntTensor:
        return self._n_ghost.to(self.upper.device)

    @n_ghost.setter
    def n_ghost(self, n_ghost: Union[Sequence[Sequence[int]], None]):
        num = torch.zeros(self.dim, 2, dtype=torch.int) if n_ghost is None else torch.tensor(n_ghost, dtype=torch.int)
        if num.shape != (self.dim, 2):
            raise ValueError("n_ghost needs to have shape `(dim, 2)`")
        self._n_ghost = num

    def refine(self, refinement_level: int, inplace=True) -> "BoxDomain":
        new_resolution = tuple(res * 2**refinement_level for res in self.resolution)
        domain = self if inplace else copy.deepcopy(self)
        domain.resolution = new_resolution
        return domain

    def coarsen(self, coarsen_level: int, inplace=True) -> "BoxDomain":
        if any(n % 2**coarsen_level != 0 for n in self.resolution):
            raise ValueError(
                f"BoxDomain with resolution {self.resolution} cannot be coarsened by {coarsen_level} levels. "
                f"All dimensions need to be multiples of 2**coarsen_level = {2**coarsen_level}."
            )
        new_resolution = tuple(res // 2**coarsen_level for res in self.resolution)
        domain = self if inplace else copy.deepcopy(self)
        domain.resolution = new_resolution
        return domain


    def grid(self, as_numpy=False, with_ghost=False) -> Union[Sequence[torch.Tensor], Sequence[np.ndarray]]:
        resolution = (torch.tensor(self.resolution, device=self.device) + self.n_ghost.sum(dim=1) if with_ghost else self.resolution)
        _lower = (self.lower - self.cell_lengths*self.n_ghost[:,0] if with_ghost else self.lower).to(device='cpu')
        _upper = (self.upper + self.cell_lengths*self.n_ghost[:,1] if with_ghost else self.upper).to(device='cpu')
        cell_boundaries_1d = (torch.tensor(np.linspace(lower, upper, step + 1, endpoint=self.endpoint), dtype=self.dtype, device=self.device)
                              for lower, upper, step in zip(_lower, _upper, resolution))
        grid = torch.meshgrid(*cell_boundaries_1d)#, indexing='ij') #TODO: indexing is not compatible with every PyTorch version

        if as_numpy:
            return tuple(g.detach().cpu().numpy() for g in grid)
        else:
            return grid


    def contains(self, points: torch.Tensor) -> torch.Tensor:
        return ((points >= self.lower) & (points <= self.upper)).all(dim=-1)


    def split(self, *coordinates: float, n_ghost: int = 1, dim: int = 0) -> Sequence["BoxDomain"]:
        grid_points = []
        bounds = [self.lower]
        upper_virtual = self.upper.clone() - self.cell_lengths + self.cell_lengths * self.n_ghost[:, 1]
        for coordinate in sorted(coordinates):
            point_fraction = ((coordinate - self.lower[dim]) / self.lengths[dim]).item()
            float_index = self.resolution[dim] * point_fraction
            int_index = round(float_index)
            # if int_index < 1 or int_index >= self.resolution[dim]:
            #     raise ValueError(
            #         f"Cannot split {self} at coordinate {coordinate} along dimension {dim}."
            #         f"Coordinate not contained in box."
            #     )
            bound = self.lower + int_index * self.cell_lengths
            if abs(int_index - float_index) > 1e-5:
                raise ValueError(
                    f"Coordinate {coordinate:.8f} too far from closest grid point  "
                    f"(at {bound[dim]:.8f}) to perform split."
                )
            grid_points.append(int_index-sum(grid_points))
            bounds.append(bound)
        grid_points.append(self.resolution[dim] - sum(grid_points) + self.n_ghost[0, 1])
        bounds.append(self.upper.clone() - self.cell_lengths + self.cell_lengths * self.n_ghost[:, 1])
        domains = []
        for i in range(len(bounds) - 1):
            lower_point, upper_point = bounds[i], bounds[i+1]
            lower = self.lower[:].clone()
            lower[dim] = lower_point[dim]
            upper = upper_virtual.clone()
            upper[dim] = upper_point[dim]
            resolution = torch.tensor(self.resolution, device=self.upper.device).clone()
            resolution[dim] = grid_points[i]
            domain = BoxDomain(
                lower=lower,
                upper=upper,
                resolution=resolution,
                mpi_rank=self.mpi_rank,
                cubic_cells=self._cubic_cells,
            )
            domain.coord[0] = i #TODO: version 1 (only for first dimension)
            domains.append(domain)
        return tuple(domains)


    def __str__(self):
        bounds_string = [f'[{a:.1f},{b:.1f}]' for a, b in zip(self.lower, self.upper)]
        res_string = 'x'.join(str(r) for r in self.resolution)
        bounds_string = 'x'.join(bounds_string)
        return f"BoxDomain(bounds={bounds_string}; res={res_string}; on rank {self.rank}; with coord {self.coord})"

