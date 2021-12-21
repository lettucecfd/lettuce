
from copy import deepcopy
from typing import Sequence
import numpy as np
from numpy import typing as npt
import torch
from typing import Tuple


from ..util import LettuceException
from ..lattices import Lattice
from ..boundary import Boundary


class Flow:
    """

    Attributes
    ----------
    boundaries : Sequence[Boundary]
    """
    def __init__(self):
        self.grid = NotImplemented
        self.units = NotImplemented
        self.initial_solution = NotImplemented

    @property
    def boundaries(self) -> Sequence[Boundary]:
        return []

    def compute_initial_f(self, lattice: Lattice) -> torch.Tensor:
        grid = self.grid
        p, u = self.initial_solution(grid)
        if not list(p.shape) == [1] + list(grid[0].shape):
            raise LettuceException(
                f"Wrong dimension of initial pressure field. "
                f"Expected {[1] + list(grid[0].shape)}, "
                f"but got {list(p.shape)}."
            )
        if not list(u.shape) == [lattice.D] + list(grid[0].shape):
            raise LettuceException(
                "Wrong dimension of initial velocity field."
                f"Expected {[lattice.D] + list(grid[0].shape)}, "
                f"but got {list(u.shape)}."
            )
        u = lattice.convert_to_tensor(self.units.convert_velocity_to_lu(u))
        rho = lattice.convert_to_tensor(self.units.convert_pressure_pu_to_density_lu(p))
        return lattice.equilibrium(rho, lattice.convert_to_tensor(u))

    def compute_masks(self, lattice: Lattice) -> Tuple[torch.Tensor, torch.Tensor]:
        grid = self.grid
        grid_shape = grid[0].shape
        f_shape = [lattice.Q, *grid_shape]
        no_stream_mask = torch.zeros(f_shape, device=lattice.device, dtype=torch.bool)
        no_collision_mask = torch.zeros(grid_shape, device=lattice.device, dtype=torch.bool)

        # Apply boundaries
        # boundaries = deepcopy(self.boundaries)  # store locally to keep the flow free from the boundary state
        for boundary in self.boundaries:
            boundary.update_mask(lattice, self.grid)
            if hasattr(boundary, "make_no_collision_mask"):
                no_collision_mask = no_collision_mask | boundary.make_no_collision_mask(f_shape)
            if hasattr(boundary, "make_no_stream_mask"):
                no_stream_mask = no_stream_mask | boundary.make_no_stream_mask(f_shape)

        return no_stream_mask, no_collision_mask




class DomainDecomposedFlow:
    def __init__(self, flow: Flow, masks: Sequence[npt.NDArray[bool]]):
        grid = flow.grid
        if not all(mask.shape == grid.shape for mask in masks):
            raise ValueError(f"At least one mask shape did not match grid shape ({grid.shape})")
        self.flow = flow
        self.masks = masks



    flow = TaylorGreenVortex3D(...)

    # manual decomposition of the flow domain into rectangular/hexagonal domains
    mask0 = flow.grid.x < 0.5
    mask1 = flow.grid.x >= 0.5

    # set up a distributed flow object
    decomposed = DomainDecomposedFlow(flow, masks=(mask0, mask1))

    # send part of the domain to a device
    decomposed.set_device(0, "cuda:0")

    # refine the domain if needed (this is important to do here; big flows will not fit on one node)
    decomposed.refine_domain(0, refinement_level=4)

    # send part of the domain to a different device
    decomposed.set_device(1, "cuda:1")

    # refine this part more coarsely
    decomposed.refine_domain(1, refinement_level=3)

    # set up the simulation with the decomposed flow
    simulation = Simulation(decomposed, ...)
